import os
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR

from l3c_baselines.dataloader import segment_iterator
from l3c_baselines.utils import Logger, log_progress, log_debug, log_warn, log_fatal
from l3c_baselines.utils import custom_load_model, noam_scheduler, LinearScheduler
from l3c_baselines.utils import Configure, DistStatistics, rewards2go, downsample
from l3c_baselines.utils import EpochManager, GeneratorBase, Logger
from l3c_baselines.dataloader import AnyMDPDataSet

import gym
import numpy
from gym.envs.toy_text.frozen_lake import generate_random_map
from l3c.anymdp import AnyMDPTaskSampler


def string_mean_var(downsample_length, res):
    string=""
    for i, (xm,xb) in enumerate(zip(res["mean"], res["bound"])):
        string += f'{downsample_length * i}\t{xm}\t{xb}\n'
    return string

@EpochManager
class AnyMDPEpoch:
    def __init__(self, **kwargs):
        for key in kwargs:
            setattr(self, key, kwargs[key])
        self.DataType=AnyMDPDataSet
        if(self.is_training):
            self.logger_keys = ["learning_rate", 
                        "loss_worldmodel_state", 
                        "loss_worldmodel_reward", 
                        "loss_policymodel",
                        "entropy"]
            self.stat = DistStatistics(*self.logger_keys[1:])
            self.reduce = 1
        else:
            self.logger_keys = ["validation_state_pred", 
                        "validation_reward_pred", 
                        "validation_policy"]
            self.stat = DistStatistics(*self.logger_keys)
            self.reduce = None
            if(self.config.has_attr("downsample_length")):
                self.downsample_length = self.config.downsample_length
            else:
                self.downsample_length = 100

    def compute(self, sarr, baarr, laarr, rarr, 
                        epoch_id=-1, 
                        batch_id=-1):
        """
        Defining the computation function for each batch
        """
        if(self.is_training):
            assert self.optimizer is not None, "optimizer is required for training"

        losses = []
        r2goarr = rewards2go(rarr)
        for sub_idx, states, bactions, lactions, rewards, r2go in segment_iterator(
                    self.config.seq_len, self.config.seg_len, self.device, 
                    (sarr, 1), baarr, laarr, rarr, (r2goarr, 1)):
            loss = self.model.module.sequential_loss(
                    r2go[:, :-1], # Prompts
                    states, 
                    rewards, # Rewards 
                    bactions, 
                    lactions, 
                    state_dropout=0.20, 
                    reward_dropout=0.20,
                    use_loss_weight=self.is_training,
                    reduce_dim=self.reduce) # Do not use loss weight for evaluation
            losses.append(loss)
            if(self.is_training):
                syn_loss = (self.config.lossweight_worldmodel_states * loss["wm-s"]
                        + self.config.lossweight_worldmodel_rewards * loss["wm-r"]
                        + self.config.lossweight_entropy * loss["ent"]
                        + self.config.lossweight_policymodel * loss["pm"]
                        + self.config.lossweight_l2 * loss["causal-l2"])
                if(self.scaler is not None):
                    self.scaler.scale(syn_loss).backward()
                else:
                    syn_loss.backward()
                self.stat.gather(self.device,
                    loss_worldmodel_state = loss["wm-s"] / loss["count"],
                    loss_worldmodel_reward = loss["wm-r"] / loss["count"],
                    loss_policymodel = loss["pm"] / loss["count"],
                    entropy = -loss["ent"] / loss["count"],
                    count = loss["count"])
        if(self.is_training):
            stat_res = self.stat()
            if(self.logger is not None):
                self.logger(self.optimizer.param_groups[0]['lr'],
                        stat_res["loss_worldmodel_state"]["mean"], 
                        stat_res["loss_worldmodel_reward"]["mean"], 
                        stat_res["loss_policymodel"]["mean"], 
                        stat_res["entropy"]["mean"],
                        epoch=epoch_id,
                        iteration=batch_id)
        else:
            loss_wm_s = torch.cat([loss["wm-s"] / loss["count"] for loss in losses], dim=1)
            loss_wm_r = torch.cat([loss["wm-r"] / loss["count"] for loss in losses], dim=1)
            loss_pm = torch.cat([loss["pm"] / loss["count"] for loss in losses], dim=1)
            counts = torch.cat([loss["count"] for loss in losses], dim=1)

            bsz = loss_wm_s.shape[0]
            seg_num = loss_wm_s.shape[1] // self.downsample_length
            valid_seq_len = seg_num * self.downsample_length

            loss_wm_s = torch.mean(loss_wm_s[:, :valid_seq_len].view(bsz, seg_num, -1), dim=-1)
            loss_wm_r = torch.mean(loss_wm_r[:, :valid_seq_len].view(bsz, seg_num, -1), dim=-1)
            loss_pm = torch.mean(loss_pm[:, :valid_seq_len].view(bsz, seg_num, -1), dim=-1)
            counts = torch.mean(counts[:, :valid_seq_len].view(bsz, seg_num, -1), dim=-1)

            for i in range(bsz):
                self.stat.gather(self.device,
                        validation_state_pred=loss_wm_s[i], 
                        validation_reward_pred=loss_wm_r[i], 
                        validation_policy=loss_pm[i],
                        count=counts[i])
            
    def epoch_end(self, epoch_id):
        if(not self.is_training):
            stat_res = self.stat()
            if(self.logger is not None):
                self.logger(stat_res["validation_state_pred"]["mean"], 
                        stat_res["validation_reward_pred"]["mean"], 
                        stat_res["validation_policy"]["mean"],
                        epoch=epoch_id)
            if(self.extra_info is not None):
                if(self.extra_info.lower() == 'validate' and self.main):
                    if not os.path.exists(self.config.output):
                        os.makedirs(self.config.output)
                    for key_name in stat_res:
                        res_text = string_mean_var(self.downsample_length, stat_res[key_name])
                        file_path = f'{self.config.output}/result_{key_name}.txt'
                        if os.path.exists(file_path):
                            os.remove(file_path)
                        with open(file_path, 'w') as f_model:
                            f_model.write(res_text)


class AnyMDPGenerator(GeneratorBase):
    def preprocess(self):
        if(self.config.env.lower().find("lake") >= 0):
            self.env = gym.make(
                'FrozenLake-v1', 
                map_name=self.config.env.replace("lake", ""), 
                is_slippery=True, 
                max_episode_steps=1000)
        elif(self.config.env.lower().find("anymdp") >= 0):
            self.env = gym.make("anymdp-v0", max_steps=self.max_steps)
            dims = self.config.env.replace("anymdp", "").split("x")
            task = AnyMDPTaskSampler(int(dims[0]), int(dims[1]))
            self.env.set_task(task)
        else:
            log_fatal("Unsupported environment:", self.config.env)
        logger_keys = ["reward", "state_prediction", "reward_prediction", "success_rate"]

        self.stat = DistStatistics(*logger_keys)
        self.logger = Logger("steps"，
                            *logger_keys, 
                            on=self.main, 
                            use_tensorboard=False)
        
    def in_context_learn_from_teacher(self):
        for folder in os.listdir(self.config.data_root):
            folder_path = os.path.join(self.config.data_root, folder)
            
            if os.path.isdir(folder_path):
                states = numpy.load(os.path.join(folder_path, 'observations.npy'))
                actions = numpy.load(os.path.join(folder_path, 'actions_behavior.npy'))
                rewards = numpy.load(os.path.join(folder_path, 'rewards.npy'))
                states = states.astype(numpy.int32)
                actions = actions.astype(numpy.int32)
                rewards = rewards.astype(numpy.float32)
                self.model.module.in_context_learn(
                    None,
                    states,
                    actions,
                    rewards,
                    single_batch=True,
                    single_step=False)
        print("Finish Learning.")

    def __call__(self):
        obs_arr = []
        act_arr = []
        rew_arr = []
        
        reward_error = []
        state_error = []
        success_list = []

        step = 0
        trail = 0

        if self.config.learn_from_data:
            self.in_context_learn_from_teacher()

        while trail < self.max_trails or step < self.max_steps:
            done = False
            previous_state, _ = self.env.reset()
            obs_arr.append(previous_state)
            
            epoch_start_step = step
            while not done:
                pred_state_dist, action, pred_reward = self.model.module.generate(
                    None,
                    previous_state,
                    action_clip=self.config.action_clip,
                    temp=self._scheduler(step))
                                
                # interact with env
                new_state, new_reward, done, *_ = self.env.step(action)

                # collect data
                act_arr.append(action)
                if not done:                     
                    obs_arr.append(new_state)   
                rew_arr.append(new_reward)
                # world model reward prediction correct count:
                # reward_correct_prob += reward_out_prob_list[0,0, int(new_reward)].item()

                reward_error.append((new_reward - pred_reward) ** 2)
                state_error.append(-numpy.log(pred_state_dist[int(new_state)].item()))

                # Judge if success
                if done:
                    if self.config.env.lower() == "lake4x4" :
                        if new_reward == 1:
                            success_list.append(int(1))
                        else:
                            success_list.append(int(0))
                    else:
                        success_list.append(int(0))

                # start learning
                self.model.module.in_context_learn(
                    None,
                    previous_state,
                    action,
                    new_reward)
                
                previous_state = new_state
                
                step += 1
                if(step > self.max_steps and trail >= self.max_trails):
                    break
            trail += 1
            self.logger(step,
                        numpy.mean(rew_arr[epoch_start_step:]), 
                        numpy.mean(state_error[epoch_start_step:]), 
                        numpy.mean(reward_error[epoch_start_step:]),
                        success_list[-1])
            
        ds_state_err = downsample(state_error, self.config.downsample_length)
        ds_reward_err = downsample(reward_error, self.config.downsample_length)
        ds_rewards = downsample(rew_arr, self.config.downsample_length)
        ds_success_rate = downsample(success_list, self.config.downsample_trail)

        print("ds_success_rate = ", ds_success_rate)

        self.stat.gather(self.device,
                         reward=ds_rewards,
                         state_prediction=ds_state_err,
                         reward_prediction=ds_reward_err,
                         success_rate = ds_success_rate)
    
    def postprocess(self):
        results=self.stat()
        self.logger("Final_Result",
                    results['reward'], 
                    results['state_prediction'], 
                    results['reward_prediction'])
        if(self.config.has_attr("output")):
            if not os.path.exists(self.config.output):
                os.makedirs(self.config.output)
            for key_name in results:
                res_text = string_mean_var(self.config.downsample_length, results[key_name])
                file_path = f'{self.config.output}/result_{key_name}.txt'
                if os.path.exists(file_path):
                    os.remove(file_path)
                with open(file_path, 'w') as f_model:
                    f_model.write(res_text)
