import os
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR

from l3c_baselines.dataloader import segment_iterator
from l3c_baselines.utils import Logger, log_progress, log_debug, log_warn, log_fatal
from l3c_baselines.utils import custom_load_model, noam_scheduler, LinearScheduler
from l3c_baselines.utils import Configure, DistStatistics, rewards2go, downsample
from l3c_baselines.utils import EpochManager, GeneratorBase, Logger
from l3c_baselines.utils import DiscreteEnvWrapper
from l3c_baselines.utils import OnlineRL
from l3c_baselines.utils import tag_vocabulary, tag_mapping_id, tag_mapping_gamma
from l3c_baselines.dataloader import AnyMDPDataSet, AnyMDPDataSetContinuousState, AnyMDPDataSetContinuousStateAction

import gym
import imageio
import numpy
import pickle
import random
from gym.envs.toy_text.frozen_lake import generate_random_map
from l3c.anymdp import AnyMDPTaskSampler
from l3c.anymdp import AnyMDPSolverOpt, AnyMDPSolverOTS, AnyMDPSolverQ
from stable_baselines3 import DQN, A2C, TD3, PPO


def string_mean_var(downsample_length, res):
    string=""
    if(numpy.size(res["mean"]) > 1):
        for i, (xm,xb) in enumerate(zip(res["mean"], res["bound"])):
            string += f'{downsample_length * i}\t{xm}\t{xb}\n'
    else:
        string =  f'{0}\t{res["mean"]}\t{res["bound"]}\n'
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
            self.stat = DistStatistics()
            self.reduce = 1
        else:
            self.logger_keys = ["validation_state_pred", 
                        "validation_reward_pred", 
                        "validation_policy"]
            self.stat = DistStatistics()
            self.reduce = None
            if(self.config.has_attr("downsample_length")):
                self.downsample_length = self.config.downsample_length
            else:
                self.downsample_length = 100
        if(self.config.has_attr('state_dropout')):
            self.state_dropout = self.config.state_dropout
        else:
            self.state_dropout = 0.20
        if(self.config.has_attr('reward_dropout')):
            self.reward_dropout = self.config.reward_dropout
        else:
            self.reward_dropout = 0.20

    def compute(self, sarr, parr, tarr, baarr, rarr, laarr,
                        epoch_id=-1, 
                        batch_id=-1):
        """
        Defining the computation function for each batch
        """
        state_dropout = 0.0
        if(self.is_training):
            assert self.optimizer is not None, "optimizer is required for training"
            state_dropout = self.state_dropout
        else:
            state_dropout = 0.0

        losses = []
        for sub_idx, states, prompts, tags, bactions, rewards, lactions in segment_iterator(
                    self.config.seq_len, self.config.seg_len, self.device, 
                    (sarr, 1), parr, tarr, baarr, rarr, laarr):
            loss = self.model.module.sequential_loss(
                    states,  # Observations
                    prompts,  # Prompts
                    tags,  # Tags
                    bactions, # Behavior Actions
                    rewards, # Rewards
                    lactions, # Reference Actions
                    state_dropout=state_dropout, 
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
                    loss_worldmodel_state = loss["wm-s"] / loss["count_s"],
                    loss_worldmodel_reward = loss["wm-r"] / loss["count_s"],
                    loss_policymodel = loss["pm"] / loss["count_a"],
                    entropy = -loss["ent"] / loss["count_a"],
                    count = loss["count_a"])
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
            loss_wm_s = torch.cat([loss["wm-s"] / torch.clamp_min(loss["count_s"], 1.0e-3) 
                    for loss in losses], dim=1)
            loss_wm_r = torch.cat([loss["wm-r"] / torch.clamp_min(loss["count_s"], 1.0e-3) 
                    for loss in losses], dim=1)
            loss_pm = torch.cat([loss["pm"] / torch.clamp_min(loss["count_a"], 1.0e-3) 
                    for loss in losses], dim=1)
            loss_ent = torch.cat([loss["ent"] / torch.clamp_min(loss["count_a"], 1.0e-3) 
                    for loss in losses], dim=1)
            counts = torch.cat([loss["count_a"] for loss in losses], dim=1)

            bsz = loss_wm_s.shape[0]

            loss_wm_s = downsample(loss_wm_s, self.downsample_length)
            loss_wm_r = downsample(loss_wm_r, self.downsample_length)
            loss_pm = downsample(loss_pm, self.downsample_length)
            loss_ent = down_sample(loss_ent, self.downsample_length)
            counts = downsample(counts, self.downsample_length)

            for i in range(bsz):
                self.stat.gather(self.device,
                        validation_state_pred=loss_wm_s[i], 
                        validation_reward_pred=loss_wm_r[i], 
                        validation_policy=loss_pm[i],
                        validation_entropy=loss_ent[i],
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

# TODO: ADAPT Generator To OPTAR @PENGTAO
# use gamma_vocabulary and tag_vocabulary
class AnyMDPGenerator(GeneratorBase):
    def preprocess(self):
        if(self.config.env.lower().find("lake") >= 0):
            self.task_sampler = self.task_sampler_lake
        elif(self.config.env.lower().find("anymdp") >= 0):
            self.env = gym.make("anymdp-v0", max_steps=self.max_steps)
            self.task_sampler = self.task_sampler_anymdp
        elif(self.config.env.lower().find("mountaincar") >= 0):
            self.task_sampler = self.task_sampler_mountain_car
        elif(self.config.env.lower().find("pendulum") >= 0):
            self.task_sampler = self.task_sampler_pendulum
        else:
            log_fatal("Unsupported environment:", self.config.env)

        if(self.config.has_attr("task_file")):
            with open(self.config.task_file, 'rb') as fr:
                self.tasks = pickle.load(fr)
            log_debug(f"Read tasks from {self.config.task_file} success")
        else:
            self.tasks = None

        logger_keys = ["step", "reward", "state_prediction", "reward_prediction", "success_rate"]
        benchmark_logger_keys = ["step", "reward", "success_rate"]

        self.stat = DistStatistics(*logger_keys)
        self.stat_opt = DistStatistics(*benchmark_logger_keys)
        self.stat_online = DistStatistics(*logger_keys)
        self.stat_random = DistStatistics(*benchmark_logger_keys)
        self.logger = Logger("trail_idx",
                            "total_steps",
                            *logger_keys, 
                            on=self.main, 
                            use_tensorboard=False)
        self.logger_benchmark = Logger("trail_idx",
                            "total_steps",
                            *benchmark_logger_keys, 
                            on=self.main, 
                            use_tensorboard=False)
        self.logger_random = Logger("trail_idx",
                            "total_steps",
                            *benchmark_logger_keys, 
                            on=self.main, 
                            use_tensorboard=False)

    def task_sampler_anymdp(self, epoch_id=0):
        task_id = None
        if(self.tasks is None):
            dims = self.config.env.lower().replace("anymdp", "").split("x")
            task = AnyMDPTaskSampler(int(dims[0]), int(dims[1]))
        else:
            task_num = len(self.tasks)
            task_id = (epoch_id * self.world_size + self.rank) % task_num
            task = self.tasks[task_id]
        self.env.set_task(task)
        return task_id

    def task_sampler_lake(self, epoch_id=0):
        self.env = gym.make(
            'FrozenLake-v1', 
            map_name=self.config.env.replace("lake", ""), 
            is_slippery=True, 
            max_episode_steps=1000,
            render_mode='rgb_array_list')
        return None
    def task_sampler_mountain_car(self, epoch_id=0):
        env = gym.make("MountainCar-v0", render_mode="rgb_array_list", max_episode_steps=self.config.max_steps)
        if self.config.map_env_discrete:
            self.env = DiscreteEnvWrapper(env=env,
                                            env_name=self.config.env.lower(),
                                            action_space=self.config.action_clip,
                                            state_space=self.config.state_clip,
                                            reward_shaping=True)
        else:
            self.env = env
        return None
    
    def task_sampler_pendulum(self, epoch_id=0):
        env = gym.make("Pendulum-v1", render_mode="rgb_array_list", g=9.81, max_episode_steps=self.config.max_steps)
        if self.config.map_env_discrete:
            self.env = DiscreteEnvWrapper(env=env,
                                            env_name=self.config.env.lower(),
                                            action_space=self.config.action_clip,
                                            state_space=self.config.state_clip)
        else:
            self.env = env
        return None

    def reward_shaping(self, done, terminated, reward):
        if(self.config.env.lower().find("lake") >= 0):
            if done and reward < 0.5:
                reward = -1.0
                return reward
        elif(self.config.env.lower().find("mountaincar") >= 0):
            if terminated:
                reward = 1.0
                return reward
        elif(self.config.env.lower().find("pendulum") >= 0):
            reward = reward / 10 + 0.3
        return reward
            

    def is_success_fail(self, reward, trail_reward, step):
        if(self.config.env.lower().find("lake") >= 0):
            if reward > 1.0e-3:
                return 1
            else:
                return 0
        elif(self.config.env.lower().find("lander") >= 0):
            if trail_reward >= 200:
                return 1
            else:
                return 0
        elif(self.config.env.lower().find("mountaincar") >= 0):
            if step < self.max_steps:
                return 1
            else:
                return 0
        else:
            return 0
    
    def reset_env(self):
        if self.config.env.lower().find("pendulum") >= 0:
            state, *_ = self.env.reset(seed=123, options={"low": -0.7, "high": 0.5})
        elif self.config.env.lower().find("mountaincar") >= 0:
            state, *_ = self.env.reset(seed=123, options={"x_init": numpy.pi/2, "y_init": 0.5})
        else:
            state, *_ = self.env.reset()
        return state

    def in_context_learn_from_teacher(self):
        # Task ID: retrieve the correpsonding teacher trajectory with task ID
        for folder in os.listdir(self.config.data_root):
            folder_path = os.path.join(self.config.data_root, folder)
            
            if os.path.isdir(folder_path):
                states = numpy.load(os.path.join(folder_path, 'observations.npy'))
                prompts = numpy.load(os.path.join(folder_path, 'prompts.npy'))
                tags = numpy.load(os.path.join(folder_path, 'tags.npy'))
                actions = numpy.load(os.path.join(folder_path, 'actions_behavior.npy'))
                rewards = numpy.load(os.path.join(folder_path, 'rewards.npy'))
                states = states.astype(numpy.int32)
                prompts = prompts.astype(numpy.int32)
                tags = tags.astype(numpy.int32)
                actions = actions.astype(numpy.int32)
                rewards = rewards.astype(numpy.float32)
                segment_len = 1000
                for start in range(0, len(states), segment_len):
                    end = min(start + segment_len, len(states))
                    self.model.module.in_context_learn(
                        states[start:end],
                        prompts[start:end],
                        tags[start:end],
                        actions[start:end],
                        rewards[start:end],
                        single_batch=True,
                        single_step=False)
        print("Finish Learning.")

    def benchmark(self):
        supported_gym_env = ["lake", "lander", "mountaincar", "pendulum"]
        # Load opt model
        if self.config.env.lower().find("anymdp") >= 0:
            model = AnyMDPSolverOpt(self.env)
            def benchmark_model(state):
                return model.policy(state)
            self.benchmark_opt_model = benchmark_model
        elif any(self.config.env.lower().find(name) == 0 for name in supported_gym_env):
            if self.config.run_benchmark.run_opt:
                model_classes = {'dqn': DQN, 'a24': A2C, 'td3': TD3, 'ppo': PPO}
                model_name = self.config.benchmark_model_name.lower()
                if model_name not in model_classes:
                    raise ValueError("Unknown policy type: {}".format())
                model = model_classes[model_name].load(f'{self.config.benchmark_model_save_path}/model/{model_name}.zip', env=self.env)
                def benchmark_model(state):
                    action, _ = model.predict(state)
                    return int(action)
                self.benchmark_opt_model = benchmark_model
        else:
            raise ValueError("Unsupported environment:", self.config.env)
        
        def run_online_rl():
            online_rl = OnlineRL(env=self.env, 
                                 env_name=self.config.env.lower(),
                                 model_name=self.config.benchmark_model_name.lower(),
                                 max_trails=self.config.max_trails,
                                 max_steps=self.config.max_steps,
                                 downsample_trail=self.config.downsample_trail)
            rew_stat, step_trail, success_rate = online_rl()
            ds_step_trail = downsample(step_trail, self.config.downsample_trail)
            ds_rewards = downsample(rew_stat, self.config.downsample_trail)
            ds_success = downsample(success_rate, self.config.downsample_trail)
            self.stat_online.gather(self.device,
                                step=ds_step_trail,
                                reward=ds_rewards,
                                success_rate=ds_success)


        # Function to run opt model or random model
        def run_benchmark(benchmark_model, logger_benchmark, stat_benchmark):
            rew_stat = []
            success_rate = []
            step_trail = []
            trail = 0
            total_steps = 0
            success_rate_f = 0.0
            
            while trail < self.max_trails:
                step = 0
                trail_reward = 0.0
                done = False
                new_state = self.reset_env()
                
                while not done:
                    action= benchmark_model(new_state)
                    new_state, new_reward, terminated, truncated, *_ = self.env.step(action)
                    if self.config.env.lower().find("anymdp") >= 0:
                        done = terminated
                    else:
                        if terminated or truncated:
                            done = True
                    shaped_reward = self.reward_shaping(done, terminated, new_reward)
                    trail_reward += new_reward

                    step += 1
                    if step > self.max_steps:
                        print("Reach max_steps, break trail.")
                        done = True
                    if done:
                        # success rate
                        succ_fail = self.is_success_fail(shaped_reward, trail_reward, step)
                        if trail + 1 < self.config.downsample_trail:
                            success_rate_f = (1-1/(trail+1)) * success_rate_f + succ_fail / (trail+1)
                        else:
                            success_rate_f = (1-1/self.config.downsample_trail) * success_rate_f + succ_fail / self.config.downsample_trail
                        rew_stat.append(trail_reward)
                        success_rate.append(success_rate_f)
                        step_trail.append(step)
                
                trail += 1
                total_steps += step
                logger_benchmark(trail,
                                total_steps,
                                step_trail[-1],
                                rew_stat[-1],
                                success_rate[-1])
            
            ds_step_trail = downsample(step_trail, self.config.downsample_trail)
            ds_rewards = downsample(rew_stat, self.config.downsample_trail)
            ds_success = downsample(success_rate, self.config.downsample_trail)

            stat_benchmark.gather(self.device,
                                step=ds_step_trail,
                                reward=ds_rewards,
                                success_rate=ds_success)
        
        if self.config.run_benchmark.run_opt:
            run_benchmark(self.benchmark_opt_model, self.logger_benchmark, self.stat_opt)
        
        if self.config.run_benchmark.run_online:
            run_online_rl()

        if self.config.run_benchmark.run_random:
            def random_model(state):
                return random.randint(0,self.config.action_clip - 1)
            run_benchmark(random_model, self.logger_random, self.stat_random)

    def dym_interactive_tag(self, success_rate):
        if self.config.env.lower().find("lake") >= 0:
            if success_rate < 0.05:
                self.interactive_tag = 6 # Random
            elif success_rate < 0.4:
                self.interactive_tag = 7 # 
            else:
                self.interactive_tag = 3 # Opt with gamma 0.994
        else:
            return
        
    def __call__(self, epoch_id):

        task_id = self.task_sampler(epoch_id=epoch_id)

        if not self.config.run_icl:
            if self.config.run_benchmark.run_opt or self.config.run_benchmark.run_online or self.config.run_benchmark.run_random:
                print("Run Benchmark Only.")
                self.benchmark()
                return
            else:
                print("run_icl & run_benchmark both False, please check config.")
                return
        else:
            if self.config.run_benchmark.run_opt or self.config.run_benchmark.run_online or self.config.run_benchmark.run_random:
                print("Run Benchmark & ICL")
                self.benchmark()
            else:
                print("Run ICL Only.")
        # Start ICL
        obs_arr = []
        act_arr = []
        rew_arr = []
        frames = []
        
        reward_error = []
        state_error = []
        rew_stat = []
        success_rate = []
        step_trail = []
        success_rate_f = 0.0

        trail = 0
        total_step = 0
        pred_state_dist = None

        interactive_prompt = numpy.array([3]) # opt3 with gamma 0.994
        self.interactive_tag = numpy.array([7]) # Unknown, let model deside current policy quality 

        if self.config.learn_from_data:
            self.in_context_learn_from_teacher()

        while trail < self.max_trails:
            step = 0
            done = False
            trail_reward = 0.0
            trail_obs_loss = 0.0
            trail_reward_loss = 0.0
            previous_state = self.reset_env()
            obs_arr.append(previous_state)
            if(pred_state_dist is not None):
                trail_obs_loss += -numpy.log(pred_state_dist[int(previous_state)].item())
            temp = self._scheduler(trail)
            self.dym_interactive_tag(success_rate_f)
            while not done:
                # Generate action, world model prediction
                if step % self.config.skip_frame == 0:
                    pred_state_dist, action, pred_reward = self.model.module.generate(
                        previous_state,
                        interactive_prompt,
                        self.interactive_tag,
                        temp=temp,
                        need_numpy=True,
                        single_batch=True,
                        future_prediction=True)
                env_action = action % self.config.action_clip 
                # Interact with environment         
                new_state, new_reward, terminated, truncated, *_ = self.env.step(env_action)
                if self.config.env.lower().find("anymdp") >= 0:
                        done = terminated
                else:
                    if terminated or truncated:
                        done = True
                # Reward shaping
                shaped_reward = self.reward_shaping(done, terminated, new_reward)

                # collect data
                act_arr.append(action)
                rew_arr.append(new_reward)
                # collect gif frame
                if (self.config.save_gif 
                and trail % self.config.save_gif_gap == 0 
                and self.config.env.lower().find("anymdp") < 0):
                    frames.extend(self.env.render())


                # start learning
                if step % self.config.skip_frame == 0 or done:        
                    self.model.module.in_context_learn(
                        previous_state,
                        interactive_prompt,
                        self.interactive_tag,
                        action,
                        shaped_reward)

                obs_arr.append(new_state) 
                previous_state = new_state

                trail_obs_loss += -numpy.log(pred_state_dist[int(new_state)].item())
                trail_reward += new_reward
                trail_reward_loss += (shaped_reward - pred_reward) ** 2

                step += 1
                if(step > self.max_steps):
                    print("Reach max_steps, break trail.")
                    done = True
                if(done):
                    act_arr.append(self.action_dim)
                    rew_arr.append(0.0)
                    self.model.module.in_context_learn(
                        new_state,
                        interactive_prompt,
                        self.interactive_tag,
                        self.action_dim,
                        0.0)
                    # success rate
                    succ_fail = self.is_success_fail(shaped_reward, trail_reward, step)
                    if trail + 1 < self.config.downsample_trail:
                        success_rate_f = (1-1/(trail+1)) * success_rate_f + succ_fail / (trail+1)
                    else:
                        success_rate_f = (1-1/self.config.downsample_trail) * success_rate_f + succ_fail / self.config.downsample_trail
                    
                    rew_stat.append(trail_reward)
                    state_error.append(trail_obs_loss / step)
                    reward_error.append(trail_reward_loss / step)
                    success_rate.append(success_rate_f)
                    step_trail.append(step)

            trail += 1
            total_step += step
            self.logger(trail,
                        total_step,
                        step_trail[-1],
                        rew_stat[-1], 
                        state_error[-1], 
                        reward_error[-1],
                        success_rate[-1])

        # Save gif
        if self.config.save_gif and self.config.env.lower().find("anymdp") < 0:
            gif_path = f'{self.config.output}/gym.gif'
            imageio.mimsave(gif_path, [numpy.array(frame) for frame in frames], fps=30)
        
        ds_state_err = downsample(state_error, self.config.downsample_trail)
        ds_reward_err = downsample(reward_error, self.config.downsample_trail)
        ds_rewards = downsample(rew_stat, self.config.downsample_trail)
        ds_success = downsample(success_rate, self.config.downsample_trail)
        ds_step_trail = downsample(step_trail, self.config.downsample_trail)

        self.stat.gather(self.device,
                         step=ds_step_trail,
                         reward=ds_rewards,
                         state_prediction=ds_state_err,
                         reward_prediction=ds_reward_err,
                         success_rate = ds_success)
    
    def postprocess(self):
        def save_results(results, prefix):
            if self.config.has_attr("output"):
                if not os.path.exists(self.config.output):
                    os.makedirs(self.config.output)
                
                for key_name in results:
                    res_text = string_mean_var(self.config.downsample_trail, results[key_name])
                    file_path = f'{self.config.output}/{prefix}_{key_name}.txt'
                    
                    if os.path.exists(file_path):
                        os.remove(file_path)
                    
                    with open(file_path, 'w') as f_model:
                        f_model.write(res_text)

        if self.config.run_icl:
            # Final Result
            final_results = self.stat()
            self.logger("Final_Result",
                        self.config.max_trails,
                        final_results['step']['mean'],
                        final_results['reward']['mean'],
                        final_results['state_prediction']['mean'],
                        final_results['reward_prediction']['mean'],
                        final_results['success_rate']['mean'])
            save_results(final_results, "result")

        # Benchmark Result
        if self.config.run_benchmark.run_opt:
            # Benchmark Result
            benchmark_results = self.stat_opt()
            self.logger_benchmark("Benchmark_Result",
                                self.config.max_trails,
                                benchmark_results['step']['mean'],
                                benchmark_results['reward']['mean'],
                                benchmark_results['success_rate']['mean'])
            save_results(benchmark_results, "benchmark_result")
        if self.config.run_benchmark.run_online:
            # Online Result
            online_results = self.stat_online()
            self.logger_benchmark("Online_Result",
                                self.config.max_trails,
                                online_results['step']['mean'],
                                online_results['reward']['mean'],
                                online_results['success_rate']['mean'])
            save_results(online_results, "online_result")
        if self.config.run_benchmark.run_random:
            # Random Result
            random_results = self.stat_random()
            self.logger_random("Random_Result",
                                self.config.max_trails,
                                random_results['step']['mean'],
                                random_results['reward']['mean'],
                                random_results['success_rate']['mean'])
            save_results(random_results, "random_result")
