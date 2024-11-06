import os
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR

from l3c_baselines.dataloader import AnyMDPDataSet, segment_iterator
from l3c_baselines.utils import Logger, log_progress, log_debug, log_warn, log_fatal
from l3c_baselines.utils import custom_load_model, noam_scheduler, LinearScheduler
from l3c_baselines.utils import Configure, DistStatistics, rewards2go
from l3c_baselines.utils import EpochManager

def string_mean_var(downsample_length, res):
    string=""
    for i, (xm,xb) in enumerate(zip(res["mean"], res["bound"])):
        string += f'{downsample_length * i}\t{xm}\t{xb}\n'
    return string

class AnyMDPEpochBase:
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

    def compute(self, device, sarr, baarr, laarr, rarr, 
                        epoch_id=-1, 
                        batch_id=-1):
        """
        Defining the computation function for each batch
        """
        if(self.is_training):
            assert self.optimizer is not None, "optimizer is required for training"
            self.logger_keys = ["learning_rate", 
                        "loss_worldmodel_state", 
                        "loss_worldmodel_reward", 
                        "loss_policymodel",
                        "entropy"]
        else:
            self.logger_keys = ["validation_state_pred", 
                        "validation_reward_pred", 
                        "validation_policy",]

        losses = []
        r2goarr = rewards2go(rarr)
        for sub_idx, states, bactions, lactions, rewards, r2go in segment_iterator(
                    self.config.seq_len, self.config.seg_len, device, 
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
                self.stat.gather(device,
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
                self.stat.gather(device,
                        validation_state_pred=loss_wm_s[i], 
                        validation_reward_pred=loss_wm_r[i], 
                        validation_policy=loss_pm[i],
                        count=counts[i])
            
@EpochManager
class AnyMDPEpochTrainValidate(AnyMDPEpochBase):
    def epoch_end(self, epoch_id):
        if(not self.is_training):
            stat_res = self.stat()
            if(self.logger is not None):
                self.logger(stat_res["validation_state_pred"]["mean"], 
                        stat_res["validation_reward_pred"]["mean"], 
                        stat_res["validation_policy"]["mean"],
                        epoch=epoch_id)

@EpochManager
class AnyMDPEpochValidate(AnyMDPEpochBase):
    def epoch_end(self, epoch_id):
        if(not self.is_training):
            stat_res = self.stat()
            if(self.main):
                if not os.path.exists(self.config.output):
                    os.makedirs(self.config.output)
                for key_name in stat_res:
                    res_text = string_mean_var(self.downsample_length, stat_res[key_name])
                    file_path = f'{self.config.output}/result_{key_name}.txt'
                    if os.path.exists(file_path):
                        os.remove(file_path)
                    with open(file_path, 'w') as f_model:
                        f_model.write(res_text)
