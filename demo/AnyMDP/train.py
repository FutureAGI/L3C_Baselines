import torch
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR

from l3c_baselines.dataloader import AnyMDPDataSet, segment_iterator
from l3c_baselines.utils import Logger, log_progress, log_debug, log_warn, log_fatal
from l3c_baselines.utils import custom_load_model, noam_scheduler, LinearScheduler
from l3c_baselines.utils import Configure, DistStatistics, rewards2go
from l3c_baselines.models import AnyMDPRSA
from l3c_baselines.utils import Runner, EpochManager

@EpochManager
class AnyMDPEpochBase:
    def __init__(self, **kwargs):
        for key in kwargs:
            setattr(self, key, kwargs[key])
        if(self.is_training):
            self.logger_keys = ["learning_rate", 
                        "loss_worldmodel_state", 
                        "loss_worldmodel_reward", 
                        "loss_policymodel",
                        "entropy"]
            self.stat = DistStatistics(*self.logger_keys[1:])
        else:
            self.logger_keys = ["validation_state_pred", 
                        "validation_reward_pred", 
                        "validation_policy"]
            self.stat = DistStatistics(*self.logger_keys)
        self.DataType=AnyMDPDataSet
        
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
                    use_loss_weight=self.is_training) # Do not use loss weight for evaluation
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
            self.logger(self.optimizer.param_groups[0]['lr'],
                    stat_res["loss_worldmodel_state"]["mean"], 
                    stat_res["loss_worldmodel_reward"]["mean"], 
                    stat_res["loss_policymodel"]["mean"], 
                    stat_res["entropy"]["mean"],
                    epoch=epoch_id,
                    iteration=batch_id)
        else:
            self.stat.gather(device,
                    validation_state_pred=[loss["wm-s"] / loss["count"] for loss in losses], 
                    validation_reward_pred=[loss["wm-r"] / loss["count"] for loss in losses], 
                    validation_policy=[loss["pm"] / loss["count"] for loss in losses],
                    count=[loss["count"] for loss in losses])
            
    def epoch_end(self, epoch_id):
        if(not self.is_training):
            stat_res = self.stat()
            self.logger(stat_res["validation_state_pred"]["mean"], 
                    stat_res["validation_reward_pred"]["mean"], 
                    stat_res["validation_policy"]["mean"],
                    epoch=epoch_id)
    

if __name__ == "__main__":
    runner=Runner()
    runner.start(AnyMDPRSA, AnyMDPEpochBase, AnyMDPEpochBase)
