import os
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
import cv2
import numpy as np

from l3c_baselines.dataloader import segment_iterator
from l3c_baselines.utils import Logger, log_progress, log_debug, log_warn, log_fatal
from l3c_baselines.utils import custom_load_model, noam_scheduler, LinearScheduler
from l3c_baselines.utils import Configure, DistStatistics, rewards2go
from l3c_baselines.utils import EpochManager, GeneratorBase, Logger
from l3c_baselines.utils import noam_scheduler, LinearScheduler
from l3c_baselines.dataloader import MazeDataSet, PrefetchDataLoader
import logging
from queue import Queue
import threading
import matplotlib.pyplot as plt
import torch.nn as nn


def string_mean_var(downsample_length, res):
    string=""
    for i, (xm,xb) in enumerate(zip(res["mean"], res["bound"])):
        string += f'{downsample_length * i}\t{xm}\t{xb}\n'
    return string

@EpochManager
class MazeEpochVAE:
    def __init__(self, **kwargs):
        for key in kwargs:
            setattr(self, key, kwargs[key])
        if(self.is_training):
            self.logger_keys = ["learning_rate", 
                        "noise",
                        "kl_weight",
                        "reconstruction_error",
                        "kl_divergence"]
            self.stat = DistStatistics(*self.logger_keys[3:])
            self.lr = self.config.lr_vae
            self.lr_decay_interval = self.config.lr_vae_decay_interval
            self.lr_start_step = self.config.lr_vae_start_step
        else:
            self.logger_keys = ["reconstruction_error", 
                        "kl_divergence"]
            self.stat = DistStatistics(*self.logger_keys)

    def preprocess(self):
        if(self.is_training):
            self.sigma_scheduler = LinearScheduler(self.config.sigma_scheduler, 
                                                   self.config.sigma_value)
            self.lambda_scheduler = LinearScheduler(self.config.lambda_scheduler, 
                                                    self.config.lambda_value)
        # use customized dataloader
        self.dataloader = PrefetchDataLoader(
            MazeDataSet(self.config.data_path, self.config.seq_len_vae, verbose=self.main),
            batch_size=self.config.batch_size_vae,
            rank=self.rank,
            world_size=self.world_size
            )
            
    def valid_epoch(self, epoch_id): # Add epoch control for VAE training
        if(self.config.has_attr('epoch_vae_stop')):
            if(epoch_id >= self.config.epoch_vae_stop):
                return False
        return True

    def compute(self, obs_arr, behavior_actid_arr, label_actid_arr, 
                behavior_act_arr, label_act_arr, 
                rew_arr, bev_arr,
                epoch_id=-1, 
                batch_id=-1):
        """
        Defining the computation function for each batch
        """
        if(self.is_training):
            assert self.optimizer is not None, "optimizer is required for training"

        losses = []
        for sub_idx, seg_obs in segment_iterator(
                            self.config.seq_len_vae, self.config.seg_len_vae,
                            self.device, obs_arr):
            # Permute (B, T, H, W, C) to (B, T, C, H, W)
            seg_obs = seg_obs.permute(0, 1, 4, 2, 3)

            if(self.is_training):
                sigma = self.sigma_scheduler()
            else:
                sigma = 0
            loss = self.model.module.vae_loss(
                    seg_obs,
                    _sigma=sigma)
            losses.append(loss)
            if(self.is_training):
                syn_loss = loss["Reconstruction-Error"] + self.lambda_scheduler() * loss["KL-Divergence"]
                if(self.scaler is not None):
                    self.scaler.scale(syn_loss).backward()
                else:
                    syn_loss.backward()
                self.stat.gather(self.device,
                    reconstruction_error = loss["Reconstruction-Error"] / loss["count"],
                    kl_divergence = loss["KL-Divergence"] / loss["count"],
                    count = loss["count"])
        if(self.is_training):
            stat_res = self.stat()
            if(self.logger is not None):
                self.logger(self.optimizer.param_groups[0]['lr'],
                            self.sigma_scheduler(), 
                            self.lambda_scheduler(), 
                            stat_res["reconstruction_error"]["mean"], 
                            stat_res["kl_divergence"]["mean"],
                            epoch=epoch_id,
                            iteration=batch_id)
            # update the scheduler
            self.sigma_scheduler.step()
            self.lambda_scheduler.step()
        else:
            self.stat.gather(self.device,
                    reconstruction_error=loss["Reconstruction-Error"] / loss["count"], 
                    kl_divergence=loss["KL-Divergence"] / loss["count"], 
                    count=loss["count"])
            
        
    def epoch_end(self, epoch_id):
        if(not self.is_training):
            stat_res = self.stat()
            if(self.logger is not None):
                self.logger(stat_res["reconstruction_error"]["mean"], 
                        stat_res["kl_divergence"]["mean"], 
                        epoch=epoch_id)

@EpochManager
class MazeEpochCausal:
    def __init__(self, **kwargs):
        for key in kwargs:
            setattr(self, key, kwargs[key])
        self.DataType=MazeDataSet
        if(self.is_training):
            self.logger_keys = ["learning_rate", 
                        "loss_worldmodel_raw",
                        "loss_worldmodel_latent",
                        "loss_policymodel"]
            self.stat = DistStatistics(*self.logger_keys[1:])
            self.lr = self.config.lr_causal
            self.lr_decay_interval = self.config.lr_causal_decay_interval
            self.lr_start_step = self.config.lr_causal_start_step
            self.reduce_dim = 1
        else:
            self.logger_keys = ["validate_worldmodel_raw",
                        "validate_worldmodel_latent",
                        "validate_policymodel"]
            self.stat = DistStatistics(*self.logger_keys)
            if(self.config.has_attr("downsample_length")):
                self.downsample_length = self.config.downsample_length
            else:
                self.downsample_length = 100
            self.reduce_dim = None
            
    def valid_epoch(self, epoch_id): # Add epoch control for VAE training
        if(self.config.has_attr('epoch_causal_stop')):
            if(epoch_id < self.config.epoch_causal_start):
                return False
        return True

    def preprocess(self):
        # use customized dataloader
        self.dataloader = PrefetchDataLoader(
            MazeDataSet(self.config.data_path, self.config.seq_len_causal, verbose=self.main),
            batch_size=self.config.batch_size_causal,
            rank=self.rank,
            world_size=self.world_size
            )

    def compute(self, obs_arr, behavior_actid_arr, label_actid_arr, 
                behavior_act_arr, label_act_arr, 
                rew_arr, bev_arr,
                epoch_id=-1, 
                batch_id=-1):
        """
        Defining the computation function for each batch
        """
        if(self.is_training):
            assert self.optimizer is not None, "optimizer is required for training"

        losses = []
        for sub_idx, seg_obs, seg_behavior_act, seg_label_act, seg_bev in segment_iterator(
                                self.config.seq_len_causal, self.config.seg_len_causal, self.device, 
                                (obs_arr, 1), behavior_actid_arr, label_actid_arr, bev_arr):

            # Permute (B, T, H, W, C) to (B, T, C, H, W)
            seg_obs = seg_obs.permute(0, 1, 4, 2, 3)
            seg_bev = seg_bev.permute(0, 1, 4, 2, 3)

            loss = self.model.module.sequential_loss(
                                    seg_obs, 
                                    seg_behavior_act, 
                                    seg_label_act, 
                                    seg_bev,
                                    state_dropout=0.20,
                                    use_loss_weight=self.is_training,
                                    reduce_dim=self.reduce_dim,) 
            losses.append(loss)
            if(self.is_training):
                syn_loss = (self.config.lossweight_worldmodel_latent * loss["wm-latent"]
                        + self.config.lossweight_worldmodel_raw * loss["wm-raw"]
                        + self.config.lossweight_policymodel * loss["pm"]
                        + self.config.lossweight_l2 * loss["causal-l2"])
                if(self.scaler is not None):
                    self.scaler.scale(syn_loss).backward()
                else:
                    syn_loss.backward()
                self.stat.gather(self.device,
                                loss_worldmodel_raw = loss["wm-raw"] / loss["count_wm"],
                                loss_worldmodel_latent = loss["wm-latent"] / loss["count_wm"],
                                loss_policymodel = loss["pm"] / loss["count_pm"])
        if(self.is_training):
            stat_res = self.stat()
            if(self.logger is not None):
                self.logger(self.optimizer.param_groups[0]['lr'],
                            stat_res["loss_worldmodel_raw"]["mean"], 
                            stat_res["loss_worldmodel_latent"]["mean"],
                            stat_res["loss_policymodel"]["mean"],
                            epoch=epoch_id,
                            iteration=batch_id)
        else:
            loss_wm_r = torch.cat([loss["wm-raw"] / loss["count_wm"] for loss in losses], dim=1)
            loss_wm_l = torch.cat([loss["wm-latent"] / loss["count_wm"] for loss in losses], dim=1)
            loss_pm = torch.cat([loss["pm"] / loss["count_pm"] for loss in losses], dim=1)
            counts = torch.cat([loss["count_pm"] for loss in losses], dim=1)

            bsz = loss_wm_r.shape[0]
            seg_num = loss_wm_l.shape[1] // self.downsample_length
            valid_seq_len = seg_num * self.downsample_length

            loss_wm_r = torch.mean(loss_wm_r[:, :valid_seq_len].view(bsz, seg_num, -1), dim=-1)
            loss_wm_l = torch.mean(loss_wm_l[:, :valid_seq_len].view(bsz, seg_num, -1), dim=-1)
            loss_pm = torch.mean(loss_pm[:, :valid_seq_len].view(bsz, seg_num, -1), dim=-1)
            counts = torch.mean(counts[:, :valid_seq_len].view(bsz, seg_num, -1), dim=-1)

            for i in range(bsz):
                self.stat.gather(self.device,
                        validate_worldmodel_raw=loss_wm_r[i], 
                        validate_worldmodel_latent=loss_wm_l[i], 
                        validate_policymodel=loss_pm[i],
                        count=counts[i])
        
    def epoch_end(self, epoch_id):
        if(not self.is_training):
            stat_res = self.stat()
            if(self.logger is not None):
                self.logger(stat_res["validate_worldmodel_raw"]["mean"], 
                        stat_res["validate_worldmodel_latent"]["mean"], 
                        stat_res["validate_policymodel"]["mean"],
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



@EpochManager
class MazeEpochTest: #TODO 
    def __init__(self, **kwargs):
        for key in kwargs:
            setattr(self, key, kwargs[key])
        self.DataType=MazeDataSet
        # self.logger_keys = ["validate_worldmodel_raw",
        #             "validate_worldmodel_latent",
        #             "validate_policymodel"]
        self.logger_keys = ["validate_worldmodel_latent"]
        self.stat = DistStatistics(*self.logger_keys)
        if(self.config.has_attr("downsample_length")):
            self.downsample_length = self.config.downsample_length
        else:
            self.downsample_length = 100
        self.reduce_dim = None
            
    # def valid_epoch(self, epoch_id): # Add epoch control for VAE training
    #     if(self.config.has_attr('epoch_causal_stop')):
    #         if(epoch_id < self.config.epoch_causal_start):
    #             return False
    #     return True

    def preprocess(self):
        # use customized dataloader
        self.dataloader = PrefetchDataLoader(
            MazeDataSet(self.config.data_path, self.config.seq_len_causal, verbose=self.main),
            batch_size=self.config.batch_size_causal,
            rank=self.rank,
            world_size=self.world_size
            )

    def compute(self, obs_arr, behavior_actid_arr, label_actid_arr, 
                behavior_act_arr, label_act_arr, 
                rew_arr, bev_arr,
                epoch_id=-1, 
                batch_id=-1):
        """
        Defining the computation function for each batch
        """
        if(self.is_training):
            assert self.optimizer is not None, "optimizer is required for training"

        losses = []
        for sub_idx, seg_obs, seg_behavior_act, seg_label_act, seg_bev in segment_iterator(
                                self.config.seq_len_causal, self.config.seg_len_causal, self.device, 
                                (obs_arr, 1), behavior_actid_arr, label_actid_arr, bev_arr):

            # Permute (B, T, H, W, C) to (B, T, C, H, W)
            seg_obs = seg_obs.permute(0, 1, 4, 2, 3)
            seg_bev = seg_bev.permute(0, 1, 4, 2, 3)

            loss = self.model.module.sequential_loss(
                                    seg_obs, 
                                    seg_behavior_act, 
                                    seg_label_act, 
                                    seg_bev,
                                    state_dropout=0.20,
                                    use_loss_weight=self.is_training,
                                    reduce_dim=self.reduce_dim,) 
            losses.append(loss)

        loss_wm_l = torch.cat([loss["wm-latent"] / loss["count_wm"] for loss in losses], dim=1)

        bsz = loss_wm_l.shape[0]
        seg_num = loss_wm_l.shape[1] // self.downsample_length
        valid_seq_len = seg_num * self.downsample_length

        loss_wm_l = torch.mean(loss_wm_l[:, :valid_seq_len].view(bsz, seg_num, -1), dim=-1)

        for i in range(bsz):
            self.stat.gather(self.device,
                    validate_worldmodel_latent=loss_wm_l[i],
            )
        
    def epoch_end(self, epoch_id):
        if(not self.is_training):
            stat_res = self.stat()
            if(self.logger is not None):
                self.logger(
                        stat_res["validate_worldmodel_latent"]["mean"], 
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

                

class PredictionCodingTestGenerator(GeneratorBase):

    def __call__(self, epoch_id):
        import copy
        N_map = 5
        folder_count = 0
        whole_peak_loss_diffs = []
        for folder in os.listdir(self.config.data_root):
            if folder_count >= N_map:
                break
            print(f"folder: {folder}")
            folder_path = os.path.join(self.config.data_root, folder)
            
            if os.path.isdir(folder_path):
                folder_count += 1
                # it is the action and observation
                states = np.load(os.path.join(folder_path, 'observations.npy'))
                actions = np.load(os.path.join(folder_path, 'actions_behavior_id.npy'))
                print("-----------------------------")
                states = states.astype(np.float32)
                print(f"states shape: {np.info(states)}")
                print("-----------------------------")
                print(f"actions shape: {np.info(actions)}")
                #some config
                in_context_len = self.config.in_context_len
                pred_len = self.config.pred_len
                start = self.config.start_position
                temp = self.config.temp
                drop_out = self.config.drop_out
                len_causal = self.config.seg_len_causal
                Npeak = self.config.Npeak
                test_len = self.config.test_len
                output_folder = self.config.output
                folder_inf = "step_" + str(self.config.pred_len) + "_Npeak_" + str(self.config.Npeak) 
                output_folder = os.path.join(output_folder, folder_inf)
                output_folder = os.path.join(output_folder, folder)
                if not os.path.exists(output_folder):
                    os.makedirs(output_folder)
                print(f"output folder: {output_folder}")

                effect_len = 10
                loss_record = []
                inference_record = []
                
                import tqdm
                history_cache = None
                for check_point in tqdm.tqdm(range(start, start + test_len)):

                    end = min(check_point + in_context_len, len(states))
                    # pred_obs_list = self.model.module.generate_step_by_step(
                    #     observations=states[start:end+1], # the context model has learned 
                    #     actions=actions[start:end], # the context model has learned 
                    #     actions_gt=actions[end:end+pred_len],
                    #     temp=temp,
                    #     drop_out = drop_out,
                    #     device=self.device,
                    #     in_context_len = in_context_len,
                    #     len_causal = len_causal,
                    #     n_step=pred_len
                    # )

                    pred_obs_list, history_cache = self.model.module.forward_by_step_with_cache(
                        observations=states[end:end+1], # the context model has learned 
                        actions_gt=actions[end:end+pred_len],
                        device=self.device,
                        n_step=pred_len,
                        cache = history_cache,
                    )
                    # z_rec, z_pred, a_pred, cache = self.forward(
                    #     inputs[:, :-1], behavior_actions, 
                    #     cache=None, need_cache=True,
                    #     update_memory=True)

                    real = states[end+1:end+1+pred_len]
                    # mse loss for every state
                    mse_loss = np.array([np.mean((real[i] - pred_obs_list[i])**2/(255*255)) for i in range(len(real))])
                    # loss_record.append(mse_loss[0])
                    loss_record.append(np.mean(mse_loss))
                    inference_record.append(pred_obs_list)
                inference_record = np.array(inference_record)
                loss_record = np.array(loss_record)
                print(f"mse loss shape for every state: {loss_record.shape}")
                print(f"mse loss for every state: {loss_record}")
                # print(f"real shape: {real.shape}")
                # plot the loss and save to the output folder
                import matplotlib.pyplot as plt
                plt.plot(loss_record, label="mse loss", alpha=0.5)
                plt.legend()
                mean_loss_record = []
                for i in range(0, len(loss_record) - 50):
                    mean_loss_record.append(np.mean(loss_record[i:i+50]))
                plt.plot(range(0, len(loss_record) - 50 ), mean_loss_record, label="mean loss")
                plt.legend()
                plt.savefig(os.path.join(output_folder, f"mse_loss_{folder_count}.png"))
                plt.close()
                print(f"Saved mse loss plot to {os.path.join(output_folder, f'mse_loss_{folder_count}.png')}")
                plt.plot(range(0, len(loss_record) - 50 ), mean_loss_record, label="mean loss")
                plt.legend()
                plt.savefig(os.path.join(output_folder, f"mean_loss_{folder_count}.png"))
                plt.close()
                print(f"Saved mean loss plot to {os.path.join(output_folder, f'mean_loss_{folder_count}.png')}")
                # plot the mean loss record every 10 steps
                
                print("mean loss: ", np.mean(loss_record))
                return
                # find the peaks point 
                peaks = []
                mask = np.ones_like(loss_record)
                for i in range(Npeak):
                    peak = np.argmax(loss_record * mask)
                    mask[peak] = 0 # remove the peak
                    print(f"peak point {i}: {peak}")
                    peaks.append(peak)
                peaks = np.array(peaks)
                print(f"peaks: {peaks}")


                # use the peak point to mask the model history
                relative_loss_diffs = []
                peaks_loss_diffs = []
                peaks_loss_10_after = []

                for peak in tqdm.tqdm(peaks):
                    # generate the mask for states 
                    pred_obs_record = []
                    state_copy = copy.deepcopy(states) # to prevent the change of the original states
                    loss_record_wo_peak = []
                    count = 0
                    for check_point in range(start, start + test_len):
                    # for end in (range(peak, peak + 10)):
                        end = min(check_point + in_context_len, len(states))
                        pred_obs_list = self.model.module.generate_step_by_step(
                            observations=state_copy[start:end+1], # the context model has learned 
                            actions=actions[start:end], # the context model has learned 
                            actions_gt=actions[end:end+pred_len],
                            temp=temp,
                            drop_out = drop_out,
                            device=self.device,
                            in_context_len = in_context_len,
                            len_causal = len_causal,
                            n_step=pred_len,
                        )
                        if count == peak:
                            state_copy[end] = pred_obs_list[0]
                        #     tmp = copy.deepcopy(pred_obs_list[0])
                        #     tmp_acount = end
                        # #     # print("test : ", pred_obs_list[0] - state_copy[end])
                        # if count > peak:
                        #     print(state_copy[tmp_acount] - tmp)
                        real = state_copy[end+1:end+1+pred_len]
                        count += 1
                        # mse loss for every state
                        mse_loss = np.array([np.mean((real[i] - pred_obs_list[i])**2/(255*255)) for i in range(len(real))])
                        # loss_record_wo_peak.append(mse_loss[0])
                        loss_record_wo_peak.append(np.mean(mse_loss))
                        
                    loss_record_wo_peak = np.array(loss_record_wo_peak)
                    peaks_loss_10_after.append((loss_record[peak], np.mean((loss_record_wo_peak[peak+1:peak+effect_len] - loss_record[peak+1:peak+effect_len])/loss_record[peak+1:peak+effect_len])))
                    whole_peak_loss_diffs.append((loss_record[peak], np.mean((loss_record_wo_peak[peak+1:peak+effect_len] - loss_record[peak+1:peak+effect_len])/loss_record[peak+1:peak+effect_len])))
                    relative_loss_diff = (loss_record_wo_peak - loss_record) / loss_record


                    # --------------- draw --------------------
                    # plot the loss and save to the output folder
                    plt.plot(range(start, start + test_len), relative_loss_diff)
                    # emphasis the peak point
                    plt.scatter(peak, (relative_loss_diff[peak]), marker="x", color="red")
                    plt.savefig(os.path.join(output_folder, f"relative_loss_diff_{folder_count}_{peak}.png"))
                    plt.close()
                    print(f"Saved relative loss diff plot to {os.path.join(output_folder, f'relative_loss_diff_{folder_count}_{peak}.png')}")
                    # plot the loss and the loss after masked together
                    plt.plot(range(start, start + test_len), loss_record_wo_peak, label="loss after mask")
                    plt.plot(range(start, start + test_len), loss_record, label="loss before mask")
                    plt.savefig(os.path.join(output_folder, f"loss_{folder_count}_{peak}.png"))
                    plt.close()
                    print(f"Saved loss plot to {os.path.join(output_folder, f'loss_{folder_count}_{peak}.png')}")

                peaks_loss_10_after = np.array(peaks_loss_10_after)
                # save the loss record without peak to the output folder
                np.save(os.path.join(output_folder, f"mse_loss_wo_peak_Npeak_{Npeak}_step_{pred_len}_effectlen_{effect_len}.npy"), loss_record_wo_peak)
                plt.scatter(peaks_loss_10_after[:, 0], peaks_loss_10_after[:, 1])
                plt.xlabel("peak loss")
                plt.ylabel("peak loss diff")
                plt.savefig(os.path.join(output_folder, f"peak_loss_diff_{folder_count}.png"))
                plt.close()

                

        whole_peak_loss_diffs = np.array(whole_peak_loss_diffs)
        plt.scatter(whole_peak_loss_diffs[:, 0], whole_peak_loss_diffs[:, 1])
        plt.xlabel("peak loss")
        plt.ylabel("peak loss diff")
        plt.savefig(os.path.join(output_folder, f"scatter_whole_peak_loss_diff.png"))
        plt.close()
        print(f"Saved scatter plot for whole peak loss diff to {os.path.join(output_folder, f'scatter_whole_peak_loss_diff.png')}")
        
        return


def img_pro(observations):
    return observations / 255

def img_post(observations):
    return observations * 255


class draw_tSNE_from_cache(GeneratorBase):

    def decode_to_np(self, z):
        pred_obs = self.model.module.vae.decoding(z)
        # pred_obs = img_post(pred_obs)
        return pred_obs.squeeze(1).squeeze(0).permute(1, 2, 0).cpu().numpy()
    def flatten_memory(self, caches):
        N_mem_layer = 18

        flat_memorys = []
        for cache in caches:
            flat_layers = []
            for n_mem_layer in range(N_mem_layer):
                flat_memory = np.append(cache[n_mem_layer]['recurrent_state'][0].flatten().cpu().numpy().T, cache[n_mem_layer]['recurrent_state'][1].flatten().cpu().numpy().T)
                flat_layers.append(flat_memory)
            flat_layers = np.array(flat_layers)
            flat_memorys.append(flat_layers)
        flat_memorys = np.array(flat_memorys)
        return flat_memorys

    def plot_tsne(self, embeddings, labels, layer_idx, perplexity=30, output_folder = "./tSNE"):
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 8))
        # print(embeddings.shape)
        # to draw the tSNE by different colors for different discrete labels
        for i in range(self.N_labels):
            mask = labels == i
            plt.scatter(embeddings[mask, 0], embeddings[mask, 1], label=f"Task {i}")

        # scatter = plt.scatter(embeddings[:, 0], embeddings[:, 1], c=labels, cmap='Spectral')

        plt.title(f'Layer {layer_idx+1} t-SNE Projection\n(Perplexity={perplexity})')
        plt.xlabel('t-SNE 1')
        plt.ylabel('t-SNE 2')
        plt.legend()

        output_name = os.path.join(output_folder, f"layer_{layer_idx+1}_tSNE.png")
        plt.savefig(output_name, dpi=300)
        plt.close()

    

    def __call__(self, epoch_id): #TODO, the data folder
        # load the cache from the npy file
        folder_count = 0
        folder_paths = []
        for folder in os.listdir(self.config.data_root):
            if folder_count >= 4:
                break
            error_folder_path = os.path.join(self.config.output_error, folder)
            error_folder_path = os.path.join(error_folder_path, "compound")
            if os.path.isdir(error_folder_path):
                folder_paths.append(error_folder_path)
            folder_count += 1

        folder_count = 0
        for folder in os.listdir(self.config.tSNE_data):
            if folder_count >= 5:
                break
            folder_path = os.path.join(self.config.tSNE_data, folder)
            for traj in os.listdir(folder_path):
                traj_path = os.path.join(folder_path, traj)
                print(traj_path)
                if os.path.isdir(traj_path):
                    folder_paths.append(os.path.join(traj_path, "error"))
                    folder_count += 1
        print(f"folder paths: {folder_paths}")
        output_folder = self.config.output_tSNE
        print(f"output folder: {output_folder}")
        self.N_labels = len(folder_paths)
        flat_caches = []
        caches_label = []
        for i in range(self.N_labels):
            # cache_root = "./test_data"
            cache_root = folder_paths[i]
            for pred in [1, 10, 100]:
                cache_file = os.path.join(cache_root, f"cache_{pred}.npy")
                caches = np.load(cache_file, allow_pickle=True)
                flat_cache = self.flatten_memory(caches)
                for m in flat_cache:
                    flat_caches.append(m)
                    caches_label.append(i)



        
        flat_caches = np.array(flat_caches)
        print(f"cache shape: {flat_caches.shape}")
        caches_label = np.array(caches_label)
        print(f"caches_label shape: {caches_label.shape}")
        for i in range(self.N_labels):
            # acount the number of different tasks
            print(f"Task {i}: {np.sum(caches_label == i)}")

        # draw tSNE for the caches
        from sklearn.manifold import TSNE
        
        # 参数配置
        PERPLEXITIES = [30]  # 测试不同复杂度参数
        LAYERS_TO_VISUALIZE = range(18)  #[0, 5, 10, 17]  # 选择代表性层

        # 分层分析
        import tqdm
        print("Start Analyzing...")
        print(f"labels: {caches_label}")
        from sklearn.preprocessing import StandardScaler

        # 标准化处理（?）
        scaler = StandardScaler()
        for layer in tqdm.tqdm(LAYERS_TO_VISUALIZE):
        # for layer in LAYERS_TO_VISUALIZE:
            # print(f"Analyzing Layer {layer}...")
            for perplexity in PERPLEXITIES:
                # 获取当前层嵌入
                tsne = TSNE(n_components=2, init='pca', random_state=501)
                layer_features = flat_caches[:, layer]
                scaled_features = scaler.fit_transform(layer_features)
                embeddings = tsne.fit_transform(scaled_features)
                
                self.plot_tsne(embeddings, caches_label, layer, perplexity)

class draw_compound_loss(GeneratorBase): #TODO

    def decode_to_np(self, z):
        pred_obs = self.model.module.vae.decoding(z)
        # pred_obs = img_post(pred_obs)
        return pred_obs.squeeze(1).squeeze(0).permute(1, 2, 0).cpu().numpy()
    def plot_tsne(self, embeddings, labels, layer_idx, perplexity=30, output_folder = "./tSNE"):
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 8))

    def __call__(self, epoch_id): #TODO
        # load the cache from the npy file
        folder_count = 0
        folder_paths = []
        for folder in os.listdir(self.config.data_root):
            if folder_count >= 4:
                break
            error_folder_path = os.path.join(self.config.output_error, folder)
            error_folder_path = os.path.join(error_folder_path, "compound")
            if os.path.isdir(error_folder_path):
                folder_paths.append(error_folder_path)
            folder_count += 1
        print(f"folder paths: {folder_paths}")
        output_folder = self.config.output_tSNE
        print(f"output folder: {output_folder}")
        self.N_labels = len(folder_paths)
        flat_caches = []
        caches_label = []
        for i in range(self.N_labels):
            # cache_root = "./test_data"
            cache_root = folder_paths[i]
            for pred in [1, 10, 100]:
                cache_file = os.path.join(cache_root, f"cache_{pred}.npy")
                caches = np.load(cache_file, allow_pickle=True)
                flat_cache = self.flatten_memory(caches)
                for m in flat_cache:
                    flat_caches.append(m)
                    caches_label.append(i)


class interactive_trajectory(GeneratorBase):
    def __call__(self, epoch_id):
        import gym
        import l3c.mazeworld
        from l3c.mazeworld import MazeTaskSampler
        from l3c.mazeworld import Resampler
        # same map with different command
        origin_task = MazeTaskSampler()
        new_task = Resampler(origin_task)
        maze_env = gym.make("mazeworld-v2", enable_render=False)
        done = False
        while not done:
            action = None # Replace it with your own policy function
            observation, reward, done, info = maze_env.step(action)
            maze_env.render()




def flatten_memory(caches):
    N_mem_layer = 18

    flat_memorys = []
    for cache in caches:
        flat_layers = []
        for n_mem_layer in range(N_mem_layer):
            flat_memory = np.append(cache[n_mem_layer]['recurrent_state'][0].flatten().cpu().numpy().T, cache[n_mem_layer]['recurrent_state'][1].flatten().cpu().numpy().T)
            flat_layers.append(flat_memory)
        flat_layers = np.array(flat_layers)
        flat_memorys.append(flat_layers)
    flat_memorys = np.array(flat_memorys)
    return flat_memorys


class cache_generator(GeneratorBase): #TODO

    def __call__(self, epoch_id):
    
        in_context_len = self.config.in_context_len
        pred_len = self.config.pred_len
        start = self.config.start_position
        
        temp = self.config.temp
        drop_out = self.config.drop_out
        len_causal = self.config.seg_len_causal
        # output_folder = self.config.output_error
        test_len = self.config.test_len

        folder_count = 0
        for folder in os.listdir(self.config.tSNE_data):
            if folder_count >= 5:
                break
            folder_path = os.path.join(self.config.tSNE_data, folder)
            for traj in os.listdir(folder_path):
                traj_path = os.path.join(folder_path, traj)
                print(traj_path)
                if os.path.isdir(traj_path):
                    folder_count += 1
                    # it is the action and observation
                    states = np.load(os.path.join(traj_path, 'observations.npy'))
                    actions = np.load(os.path.join(traj_path, 'actions_behavior_id.npy'))
                    states = states.astype(np.float32) # it is in range [0, 255]
                    print("-----------------------------")
                    print(f"states shape: {states.shape}")
                    print(f"actions shape: {actions.shape}")
                    error_folder = "error"
                    output_folder = os.path.join(traj_path, error_folder)
                    if not os.path.exists(output_folder):
                        os.makedirs(output_folder)
                    print(f"output folder: {output_folder}")
                    print("-----------------------------")
                    history_cache = None # TODO


                    import tqdm
                    # loss_record = []
                    # inference_record = []
                    for pred_len in [1, 10, 100]:
                        print(f"pred_len: {pred_len}")
                        loss_records = []
                        cache_records = []
                        for check_point in tqdm.tqdm(range(start, start + test_len)):
                            end = min(check_point, len(states))
                            # pred_obs_list, cache = self.model.module.generate_step_by_step_with_cache(
                            #     observations=states[start:end+1], # the context model has learned 
                            #     actions=actions[start:end], # the context model has learned 
                            #     actions_gt=actions[end:end+pred_len],
                            #     device= self.device, # torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                            #     n_step=pred_len,
                            #     cache = history_cache,
                            # )
                            if check_point > 1:
                                tmp_cache = self.model.module.get_memory()#['recurrent_state'].copy()
                                tmp_cache = flatten_memory(tmp_cache)
                                print(f"tmp_cache shape: {tmp_cache.shape}")
                                

                            pred_obs_list, history_cache = self.model.module.forward_by_step_with_cache(
                                observations=states[end:end+1], # the context model has learned 
                                actions_gt=actions[end:end+pred_len],
                                device=self.device,
                                n_step=pred_len,
                                cache = history_cache,
                            )
                            if check_point > 1:
                                print("-----------------------------")
                                current_cache = self.model.module.get_memory()#['recurrent_state'].copy()
                                current_cache = flatten_memory(current_cache)
                                cache_diff = current_cache - tmp_cache
                                # mean of the cache_diff
                                cache_diff = np.mean(cache_diff)
                                print(f"cache diff: {cache_diff}")

                            
                            real = states[end+1:end+1+pred_len]
                            # mse loss for every state
                            mse_loss = np.array([np.mean((real[i] - pred_obs_list[i])**2/(255*255) ) for i in range(len(real))])
                            # loss_records[end] = np.mean(mse_loss) # consider the mean of whole steps' loss
                            loss_records.append(mse_loss)
                            if check_point > 500 and check_point % 25 == 0:
                                cache_records.append(history_cache)
                        cache_records = np.array(cache_records)
                        print(f"cache shape: {cache_records.shape}")
                        loss_records = np.array(loss_records)
                        print("mean loss: ", np.mean(loss_records))
                        # save the records locally in npy
                        np.save(os.path.join(output_folder, f"cache_{pred_len}.npy"), cache_records)
                        np.save(os.path.join(output_folder, f"loss_{pred_len}.npy"), loss_records)
                        # the format of the loss is : [context_length, pred_len]
                        print(f"Saved cache and loss records to {os.path.join(output_folder, f'cache_{pred_len}.npy')} and {os.path.join(output_folder, f'loss_{pred_len}.npy')}")

class test_generator(GeneratorBase): #TODO

    def __call__(self, epoch_id):
    
        in_context_len = self.config.in_context_len
        pred_len = self.config.pred_len
        start = self.config.start_position
        
        temp = self.config.temp
        drop_out = self.config.drop_out
        len_causal = self.config.seg_len_causal
        test_len = self.config.test_len
        import matplotlib.pyplot as plt
        output_image_folder = "./test_image"
        folder_count = 0
        for folder in os.listdir(self.config.tSNE_data):
            folder_path = os.path.join(self.config.tSNE_data, folder)
            for traj in os.listdir(folder_path):
                traj_path = os.path.join(folder_path, traj)
                print(traj_path)
                if os.path.isdir(traj_path):
                    if folder_count >= 1:
                        break
                    folder_count += 1
                    # it is the action and observation
                    states = np.load(os.path.join(traj_path, 'observations.npy'))
                    actions = np.load(os.path.join(traj_path, 'actions_behavior_id.npy'))
                    states = states.astype(np.float32) # it is in range [0, 255]
                    print("-----------------------------")
                    print(f"states shape: {states.shape}")
                    print(f"actions shape: {actions.shape}")
                    error_folder = "error"
                    output_folder = os.path.join(traj_path, error_folder)
                    if not os.path.exists(output_folder):
                        os.makedirs(output_folder)
                    print(f"output folder: {output_folder}")
                    print("-----------------------------")
                     # TODO

                    data_generate = False
                    import tqdm
                    # loss_record = []
                    # inference_record = []
                    for pred_len in [1]:
                        history_cache = None
                        print(f"pred_len: {pred_len}")
                        loss_records = []
                        cache_records = []
                        cache_diffs = []
                        for check_point in tqdm.tqdm(range(start+1, start + test_len)):
                            end = min(check_point, len(states))
                            if check_point > 0:
                                tmp_cache = history_cache
                            # pred_obs_list, history_cache = self.model.module.generate_states_only(
                            # current_observation = states[end:end+1], 
                            # action_trajectory = actions[end:end+pred_len],
                            # history_observation=None,
                            # history_action=None,
                            # history_update_memory=True, 
                            # autoregression_update_memory=False, # TOTEST
                            # cache=None,
                            # single_batch=True,
                            # history_single_step=False,
                            # future_single_step=False,
                            # raw_images=True,
                            # need_numpy=True)


                            pred_obs_list, history_cache = self.model.module.forward_by_step_with_cache(
                                observations=states[end:end+1], # the context model has learned 
                                actions_gt=actions[end:end+pred_len],
                                device=self.device,
                                n_step=pred_len,
                                cache = history_cache,
                            )

                            # if check_point > 0:
                            #     cache_diff = tmp_cache - history_cache
                            #     # mean of the cache_diff
                            #     cache_diff = np.mean(cache_diff)
                            #     cache_diffs.append(cache_diff)
                            # print(history_cache)
                            real = states[end+1:end+1+pred_len]
                            if check_point % 100 == 0:
                                tmp_folder = os.path.join(output_image_folder, "test_image_decoding")
                                if not os.path.exists(tmp_folder):
                                    os.makedirs(tmp_folder)
                                plt.imsave(os.path.join(tmp_folder, f"real_{check_point}.png"), real[0]/255)
                                plt.imsave(os.path.join(tmp_folder, f"pred_{check_point}.png"), pred_obs_list[0]/255)
                                print(f"Saved real and pred images to {os.path.join(tmp_folder, 'real.png')} and {os.path.join(tmp_folder, 'pred.png')}")

                            # mse loss for every state
                            mse_loss = np.array([np.mean((real[i] - pred_obs_list[i])**2/(255*255) ) for i in range(len(real))])
                            # loss_records[end] = np.mean(mse_loss) # consider the mean of whole steps' loss
                            loss_records.append(mse_loss)
                            if data_generate == True:
                                if check_point > 500 and check_point % 25 == 0:
                                    cache_records.append(history_cache)
                        # end of the loop for check_point in tqdm.tqdm(range(start, start + test_len)):

                        loss_records = np.array(loss_records)
                        cache_records = np.array(cache_records)
                        cache_diffs = np.array(cache_diffs)

                        if data_generate == True:
                            print(f"cache shape: {cache_records.shape}")
                            # save the records locally in npy
                            np.save(os.path.join(output_folder, f"cache_{pred_len}.npy"), cache_records)
                            print("mean loss: ", np.mean(loss_records))
                            # save the records locally in npy
                            np.save(os.path.join(output_folder, f"loss_{pred_len}.npy"), loss_records)
                            # the format of the loss is : [context_length, pred_len]
                            print(f"Saved cache and loss records to {os.path.join(output_folder, f'cache_{pred_len}.npy')} and {os.path.join(output_folder, f'loss_{pred_len}.npy')}")
                        

                        if not os.path.exists(output_image_folder):
                            os.makedirs(output_image_folder)
                        plt.plot(loss_records, label="mse loss", alpha=0.5)
                        plt.legend()
                        mean_loss_record = []
                        for i in range(0, len(loss_records) - 50):
                            mean_loss_record.append(np.mean(loss_records[i:i+50]))
                        plt.plot(range(0, len(loss_records) - 50 ), mean_loss_record, label="mean loss")
                        plt.legend()
                        plt.savefig(os.path.join(output_image_folder, f"mse_loss_{folder_count}.png"))
                        plt.close()
                        print(f"Saved mse loss plot to {os.path.join(output_image_folder, f'mse_loss_{folder_count}.png')}")
                        plt.plot(range(0, len(loss_records) - 50 ), mean_loss_record, label="mean loss")
                        plt.legend()
                        plt.savefig(os.path.join(output_image_folder, f"mean_loss_{folder_count}.png"))
                        plt.close()
                        print(f"Saved mean loss plot to {os.path.join(output_image_folder, f'mean_loss_{folder_count}.png')}")
                        # plot the mean loss record every 10 steps
                        print("mean loss: ", np.mean(loss_records))
                        # plot the loss and the cache diffs in one plot and with different colors but the same x-axis and similar y-axis ranges
                        plt.plot(range(0, len(cache_diffs)), cache_diffs, label="cache diff")
                        plt.plot(range(0, len(loss_records)), loss_records, label="loss")
                        plt.legend()
                        plt.savefig(os.path.join(output_image_folder, f"loss_cache_diff_{folder_count}.png"))
                        plt.close()


                    # end of the loop for pred_len in [1]:


class MAZEGenerator(GeneratorBase):

    def __call__(self, epoch_id):
    
        folder_count = 0

        for folder in os.listdir(self.config.data_root):
            folder_path = os.path.join(self.config.data_root, folder)
            
            if os.path.isdir(folder_path):
                states = np.load(os.path.join(folder_path, 'observations.npy'))
                actions = np.load(os.path.join(folder_path, 'actions_behavior_id.npy'))

                in_context_len = self.config.in_context_len
                pred_len = self.config.pred_len
                start = self.config.start_position
                temp = self.config.temp
                drop_out = self.config.drop_out
                len_causal = self.config.seg_len_causal
                output_folder = self.config.output
                
                end = min(start + in_context_len, len(states))

                pred_obs_list = self.model.module.generate_step_by_step(
                    observations=states[start:end+1],
                    actions=actions[start:end],
                    actions_gt=actions[end:end+pred_len],
                    temp=temp,
                    drop_out = drop_out,
                    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                    in_context_len = in_context_len,
                    len_causal = len_causal,
                    n_step=pred_len
                )

                real = [states[i] for i in range(end+1, end + 1 + pred_len)] 

                pred_obs_list_with_initial = pred_obs_list
                
                
                video_folder = os.path.join(output_folder, f'video_{folder_count}')
                if not os.path.exists(video_folder):
                    os.makedirs(video_folder)

                video_filename = os.path.join(video_folder, f"pred_obs_video_{folder_count}.avi")
                fourcc = cv2.VideoWriter_fourcc(*'XVID') 
                frame_height, frame_width = pred_obs_list_with_initial[0].shape[:2]
                video_writer = cv2.VideoWriter(video_filename, fourcc, 10.0, (frame_width * 2, frame_height))

                for real_frame, pred_frame in zip(real, pred_obs_list_with_initial):
                    rotated_real = cv2.rotate(real_frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
                    rotated_pred = cv2.rotate(pred_frame, cv2.ROTATE_90_COUNTERCLOCKWISE)

                    concatenated_img = np.hstack((rotated_real, rotated_pred))

                    img = np.clip(concatenated_img, 0, 255).astype(np.uint8)
                    video_writer.write(img)

                video_writer.release() 

                print(f"Saved video with {len(real)} frames to {video_filename}")

                
                updated_cache = None
                print(f"Cache cleared after generating {len(real)} frames.")

                folder_count += 1  

                if folder_count >= 16:
                    print("Processed 16 folders. Stopping.")
                    break 



class MAZEGenerator_loss(GeneratorBase):

    def __call__(self, epoch_id):
        if epoch_id == 0:
            output_queue = Queue()  # Create a queue to store logs
            log_lock = threading.Lock()  # Place the lock inside the __call__ method
            all_results = {}  # Store results for all folders

        folder_count = 0

        for folder in os.listdir(self.config.data_root):
            folder_path = os.path.join(self.config.data_root, folder)

            if os.path.isdir(folder_path):
                states = np.load(os.path.join(folder_path, 'observations.npy'))
                actions = np.load(os.path.join(folder_path, 'actions_behavior_id.npy'))

                folder_results = set()  # Use a set to remove duplicates and store (folder, in context learn, world loss) data

                # Outer loop for `i` in range 1 to 1001 with a step of 10 (i.e., process in batches of 10)
                for i in range(1, 101, 10):  # i starts at 1 and increments by 10
                    batch_end = min(i + 10, 101)  # Ensure we don't go beyond 1000
                    for j in range(i, batch_end):
                        in_context_len = j
                        pred_len = self.config.pred_len
                        start = self.config.start_position
                        temp = self.config.temp
                        drop_out = self.config.drop_out
                        len_causal = self.config.seg_len_causal

                        end = min(start + in_context_len, len(states))

                        world_loss = self.model.module.world_generate_loss(
                            observations=states[start:end + 1],
                            actions=actions[start:end],
                            actions_gt=actions[end:end + pred_len],
                            obss_gt=states[end + 1:end + 1 + pred_len],
                            temp=temp,
                            drop_out=drop_out,
                            device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                            in_context_len=in_context_len,
                            len_causal=len_causal,
                            n_step=pred_len
                        )

                        # Add (folder, in context learn, world_loss) to the set, which will automatically remove duplicates
                        folder_results.add((folder, in_context_len, world_loss.item()))

                        # Only put logs into the queue when epoch_id == 0
                        if epoch_id == 0:
                            output_queue.put(f"Epoch {epoch_id}, Folder {folder}, in context learn {in_context_len}, world loss {world_loss.item()}")

                # Store the deduplicated results in all_results
                all_results[folder] = sorted(folder_results, key=lambda x: x[1])  # Sort by in-context length

                folder_count += 1
                if folder_count >= 1:
                    break  # Only process one folder

        # If epoch_id == 0, process logs in the queue
        if epoch_id == 0:
            # Use the lock to ensure only one thread prints the logs
            with log_lock:
                self.process_logs(output_queue)

            # Plot the results
            self.plot_results(all_results)

    def process_logs(self, output_queue):
        """Process or print all logs in the queue"""
        while not output_queue.empty():
            print(output_queue.get())

    def plot_results(self, all_results):
        """Plot the results"""
        # All folder results are stored in all_results
        for folder, results in all_results.items():
            in_context_lens = [result[1] for result in results]  # Extract in-context lengths
            world_losses = [result[2] for result in results]  # Extract world losses

            plt.plot(in_context_lens, world_losses, label=f"Folder {folder}")

        plt.xlabel('in context learn')
        plt.ylabel('world loss')
        plt.legend()
        plt.title('World Loss vs In Context Learn')
        plt.grid(True)

        # Save the image
        save_path = 'world_loss_vs_in_context_learn.png'  # Path and filename to save the plot
        plt.savefig(save_path)  # Save the plot
        plt.close()  # Close the plot to release memory
