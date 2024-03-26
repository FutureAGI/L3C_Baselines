#!/usr/bin/env python
# coding=utf8
# File: models.py
import torch
from torch import nn
from torch.nn import functional as F
from modules import Encoder, Decoder, ResBlock, MapDecoder
from torch.utils.checkpoint import checkpoint  
from ar_transformer import ARTransformerEncoder

def print_memory(info="Default"):
    print(info, "Memory allocated:", torch.cuda.memory_allocated(), "Memory cached:", torch.cuda.memory_cached())


class MazeModels(nn.Module):
    def __init__(self, 
                 image_size=128,
                 action_size=4,
                 map_size=5,
                 hidden_size=512,
                 max_steps=512,
                 nhead=16,
                 n_res_block=6,
                 n_trn_block=12):
        super().__init__()

        self.encoder = Encoder(image_size, 3, hidden_size, n_res_block)

        self.decoder = Decoder(image_size, hidden_size, 3, n_res_block)

        self.map_decoder = MapDecoder(hidden_size, hidden_size, 3, map_size)

        # 创建动作编码层
        self.action_embedding = nn.Embedding(action_size, hidden_size)
        self.action_size = action_size
        self.action_decoder = nn.Sequential(nn.Linear(hidden_size, 4 * hidden_size), nn.GELU(), nn.Linear(4 * hidden_size, action_size), nn.Softmax(dim=-1))
        self.reward_decoder = nn.Sequential(nn.Linear(hidden_size, 4 * hidden_size), nn.GELU(), nn.Linear(4 * hidden_size, 1))

        # 创建Transformer编码器层
        self.temporal_encoder = ARTransformerEncoder(n_trn_block, hidden_size, nhead, dim_feedforward=4*hidden_size)

        # 创建位置编码和Query向量[1, NT, 1, C]
        self.max_steps = max_steps
        temporal_embeddings = torch.randn(1, self.max_steps + 1, 1, hidden_size)
        self.temporal_query = nn.Parameter(temporal_embeddings, requires_grad=True)

        # 创建Type向量[1, 1, NP, C]
        type_embeddings = torch.randn(1, 1, 2, hidden_size)
        self.type_query = nn.Parameter(type_embeddings, requires_grad=True)

        self.hidden_size = hidden_size

    def forward(self, observations, actions, cache=None, need_cache=True):
        """
        Input Size:
            observations:[B, NT, C, W, H]
            actions:[B, NT / (NT - 1)] 
            cache: [B, NC, H]
        """
        B, NT, C, W, H = observations.shape
        assert actions.shape[0] == B and actions.shape[1] == NT, "The shape of actions should be [%s, %s], but get %s" % (B, NT, actions.shape)

        # Preprocessing, normalize to [-1, 1]
        # Output Shape: [B, NT, H]
        outputs = observations

        outputs = (outputs.reshape(-1, C, W, H) - 127) / 32
        outputs = self.encoder(outputs)
        #torch.cuda.empty_cache() 
        #print_memory("Stage Encoder")

        # VAE
        err = 0.01 * torch.randn_like(outputs).to(observations.device)
        rec_img_out = self.decoder(outputs.reshape(B * NT, -1) + err.reshape(B * NT, -1))
        _, n_c, n_w, n_h = rec_img_out.shape
        rec_img_out = rec_img_out.reshape(B, NT, n_c, n_w, n_h)

        # Input observations: [B, NT, 1, H]
        # Stop the gradients here
        outputs = outputs.detach().view(B, NT, 1, -1)
        # Input actions: [B, NT, 1, H]
        action_in = self.action_embedding(actions).view(B, NT, 1, -1)

        # [B, NT, 2, H]
        outputs = torch.cat([outputs, action_in], dim=2)
        # Add Temporal Position Embedding
        outputs = outputs + self.temporal_query[:, :NT]
        # Add Type Embedding
        outputs = outputs + self.type_query

        # Concatenate [s_0, a_0, s_1, a_1, s_2, ...] to acquire the size of [B, NT * 2, H]
        outputs = outputs.view(B, NT * 2, -1)

        # Auto Regressive Mask
        seq_len = NT * 2

        # Temporal Encoders
        outputs, new_cache = checkpoint(lambda x:self.temporal_encoder(x, cache=cache, need_cache=need_cache), outputs)

        # Acqure Outputs: [a_0, s_1, a_1, ...]
        outputs = outputs.reshape(B, NT, 2, -1)

        act_output = outputs[:, :, 0]
        obs_output = outputs[:, :, 1]
        #torch.cuda.empty_cache() 
        #print_memory("Stage Temporal Decoder")

        # Decode Observation [B, N_T, C, W, H]
        # Pass the gradients without adjust the weights of self.decoder
        self.decoder.requires_grad_(False)
        img_out = self.decoder(obs_output.reshape(B * NT, -1))
        self.decoder.requires_grad_(True)

        _, n_c, n_w, n_h = img_out.shape
        img_out = img_out.reshape(B, NT, n_c, n_w, n_h)
        rew_out = self.reward_decoder(obs_output)

        # Decode Action [B, N_T, action_size], without softmax!
        act_out = self.action_decoder(act_output)

        # Decode Map [B, N_T, C, W, H], shares the output with act_out
        map_out = self.map_decoder(act_output.reshape(B * NT, -1))
        _, n_c, n_w, n_h = map_out.shape
        map_out = map_out.reshape(B, NT, n_c, n_w, n_h)

        #torch.cuda.empty_cache() 
        #print_memory("Stage Output")

        return rec_img_out, img_out, act_out, map_out, rew_out, new_cache

    def mse_loss_img(self, img_out, img_gt, mask = None):
        mse_loss = torch.mean(((img_out - img_gt / 255)) ** 2, dim=[2, 3, 4])
        if mask is not None:
            mse_loss = mse_loss * mask
            sum_mask = torch.sum(mask)
            sum_loss = torch.sum(mse_loss)
            mse_loss = sum_loss / sum_mask
        else:
            mse_loss = torch.mean(mse_loss)

        return mse_loss

    def ce_loss_act(self, act_out, act_gt, mask = None):
        act_logits = F.one_hot(act_gt, self.action_size)
        ce_loss = -torch.mean(torch.log(act_out) * act_logits, dim=-1)
        if mask is not None:
            ce_loss = ce_loss * mask
            sum_mask = torch.sum(mask)
            sum_loss = torch.sum(ce_loss)
            ce_loss = sum_loss / sum_mask
        else:
            ce_loss = torch.mean(ce_loss)

        return ce_loss

    def train_loss(self, observations, actions, rewards, local_maps):
        rec_img_out, img_out, act_out, map_out, rew_out, _ = self.forward(observations[:, :-1], actions, cache=None, need_cache=False)
        lmse_rec = self.mse_loss_img(rec_img_out, observations[:, :-1])
        lmse_obs = self.mse_loss_img(img_out, observations[:, 1:])
        lmse_map = self.mse_loss_img(map_out, local_maps)
        lmse_rew = F.mse_loss(rewards, rew_out.squeeze(-1))
        ce_loss = self.ce_loss_act(act_out, actions)
        B, NT = actions.shape
        return lmse_rec, lmse_obs, ce_loss, lmse_map, lmse_rew, torch.tensor(B * NT, dtype=torch.int, device='cuda')

    def inference_next(self, observations, actions, cache=None):
        """
        Inference next observation and action
        """
        B, NT, C, W, H = observations.shape
        device = observations.device
        add_act = torch.zeros((B, 1), dtype=torch.int).to(device)
        add_obs = torch.zeros((B, 1, C, W, H), dtype=torch.float).to(device)

        if(cache is not None):
            l_cached = cache[0].shape[1] // 2
            valid_obs = observations[l_cached:]
            valid_act = actions[l_cached]
        else:
            valid_obs = observations
            valid_act = actions
        B, NT, C, W, H = valid_obs.shape

        if(NT < 2):
            ext_act = add_act
        else:
            ext_act = torch.cat([valid_act, add_act], dim=1)

        # Inference Action First
        with torch.no_grad():
            rec_img_out, img_out, act_out, map_out, rew_out, new_cache = self.forward(valid_obs, ext_act, cache=cache, need_cache=True)
        # Softmax Sampling
        n_action = torch.multinomial(act_out[:, -1], num_samples=1)
        print("Model decision", act_out[:, -1], n_action)

        if(NT < 2):
            ext_act = n_action
        else:
            ext_act = torch.cat([valid_act, n_action], dim=1)

        # Inference Next Observation based on Sampled Action
        with torch.no_grad():
            rec_img_out, img_out, act_out, map_out, rew_out, new_cache = self.forward(valid_obs, ext_act, cache=cache, need_cache=True)

        return 255 * img_out[:, -1], n_action.squeeze(1), map_out[:, -1], cache
        

if __name__=="__main__":
    model = MazeModels()
    observation = torch.randn(8, 33, 3, 128, 128)
    action = torch.randint(4, (8, 32)) 
    reward = torch.randn(8, 32)
    local_map = torch.randn(8, 32, 3, 5, 5)

    losses = model.train_loss(observation, action, reward, local_map)
    img_out, act_out, map_out = model.inference_next(observation, action)
    print(losses)
    print(img_out.shape, act_out.shape, map_out.shape)
