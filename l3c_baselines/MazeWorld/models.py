#!/usr/bin/env python
# coding=utf8
# File: models.py
import torch
from torch import nn
from torch.nn import functional as F
from modules import Encoder, Decoder, ResBlock, MapDecoder
from torch.utils.checkpoint import checkpoint  

def gen_mask(actions, is_train=True, is_policy=True):
    B, NT = actions.shape
    if(is_train):
        mask_obs = torch.rand((B, NT), dtype=torch.float).lt(0.15).float().detach()
        mask_act = torch.rand((B, NT), dtype=torch.float).lt(0.15).float().detach()
    else:
        mask_obs = torch.zeros((B, NT), dtype=torch.float)
        mask_act = torch.zeros((B, NT), dtype=torch.float)
        mask_obs[:, -1] = 1
        if(is_policy):
            mask_act[:, -1] = 1
    return mask_obs, mask_act

def print_memory(info="Default"):
    print(info, "Memory allocated:", torch.cuda.memory_allocated(), "Memory cached:", torch.cuda.memory_cached())


class MazeModels(nn.Module):
    def __init__(self, 
                 image_size=128,
                 action_size=4,
                 map_size=5,
                 hidden_size=512,
                 max_steps=512,
                 nhead=8,
                 n_res_block=3,
                 n_trn_block=12):
        super().__init__()

        self.encoder = Encoder(image_size, 3, hidden_size, n_res_block)

        self.decoder = Decoder(image_size, hidden_size, 3, n_res_block)

        self.map_decoder = MapDecoder(hidden_size, 3, hidden_size // 8, map_size, n_res_block, hidden_size // 4)

        # 创建动作编码层
        self.action_embedding = nn.Embedding(action_size, hidden_size)
        self.action_size = action_size
        self.action_decoder = nn.Sequential(nn.Linear(hidden_size, action_size), nn.Softmax(dim=1))

        # 创建Transformer编码器层
        temporal_encoder_layer = nn.TransformerEncoderLayer(hidden_size, nhead)
        self.temporal_encoder_1 = nn.TransformerEncoder(temporal_encoder_layer, num_layers=n_trn_block // 3)
        self.temporal_encoder_2 = nn.TransformerEncoder(temporal_encoder_layer, num_layers=n_trn_block // 3)
        self.temporal_encoder_3 = nn.TransformerEncoder(temporal_encoder_layer, num_layers=n_trn_block // 3)

        # 创建位置编码和Query向量[1, NT, 1, C]
        self.max_steps = max_steps
        temporal_embeddings = torch.randn(1, self.max_steps + 1, 1, hidden_size)
        self.temporal_query = nn.Parameter(temporal_embeddings, requires_grad=True)

        # 创建位置编码和Query向量[1, 1, NP, C]
        obs_embeddings = torch.randn(1, 1, 1, hidden_size)
        self.obs_query = nn.Parameter(obs_embeddings, requires_grad=True)

        #self.position_query.weight.register_backward_hook(backward_hook)
        # 创建地图Query向量[1, 1, 1, C]
        map_embeddings = torch.randn(1, 1, 1, hidden_size)
        self.map_query = nn.Parameter(map_embeddings, requires_grad=True)
        mask_action_embeddings = torch.randn(1, 1, 1, hidden_size)
        self.action_query = nn.Parameter(mask_action_embeddings, requires_grad=True)

        self.hidden_size = hidden_size

    def forward(self, observations, actions, rewards, mask_obs, mask_act):
        """
        Input Size:
            observations:[B, NT + 1, C, W, H]
            actions:[B, NT] 
            rewards:[B, NT]
        """
        B, NT, C, W, H = observations.shape
        NT = NT - 1
        assert actions.shape[0] == B and actions.shape[1] == NT, "The shape of actions should be [%s, %s], but get %s" % (B, NT, actions.shape)
        assert rewards.shape[0] == B and rewards.shape[1] == NT, "The shape of rewards should be [%s, %s], but get %s" % (B, NT, rewards.shape)
        outputs = observations.view(-1, C, W, H)
        outputs = checkpoint(self.encoder, outputs)
        torch.cuda.empty_cache() 
        #print_memory("Stage Encoder")

        # Input observations: [B, NT + 1, H]
        outputs = outputs.view(B, NT + 1, 1, -1)
        # Input actions: [B, NT, H]
        action_in = self.action_embedding(actions).view(B, NT, 1, -1)

        # 创建动作Query向量
        obs_mask = (1 - mask_obs.unsqueeze(-1).unsqueeze(-1)) * outputs[:, 1:] + self.obs_query
        act_mask = (1 - mask_act.unsqueeze(-1).unsqueeze(-1)) * action_in + self.action_query
        map_mask = self.map_query.repeat((B, NT, 1, 1))

        # Get the size of [B, NT, 3, H]
        inputs = torch.cat([act_mask, map_mask, obs_mask], dim=2)
        inputs = inputs + self.temporal_query[:, 1:(NT+1)]

        # Get the size of [B, NT * 3, H]
        inputs = inputs.view(B, NT * 3, -1)

        # Get the size of [B, 1 + NT * 3, H]
        outputs = torch.cat([outputs[:, 0] + self.temporal_query[:, 0], inputs], dim=1)

        # Get the output size of [B, NT, 3, H]
        seq_len = outputs.size(1)
        mask = (torch.triu(torch.ones(seq_len, seq_len, device=outputs.device)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        outputs = checkpoint(lambda x:self.temporal_encoder_1(x.permute(1, 0, 2), mask=mask), outputs)
        outputs = checkpoint(lambda x:self.temporal_encoder_2(x, mask=mask), outputs)
        outputs = checkpoint(lambda x:self.temporal_encoder_3(x, mask=mask).permute(1, 0, 2), outputs)
        outputs = outputs[:, 1:].view(B, NT, 3, -1)

        act_output = outputs[:, :, 0]
        map_output = outputs[:, :, 1]
        obs_output = outputs[:, :, 2]
        torch.cuda.empty_cache() 
        #print_memory("Stage Temporal Decoder")

        # Decode Observation [B, N_T, C, W, H]
        img_out = self.decoder(obs_output.reshape(B * NT, -1))
        _, n_c, n_w, n_h = img_out.shape
        img_out = img_out.reshape(B, NT, n_c, n_w, n_h)

        # Decode Action [B, N_T, action_size], without softmax!
        act_out = self.action_decoder(act_output.reshape(B * NT, -1)).reshape(B, NT, -1)

        # Decode Map [B, N_T, C, W, H]
        map_out = self.map_decoder(map_output.reshape(B * NT, -1))
        _, n_c, n_w, n_h = map_out.shape
        map_out = map_out.reshape(B, NT, n_c, n_w, n_h)
        torch.cuda.empty_cache() 
        #print_memory("Stage Output")

        return img_out, act_out, map_out

    def mse_loss_img(self, img_out, img_gt, mask = None):
        mse_loss = torch.mean(((img_out - img_gt) / 256)**2, dim=[2, 3, 4])
        B, T, _, _, _ = img_out.shape
        if mask is not None:
            mse_loss = mse_loss * mask
            sum_mask = torch.sum(mask)
            sum_loss = torch.sum(mse_loss)
            mse_loss = sum_loss / sum_mask
        else:
            mse_loss = torch.mean(mse_loss)
            sum_mask = torch.tensor(B * T).to(img_out.device)

        return mse_loss, sum_mask

    def ce_loss_act(self, act_out, act_gt, mask = None):
        B, T = act_gt.shape
        act_logits = F.one_hot(act_gt, self.action_size)
        ce_loss = -torch.mean(F.log_softmax(act_out, dim=-1) * act_logits, dim=-1)
        if mask is not None:
            ce_loss = ce_loss * mask
            sum_mask = torch.sum(mask)
            sum_loss = torch.sum(ce_loss)
            ce_loss = sum_loss / sum_mask
        else:
            ce_loss = torch.mean(ce_loss)
            sum_mask = torch.tensor(B * T).to(act_out.device)

        return ce_loss, sum_mask

    def train_loss(self, observations, actions, rewards, local_maps, mask_obs, mask_act):
        img_out, act_out, map_out = self.forward(observations, actions, rewards, mask_obs, mask_act)
        mse_loss_1, sum_1 = self.mse_loss_img(img_out, observations[:, 1:], mask_obs)
        mse_loss_2, sum_2 = self.mse_loss_img(map_out, local_maps)
        ce_loss, sum_ce = self.ce_loss_act(act_out, actions, mask_act)
        return mse_loss_1, sum_1, mse_loss_2, sum_2, ce_loss, sum_ce

    def inference_next(self, observations, actions, rewards):
        B, NT, C, W, H = observations.shape
        device = observations.device
        add_act = torch.zeros((B, 1), dtype=torch.int).to(device)
        add_rew = torch.zeros((B, 1), dtype=torch.float).to(device)
        add_obs = torch.zeros((B, 1, C, W, H), dtype=torch.float).to(device)

        if(NT < 2):
            ext_act = add_act
            ext_rew = add_rew
        else:
            ext_act = torch.cat([actions, add_act], dim=1)
            ext_rew = torch.cat([rewards, add_rew], dim=1)
        ext_obs = torch.cat([observations, add_obs], dim=1)

        # Inference Action First
        mask_obs, mask_act = gen_mask(ext_act, is_train=False, is_policy=True) 
        mask_obs = mask_obs.to(device)
        mask_act = mask_act.to(device)
        with torch.no_grad():
            img_out, act_out, map_out = self.forward(ext_obs, ext_act, ext_rew, mask_obs, mask_act)
        # Softmax Sampling
        n_action = torch.multinomial(F.softmax(act_out[:, -1], dim=-1), num_samples=1)

        if(NT < 2):
            ext_act = n_action
        else:
            ext_act = torch.cat([actions, n_action], dim=1)

        # Inference Next Observation based on Sampled Action
        _, mask_act = gen_mask(ext_act, is_train=False, is_policy=False) 
        mask_act = mask_act.to(device)
        with torch.no_grad():
            img_out, act_out, map_out = self.forward(ext_obs, ext_act, ext_rew, mask_obs, mask_act)

        return img_out[:, -1], n_action.squeeze(1), map_out[:, -1]
        

if __name__=="__main__":
    model = MazeModels()
    observation = torch.randn(8, 33, 3, 128, 128)
    action = torch.randint(4, (8, 32)) 
    reward = torch.randn(8, 32)
    local_map = torch.randn(8, 32, 3, 5, 5)

    mask_obs, mask_act = gen_mask(action) 
    losses = model.train_loss(observation, action, reward, local_map, mask_obs, mask_act)
    img_out, act_out, map_out = model.inference_next(observation, action, reward)
    print(losses)
    print(img_out.shape, act_out.shape, map_out.shape)
