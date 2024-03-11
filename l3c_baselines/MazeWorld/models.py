#!/usr/bin/env python
# coding=utf8
# File: models.py
import torch
from torch import nn
from torch.nn import functional as F
from modules import Encoder, Decoder, ResBlock, MapDecoder

class Models(nn.Module):
    def __init__(self, 
                 image_size=128,
                 action_size=4,
                 map_size=5,
                 hidden_size=256,
                 nhead=8,
                 n_res_block=4,
                 n_trn_block=8):
        super().__init__()

        self.encoder_1 = Encoder(3, hidden_size // 4, n_res_block, hidden_size // 4)
        self.encoder_2 = Encoder(hidden_size // 4, hidden_size, n_res_block, hidden_size // 4)
        self.decoder = Decoder(hidden_size, 3, hidden_size, n_res_block, hidden_size)
        self.map_decoder = MapDecoder(hidden_size, 3, hidden_size, n_res_block, hidden_size)

        # 创建动作编码层
        self.action_embedding = nn.Embedding(action_size + 1, hidden_size)
        self.action_size = action_size
        self.action_decoder = nn.Sequential(nn.Linear(hidden_size, action_size), nn.Softmax(dim=1))

        # 创建Transformer编码器层
        temporal_encoder_layer = nn.TransformerEncoderLayer(hidden_size, nhead)
        self.temporal_encoder = nn.TransformerEncoder(temporal_encoder_layer, num_layers=n_trn_block)

        # 创建位置编码和Query向量[1, 1, NP, C]
        self.position_encoding_size = image_size // 16
        self.position_vocab_size = self.position_encoding_size * self.position_encoding_size
        self.position_embedding = nn.Embedding(self.position_vocab_size, hidden_size)
        self.position_query = self.position_embedding.weight.data.view(1, 1, self.position_vocab_size, hidden_size)
        
        # 创建地图Query向量[1, 1, NM, C]
        self.map_size = map_size
        self.map_vocab_size = map_size * map_size
        self.map_embedding = nn.Embedding(self.map_vocab_size, hidden_size)
        self.map_query = self.map_embedding.weight.data.view(1, 1, self.map_vocab_size, hidden_size)

        # 创建动作Query向量
        self.action_query = self.action_embedding(torch.tensor([action_size], dtype=torch.long)).view(1, 1, -1, hidden_size)

    def forward(self, observations, actions, rewards, is_train=True):
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
        reshaped_input = observations.view(-1, C, W, H)
        enc_out = self.encoder_1(reshaped_input)
        enc_out = self.encoder_2(enc_out)
        _, eC, eW, eH = enc_out.shape
        
        # Change it to [B * (NT + 1), eW, eH, eC]
        enc_out = enc_out.permute(0, 2, 3, 1)
        # Input observations: [B, NT + 1, eW * eH, eC]
        enc_out = enc_out.view(B, NT + 1, eW * eH, eC)
        # Input actions: [B, NT, eC]
        action_in = self.action_embedding(actions).view(B, NT, 1, eC)

        if(is_train):
            mask_obs = torch.rand_like(actions, dtype=torch.float).lt(0.15).float()
            mask_act = torch.rand_like(actions, dtype=torch.float).lt(0.15).float()
            map_in = self.map_query.repeat((B, NT, 1, 1))
        else:
            #change NT to NT + 1 for all the inputs
            enc_out = torch.cat([enc_out, torch.zeros((B, 1, eW * eH, eC))], dim=1)
            action_in = torch.cat([action_in, torch.zeros((B, 1, 1, eC))], dim=1)
            map_in = self.map_query.repeat((B, NT + 1, 1, 1))
            mask_obs = torch.zeros((B, NT + 1), dtype=torch.float)
            mask_act = torch.zeros((B, NT + 1), dtype=torch.float)
            mask_obs[:, -1] = 1
            mask_act[:, -1] = 1
            NT = NT + 1

        obs_mask = (1 - mask_obs.unsqueeze(-1).unsqueeze(-1)) * enc_out[:, 1:, :, :] + self.position_query
        act_mask = (1 - mask_act.unsqueeze(-1).unsqueeze(-1)) * action_in + self.action_query
        map_mask = self.map_query.repeat((B, NT, 1, 1))

        # Get the size of [B, NT * (eW * eH + Nm * Nm + 1), eC]
        inputs = torch.cat([obs_mask, act_mask, map_mask], dim=2).view(B, -1, eC)

        # Get the size of [B, eWeH + NT * (eWeH + Nm^2 + 1), eC]
        inputs = torch.cat([enc_out[:, 0, :, :], inputs], dim=1)

        # Get the output size of [B, NT, eWeH + Nm^2 + 1, eC]
        outputs = self.temporal_encoder(inputs)[:, eW * eH:, :].view(B, NT, -1, eC)

        obs_output = outputs[:, :, :eW * eH]
        act_output = outputs[:, :, eW * eH]
        map_output = outputs[:, :, eW * eH + 1:]

        img_out = self.decoder(obs_output.reshape(B * NT, eW, eH, eC).permute(0, 3, 1, 2))
        _, n_c, n_w, n_h = img_out.shape
        # [B, N_T, C, eW, eH]
        img_out = img_out.reshape(B, NT, n_c, n_w, n_h)
        # [B, N_T, action_size]
        act_out = self.action_decoder(act_output.reshape(B * NT, eC)).reshape(B, NT, self.action_size)
        map_out = self.map_decoder(map_output.reshape(B * NT, self.map_size, self.map_size, eC).permute(0, 3, 1, 2))
        _, n_c, n_w, n_h = map_out.shape
        # [B, N_T, C, W, H]
        map_out = map_out.reshape(B, NT, n_c, n_w, n_h)

        return img_out, act_out, map_out, mask_obs, mask_act

    def mse_loss_img(self, img_out, img_gt, mask = None):
        mse_loss = torch.mean((img_out - img_gt)**2, dim=[2, 3, 4])
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
        img_out, act_out, map_out, mask_obs, mask_act = self.forward(observations, actions, rewards, is_train=True)
        mse_loss_1 = self.mse_loss_img(img_out, observations[:, 1:], mask_obs)
        mse_loss_2 = self.mse_loss_img(map_out, local_maps)
        ce_loss = self.ce_loss_act(act_out, actions, mask_act)
        return mse_loss_1 + mse_loss_2 + ce_loss

    def inference(self, observations, actions, rewards):
        img_out, act_out, map_out, mask_obs, mask_act = self.forward(observations, actions, rewards, is_train=False)
        return img_out[:, -1], act_out[:, -1], map_out[:, -1]
        

if __name__=="__main__":
    model = Models()
    observation = torch.randn(8, 33, 3, 128, 128)
    action = torch.randint(4, (8, 32)) 
    reward = torch.randn(8, 32)
    local_map = torch.randn(8, 32, 3, 5, 5)

    losses = model.train_loss(observation, action, reward, local_map)
    img_out, act_out, map_out = model.inference(observation, action, reward)
    print(losses)
    print(img_out.shape, act_out.shape, map_out.shape)
