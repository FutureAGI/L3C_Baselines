import torch
from torch import nn
from torch.nn import functional as F

class ResBlock(nn.Module):
    def __init__(self, in_channel, hidden_size):
        super().__init__()

        self.conv = nn.Sequential(
            nn.LeakyReLU(),
            nn.Conv2d(in_channel, hidden_size, 3, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(hidden_size, in_channel, 1),
        )

    def forward(self, input):
        out = self.conv(input)
        out += input

        return out

class Encoder(nn.Module):
    """
    Change [B*NT, C_in, 128, 128] to [B*NT, C_out]
    """
    def __init__(self, img_size, in_channel, out_channel, n_res_block):
        super().__init__()

        channel_b1 = out_channel // 16
        channel_b2 = out_channel // 8
        channel_b3 = out_channel // 4
        cur_size = img_size // 8
        fin_channel = cur_size * cur_size * channel_b3

        blocks = [
            nn.Conv2d(in_channel, channel_b1, 3, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(channel_b1, channel_b2, 4, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(channel_b2, channel_b2, 3, padding=1),
        ]

        for i in range(n_res_block):
            blocks.append(ResBlock(channel_b2, out_channel))

        blocks.extend([
            nn.Conv2d(channel_b2, channel_b2, 3, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(channel_b2, channel_b3, 4, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(channel_b3, channel_b3, 3, padding=1),
        ])

        for i in range(n_res_block):
            blocks.append(ResBlock(channel_b3, out_channel))

        blocks.extend([
            nn.Conv2d(channel_b3, channel_b3, 3, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(channel_b3, channel_b3, 4, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(channel_b3, channel_b3, 3, padding=1),
        ])

        blocks.extend([
            nn.Flatten(start_dim=1, end_dim=-1),
            nn.Linear(fin_channel, out_channel),
            nn.LeakyReLU(),
        ])

        self.blocks = nn.Sequential(*blocks)
        self.hidden_size = out_channel

    def forward(self, input):
        return self.blocks(input)

class Decoder(nn.Module):
    """
    Change [B*NT, C_in] to [B*NT, C_out, 128, 128]
    """
    def __init__(
        self, 
        img_size,
        in_channel, 
        hidden_size,
        out_channel, 
        n_res_block, 
    ):
        super().__init__()

        channel_b1 = hidden_size // 2
        self.ini_size = img_size // 8
        self.ini_channel = hidden_size // 4
        ini_mapping = self.ini_size * self.ini_size * self.ini_channel

        self.input_mapping = nn.Sequential(nn.Linear(in_channel, hidden_size), nn.LeakyReLU(), nn.Linear(hidden_size, ini_mapping), nn.LeakyReLU())

        blocks = [nn.Conv2d(self.ini_channel, channel_b1, 3, padding=1)]

        for i in range(n_res_block):
            blocks.append(ResBlock(channel_b1, in_channel))

        blocks.append(nn.LeakyReLU())

        blocks.extend(
            [
                nn.ConvTranspose2d(channel_b1, channel_b1 // 2, 4, stride=2, padding=1),
                nn.LeakyReLU(),
                nn.ConvTranspose2d(channel_b1 // 2, channel_b1 // 4, 4, stride=2, padding=1),
                nn.LeakyReLU(),
                nn.ConvTranspose2d(channel_b1 // 4, channel_b1 // 8, 4, stride=2, padding=1),
                nn.LeakyReLU(),
                nn.Conv2d(channel_b1 // 8, out_channel, 3, padding=1),
                nn.Sigmoid(),
            ]
        )

        self.blocks = nn.Sequential(*blocks)

    def forward(self, inputs):
        img = self.input_mapping(inputs)
        img = img.view(-1, self.ini_channel, self.ini_size, self.ini_size)
        return self.blocks(img)

class MapDecoder(nn.Module):
    def __init__(
        self, 
        in_channel, 
        hidden,
        out_channel, 
        map_size,
    ):
        super().__init__()

        self.input_mapping = nn.Linear(in_channel, map_size * map_size * hidden)
        self.map_size = map_size
        self.hidden = hidden

        blocks = [
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, out_channel),
        ]

        self.blocks = nn.Sequential(*blocks)

    def forward(self, input):
        out = self.input_mapping(input)
        out = out.view(-1, self.map_size, self.map_size, self.hidden)
        out = self.blocks(out).permute(0, 3, 1, 2)
        return out

class ActionDecoder(nn.Module):
    def __init__(
        self,
        input_size,
        hidden_size,
        output_size,
        dropout=0.10
    ):
        super().__init__()

        self.layer_norm = nn.LayerNorm(input_size, eps=1.0e-5)

        self.act_decoder_pre = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, input_size),
            nn.GELU())

        self.act_decoder_post = nn.Sequential(
            nn.Linear(input_size, output_size),
            nn.Softmax(dim=-1))

    def forward(self, input):
        src = self.layer_norm(input)
        out = self.act_decoder_pre(src)
        out = self.act_decoder_post(out + src)
        return out
