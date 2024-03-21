import torch
from torch import nn

class ResBlock(nn.Module):
    def __init__(self, in_channel, hidden_size):
        super().__init__()

        self.conv = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(in_channel, hidden_size, 3, padding=1),
            nn.ReLU(inplace=True),
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

        channel_b1 = out_channel // 2
        channel_b2 = out_channel // 4
        channel_b3 = out_channel // 8
        cur_size = img_size // 16
        fin_channel = cur_size * cur_size * channel_b3

        blocks = [
            nn.Conv2d(in_channel, channel_b1, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel_b1, channel_b1, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel_b1, channel_b1, 3, padding=1),
        ]

        for i in range(n_res_block):
            blocks.append(ResBlock(channel_b1, channel_b1))

        blocks.extend([
            nn.Conv2d(channel_b1, channel_b2, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel_b2, channel_b2, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel_b2, channel_b2, 3, padding=1),
        ])

        for i in range(n_res_block):
            blocks.append(ResBlock(channel_b2, channel_b2))

        blocks.extend([
            nn.Conv2d(channel_b2, channel_b3, 3, padding=1),
            nn.Flatten(start_dim=1, end_dim=-1),
            nn.Linear(fin_channel, out_channel),
            nn.ReLU(inplace=True),
            nn.Linear(out_channel, out_channel)
        ])

        self.blocks = nn.Sequential(*blocks)

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
        out_channel, 
        n_res_block, 
    ):
        super().__init__()

        channel_b1 = in_channel // 2
        channel_b2 = in_channel // 4
        self.ini_size = img_size // 16
        self.ini_channel = in_channel // 8
        ini_mapping = self.ini_size * self.ini_size * self.ini_channel
        print(ini_mapping)

        self.input_mapping = nn.Linear(in_channel, ini_mapping)

        blocks = [nn.Conv2d(self.ini_channel, channel_b1, 3, padding=1)]

        for i in range(n_res_block):
            blocks.append(ResBlock(channel_b1, channel_b1))

        blocks.append(nn.ReLU(inplace=True))

        blocks.extend(
            [
                nn.ConvTranspose2d(channel_b1, channel_b1, 4, stride=2, padding=1),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(channel_b1, channel_b1, 4, stride=2, padding=1),
            ]
        )

        for i in range(n_res_block):
            blocks.append(ResBlock(channel_b1, channel_b1))

        blocks.extend(
            [
                nn.ConvTranspose2d(channel_b1, channel_b2, 4, stride=2, padding=1),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(channel_b2, channel_b2, 4, stride=2, padding=1),
            ]
        )

        blocks.extend([
                nn.ConvTranspose2d(channel_b2, channel_b2, 3, stride=1, padding=1),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(channel_b2, out_channel, 3, stride=1, padding=1)]
        )

        self.blocks = nn.Sequential(*blocks)

    def forward(self, input):
        img = self.input_mapping(input)
        img = img.view(-1, self.ini_channel, self.ini_size, self.ini_size)
        return self.blocks(img)

class MapDecoder(nn.Module):
    def __init__(
        self, 
        in_channel, 
        out_channel, 
        channel,
        map_size,
        n_res_block, 
        n_res_channel
    ):
        super().__init__()

        self.input_mapping = nn.Linear(in_channel, map_size * map_size * channel)
        self.map_size = map_size
        self.channel = channel

        blocks = [nn.Conv2d(channel, channel, 3, padding=1)]
        for i in range(n_res_block):
            blocks.append(ResBlock(channel, n_res_channel))

        blocks.append(nn.ReLU(inplace=True))

        blocks.extend([
                nn.ConvTranspose2d(channel, channel, 3, stride=1, padding=1),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(channel, out_channel, 3, stride=1, padding=1)]
        )

        self.blocks = nn.Sequential(*blocks)

    def forward(self, input):
        out = self.input_mapping(input)
        out = out.view(-1, self.channel, self.map_size, self.map_size)
        return self.blocks(out)
