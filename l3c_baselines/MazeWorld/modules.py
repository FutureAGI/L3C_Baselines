import torch
from torch import nn

class ResBlock(nn.Module):
    def __init__(self, in_channel, channel):
        super().__init__()

        self.conv = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(in_channel, channel, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel, in_channel, 1),
        )

    def forward(self, input):
        out = self.conv(input)
        out += input

        return out

class Encoder(nn.Module):
    def __init__(self, in_channel, channel, n_res_block, n_res_channel):
        super().__init__()

        blocks = [
            nn.Conv2d(in_channel, channel // 2, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // 2, channel, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel, channel, 3, padding=1),
        ]

        for i in range(n_res_block):
            blocks.append(ResBlock(channel, n_res_channel))

        blocks.append(nn.ReLU(inplace=True))

        self.blocks = nn.Sequential(*blocks)

    def forward(self, input):
        return self.blocks(input)

class Decoder(nn.Module):
    def __init__(
        self, 
        in_channel, 
        out_channel, 
        channel, 
        n_res_block, 
        n_res_channel
    ):
        super().__init__()

        blocks = [nn.Conv2d(in_channel, channel, 3, padding=1)]

        for i in range(n_res_block):
            blocks.append(ResBlock(channel, n_res_channel))

        blocks.append(nn.ReLU(inplace=True))

        blocks.extend(
            [
                nn.ConvTranspose2d(channel, channel // 2, 4, stride=2, padding=1),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(channel // 2, channel // 4, 4, stride=2, padding=1),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(channel // 4, channel // 8, 4, stride=2, padding=1),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(channel // 8, out_channel, 4, stride=2, padding=1),
            ]
        )

        self.blocks = nn.Sequential(*blocks)

    def forward(self, input):
        return self.blocks(input)

class MapDecoder(nn.Module):
    def __init__(
        self, 
        in_channel, 
        out_channel, 
        channel, 
        n_res_block, 
        n_res_channel
    ):
        super().__init__()

        blocks = [nn.Conv2d(in_channel, channel, 3, padding=1)]

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
        return self.blocks(input)
