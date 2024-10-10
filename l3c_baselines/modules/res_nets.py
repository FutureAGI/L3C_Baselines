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

class ImageEncoder(nn.Module):
    """
    Change [B*NT, C_in, 128, 128] to [B*NT, C_out]
    """
    def __init__(self, img_size, in_channel, out_channel, n_res_block):
        super().__init__()

        channel_b1 = out_channel // 64
        channel_b2 = out_channel // 32
        channel_b3 = out_channel // 16
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

class ImageDecoder(nn.Module):
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

        channel_b1 = hidden_size // 8
        self.ini_channel = hidden_size // 32
        self.ini_size = img_size // 8
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

class MLPEncoder(nn.Module):
    def __init__(
        self,
        input_type,
        hidden_size, # Can be a int or a list of ints
        dropout=0.0
    ):
        super().__init__()

        if(input_type.startswith("Discrete")):
            input_size = int(input_type.replace("Discrete", ""))
            self.is_continuous = False
            self.encoder_layer = nn.Embedding(input_size, hidden_size)
        elif(input_type.startswith("Continuous")):
            input_size = int(input_type.replace("Continuous", ""))
            self.is_continuous = True
            if(isinstance(hidden_size, tuple) or isinstance(hidden_size, list)):
                layers = []
                ph = input_size
                for h in hidden_size[:-1]:
                    layers.append(nn.Linear(ph, h))
                    layers.append(nn.GELU())
                    layers.append(nn.Dropout(dropout))
                    ph = h
                layers.append(nn.Linear(ph, hidden_size[-1]))
                self.encoder_layer = nn.Sequential(*layers)
            else:
                self.encoder_layer = nn.Linear(input_size, hidden_size)
        else:
            raise Exception(f"action input type must be ContinuousXX or DiscreteXX, unrecognized `{output_type}`")

    def forward(self, input):
        return self.encoder_layer(input)

class ResidualMLPDecoder(nn.Module):
    def __init__(
        self,
        input_size,
        hidden_size,  # Can be a int or a list of ints
        output_type,  # Output Type is selected between "ContinousXX" and "DiscreteXX"
        dropout=0.10,
        layer_norm=True,
        residual_connect=True
    ):
        super().__init__()

        if(layer_norm):
            self.layer_norm = nn.LayerNorm(input_size, eps=1.0e-5)
        else:
            self.layer_norm = nn.Identity()

        self.residual_connect = residual_connect
        if(residual_connect):
            if(isinstance(hidden_size, tuple) or isinstance(hidden_size, list)):
                hiddens = list(hidden_size) + [input_size]
            else:
                hiddens = [hidden_size, input_size]
        else:
            if(isinstance(hidden_size, tuple) or isinstance(hidden_size, list)):
                hiddens = list(hidden_size)
            else:
                hiddens = [hidden_size]

        layers = []
        ph = input_size
        for h in hidden_size[:-1]:
            layers.append(nn.Linear(ph, h))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(dropout))
            ph = h
        layers.append(nn.Linear(ph, input_size))
        self.decoder_layer_pre = nn.Sequential(*layers)

        if(output_type.startswith("Discrete")):
            output_size = int(output_type.replace("Discrete", ""))
            self.is_continuous = False
            self.decoder_layer_post = nn.Linear(hiddens[-1], output_size)
            self.decoder_layer_output = nn.Softmax(dim=-1)
        elif(output_type.startswith("Continuous")):
            output_size = int(output_type.replace("Continuous", ""))
            self.is_continuous = True
            self.decoder_layer_post = nn.Linear(hiddens[-1], output_size)
            self.decoder_layer_output = nn.Identity()
        else:
            raise Exception(f"action output type must be ContinuousXX or DiscreteXX, unrecognized `{output_type}`")

    def forward(self, input, T=1.0):
        src = self.layer_norm(input)
        out = self.act_decoder_pre(src)
        if(self.residual_connect):
            out = self.act_decoder_post(out + src)
        else:
            out = self.act_decoder_post(out)
        return self.act_decoder_output(out / T)