import torch
from torch import nn
from torch.nn import functional as F

class VAE(nn.Module):
    def __init__(
        self,
        hidden_size,
        encoder,
        decoder):
        super().__init__()
        self.hidden_size = hidden_size
        self.encoder = encoder
        self.decoder = decoder
        self.layer_mean = nn.Linear(encoder.hidden_size, hidden_size)
        self.layer_var = nn.Linear(encoder.hidden_size, hidden_size)
    
    def forward(self, inputs):
        # input shape: [B, NT, C, W, H]
        nB, nT, nC, nW, nH = inputs.shape
        hidden = self.encoder(inputs.reshape(nB * nT, nC, nW, nH))
        z_exp = self.layer_mean(hidden)
        z_log_var = self.layer_var(hidden)
        return z_exp.reshape(nB, nT, self.hidden_size), z_log_var.reshape(nB, nT, self.hidden_size)

    def reconstruct(self, inputs, _sigma=1.0):
        nB, nT, nC, nW, nH = inputs.shape
        z_exp, z_log_var = self.forward(inputs)
        epsilon = torch.randn_like(z_log_var).to(z_log_var.device)
        z = z_exp + _sigma * torch.exp(z_log_var / 2) * epsilon
        outputs = self.decoder(z.reshape(nB * nT, self.hidden_size))
        outputs = outputs.reshape(nB, nT, nC, nW, nH)
        return outputs, z_exp, z_log_var

    def loss(self, inputs, _lambda=1.0e-5, _sigma=1.0):
        outputs, z_exp, z_log_var = self.reconstruct(inputs, _sigma = _sigma)
        kl_loss = torch.mean(-0.5 * torch.sum(1 + z_log_var - torch.square(z_exp) - torch.exp(z_log_var), axis=1))
        reconstruction_loss = F.mse_loss(outputs, inputs)

        return reconstruction_loss + _lambda * kl_loss
