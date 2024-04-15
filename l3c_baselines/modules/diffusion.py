import torch
from torch import nn
from torch.nn import functional as F
class DiffusionLayers(nn.Module):
    def __init__(self, T, hidden_size, condition_size, inner_hidden_size, beta=(0.05, 0.20)):
        super().__init__()
        self.betas = torch.linspace(beta[0], beta[1], T)
        self.betas = torch.cat([torch.tensor([0.0]), self.betas], dim=0)
        self.alphas = 1 - self.betas
        self._alphas = torch.cumprod(self.alphas, axis=0, dtype=torch.float)

        self.condition_size = condition_size
        self.hidden_size = hidden_size
        self.input_size = hidden_size + condition_size
        self.diffusion_layers_1 = nn.Sequential(
            nn.LayerNorm(self.input_size, eps=1.0e-5),
            nn.Linear(self.input_size, inner_hidden_size), 
            nn.GELU(),
            nn.Dropout(0.10),
            nn.Linear(inner_hidden_size, hidden_size),
            nn.GELU()) 
        self.diffusion_layers_2 = nn.Linear(hidden_size, hidden_size)
        self.T = T
        self.t_embedding = nn.Embedding(T, hidden_size)

    def forward(self, xt, t, cond):
        """
        xt: [B, NT, H], float
        t: [B, NT], int
        cond: [B, NT, HC], float
        """
        assert cond.shape[:2] == xt.shape[:2] and cond.shape[2] == self.condition_size
        t_emb = self.t_embedding(t - 1)
        outputs = torch.cat([xt + t_emb, cond], dim=-1)
        outputs = self.diffusion_layers_1(outputs) + xt
        outputs = self.diffusion_layers_2(outputs)

        return outputs
    
    def loss(self, x0, cond):
        t = torch.randint(low=1, high=self.T + 1, size=x0.shape[:2], dtype=torch.int64, device=x0.device)
        x_t, eps = self._forward(x0, t)
        eps_t = self.forward(x_t, t, cond)
        loss = F.mse_loss(eps_t, eps)
        return loss

    def _forward(self, x0, t):
        eps = torch.randn_like(x0).to(x0.device)
        a_t = torch.take(self._alphas.to(x0.device), t).unsqueeze(-1)
        x_t = torch.sqrt(a_t) * x0 + torch.sqrt(1 - a_t) * eps
        return x_t, eps

    def inference(self, cond):
        assert cond.shape[2] == self.condition_size
        z_list = []
        steps = [self.T, 2 * self.T // 3, self.T // 3, 1]
        with torch.no_grad():
            x_T = torch.randn(*cond.shape[:2], self.hidden_size)
            x_t = x_T.to(cond.device)
            for t in range(self.T, 0, -1):
                _t = torch.full(cond.shape[:2], t, dtype=torch.int64, device=cond.device)
                a_t = self._alphas[t]
                a_t_ = self._alphas[t-1]
                b_t = self.betas[t]

                #eps = torch.randn_like(x_t)
                #if(t == 1):
                #    eps = eps * 0
                #x_t = 1.0 / torch.sqrt(1 - b_t) * (x_t - (b_t / torch.sqrt(1 - a_t)) * self.forward(x_t, _t, cond)) + torch.sqrt(b_t) * eps
                eps_t = self.forward(x_t, _t, cond)
                x_t = torch.sqrt(a_t_ / a_t) * (x_t - torch.sqrt(1 - a_t) * eps_t) + torch.sqrt(1 - a_t_) * eps_t
                if(t in steps):
                    z_list.append(x_t.detach())
        return z_list

if __name__=="__main__":
    layer = DiffusionLayers(24, 32, 512, 512)
    x0 = torch.randn(4, 8, 32)
    cond = torch.randn(4, 8, 512)
    loss = layer.loss(x0, cond)
    output = layer.inference(cond)
    print("Loss:", loss)
    print("Output:", output.shape)
