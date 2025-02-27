import torch
import math
from torch import nn
from torch.nn import functional as F
from airsoul.utils import weighted_loss

class BasicModel(nn.Module):
    def __init__(self, 
                 hidden_size: int,
                 condition_size: int,
                 t_embedding_size: int,
                 inner_hidden_size: int,
                 diffusion_T: int,
                 dropout: float = 0.1,
                 use_cond_layer_norm: bool = True):
        super().__init__()
        
        # Calculate input size
        self.input_size = hidden_size + condition_size + t_embedding_size
        
        # Normalization layer
        self.pre_diffusion_norm = nn.LayerNorm(t_embedding_size, eps=1e-5)
        self.cond_layer_norm = nn.LayerNorm(condition_size, eps=1e-5) if use_cond_layer_norm else nn.Identity()
        self.t_embedding = nn.Embedding(diffusion_T, t_embedding_size)
        # Core network layer
        self.diffusion_layers_1 = nn.Sequential(
            nn.Linear(self.input_size, inner_hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(inner_hidden_size, hidden_size),
            nn.GELU()
        )
        self.diffusion_layers_2 = nn.Linear(hidden_size, hidden_size)

    def forward(self, 
               xt: torch.Tensor, 
               t: int, 
               cond: torch.Tensor) -> torch.Tensor:
        """
        xt:    [B, NT, H]
        t_emb: [B, NT, D_t]
        cond:  [B, NT, D_c]
        """
        # Normalization processing
        t_emb = self.pre_diffusion_norm(self.t_embedding(t))
        cond = self.cond_layer_norm(cond)
        
        # Feature concatenation and processing
        combined = torch.cat([xt, t_emb, cond], dim=-1)
        x = self.diffusion_layers_1(combined)
        x = self.diffusion_layers_2(x)
        return x


class DiffusionLayers(nn.Module):
    # def __init__(self, T, hidden_size, condition_size, inner_hidden_size, beta=(0.05, 0.20)):
    def __init__(self, config):
        super().__init__()

        if config.schedule == "linear":
            self.betas = torch.linspace(config.beta[0], config.beta[1], config.T)
            self.betas = torch.cat([torch.tensor([0.0]), self.betas], dim=0)
        elif config.schedule == "cosine":
            betas = []
            for i in range(config.T):
                t1 = i / config.T
                beta_t = min(config.beta[1], (1 - math.cos(math.pi * t1)) / 2 * (config.beta[1] - config.beta[0]) + config.beta[0])
                betas.append(beta_t)
            self.betas = torch.tensor(betas, dtype=torch.float32)
            self.betas = torch.cat([torch.tensor([0.0]), self.betas], dim=0)
        elif config.schedule == "scaled_linear":
            # this schedule is very specific to the latent diffusion model.
            self.betas = torch.linspace(config.beta[0]**0.5, config.beta[1]**0.5, config.T, dtype=torch.float32) ** 2
            self.betas = torch.cat([torch.tensor([0.0]), self.betas], dim=0)
        self.alphas = 1 - self.betas
        self._alphas = torch.cumprod(self.alphas, axis=0, dtype=torch.float)

        self.prediction_type = config.prediction_type

        self.condition_size = config.condition_size
        self.hidden_size = config.hidden_size
        self.t_embedding_size = config.t_embedding_size
        self.T = config.T

        model_name = config.diffusion_model_name
        if model_name == "basic":
            self.model = BasicModel(
                hidden_size=config.hidden_size,
                condition_size=config.condition_size,
                t_embedding_size=config.t_embedding_size,
                inner_hidden_size=config.inner_hidden_size,
                diffusion_T=config.T,
                dropout=config.dropout,
                use_cond_layer_norm=config.cond_layer_norm
            )
        elif model_name == "latentlm":
            self.model = LatentLMDiffusionBlock(
                mlp_ratio=config.mlp_ratio,
                hidden_size=config.hidden_size,
                drop=config.dropout, 
                diffusion_depth = config.block_size)
        

        self.inference_sample_steps = config.inference_sample_steps
        self.need_clip = config.need_clip
        self.clip_threshold = config.clip_threshold
        self.eta = config.eta

    def get_velocity_targets(self, x0, eps, t):  
        device = t.device
        alpha_t = self._alphas.to(device)[t].unsqueeze(-1)  
        sqrt_alpha_t = torch.sqrt(alpha_t)  
        sqrt_sigma_t = torch.sqrt(1. - alpha_t)
    
        v = sqrt_alpha_t * eps - sqrt_sigma_t * x0
        return v

    def step_forward(self, xt, t, cond):
        """
        xt: [B, NT, H], float
        t: [B, NT], int
        cond: [B, NT, HC], float
        """
        
        """
        print("cond.shape[:2]  ",cond.shape[:2])
        print("xt.shape[:2]  ",xt.shape[:2])
        print("self.condition_size  ",self.condition_size)
        print("cond.shape[2]  ",cond.shape[2])
        """ 
        
        assert cond.shape[:2] == xt.shape[:2] and cond.shape[2] == self.condition_size

        t_indices = t - 1  # For t_emb indexing
        outputs = self.model(xt, t_indices, cond)

        return outputs
    
    def loss_DDPM(self, x0, cond, mask=None, reduce_dim=1, t=None, need_cnt=False):
        if(t is None):
            _t = torch.randint(low=1, high=self.T + 1, size=x0.shape[:2], dtype=torch.int64, device=x0.device)
        else:
            _t = torch.full(cond.shape[:2], t, dtype=torch.int64, device=cond.device)
        x_t, eps, _ = self.diffusion_forward(x0, _t)
        eps_t = self.step_forward(x_t, _t, cond)
        
        if self.prediction_type == 'velocity':  
            target = self.get_velocity_targets(x0, eps, _t)
            pred = eps_t
        elif self.prediction_type == 'epslion':
            target = eps
            pred = eps_t
        
        if  need_cnt:
            loss,loss_count_s = weighted_loss(pred.float(), gt=target.float(), loss_type="mse", loss_wht=mask, reduce_dim=reduce_dim, need_cnt=need_cnt)
            return loss, loss_count_s
        else:
            loss = weighted_loss(pred, gt=target, loss_type="mse", loss_wht=mask, reduce_dim=reduce_dim, need_cnt=need_cnt)
            return loss

    def forward(self, x0, cond, mask=None, reduce_dim=1, t=None):
        """
        Allows to back propagate through the whole process
        """

        if(t is None):
            _t = torch.randint(low=1, high=self.T + 1, size=x0.shape[:2], dtype=torch.int64, device=x0.device)
        else:
            _t = torch.full(cond.shape[:2], t, dtype=torch.int64, device=cond.device)
        x_t, eps, a_t = self.diffusion_forward(x0, _t)
        eps_t = self.step_forward(x_t, _t, cond)
        a_0 = torch.full(a_t.shape, 0, dtype=torch.int64, device=cond.device)
        print(x_t.shape, eps.shape, eps_t.shape, a_t.shape, a_0.shape)

        return torch.sqrt(a_0 / a_t) * x_t + (torch.sqrt((1 - a_0) / a_0) - torch.sqrt((1 - a_t) / a_t)) * eps_t

    def diffusion_forward(self, x0, t):
        # x0  = x0.to(torch.float)
        eps = torch.randn_like(x0).to(x0.device)
        a_t = torch.take(self._alphas.to(x0.device), t).unsqueeze(-1)
        x_t = torch.sqrt(a_t) * x0 + torch.sqrt(1 - a_t) * eps
        x_t = torch.clamp(x_t, -self.clip_threshold, self.clip_threshold) if self.need_clip else x_t
        return x_t, eps, a_t

    def inference(self, cond, gt=None, mask=None, reduce_dim=1):
        assert cond.shape[2] == self.condition_size
        z_list = []
        steps = [2 * self.T // 3, self.T // 3, 1]
        
        with torch.no_grad():
            x_T = torch.randn(*cond.shape[:2], self.hidden_size)
            x_t = x_T.to(cond.device)
            z_list.append(x_t.detach())

            for t in range(self.T, 0, -self.inference_sample_steps):
                _t = torch.full(cond.shape[:2], t, dtype=torch.int64, device=cond.device)
                a_t = self._alphas[t]
                last_step = t <= self.inference_sample_steps

                if not last_step:
                    assert t - self.inference_sample_steps > 0
                    a_t_ = self._alphas[t - self.inference_sample_steps]
                    # Construct sigma_t with eta 
                    variance = self._get_variance(a_t, a_t_)
                    std_dev_t = self.eta * variance ** (0.5)
                else:
                    a_t_ = self._alphas[0]
                    variance = self._get_variance(a_t, a_t_)
                    std_dev_t = 0.0 * variance ** (0.5) # Enforces certainty at last step

                model_out = self.step_forward(x_t, _t, cond)
                if self.prediction_type == 'velocity':
                    v = model_out
                    pred_x0 = torch.sqrt(a_t) * x_t + torch.sqrt(1 - a_t) * v
                    pred_epsilon = torch.sqrt(a_t) * v + torch.sqrt(1 - a_t) * x_t
                elif self.prediction_type == 'epslion':
                    pred_epsilon = model_out
                    pred_x0 = (x_t - torch.sqrt(1 - a_t)*pred_epsilon) / torch.sqrt(a_t)
                
                # Compute "direction pointing to x_t" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
                pred_sample_direction = torch.sqrt(1 - a_t_ - std_dev_t**2)  * pred_epsilon
                # Clip for xt stability
                pred_x0 = torch.clamp(pred_x0, -self.clip_threshold, self.clip_threshold) if self.need_clip else pred_x0
                
                # Compute x_t without "random noise" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
                x_t = torch.sqrt(a_t_) * pred_x0 + pred_sample_direction
                variance = std_dev_t * torch.randn_like(x_t)
                x_t = x_t + variance
                
                if (t in steps) or last_step:
                    z_list.append(x_t.detach())

                # Debug, test loss
                if gt is not None:
                    _, eps, _ = self.diffusion_forward(gt, _t)
                    loss, cnt = weighted_loss(pred_epsilon, gt=eps, loss_type="mse", loss_wht=mask, reduce_dim=reduce_dim, need_cnt=True)
                    print("epsilon loss:", loss.item()/cnt)

        return z_list
    
    def _get_variance(self, a_t, a_t_):
        beta_prod_t = 1 - a_t
        beta_prod_t_prev = 1 - a_t_

        variance = (beta_prod_t_prev / beta_prod_t) * (1 - a_t / a_t_)

        return variance

############## Latent LM diffusion block ############

class LatentLMDiffusionBlock(nn.Module):
    def __init__(self, hidden_size, mlp_ratio=4.0, drop=0.0, diffusion_depth = 3):
        super().__init__()
        self.diffusion_blocks = nn.ModuleList([
            MLPBlock(hidden_size, mlp_ratio=mlp_ratio) for _ in range(diffusion_depth)
        ])
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.final_layer = FinalLayer(hidden_size, hidden_size)

        cold_start = True
        if cold_start:
            # Initialize timestep embedding MLP:
            nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
            nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)
            # Zero-out adaLN modulation layers in DiT blocks:
            for block in self.diffusion_blocks:
                nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            # Zero-out output layers:
            nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(self.final_layer.linear.weight, 0)

    def forward(self, x, t, condition):
        bsz, seq_len = t.shape if t.dim() > 1 else (t.shape[0], 1)
        t = self.t_embedder(t.view(-1)).view(bsz, seq_len, -1)
        c = condition + t
        #x = self.noisy_x_embedder(x) # nn.Linear(in_channels, hidden_size, bias=False)
        
        for block in self.diffusion_blocks:
            x = block(x, c)
            
        x = self.final_layer(x, c)
        return x


class MLPBlock(nn.Module):
    def __init__(self, hidden_size,  mlp_ratio=4.0, drop=0.0, **block_kwargs):
        super().__init__()
        self.norm = RMSNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio * 2 / 3 / 64) * 64
        self.mlp = SwiGLU(hidden_size, mlp_hidden_dim, drop=drop)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 3 * hidden_size, bias=False)
        )

    def forward(self, x, c):
        shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(3, dim=-1)
        x = x + gate_mlp * self.mlp(modulate(self.norm(x), shift_mlp, scale_mlp))
        return x
    
class SwiGLU(nn.Module):
    def __init__(
        self,
        embed_dim,
        ffn_dim,
        drop=0.,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.fc1 = nn.Linear(self.embed_dim, ffn_dim, bias=False)
        self.gate = nn.Linear(self.embed_dim, ffn_dim, bias=False)
        self.fc2 = nn.Linear(ffn_dim, self.embed_dim, bias=False)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x_shape = x.shape
        x = x.reshape(-1, x.size(-1))
        x = F.silu(self.fc1(x)) * self.gate(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        output = x.view(x_shape)
        return output
    
class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6, elementwise_affine=True):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if self.elementwise_affine:
            self.weight = nn.Parameter(torch.ones(dim))
        else:
            self.register_parameter('weight', None)

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        if self.weight is not None:
            output = output * self.weight
        return output

    def extra_repr(self) -> str:
        return f'dim={self.dim}, eps={self.eps}, elementwise_affine={self.elementwise_affine}'

class ConditionLayer(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.norm_final = RMSNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, x):
        x = self.norm_final(x)
        x = self.linear(x)
        return x

class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=False),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=False),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        # Force float dtype
        t = t.float()  
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding.to(t.dtype)

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb

class FinalLayer(nn.Module):
    """
    The final layer of DiT.
    """
    def __init__(self, hidden_size, output_size):
        super().__init__()
        self.norm_final = RMSNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, output_size, bias=False)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=False)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=-1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x

def modulate(x, shift, scale):
    return x * (1 + scale) + shift
