import torch as t
import torch.nn as nn
from torch import Tensor
from jaxtyping import Float, Int
from gpt2.config import Config

class LayerNorm(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.w = nn.Parameter(t.ones(cfg.d_model))
        self.b = nn.Parameter(t.zeros(cfg.d_model))

    def forward(self, residual: Float[Tensor, "batch posn d_model"]) -> Float[Tensor, "batch posn d_model"]:
        mean = residual.mean(dim=-1, keepdim=True)
        std = (residual.var(dim=-1, keepdim=True, correction=0) + self.cfg.layer_norm_eps).sqrt()

        # einops.reduce(residual, "batch posn d_model -> posn d_model", "mean")
        residual = (residual - mean) / std
        return residual * self.w + self.b