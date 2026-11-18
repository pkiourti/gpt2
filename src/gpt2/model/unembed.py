import torch
import torch.nn as nn
from torch import Tensor
import einops
from jaxtyping import Float, Int
from gpt2.config import Config

class Unembed(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.W_U = nn.Parameter(torch.empty((cfg.d_model, cfg.d_vocab)))
        self.b_U = nn.Parameter(torch.zeros((cfg.d_vocab)), requires_grad=False)
        nn.init.normal_(self.W_U, std=self.cfg.init_range)

    def forward(
        self,
        normalized_resid_final: Float[Tensor, "batch position d_model"]
    ) -> Float[Tensor, "batch position d_vocab"]:
        out = (
            einops.einsum(
                normalized_resid_final,
                self.W_U,
                "batch position d_model, d_model d_vocab -> batch position d_vocab"
            )
            + self.b_U
        )
        return out