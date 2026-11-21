import torch
import torch.nn as nn
from torch import Tensor
from jaxtyping import Float, Int

from gpt2.config import Config
from gpt2.model.layer_norm import LayerNorm
from gpt2.model.attention import Attention
from gpt2.model.mlp import MLP

class TransformerBlock(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.ln1 = LayerNorm(cfg)
        self.attn = Attention(cfg)
        if not cfg.attn_only:
            self.ln2 = LayerNorm(cfg)
            self.mlp = MLP(cfg)

    def forward(
        self,
        resid_pre: Float[Tensor, "batch position d_model"]
    ) -> Float[Tensor, "batch position d_model"]:
        resid_mid = self.attn(self.ln1(resid_pre)) + resid_pre
        if self.cfg.attn_only:
            return resid_mid
        resid_post = self.mlp(self.ln2(resid_mid)) + resid_mid
        return resid_post