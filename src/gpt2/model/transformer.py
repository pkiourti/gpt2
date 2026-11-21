import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
from gpt2.model.embed import Embed, PosEmbed
from gpt2.model.block import TransformerBlock
from gpt2.model.layer_norm import LayerNorm
from gpt2.model.unembed import Unembed
from gpt2.config import Config
from jaxtyping import Float, Int

class Transformer(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.embed = Embed(cfg)
        self.pos_embed = PosEmbed(cfg)
        self.blocks = nn.ModuleList([TransformerBlock(cfg) for _ in range(cfg.n_layers)])
        self.ln_final = LayerNorm(cfg)
        self.unembed = Unembed(cfg)

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def forward(
        self,
        tokens: Int[Tensor, "batch position"]
    ) -> Float[Tensor, "batch position d_vocab"]:
        residual = self.embed(tokens) + self.pos_embed(tokens)
        for block in self.blocks:
            residual = block(residual)
        logits = self.unembed(self.ln_final(residual))
        return logits

    def get_log_probs(
        self,
        logits: Float[Tensor, "batch posn d_vocab"], tokens: Int[Tensor, "batch posn"]
    ) -> Float[Tensor, "batch posn-1"]:
        batch, size = tokens.shape
        nll = F.cross_entropy(
            logits[:, :-1].reshape(-1, self.cfg.d_vocab),
            tokens[:, 1:].reshape(-1),
            reduction="none"
        )
        return -nll.reshape(batch, size - 1)