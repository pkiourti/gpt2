import torch as t
import torch.nn as nn
from torch import Tensor
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
    def device(self) -> t.device:
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
        log_probs = logits.log_softmax(dim=-1)
        # Get logprobs for the first seq_len-1 predictions (so we can compare them with the actual next tokens)
        log_probs_for_tokens = log_probs[:, :-1].gather(dim=-1, index=tokens[:, 1:].unsqueeze(-1)).squeeze(-1)
        return log_probs_for_tokens