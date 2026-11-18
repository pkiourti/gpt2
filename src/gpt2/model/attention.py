import torch as t
import torch.nn as nn
from torch import Tensor
import einops
from jaxtyping import Float, Int
from gpt2.config import Config

class Attention(nn.Module):
    IGNORE: Float[Tensor, ""]

    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.W_Q = nn.Parameter(t.empty((cfg.n_heads, cfg.d_model, cfg.d_head)))
        self.W_K = nn.Parameter(t.empty((cfg.n_heads, cfg.d_model, cfg.d_head)))
        self.W_V = nn.Parameter(t.empty((cfg.n_heads, cfg.d_model, cfg.d_head)))
        self.W_O = nn.Parameter(t.empty((cfg.n_heads, cfg.d_head, cfg.d_model)))
        self.b_Q = nn.Parameter(t.zeros((cfg.n_heads, cfg.d_head)))
        self.b_K = nn.Parameter(t.zeros((cfg.n_heads, cfg.d_head)))
        self.b_V = nn.Parameter(t.zeros((cfg.n_heads, cfg.d_head)))
        self.b_O = nn.Parameter(t.zeros((cfg.d_model)))
        nn.init.normal_(self.W_Q, std=self.cfg.init_range)
        nn.init.normal_(self.W_K, std=self.cfg.init_range)
        nn.init.normal_(self.W_V, std=self.cfg.init_range)
        nn.init.normal_(self.W_O, std=self.cfg.init_range)
        self.register_buffer("IGNORE", t.tensor(float("-inf"), dtype=t.float32))

    def forward(self,
                normalized_resid_pre: Float[Tensor, "batch pos d_model"]
                ) -> Float[Tensor, "batch pos d_model"]:
        q = (
                einops.einsum(
                    normalized_resid_pre,
                    self.W_Q,
                    "batch pos d_model, n_heads d_model d_head -> batch pos n_heads d_head",
                )
                + self.b_Q
        )
        k = (
                einops.einsum(
                    normalized_resid_pre,
                    self.W_K,
                    "batch pos d_model, n_heads d_model d_head -> batch pos n_heads d_head"
                )
                + self.b_K
        )
        v = (
                einops.einsum(
                    normalized_resid_pre,
                    self.W_V,
                    "batch pos d_model, n_heads d_model d_head -> batch pos n_heads d_head"
                )
                + self.b_V
        )
        attn_scores = einops.einsum(
            q,
            k,
            "batch posq n_heads d_head, batch posk n_heads d_head -> batch n_heads posq posk"
        )
        attn_scores_masked = self.apply_causal_mask(attn_scores / self.cfg.d_head ** 0.5)
        attn_pattern = attn_scores_masked.softmax(-1)

        z = einops.einsum(
            v,
            attn_pattern,
            "batch posk n_heads d_head, batch n_heads posq posk -> batch posq n_heads d_head"
        )

        attn_out = (
                einops.einsum(
                    z,
                    self.W_O,
                    "batch posq n_heads d_head, n_heads d_head d_model -> batch posq d_model"
                )
                + self.b_O
        )

        return attn_out

    def apply_causal_mask(
            self,
            attn_scores: Float[Tensor, "batch n_heads query_pos key_pos"],
    ) -> Float[Tensor, "batch n_heads query_pos key_pos"]:
        q_size, k_size = attn_scores.size(-2), attn_scores.size(-1)
        all_ones = t.ones(q_size, k_size, device=attn_scores.device)
        mask = t.triu(all_ones, diagonal=1).bool()
        return attn_scores.masked_fill_(mask, self.IGNORE)