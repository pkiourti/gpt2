"""Measurements for watching induction heads form during training.

An induction head implements "[A][B] ... [A] -> predict [B]". It needs two heads in different layers:
a previous-token head (each position attends to the one before it), and an induction head whose keys
read that previous-token information, so the current [A] attends to the [B] that followed the last [A].

The measurements here:
- induction_metrics: feed random tokens repeated twice. The first copy is unpredictable (loss ~ ln d_vocab),
  the second copy is fully predictable by copying. Also scores every head for previous-token and induction
  attention on that sequence.
- icl_score: loss at a late position minus loss at an early position on real text. It goes negative once
  the model uses earlier context, and drops sharply when induction heads form (Olsson et al., 2022).
"""

from contextlib import contextmanager
from typing import Iterator

import torch
from jaxtyping import Float, Int
from torch import Tensor

from gpt2.model import Transformer

GPT2_BOS = 50256


@contextmanager
def capture_patterns(model: Transformer) -> Iterator[list]:
    """Record the attention patterns of the last forward pass run inside this block.

    Yields a list that is filled when the block exits: one [batch, n_heads, query_pos, key_pos] tensor per layer.

        with capture_patterns(model) as patterns:
            model(tokens)
        patterns[1][0, 3]   # layer 1, head 3, first sequence
    """
    patterns: list[Tensor] = []
    for block in model.blocks:
        block.attn.store_pattern = True
    try:
        yield patterns
    finally:
        patterns.extend(block.attn.pattern for block in model.blocks)
        for block in model.blocks:
            block.attn.store_pattern = False
            block.attn.pattern = None


def repeated_random_tokens(
    batch: int, seq_len: int, device: torch.device, seed: int = 0, bos: int = GPT2_BOS
) -> Int[Tensor, "batch 2*seq_len+1"]:
    """[BOS, r_1 .. r_seq_len, r_1 .. r_seq_len], with r drawn from ordinary (non-special) token ids."""
    generator = torch.Generator().manual_seed(seed)
    rand = torch.randint(1000, 10000, (batch, seq_len), generator=generator)
    prefix = torch.full((batch, 1), bos)
    return torch.cat([prefix, rand, rand], dim=1).to(device)


def head_scores(
    patterns: list[Float[Tensor, "batch n_heads q k"]], seq_len: int
) -> tuple[Float[Tensor, "n_layers n_heads"], Float[Tensor, "n_layers n_heads"]]:
    """Previous-token and induction scores per head, from patterns on a repeated_random_tokens sequence.

    previous-token score: mean attention from position i to i-1.
    induction score: mean attention from position i in the second copy to i-seq_len+1, the token right after
    position i's earlier occurrence. Both are between 0 and 1.
    """
    prev_token, induction = [], []
    for pattern in patterns:
        prev_token.append(pattern.diagonal(offset=-1, dim1=-2, dim2=-1).mean(dim=(0, -1)))
        induction_diag = pattern.diagonal(offset=-(seq_len - 1), dim1=-2, dim2=-1)
        induction.append(induction_diag[..., -seq_len:].mean(dim=(0, -1)))
    return torch.stack(prev_token).float(), torch.stack(induction).float()


@torch.inference_mode()
def induction_metrics(model: Transformer, seq_len: int = 50, batch: int = 16, seed: int = 0) -> dict[str, float]:
    """Loss on each copy of a repeated random sequence, plus per-head previous-token and induction scores."""
    assert 2 * seq_len + 1 <= model.cfg.n_ctx, "repeated sequence must fit in the context window"
    was_training = model.training
    model.eval()
    tokens = repeated_random_tokens(batch, seq_len, model.device, seed)
    with capture_patterns(model) as patterns:
        logits = model(tokens)
    log_probs = model.get_log_probs(logits, tokens)  # log_probs[:, i] is the prediction of tokens[:, i + 1]
    model.train(was_training)

    # First copy: tokens 1..seq_len. Second copy: tokens seq_len+2..2*seq_len (its first token isn't predictable).
    metrics = {
        "induction/loss_first_copy": -log_probs[:, :seq_len].mean().item(),
        "induction/loss_second_copy": -log_probs[:, seq_len + 1 :].mean().item(),
    }
    prev_token, induction = head_scores(patterns, seq_len)
    for layer in range(prev_token.shape[0]):
        metrics[f"induction/max_induction_score_L{layer}"] = induction[layer].max().item()
        metrics[f"induction/max_prev_token_score_L{layer}"] = prev_token[layer].max().item()
        for head in range(prev_token.shape[1]):
            metrics[f"heads/induction_L{layer}H{head}"] = induction[layer, head].item()
            metrics[f"heads/prev_token_L{layer}H{head}"] = prev_token[layer, head].item()
    return metrics


@torch.inference_mode()
def icl_score(
    model: Transformer, tokens: Int[Tensor, "batch seq"], early: int = 50, late: int | None = None
) -> dict[str, float]:
    """Loss at token `late` minus loss at token `early`, on real text. Negative = later tokens are easier.

    Olsson et al. use late=500, early=50. `late` defaults to min(500, seq - 1), so short contexts still work.
    """
    seq = tokens.shape[1]
    late = min(500, seq - 1) if late is None else late
    assert early < late < seq, f"need early < late < seq, got {early=}, {late=}, {seq=}"
    was_training = model.training
    model.eval()
    tokens = tokens.to(model.device)
    log_probs = model.get_log_probs(model(tokens), tokens)  # index i predicts token i+1
    model.train(was_training)
    loss_early = -log_probs[:, early - 1].mean().item()
    loss_late = -log_probs[:, late - 1].mean().item()
    return {
        "icl/score": loss_late - loss_early,
        f"icl/loss_at_token_{early}": loss_early,
        f"icl/loss_at_token_{late}": loss_late,
    }
