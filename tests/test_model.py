"""Load GPT-2 small's weights into our Transformer and check it matches transformer_lens's reference."""

import pytest
import torch
from transformer_lens import HookedTransformer

from gpt2 import Config, Transformer
from gpt2.induction import icl_score, induction_metrics

TEXT = "I am an amazing autoregressive, decoder-only, GPT-2 style transformer."


@pytest.fixture(scope="module")
def reference() -> HookedTransformer:
    return HookedTransformer.from_pretrained(
        "gpt2-small",
        fold_ln=False,
        center_unembed=False,
        center_writing_weights=False,
        device="cpu",
    )


@pytest.fixture(scope="module")
def model(reference: HookedTransformer) -> Transformer:
    model = Transformer(Config(debug=False))
    missing, _unexpected = model.load_state_dict(reference.state_dict(), strict=False)
    assert missing == [], f"Our model has weights the reference doesn't: {missing}"
    return model.eval()


@pytest.fixture(scope="module")
def tokens(reference: HookedTransformer) -> torch.Tensor:
    return reference.to_tokens(TEXT)


@torch.inference_mode()
def test_logits_match_reference(model, reference, tokens):
    torch.testing.assert_close(model(tokens), reference(tokens), atol=1e-4, rtol=1e-4)


@torch.inference_mode()
def test_each_block_matches_reference(model, reference, tokens):
    _, cache = reference.run_with_cache(tokens)
    for i, block in enumerate(model.blocks):
        out = block(cache[f"blocks.{i}.hook_resid_pre"])
        torch.testing.assert_close(out, cache[f"blocks.{i}.hook_resid_post"], atol=1e-4, rtol=1e-4)


@torch.inference_mode()
def test_log_probs_shape(model, tokens):
    log_probs = model.get_log_probs(model(tokens), tokens)
    assert log_probs.shape == (tokens.shape[0], tokens.shape[1] - 1)


def test_gpt2_small_has_known_induction_heads(model):
    # GPT-2 small's best-known induction heads are 5.1, 5.5, 6.9 and 7.2 (Olsson et al.; ARENA 1.2).
    metrics = induction_metrics(model, seq_len=50, batch=4)
    assert metrics["induction/loss_second_copy"] < 1.0 < 5.0 < metrics["induction/loss_first_copy"]
    assert metrics["heads/induction_L5H5"] > 0.5
    assert metrics["heads/prev_token_L4H11"] > 0.5  # a well-known previous-token head


def test_attn_only_model_has_no_mlp():
    model = Transformer(Config(d_model=64, n_heads=2, d_head=32, n_layers=2, n_ctx=32, attn_only=True))
    assert not any("mlp" in name or "ln2" in name for name, _ in model.named_parameters())
    tokens = torch.randint(0, 50257, (2, 16))
    assert model(tokens).shape == (2, 16, 50257)
    metrics = icl_score(model, tokens, early=2, late=10)
    assert set(metrics) == {"icl/score", "icl/loss_at_token_2", "icl/loss_at_token_10"}
