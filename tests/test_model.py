"""Load GPT-2 small's weights into our Transformer and check it matches transformer_lens's reference."""

import pytest
import torch
from transformer_lens import HookedTransformer

from gpt2 import Config, Transformer

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
