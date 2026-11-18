import torch as t
import numpy as np
from torch import Tensor
from jaxtyping import Float, Int
from transformers import GPT2TokenizerFast

from gpt2.model import Transformer


class TransformerSampler:
    def __init__(self, model: Transformer, tokenizer: GPT2TokenizerFast):
        self.model = model
        self.device = model.device
        self.cfg = model.cfg
        self.tokenizer = tokenizer

    @t.inference_mode()
    def sample(self, prompt: str, max_tokens_generated=100, verbose=False, **kwargs) -> str:
        """
        Returns a string of autoregressively generated text, starting from the prompt.

        Sampling terminates at max_tokens_generated, or when the model generates an end-of-sequence token. kwargs are
        passed to sample_next_token, to give detailed instructions on how new tokens are chosen.
        Pass `seed` to make generation reproducible.
        """
        self.model.eval()
        seed = kwargs.pop("seed", None)
        if seed is not None:
            t.manual_seed(seed)
            np.random.seed(seed)
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)[0]
        for i in range(max_tokens_generated):
            logits = self.model(input_ids[None, -self.cfg.n_ctx:])
            logits = logits[0, -1]
            sample = TransformerSampler.sample_next_token(input_ids, logits, **kwargs)
            input_ids = t.cat([input_ids, t.tensor([sample], device=self.device)], dim=-1)
            if verbose:
                print(self.tokenizer.decode(input_ids, skip_special_tokens=True, end="\r"))
            if sample == getattr(self.tokenizer, "eos_token_id", None):
                break
        return self.tokenizer.decode(input_ids, skip_special_tokens=True)

    @staticmethod
    def sample_next_token(
        input_ids: Int[Tensor, " seq_len"],
        logits: Float[Tensor, "d_vocab"],
        temperature=1.0,
        top_k=0,
        top_p=0.0,
        frequency_penalty=0.0,
    ) -> int:
        assert input_ids.ndim == 1, "input_ids should be a 1D sequence of token ids"
        assert logits.ndim == 1, "logits should be a 1D tensor of shape (d_vocab,)"
        assert temperature >= 0, "Temperature should be non-negative"
        assert 0 <= top_p <= 1.0, "Top-p must be a probability"
        assert 0 <= top_k, "Top-k must be non-negative"
        assert not (top_p != 0 and top_k != 0), "At most one of top-p and top-k supported"

        # Apply all the specialized sampling methods. The frequency penalty goes first, so that it
        # still applies when temperature == 0, and so that its strength doesn't depend on temperature.
        if frequency_penalty != 0.0:
            logits = TransformerSampler.apply_frequency_penalty(input_ids, logits, frequency_penalty)
        if temperature == 0:
            return TransformerSampler.greedy_search(logits)
        elif temperature != 1.0:
            logits = TransformerSampler.apply_temperature(logits, temperature)
        if top_k > 0:
            return TransformerSampler.sample_top_k(logits, top_k)
        if top_p > 0.0:
            return TransformerSampler.sample_top_p(logits, top_p)
        return TransformerSampler.sample_basic(logits)

    @staticmethod
    def greedy_search(logits: Float[Tensor, "d_vocab"]) -> int:
        """
        Returns the most likely token (as an int).
        """
        return logits.argmax()

    @staticmethod
    def apply_temperature(logits: Float[Tensor, "d_vocab"], temperature: float) -> Float[Tensor, "d_vocab"]:
        """
        Applies temperature scaling to the logits.
        """
        return logits / temperature

    @staticmethod
    def apply_frequency_penalty(
        input_ids: Int[Tensor, " seq_len"], logits: Float[Tensor, "d_vocab"], freq_penalty: float
    ) -> Float[Tensor, "d_vocab"]:
        """
        Applies a frequency penalty to the logits.
        """
        d_vocab = logits.size(0)
        token_freq = t.bincount(input_ids, minlength=d_vocab)
        return logits - freq_penalty * token_freq

    @staticmethod
    def sample_basic(logits: Float[Tensor, "d_vocab"]) -> int:
        """
        Samples from the distribution defined by the logits.
        """
        sampled_token = t.distributions.categorical.Categorical(logits=logits).sample()
        return sampled_token.item()

    @staticmethod
    def sample_top_k(logits: Float[Tensor, "d_vocab"], k: int) -> int:
        """
        Samples from the top k most likely tokens.
        """
        top_k = logits.argsort()[-1*k:]
        sampled_token = t.distributions.categorical.Categorical(logits=logits[top_k]).sample()
        return top_k[sampled_token].item() #sampled_token.item()

    @staticmethod
    def sample_top_p(logits: Float[Tensor, "d_vocab"], top_p: float, min_tokens_to_keep: int = 1) -> int:
        """
        Samples from the most likely tokens which make up at least p cumulative probability.
        """
        sorted_logits, sorted_indices = logits.sort(descending=True, stable=True)
        cumsum = t.cumsum(sorted_logits.softmax(-1), dim=-1)
        ids = t.where(cumsum >= top_p)[0]
        sorted_logits[ids[0]+1:] = 0
        len_tokens = max(len(sorted_logits[:ids[0]+1]), min_tokens_to_keep)
        keep_logits = sorted_logits[sorted_indices[:len_tokens]]
        next_token = t.distributions.categorical.Categorical(logits=keep_logits).sample()
        return sorted_indices[next_token].item()

    @t.inference_mode()
    def beam_search(
        self,
        prompt: str,
        num_return_sequences: int,
        num_beams: int,
        max_new_tokens: int,
        no_repeat_ngram_size: int | None = None,
    ) -> list[tuple[float, str]]:
        """
        Implements a beam search, by repeatedly performing the `generate` and `filter` steps (starting from the initial
        prompt) until either of the two stopping criteria are met: (1) we've generated `max_new_tokens` tokens, or (2)
        we've generated `num_return_sequences` terminating sequences.
        """
        raise NotImplementedError()
