import datasets
from transformer_lens.utils import tokenize_and_concatenate
from transformers import PreTrainedTokenizerBase


def load_tinystories(
    tokenizer: PreTrainedTokenizerBase,
    n_ctx: int,
    n_samples: int = 200_000,
    test_size: int = 1000,
    num_proc: int = 8,
) -> datasets.DatasetDict:
    """Load TinyStories, tokenize it into sequences of length n_ctx, and split into train/test."""
    dataset = datasets.load_dataset("roneneldan/TinyStories", split="train")
    dataset = dataset.select(range(n_samples))
    tokenized = tokenize_and_concatenate(
        dataset,
        tokenizer,
        streaming=False,
        max_length=n_ctx,
        column_name="text",
        add_bos_token=True,
        num_proc=num_proc,
    )
    return tokenized.train_test_split(test_size=test_size)
