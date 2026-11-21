import os

import datasets
from transformer_lens.utils import tokenize_and_concatenate
from transformers import PreTrainedTokenizerBase


def load_text_dataset(
    path: str,
    tokenizer: PreTrainedTokenizerBase,
    n_ctx: int,
    n_samples: int,
    name: str | None = None,
    column: str = "text",
    streaming: bool = False,
    test_size: int = 1000,
    num_proc: int | None = None,
) -> datasets.DatasetDict:
    """Load a Hugging Face text dataset, tokenize it into n_ctx-token sequences, and split into train/test.

    Documents are joined with BOS tokens between them and cut into rows of exactly n_ctx tokens.
    streaming=True downloads only the first n_samples documents, for huge datasets like FineWeb-Edu.
    """
    if streaming:
        stream = datasets.load_dataset(path, name=name, split="train", streaming=True)
        dataset = datasets.Dataset.from_list([{column: row[column]} for row in stream.take(n_samples)])
    else:
        dataset = datasets.load_dataset(path, name=name, split="train")
        dataset = dataset.select(range(min(n_samples, len(dataset))))
    dataset = dataset.select_columns([column])
    tokenized = tokenize_and_concatenate(
        dataset,
        tokenizer,
        streaming=False,
        max_length=n_ctx,
        column_name=column,
        add_bos_token=True,
        num_proc=num_proc or os.cpu_count() or 1,
    )
    return tokenized.train_test_split(test_size=test_size)


def load_tinystories(
    tokenizer: PreTrainedTokenizerBase, n_ctx: int, n_samples: int = 200_000, **kwargs
) -> datasets.DatasetDict:
    """TinyStories: ~2M short synthetic children's stories (~200 tokens each)."""
    return load_text_dataset("roneneldan/TinyStories", tokenizer, n_ctx, n_samples, **kwargs)


def load_fineweb_edu(
    tokenizer: PreTrainedTokenizerBase, n_ctx: int, n_samples: int = 100_000, **kwargs
) -> datasets.DatasetDict:
    """FineWeb-Edu (10B-token sample), streamed: educational web pages, ~1000 tokens each on average.

    Long documents with repeated names and terms make in-context copying pay off, so induction heads
    form clearly. 100k documents is roughly 100M tokens.
    """
    return load_text_dataset(
        "HuggingFaceFW/fineweb-edu", tokenizer, n_ctx, n_samples, name="sample-10BT", streaming=True, **kwargs
    )
