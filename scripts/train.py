import torch
from transformers import GPT2TokenizerFast

from gpt2 import Config, Transformer
from gpt2.training import TransformerTrainer, TransformerTrainingArgs, load_tinystories


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def main():
    config = Config()
    args = TransformerTrainingArgs()

    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token  # GPT-2 has no pad token; same as HookedTransformer sets

    model = Transformer(config).to(get_device())
    dataset_dict = load_tinystories(tokenizer, n_ctx=config.n_ctx)

    trainer = TransformerTrainer(args, model, tokenizer, dataset_dict)
    trainer.train()


if __name__ == "__main__":
    main()
