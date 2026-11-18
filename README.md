# GPT-2

GPT-2 small written from scratch (following ARENA), plus a sampler and a trainer for TinyStories.

## Layout

```
src/gpt2/
  config.py          Config: model hyperparameters
  model/             Embed, PosEmbed, LayerNorm, Attention, MLP, TransformerBlock, Unembed, Transformer
  sampling.py        TransformerSampler: greedy, temperature, top-k, top-p, frequency penalty
  training/          TransformerTrainingArgs, load_tinystories, TransformerTrainer
scripts/train.py     Train on TinyStories and log to wandb
tests/               Compare the model against transformer_lens's reference GPT-2
```

## Usage

From the ARENA workspace root:

```bash
uv sync --all-packages
uv run python gpt2/scripts/train.py
uv run pytest gpt2
```

Set `WANDB_API_KEY` (or run `uv run wandb login`) before training.
