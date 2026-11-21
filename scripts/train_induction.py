"""Train a small attention-only transformer and log induction-head metrics as it learns.

    python scripts/train_induction.py --dataset fineweb --steps 10000 --checkpoint-dir checkpoints/induction

Watch for the phase change: train loss has a bump, icl/score and induction/loss_second_copy drop sharply,
and one layer-1 head's induction score jumps, all at the same step.
"""

import argparse

import torch
from transformers import GPT2TokenizerFast

from gpt2 import Config, Transformer
from gpt2.induction import icl_score, induction_metrics
from gpt2.training import TransformerTrainer, TransformerTrainingArgs, load_fineweb_edu, load_tinystories


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", choices=["fineweb", "tinystories"], default="fineweb")
    p.add_argument("--n-samples", type=int, default=100_000, help="documents to load")
    p.add_argument("--n-layers", type=int, default=2)
    p.add_argument("--d-model", type=int, default=256)
    p.add_argument("--n-heads", type=int, default=8)
    p.add_argument("--n-ctx", type=int, default=512)
    p.add_argument("--with-mlp", action="store_true", help="keep the MLPs (default: attention-only)")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--steps", type=int, default=10_000)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--warmup-steps", type=int, default=200)
    p.add_argument("--metrics-every", type=int, default=50)
    p.add_argument("--snapshot-every", type=int, default=500)
    p.add_argument("--eval-every", type=int, default=1000, help="test accuracy, sample text, save latest.pt")
    p.add_argument("--checkpoint-dir", default="checkpoints/induction")
    p.add_argument("--resume", action="store_true", help="continue from <checkpoint-dir>/latest.pt")
    p.add_argument("--wandb-project", default="induction-heads", help="pass '' to disable wandb")
    p.add_argument("--wandb-name", default=None)
    return p.parse_args()


def main():
    a = parse_args()
    device = get_device()
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    cfg = Config(
        d_model=a.d_model,
        n_heads=a.n_heads,
        d_head=a.d_model // a.n_heads,
        d_mlp=4 * a.d_model,
        n_layers=a.n_layers,
        n_ctx=a.n_ctx,
        attn_only=not a.with_mlp,
        debug=False,
    )
    model = Transformer(cfg).to(device)
    print(f"{sum(p.numel() for p in model.parameters()) / 1e6:.1f}M parameters on {device}")

    load = load_fineweb_edu if a.dataset == "fineweb" else load_tinystories
    dataset_dict = load(tokenizer, n_ctx=cfg.n_ctx, n_samples=a.n_samples)
    print(f"{len(dataset_dict['train']) * cfg.n_ctx / 1e6:.0f}M training tokens")

    # A fixed batch of held-out text for the ICL score, so the measurement is comparable across steps.
    icl_tokens = dataset_dict["test"][:32]["tokens"]

    def metrics_fn(model: Transformer) -> dict[str, float]:
        return {**induction_metrics(model), **icl_score(model, icl_tokens)}

    args = TransformerTrainingArgs(
        batch_size=a.batch_size,
        epochs=max(1, a.steps // a.eval_every),
        max_steps_per_epoch=a.eval_every,
        lr=a.lr,
        warmup_steps=a.warmup_steps,
        wandb_project=a.wandb_project or None,
        wandb_name=a.wandb_name,
        eval_prompt="Once upon a time" if a.dataset == "tinystories" else "The history of",
        checkpoint_dir=a.checkpoint_dir,
        save_every_steps=a.snapshot_every,
        snapshot_every_steps=a.snapshot_every,
        metrics_every_steps=a.metrics_every,
    )
    trainer = TransformerTrainer(args, model, tokenizer, dataset_dict, metrics_fn=metrics_fn)
    if a.resume:
        trainer.load_checkpoint(f"{a.checkpoint_dir}/latest.pt")
        print(f"Resuming from step {trainer.step}")
    trainer.train()


if __name__ == "__main__":
    main()
