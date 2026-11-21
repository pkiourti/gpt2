from dataclasses import dataclass


@dataclass
class TransformerTrainingArgs:
    batch_size: int = 32
    # Training runs for epochs * max_steps_per_epoch steps; "epoch" just marks when to evaluate and sample.
    epochs: int = 10
    max_steps_per_epoch: int = 500
    lr: float = 1e-3
    weight_decay: float = 1e-2
    warmup_steps: int = 0  # linear LR warmup from 0 to lr
    max_grad_norm: float | None = 1.0  # gradient clipping; None disables
    mixed_precision: bool = True  # CUDA only: bf16 where supported (A100, L4), else fp16 with loss scaling (T4)
    wandb_project: str | None = "demotransformer"  # None disables wandb
    wandb_name: str | None = None
    eval_prompt: str = "Once upon a time"
    checkpoint_dir: str | None = "checkpoints"  # None disables saving; on Colab, point this at Google Drive
    save_every_steps: int | None = None  # also overwrite checkpoint_dir/latest.pt this often (always at epoch end)
    snapshot_every_steps: int | None = None  # keep model-only fp16 weights at snapshots/step_XXXXXX.pt
    metrics_every_steps: int | None = None  # call the trainer's metrics_fn this often; logged to wandb and metrics.jsonl
