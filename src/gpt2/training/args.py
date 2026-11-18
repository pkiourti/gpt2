from dataclasses import dataclass

@dataclass
class TransformerTrainingArgs:
    batch_size: int = 32
    epochs: int = 10
    max_steps_per_epoch: int = 500
    lr: float = 1e-3
    weight_decay: float = 1e-2
    wandb_project: str | None = "demotransformer"
    wandb_name: str | None = None
    eval_prompt: str = "Once upon a time"
    checkpoint_dir: str | None = "checkpoints"  # None disables saving; on Colab, point this at Google Drive
