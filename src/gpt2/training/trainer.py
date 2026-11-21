import json
from contextlib import nullcontext
from pathlib import Path
from typing import Callable, Iterator

import torch
from jaxtyping import Float, Int
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import GPT2TokenizerFast
import wandb

from gpt2.model import Transformer
from gpt2.sampling import TransformerSampler
from gpt2.training.args import TransformerTrainingArgs

MetricsFn = Callable[[Transformer], dict[str, float]]


class TransformerTrainer:
    def __init__(
        self,
        args: TransformerTrainingArgs,
        model: Transformer,
        tokenizer: GPT2TokenizerFast,
        dataset_dict: dict[str, Int[Tensor, "batch seq"]],
        metrics_fn: MetricsFn | None = None,
    ):
        self.model = model
        self.device = model.device
        self.args = args
        self.sampler = TransformerSampler(self.model, tokenizer)
        self.metrics_fn = metrics_fn
        self.step = 0
        self.total_steps = args.epochs * args.max_steps_per_epoch

        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer, lambda step: min(1.0, (step + 1) / args.warmup_steps) if args.warmup_steps else 1.0
        )

        # Mixed precision: autocast runs matmuls in 16-bit. fp16 needs loss scaling to avoid gradient underflow.
        use_amp = args.mixed_precision and self.device.type == "cuda"
        self.amp_dtype = (torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16) if use_amp else None
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.amp_dtype == torch.float16)

        self.train_loader = DataLoader(dataset_dict["train"], batch_size=args.batch_size, shuffle=True)
        self.test_loader = DataLoader(dataset_dict["test"], batch_size=args.batch_size, shuffle=False)

        self.checkpoint_dir = Path(args.checkpoint_dir) if args.checkpoint_dir else None

    def autocast(self):
        if self.amp_dtype is None:
            return nullcontext()
        return torch.autocast(device_type="cuda", dtype=self.amp_dtype)

    def batches(self) -> Iterator[dict[str, Tensor]]:
        """Loop over the training set forever, reshuffling each pass."""
        while True:
            yield from self.train_loader

    def training_step(self, batch: dict[str, Int[Tensor, "batch seq"]]) -> float:
        tokens = batch["tokens"].to(self.device)
        with self.autocast():
            logits = self.model(tokens)
            loss = -self.model.get_log_probs(logits, tokens).mean()
        self.scaler.scale(loss).backward()
        if self.args.max_grad_norm is not None:
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad(set_to_none=True)
        self.scheduler.step()
        self.step += 1
        loss = loss.item()
        wandb.log({"train_loss": loss, "lr": self.scheduler.get_last_lr()[0]}, step=self.step)
        return loss

    @torch.inference_mode()
    def evaluate(self) -> float:
        self.model.eval()
        total_correct, total_samples = 0, 0
        for batch in tqdm(self.test_loader, desc="Evaluating", leave=False):
            tokens = batch["tokens"].to(self.device)
            with self.autocast():
                logits = self.model(tokens)[:, :-1]
            predicted_tokens = logits.argmax(dim=-1)
            total_correct += (predicted_tokens == tokens[:, 1:]).sum().item()
            total_samples += tokens.size(0) * (tokens.size(1) - 1)
        accuracy = total_correct / total_samples
        self.model.train()
        return accuracy

    def log_metrics(self, metrics: dict[str, float]) -> None:
        """Send metrics to wandb and append them to checkpoint_dir/metrics.jsonl (readable without wandb)."""
        wandb.log(metrics, step=self.step)
        if self.checkpoint_dir is not None:
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
            with open(self.checkpoint_dir / "metrics.jsonl", "a") as f:
                f.write(json.dumps({"step": self.step, **metrics}) + "\n")

    def save_checkpoint(self) -> Path | None:
        """Overwrite checkpoint_dir/latest.pt with everything needed to resume training."""
        if self.checkpoint_dir is None:
            return None
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        checkpoint = {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "scaler": self.scaler.state_dict(),
            "step": self.step,
            "config": self.model.cfg,
        }
        path = self.checkpoint_dir / "latest.pt"
        tmp = path.with_suffix(".tmp")
        torch.save(checkpoint, tmp)
        tmp.replace(path)  # so a disconnect mid-save can't corrupt the previous checkpoint
        return path

    def save_snapshot(self) -> Path | None:
        """Keep the model weights (fp16, no optimizer) at this step, for analysing how training unfolded."""
        if self.checkpoint_dir is None:
            return None
        snapshot_dir = self.checkpoint_dir / "snapshots"
        snapshot_dir.mkdir(parents=True, exist_ok=True)
        state = {k: v.half() if v.is_floating_point() else v for k, v in self.model.state_dict().items()}
        path = snapshot_dir / f"step_{self.step:06d}.pt"
        torch.save({"model": state, "step": self.step, "config": self.model.cfg}, path)
        return path

    def load_checkpoint(self, path: str | Path) -> None:
        """Restore a latest.pt; train() then continues from the saved step (with freshly shuffled data)."""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint["model"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.step = checkpoint["step"]
        if "scheduler" in checkpoint:
            self.scheduler.load_state_dict(checkpoint["scheduler"])
            self.scaler.load_state_dict(checkpoint["scaler"])

    def _due(self, every: int | None) -> bool:
        return every is not None and self.step % every == 0

    def train(self):
        args = self.args
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_name,
            config={**vars(args), "model": vars(self.model.cfg)},
            mode="disabled" if args.wandb_project is None else None,
        )
        self.model.train()
        if self.step == 0:
            if self.metrics_fn is not None and args.metrics_every_steps:
                self.log_metrics(self.metrics_fn(self.model))
            if args.snapshot_every_steps:
                self.save_snapshot()
            print(self.sampler.sample(args.eval_prompt, max_tokens_generated=50))

        accuracy = float("nan")
        progress_bar = tqdm(initial=self.step, total=self.total_steps)
        batches = self.batches()
        recent_losses: list[float] = []
        while self.step < self.total_steps:
            loss = self.training_step(next(batches))
            progress_bar.update()
            epoch = (self.step - 1) // args.max_steps_per_epoch + 1
            progress_bar.set_description(f"Epoch {epoch}, loss: {loss:.3f}, accuracy: {accuracy:.3f}")

            recent_losses.append(loss)
            if self.metrics_fn is not None and self._due(args.metrics_every_steps):
                # Mean train loss since the last measurement, so metrics.jsonl alone can plot the loss curve.
                avg_loss = sum(recent_losses) / len(recent_losses)
                recent_losses.clear()
                self.log_metrics({"train_loss_avg": avg_loss, **self.metrics_fn(self.model)})
            if self._due(args.snapshot_every_steps):
                self.save_snapshot()
            if self._due(args.max_steps_per_epoch):
                accuracy = self.evaluate()
                self.log_metrics({"accuracy": accuracy})
                print(self.sampler.sample(args.eval_prompt, max_tokens_generated=50))
                self.save_checkpoint()
            elif self._due(args.save_every_steps):
                self.save_checkpoint()

        progress_bar.close()
        wandb.finish()
