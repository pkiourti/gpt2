from pathlib import Path

import torch
import numpy as np
from torch import Tensor
from jaxtyping import Float, Int
from transformers import GPT2TokenizerFast
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import wandb

from gpt2.training.args import TransformerTrainingArgs
from gpt2.model import Transformer
from gpt2.sampling import TransformerSampler


class TransformerTrainer:
    def __init__(
        self,
        args: TransformerTrainingArgs,
        model: Transformer,
        tokenizer: GPT2TokenizerFast,
        dataset_dict: dict[str, Int[Tensor, "batch seq"]],
    ):
        super().__init__()
        self.model = model
        self.device = model.device
        self.args = args
        self.sampler = TransformerSampler(self.model, tokenizer)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        self.step = 0
        self.start_epoch = 0

        self.train_loader = DataLoader(
            dataset_dict["train"],
            batch_size=args.batch_size,
            shuffle=True,
        )

        self.test_loader = DataLoader(
            dataset_dict["test"],
            batch_size=args.batch_size,
            shuffle=False,
        )

    def training_step(self, batch: dict[str, Int[Tensor, "batch seq"]]) -> Float[Tensor, ""]:
        tokens = batch["tokens"].to(self.device)
        logits = self.model(tokens)
        loss = -self.model.get_log_probs(logits, tokens).mean()
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        self.step += 1
        wandb.log({"train_loss": loss}, step=self.step)
        return loss

    @torch.inference_mode()
    def evaluate(self) -> float:
        self.model.eval()
        total_correct, total_samples = 0, 0
        for batch in tqdm(self.test_loader, desc="Evaluating"):
            tokens = batch["tokens"].to(self.device)
            logits = self.model(tokens)[:, :-1]
            predicted_tokens = logits.argmax(dim=-1)
            total_correct += (predicted_tokens == tokens[:, 1:]).sum().item()
            total_samples += tokens.size(0) * (tokens.size(1) - 1)
        accuracy = total_correct / total_samples
        wandb.log({"accuracy": accuracy}, step=self.step)
        self.model.train()
        return accuracy

    def save_checkpoint(self, epoch: int) -> Path | None:
        """Save model and optimizer state so training can resume after a crash or Colab disconnect."""
        if self.args.checkpoint_dir is None:
            return None
        checkpoint_dir = Path(self.args.checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        checkpoint = {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "step": self.step,
            "epoch": epoch,
            "config": self.model.cfg,
        }
        path = checkpoint_dir / f"epoch_{epoch + 1:03d}.pt"
        torch.save(checkpoint, path)
        torch.save(checkpoint, checkpoint_dir / "latest.pt")
        return path

    def load_checkpoint(self, path: str | Path) -> None:
        """Restore model and optimizer state; train() then continues from the epoch after the saved one."""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint["model"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.step = checkpoint["step"]
        self.start_epoch = checkpoint["epoch"] + 1

    def train(self):
        wandb.init(project=self.args.wandb_project, name=self.args.wandb_name, config=self.args)
        accuracy = np.nan

        progress_bar = tqdm(total=self.args.max_steps_per_epoch * (self.args.epochs - self.start_epoch))

        print(self.sampler.sample(self.args.eval_prompt, max_tokens_generated=50))
        for epoch in range(self.start_epoch, self.args.epochs):
            for i, batch in enumerate(self.train_loader):
                loss = self.training_step(batch)
                progress_bar.update()
                progress_bar.set_description(f"Epoch {epoch + 1}, loss: {loss:.3f}, accuracy: {accuracy:.3f}")
                if i + 1 >= self.args.max_steps_per_epoch:
                    break

            accuracy = self.evaluate()
            print(self.sampler.sample(self.args.eval_prompt, max_tokens_generated=50))
            self.save_checkpoint(epoch)

        wandb.finish()