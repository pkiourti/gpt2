import torch as t
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
        self.optimizer = t.optim.AdamW(self.model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        self.step = 0

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

    @t.inference_mode()
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

    def train(self):
        wandb.init(project=self.args.wandb_project, name=self.args.wandb_name, config=self.args)
        accuracy = np.nan

        progress_bar = tqdm(total=self.args.max_steps_per_epoch * self.args.epochs)

        print(self.sampler.sample(self.args.eval_prompt, max_tokens_generated=50))
        for epoch in range(self.args.epochs):
            for i, batch in enumerate(self.train_loader):
                loss = self.training_step(batch)
                progress_bar.update()
                progress_bar.set_description(f"Epoch {epoch + 1}, loss: {loss:.3f}, accuracy: {accuracy:.3f}")
                if i + 1 >= self.args.max_steps_per_epoch:
                    break

            accuracy = self.evaluate()
            print(self.sampler.sample(self.args.eval_prompt, max_tokens_generated=50))

        wandb.finish()