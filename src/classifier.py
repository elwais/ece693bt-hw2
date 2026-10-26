import json
from collections.abc import Iterator
from dataclasses import asdict
from pathlib import Path
from typing import override

import torch
from torch import Tensor, nn  # Super frequent and important stuff
from torch.utils.data import DataLoader  # Misc things

from data import Sample, batches, gan_loaders
from trainer import Batch, Stateful, Stats, Trainer, optimizer_step
from utils import (
    Metrics,
    Splits,
    as_tensor,
    built,
    eval_mode,
    get_preferred_device,
    nn_dataclass,
    save_params,
    seed_everything,
)


@nn_dataclass
class Classifier(nn.Module):
    # Boilerplate to ensure that forward pass calls are typed appropriately
    # May need to uncomment, but keep hidden for now to make the lesson stick
    def __call__(self, x: Tensor):
        return as_tensor(super().__call__(x))

    # `features` computes the ϕ(x) we'll need for FID
    features: nn.Module = built(
        lambda: nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.25),
            nn.Flatten(),
            nn.Linear(9216, 128),
            nn.ReLU(),
        )
    )

    # This takes care of the rest
    head: nn.Module = built(lambda: nn.Sequential(nn.Dropout(0.5), nn.Linear(128, 10)))

    def phi(self, x: Tensor):
        with eval_mode(self.features):
            return as_tensor(self.features(x))

    def forward(self, x: Tensor):
        return as_tensor(self.head(self.features(x)))


######################## Training ######################################


OUT_DIR = Path("outputs")  # the FID code and the report read these two files
CHECKPOINT = OUT_DIR / "classifier.pt"
HISTORY = OUT_DIR / "classifier_history.json"


class ClassifierTrainer(Trainer):
    """Task 0: Adam, cross-entropy, one pass per epoch; evaluate every split each epoch.

    The checkpoint is the epoch with the lowest validation loss (a running minimum).
    """

    model: Classifier
    opt: torch.optim.Optimizer
    loaders: Splits[DataLoader[Sample]]
    history: Splits[Metrics]
    best_val_loss: float

    def __init__(self, epochs: int = 30, seed: int = 0) -> None:
        device = get_preferred_device()
        seed_everything(seed)
        super().__init__(epochs=epochs, device=device)

        with torch.device(self.device):
            self.model = Classifier()
        # Loaders for train, val and test, all in the [-1, 1] scaling
        self.loaders = gan_loaders()
        self.train_loader = self.loaders.train
        # By default Adam uses lr=1e-3, so no need to make it explicit
        self.opt = torch.optim.Adam(self.model.parameters())

        self.history = Splits(train=Metrics(), val=Metrics(), test=Metrics())
        self.best_val_loss = float("inf")

    @override
    def state(self) -> dict[str, Stateful]:
        return {"model": self.model, "opt": self.opt}

    @override
    def train_step(self, stream: Iterator[Batch]) -> Stats:
        sample, label = next(stream)
        sample, label = sample.to(self.device), label.to(self.device)
        loss = nn.functional.cross_entropy(self.model(sample), label)
        optimizer_step(self.opt, loss)
        return {"loss": loss.detach()}

    def evaluate(self, loader: DataLoader[Sample]) -> tuple[float, float]:
        """Mean cross-entropy and accuracy"""
        loss_sum, correct, n = 0.0, 0, 0
        with self.evaluating():
            for sample, label in batches(loader):
                sample, label = sample.to(self.device), label.to(self.device)
                logits = self.model(sample)
                loss_sum += nn.functional.cross_entropy(
                    logits, label, reduction="sum"
                ).item()
                correct += (logits.argmax(dim=1) == label).sum().item()
                n += label.shape[0]
        return loss_sum / n, correct / n

    @override
    def end_epoch(self) -> None:
        # Evaluating over all splits and saving results
        for split, loader in self.loaders.items():
            loss, acc = self.evaluate(loader)
            self.history[split].loss.append(loss)
            self.history[split].acc.append(acc)

        # A running min of val loss; the policy we use to choose our checkpoint
        val_loss = self.history.val.loss[-1]
        if val_loss < self.best_val_loss:
            save_params(self.model, CHECKPOINT)
            self.best_val_loss = val_loss

        with open(HISTORY, "w") as f:
            json.dump(asdict(self.history), f)
        print(
            f"Epoch {self.curr_epoch:02d}  val loss {val_loss:.4f}  "
            f"val acc {self.history.val.acc[-1]:.4f}"
        )


if __name__ == "__main__":
    ClassifierTrainer().fit()
