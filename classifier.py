from itertools import chain
from typing import Literal
from pathlib import Path
import json

import torch
from torch import Tensor, nn  # Super frequent and important stuff
from torch.utils.data import DataLoader  # Misc things

from data import Sample, get_test_loader, get_train_val_loaders, unpack_batch
from utils import (
    as_tensor,
    built,
    eval_mode,
    nn_dataclass,
    seed_everything,
    save_params,
    get_preferred_device,
)


@nn_dataclass
class Classifier(nn.Module):
    # Boilerplate to ensure that forward pass calls are typed appropriately
    # May need to uncomment, but keep hidden for now to make the lesson stick
    def __call__(self, x: Tensor) -> Tensor:
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

    @torch.inference_mode()  # Drops autograd computations
    def phi(self, x: Tensor) -> Tensor:
        was_training = self.features.training
        self.features.eval()  # Ensures that dropout is identity, yielding determinism
        try:
            return as_tensor(self.features(x))
        finally:
            self.features.train(was_training)

    def forward(self, x: Tensor) -> Tensor:
        return as_tensor(self.head(self.features(x)))


######################## Here we try to train manually #########################

if __name__ == "__main__":
    DEVICE = get_preferred_device()
    seed_everything(0)

    type SplitLabel = Literal["train", "val", "test"]

    # Make our model
    with torch.device(DEVICE):
        model = Classifier()

    # Prepare loaders, which handle datasets but also generation and the like
    loaders: dict[SplitLabel, DataLoader[Sample]] = {
        name: loader
        for name, loader in zip(
            ("train", "val", "test"),
            chain(get_train_val_loaders("gan"), (get_test_loader("gan"),)),
        )
    }

    # Our optimizer; by default uses lr=1e-3, so no need to make it explicit
    opt = torch.optim.Adam(model.parameters())

    def loader_iterate(loader: DataLoader[Sample]) -> "map[tuple[Tensor, Tensor]]":
        return map(
            lambda p: (p[0].to(DEVICE), p[1].to(DEVICE)), map(unpack_batch, loader)
        )

    def evaluate(model: Classifier, loader: DataLoader[Sample]) -> tuple[float, float]:
        """Mean cross-entropy and accuracy"""
        loss_sum, correct, n = 0.0, 0, 0
        with eval_mode(model):
            for sample, label in loader_iterate(loader):
                logits = model(sample)
                loss_sum += nn.functional.cross_entropy(
                    logits, label, reduction="sum"
                ).item()
                correct += (logits.argmax(dim=1) == label).sum().item()
                n += label.shape[0]
        return loss_sum / n, correct / n

    # To be populated in eval loops
    history: dict[SplitLabel, dict[Literal["acc", "loss"], list[int | float]]] = {
        "train": {"acc": [], "loss": []},
        "val": {"acc": [], "loss": []},
        "test": {"acc": [], "loss": []},
    }

    min_epoch_loss = float("inf")

    # A running min of loss; the policy we will use to choose our checkpoint
    for epoch in range(1, 31):
        print(f"Epoch {epoch:02d}")

        # Training
        model.train()
        for sample, label in loader_iterate(loaders["train"]):
            opt.zero_grad()
            nn.functional.cross_entropy(model(sample), label).backward()  # pyright: ignore[reportUnknownMemberType]
            opt.step()  # pyright: ignore[reportUnknownMemberType]

        # Evaluating over all splits and saving results
        for split, loader in loaders.items():
            loss, acc = evaluate(model, loader)
            history[split]["loss"].append(loss)
            history[split]["acc"].append(acc)

        current_epoch_loss = history["val"]["loss"][-1]
        if current_epoch_loss < min_epoch_loss:
            save_params(model, Path("outputs/classifier.pt"))
            min_epoch_loss = current_epoch_loss

        with open("outputs/classifier_history.json", "w") as f:
            json.dump(history, f)
