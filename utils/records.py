"""Typed containers for loaders and training histories.

Call sites construct records and push values; the mechanics (saving, loading,
key checks) live here so training scripts stay short.
"""

from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, Literal, Self, cast

import torch

type SplitLabel = Literal["train", "val", "test"]


@dataclass
class Splits[T]:
    """One value per split. A misspelt split name is a type error, not a KeyError."""

    train: T
    val: T
    test: T

    def items(self) -> tuple[tuple[SplitLabel, T], ...]:
        return (("train", self.train), ("val", self.val), ("test", self.test))

    def __getitem__(self, split: SplitLabel) -> T:
        match split:
            case "train":
                return self.train
            case "val":
                return self.val
            case "test":
                return self.test


@dataclass
class Metrics:
    """Per-epoch series for one split (Task 0)."""

    acc: list[float] = field(default_factory=list)
    loss: list[float] = field(default_factory=list)


@dataclass
class Log:
    """Base for flat histories. Saved as a plain dict of lists, matching the spec."""

    def save(self, path: Path) -> None:
        torch.save(asdict(self), path)

    @classmethod
    def load(cls, path: Path) -> Self:
        """Load a saved log. Fails if the file's keys are not exactly this class's fields."""
        # torch.load returns untyped data. The key check below guards its structure,
        # and save() wrote the values with the field types, so the Any stays here.
        raw = cast("dict[str, Any]", torch.load(path, weights_only=True))  # pyright: ignore[reportExplicitAny]
        expected = {f.name for f in fields(cls)}
        if raw.keys() != expected:
            raise ValueError(
                f"{path}: missing {sorted(expected - raw.keys())}, "
                f"unexpected {sorted(raw.keys() - expected)}"
            )
        return cls(**raw)


@dataclass
class StepLog(Log):
    """Fields shared by every per-step log. Listed first, so saved key order matches."""

    step: list[int] = field(default_factory=list)
    epoch: list[int] = field(default_factory=list)


@dataclass
class GANLog(StepLog):
    """DCGAN (Setup_GAN.md Task 1). Field names are fixed by the spec's losses.pth."""

    loss_G: list[float] = field(default_factory=list)
    loss_D: list[float] = field(default_factory=list)
    D_x: list[float] = field(default_factory=list)
    D_G_z: list[float] = field(default_factory=list)

    def push(
        self,
        *,
        step: int,
        epoch: int,
        loss_G: float,
        loss_D: float,
        D_x: float,
        D_G_z: float,
    ) -> None:
        self.step.append(step)
        self.epoch.append(epoch)
        self.loss_G.append(loss_G)
        self.loss_D.append(loss_D)
        self.D_x.append(D_x)
        self.D_G_z.append(D_G_z)


@dataclass
class WGANLog(StepLog):
    """WGAN (Part 2). The critic plays D's role, so loss_D is the critic loss."""

    loss_G: list[float] = field(default_factory=list)
    loss_D: list[float] = field(default_factory=list)
    w_dist: list[float] = field(default_factory=list)

    def push(
        self, *, step: int, epoch: int, loss_G: float, loss_D: float, w_dist: float
    ) -> None:
        self.step.append(step)
        self.epoch.append(epoch)
        self.loss_G.append(loss_G)
        self.loss_D.append(loss_D)
        self.w_dist.append(w_dist)


@dataclass
class VAELog(StepLog):
    """VAE (Part 3): per-step training terms plus a per-epoch validation curve."""

    beta: list[float] = field(default_factory=list)
    loss: list[float] = field(default_factory=list)
    recon: list[float] = field(default_factory=list)
    kl: list[float] = field(default_factory=list)
    val_epoch: list[int] = field(default_factory=list)
    val_loss: list[float] = field(default_factory=list)

    def push(
        self, *, step: int, epoch: int, beta: float, loss: float, recon: float, kl: float
    ) -> None:
        self.step.append(step)
        self.epoch.append(epoch)
        self.beta.append(beta)
        self.loss.append(loss)
        self.recon.append(recon)
        self.kl.append(kl)

    def push_val(self, *, epoch: int, loss: float) -> None:
        self.val_epoch.append(epoch)
        self.val_loss.append(loss)
