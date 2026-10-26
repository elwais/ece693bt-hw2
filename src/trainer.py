"""Trainer: the training loop that the classifier, DCGAN and WGAN share.

fit() is a template method. It owns the epoch loop, the train() switch, the
step counter and the batch stream. A subclass fills in what makes it that
model; the loop around it is written once.

Hooks, in the order fit() calls them:

    epoch_batches()   a stream of batches for this epoch      (default: one pass)
    steps_per_epoch() how many train_step calls make an epoch (default: len(loader))
    train_step(...)   ONE optimizer step, returns what to log (abstract)
    log_step(...)     every `log_every` steps, with floats    (default: nothing)
    end_epoch()       evaluation, sample grids, checkpoints   (default: nothing)

state() is the other abstract hook: {name: model or optimizer}. fit() puts the
models in train(), checkpoint() saves every entry, and evaluating() switches
the models to eval() and back.

Counters: `curr_epoch` is 1-based (the epoch in progress, or the last finished one). `step` is 0-based and counts steps across epochs. During train_step and log_step it is the index of the step in progress, so the first step is 0; after the step it is incremented.

Not here (a subclass does it): the run folder layout, what a checkpoint file is
called, and how the history is stored. `make_run_dir` and `save_checkpoints`
are there for the trainers that use the usual layout.
"""

import json
from abc import ABC, abstractmethod
from collections.abc import Generator, Iterator, Mapping
from contextlib import ExitStack, contextmanager
from datetime import datetime
from pathlib import Path
from typing import Protocol

import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader

from data import Sample, batches
from utils import eval_mode, save_atomic

# One batch: (images, labels).
type Batch = tuple[Tensor, Tensor]
# What train_step returns: 0-dim tensors, not floats. fit() calls .item() only on the
# steps that are logged, so the device is not synchronised on every step.
type Stats = dict[str, Tensor]


class Stateful(Protocol):
    """Anything a checkpoint can hold. nn.Module and torch.optim.Optimizer both match."""

    def state_dict(self) -> Mapping[str, object]: ...


def optimizer_step(opt: torch.optim.Optimizer, loss: Tensor) -> None:
    """Clear the gradients, backpropagate `loss`, step `opt`.

    Only `opt`'s parameters move. Gradients that `loss` leaves on other networks
    (D after a generator step) are cleared by that network's own next step.
    """
    opt.zero_grad()
    loss.backward()  # pyright: ignore[reportUnknownMemberType]  # backward's args are untyped
    opt.step()  # pyright: ignore[reportUnknownMemberType]


def make_run_dir(
    family: str,
    name: str | None,
    config: Mapping[str, object],
    subdirs: tuple[str, ...] = ("checkpoints", "samples", "logs"),
) -> Path:
    """outputs/<family>/<YYYYmmdd-HHMMSS>[_<name>]/ with its subfolders and args.json."""
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run = Path("outputs") / family / (f"{stamp}_{name}" if name else stamp)
    for sub in subdirs:
        (run / sub).mkdir(parents=True, exist_ok=False)
    _ = (run / "args.json").write_text(json.dumps(config, indent=2))
    return run


class Trainer(ABC):
    """One training run. A subclass builds its models, then calls fit().

    A subclass's __init__ seeds, builds the models and optimizers and sets
    `train_loader`, in that order, because weight initialization draws from the
    global RNG. It passes the settings of this base class to super().__init__().
    """

    device: torch.device
    log_every: int  # 0 = never call log_step
    train_loader: DataLoader[Sample]  # set by the subclass
    curr_epoch: int  # 1-based: the epoch in progress, or the last finished one
    final_epoch: int
    step: int  # 0-based: see the module docstring

    def __init__(
        self, *, epochs: int, device: str | torch.device, log_every: int = 0
    ) -> None:
        self.final_epoch = epochs
        self.device = torch.device(device)
        self.log_every = log_every
        self.curr_epoch = 0
        self.step = 0

    # ------------------------------------------------------------------
    # Hooks
    # ------------------------------------------------------------------

    @abstractmethod
    def state(self) -> dict[str, Stateful]:
        """Name -> model or optimizer. The checkpoint holds one state_dict per entry."""

    @abstractmethod
    def train_step(self, stream: Iterator[Batch]) -> Stats:
        """One optimizer step. Take what you need with next(stream); return 0-dim tensors."""

    def steps_per_epoch(self) -> int:
        """Default: one pass over the train loader."""
        return len(self.train_loader)

    def epoch_batches(self) -> Iterator[Batch]:
        """Default: a fresh, reshuffled pass over the train loader every epoch."""
        return batches(self.train_loader)

    def log_step(self, stats: dict[str, float]) -> None:
        """Called with the stats of every `log_every`-th step, as floats."""

    def end_epoch(self) -> None:
        """Called after the last step of an epoch: evaluate, save samples, checkpoint."""

    # ------------------------------------------------------------------
    # The loop
    # ------------------------------------------------------------------

    def fit(self) -> None:
        """Train from the epoch after `curr_epoch` through `final_epoch`."""
        for epoch in range(self.curr_epoch + 1, self.final_epoch + 1):
            self.curr_epoch = epoch
            for module in self.modules():
                module.train()
            stream = self.epoch_batches()
            for _ in range(self.steps_per_epoch()):
                stats = self.train_step(stream)
                if self.log_every > 0 and self.step % self.log_every == 0:
                    self.log_step({k: float(v.item()) for k, v in stats.items()})
                self.step += 1
            self.end_epoch()

    # ------------------------------------------------------------------
    # Helpers for subclasses
    # ------------------------------------------------------------------

    def modules(self) -> list[nn.Module]:
        """The models in state(); the optimizers are left out."""
        return [obj for obj in self.state().values() if isinstance(obj, nn.Module)]

    @contextmanager
    def evaluating(self) -> Generator[None, None, None]:
        """eval() and inference_mode() for every model; each model's mode is restored after."""
        with ExitStack() as stack:
            for module in self.modules():
                _ = stack.enter_context(eval_mode(module))
            yield

    def checkpoint(self) -> dict[str, object]:
        """Every state() entry as a state_dict, plus the epoch. Extend it by overriding."""
        payload: dict[str, object] = {
            key: obj.state_dict() for key, obj in self.state().items()
        }
        payload["epoch"] = self.curr_epoch
        return payload

    def save_checkpoints(self, directory: Path, every: int) -> None:
        """last.pt after every epoch, epoch_XXX.pt every `every` epochs, final.pt at the end."""
        ckpt = self.checkpoint()
        save_atomic(ckpt, directory / "last.pt")
        if self.curr_epoch % every == 0:
            save_atomic(ckpt, directory / f"epoch_{self.curr_epoch:03d}.pt")
        if self.curr_epoch == self.final_epoch:
            save_atomic(ckpt, directory / "final.pt")
