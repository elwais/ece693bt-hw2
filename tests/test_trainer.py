"""Trainer: the loop, the counters, mode switching, logging gate, checkpoints."""

from collections.abc import Iterator
from pathlib import Path
from typing import cast

import pytest
import torch
from torch import nn

from utils import as_tensor
from trainer import Batch, Stateful, Stats, Trainer, make_run_dir, optimizer_step


class Toy(Trainer):
    """A one-layer model on a fixed stream. Records what the loop did."""

    def __init__(self, *, epochs: int, per_epoch: int, log_every: int = 0) -> None:
        super().__init__(epochs=epochs, device="cpu", log_every=log_every)
        self.per_epoch = per_epoch
        self.net = nn.Sequential(nn.Linear(2, 1), nn.Dropout(0.5))
        self.opt = torch.optim.SGD(self.net.parameters(), lr=0.1)
        self.calls: list[tuple[int, int, bool]] = []  # (epoch, step, net.training)
        self.logged: list[tuple[int, dict[str, float]]] = []
        self.ended: list[int] = []

    def state(self) -> dict[str, Stateful]:
        return {"net": self.net, "opt": self.opt}

    def steps_per_epoch(self) -> int:
        return self.per_epoch

    def epoch_batches(self) -> Iterator[Batch]:
        while True:
            yield torch.ones(4, 2), torch.zeros(4)

    def train_step(self, stream: Iterator[Batch]) -> Stats:
        x, _ = next(stream)
        self.calls.append((self.epoch, self.step, self.net.training))
        loss = as_tensor(self.net(x)).pow(2).mean()
        optimizer_step(self.opt, loss)
        return {"loss": loss.detach()}

    def log_step(self, stats: dict[str, float]) -> None:
        self.logged.append((self.step, stats))

    def end_epoch(self) -> None:
        self.ended.append(self.epoch)


def test_loop_counts_and_indices() -> None:
    t = Toy(epochs=3, per_epoch=4)
    t.fit()
    assert [c[1] for c in t.calls] == list(range(12))  # 0-based, across epochs
    assert [c[0] for c in t.calls] == [1] * 4 + [2] * 4 + [3] * 4
    assert t.ended == [1, 2, 3]
    assert (t.epoch, t.step) == (3, 12)


def test_models_are_in_train_mode_during_steps() -> None:
    t = Toy(epochs=2, per_epoch=2)
    t.net.eval()  # fit() must switch it back
    t.fit()
    assert all(training for _, _, training in t.calls)


def test_log_gate_and_float_conversion() -> None:
    t = Toy(epochs=1, per_epoch=11, log_every=5)
    t.fit()
    assert [s for s, _ in t.logged] == [0, 5, 10]
    assert all(isinstance(v, float) for _, st in t.logged for v in st.values())
    quiet = Toy(epochs=1, per_epoch=11)  # log_every = 0: log_step is never called
    quiet.fit()
    assert quiet.logged == []


def test_evaluating_switches_and_restores() -> None:
    t = Toy(epochs=1, per_epoch=1)
    t.net.train()
    with t.evaluating():
        assert not t.net.training
        assert not t.net[1].training
        assert torch.is_inference_mode_enabled()
    assert t.net.training and t.net[1].training
    t.net.eval()
    with t.evaluating():
        pass
    assert not t.net.training  # it was in eval before, so it stays there


def test_checkpoint_payload_and_schedule(tmp_path: Path) -> None:
    t = Toy(epochs=5, per_epoch=1)
    for epoch in range(1, 6):
        t.epoch = epoch
        t.save_checkpoints(tmp_path, every=2)
    assert sorted(p.name for p in tmp_path.glob("*.pt")) == [
        "epoch_002.pt",
        "epoch_004.pt",
        "final.pt",
        "last.pt",
    ]
    final = cast("dict[str, object]", torch.load(tmp_path / "final.pt", weights_only=True))
    assert set(final) == {"net", "opt", "epoch"}
    assert final["epoch"] == 5
    epoch_4 = cast("dict[str, object]", torch.load(tmp_path / "epoch_004.pt", weights_only=True))
    assert epoch_4["epoch"] == 4


def test_optimizer_step_moves_only_its_parameters() -> None:
    a, b = nn.Linear(2, 1), nn.Linear(2, 1)
    opt_a = torch.optim.SGD(a.parameters(), lr=0.1)
    b_before = b.weight.detach().clone()
    a_before = a.weight.detach().clone()
    loss = (as_tensor(a(torch.ones(1, 2))) + as_tensor(b(torch.ones(1, 2)))).sum()
    optimizer_step(opt_a, loss)
    assert not torch.equal(a.weight, a_before)
    assert torch.equal(b.weight, b_before)


def test_make_run_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    run = make_run_dir("dcgan", "tag", {"lr": 0.1, "name": "tag"})
    assert run.parent == Path("outputs/dcgan")
    assert run.name.endswith("_tag")
    for sub in ("checkpoints", "samples", "logs"):
        assert (run / sub).is_dir()
    assert '"lr": 0.1' in (run / "args.json").read_text()
    unnamed = make_run_dir("wgan", None, {}, subdirs=("logs",))
    assert "_" not in unnamed.name and (unnamed / "logs").is_dir()
