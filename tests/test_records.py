"""Records: split containers and flat logs, including the save/load key check."""

from pathlib import Path
from typing import cast

import pytest
import torch

from utils import DCGANLog, Metrics, Splits, VAELog, WGANLog


def test_splits_items_are_in_order_and_index_by_label() -> None:
    s = Splits(train=1, val=2, test=3)
    assert s.items() == (("train", 1), ("val", 2), ("test", 3))
    assert s["val"] == 2


def test_metrics_default_to_independent_lists() -> None:
    a, b = Metrics(), Metrics()
    a.loss.append(0.5)
    assert b.loss == []


def test_dcgan_log_saves_spec_keys_in_order(tmp_path: Path) -> None:
    log = DCGANLog()
    log.push(step=0, epoch=1, loss_G=0.7, loss_D=1.2, D_x=0.9, D_G_z=0.2)
    path = tmp_path / "losses.pth"
    log.save(path)
    raw = cast(dict[str, object], torch.load(path, weights_only=True))
    assert list(raw) == ["step", "epoch", "loss_G", "loss_D", "D_x", "D_G_z"]
    assert raw["loss_G"] == [0.7]


def test_log_round_trips(tmp_path: Path) -> None:
    log = VAELog()
    log.push(step=0, epoch=1, beta=0.01, loss=100.0, recon=90.0, kl=10.0)
    log.push_val(epoch=1, loss=101.0)
    path = tmp_path / "vae.pth"
    log.save(path)
    assert VAELog.load(path) == log


def test_load_rejects_file_from_another_log_class(tmp_path: Path) -> None:
    path = tmp_path / "vae.pth"
    VAELog().save(path)
    with pytest.raises(ValueError, match="missing"):
        WGANLog.load(path)


def test_push_requires_keyword_arguments() -> None:
    with pytest.raises(TypeError):
        DCGANLog().push(0, 1, 0.7, 1.2, 0.9, 0.2)  # type: ignore[misc]
