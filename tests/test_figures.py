"""figures.py: the losses.pth reader follows the run's family folder."""

import json
from pathlib import Path
from typing import cast

import pytest

from figures import export_losses, load_log, plot_losses
from utils import DCGANLog, VAELog, WGANLog


def make_run(base: Path, family: str, log: DCGANLog | WGANLog | VAELog) -> Path:
    run = base / "outputs" / family / "20260101-000000_x"
    (run / "logs").mkdir(parents=True)
    log.save(run / "logs" / "losses.pth")
    return run


def dcgan_log() -> DCGANLog:
    log = DCGANLog()
    log.push(step=0, epoch=1, loss_G=0.7, loss_D=1.2, D_x=0.9, D_G_z=0.2)
    return log


def wgan_log() -> WGANLog:
    log = WGANLog()
    log.push(step=0, epoch=1, loss_G=-0.5, loss_D=-1.5, w_dist=1.5)
    return log


def vae_log() -> VAELog:
    log = VAELog()
    log.push(step=0, epoch=1, beta=0.01, loss=100.0, recon=90.0, kl=1000.0)
    log.push_val(epoch=1, loss=120.0, recon=95.0, kl=25.0)
    return log


def test_load_log_picks_the_class_from_the_family_folder(tmp_path: Path) -> None:
    assert isinstance(
        load_log(make_run(tmp_path / "a", "dcgan", dcgan_log())), DCGANLog
    )
    assert isinstance(load_log(make_run(tmp_path / "b", "wgan", wgan_log())), WGANLog)
    assert isinstance(load_log(make_run(tmp_path / "c", "vae", vae_log())), VAELog)


def test_a_wgan_file_is_not_read_as_a_dcgan_log(tmp_path: Path) -> None:
    run = make_run(tmp_path, "dcgan", wgan_log())  # wrong folder for the contents
    with pytest.raises(ValueError, match="missing"):
        _ = load_log(run)


def test_unknown_family_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(SystemExit, match="flow"):
        _ = load_log(make_run(tmp_path, "flow", wgan_log()))


def exported_keys(out: Path) -> list[str]:
    return list(cast("dict[str, object]", json.loads(out.read_text())))


def test_export_losses_writes_the_runs_own_keys(tmp_path: Path) -> None:
    out = export_losses(make_run(tmp_path / "a", "wgan", wgan_log()))
    assert exported_keys(out) == ["step", "epoch", "loss_G", "loss_D", "w_dist"]
    out = export_losses(make_run(tmp_path / "b", "dcgan", dcgan_log()))
    assert exported_keys(out) == ["step", "epoch", "loss_G", "loss_D", "D_x", "D_G_z"]


def test_plot_losses_draws_every_family(tmp_path: Path) -> None:
    assert plot_losses(make_run(tmp_path / "a", "dcgan", dcgan_log())).exists()
    assert plot_losses(make_run(tmp_path / "b", "wgan", wgan_log())).exists()
    assert plot_losses(make_run(tmp_path / "c", "vae", vae_log())).exists()
