# pyright: reportUnknownMemberType=false
# matplotlib's keyword arguments are untyped, so every plotting call would be flagged.
# Only this rule is relaxed; all other strict checks still apply to this file.
"""Quick-look matplotlib figures for DCGAN runs, for checking runs while they train.

The report does not use these images: it reads outputs/manifest.json (see manifest.py)
and draws its plots and sample grids in Typst. `export` is used by manifest.py.

    python figures.py losses RUN_DIR
        -> RUN_DIR/figures/losses.png: loss_D and loss_G vs step, with D_x and D_G_z
           (dcgan) or the Wasserstein estimate w_dist (wgan) beside them

    python figures.py grids RUN_A RUN_B [...] -o OUT.png
        -> the final sample grid of each run, side by side

    python figures.py progress RUN [-o OUT.png]
        -> the fixed-z grid at several points in training (default RUN/figures/progress.png)

    python figures.py fid RUN_A RUN_B [...] -o OUT.png
        -> val-5k FID vs epoch for each run (from RUN/fid.json), with the real-data
           floor from outputs/diagnostics.json when present

    python figures.py export RUN
        -> RUN/logs/losses.json: losses.pth as plain JSON (for plotting in Typst)
"""

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import cast

import matplotlib

matplotlib.use("Agg")  # no display needed; set before pyplot is imported

import matplotlib.image as mpimg
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from utils import DCGANLog, VAELog, WGANLog


def final_sample(run: Path) -> Path:
    """The last per-epoch grid, e.g. samples/epoch_025.png. Excludes the epoch-1 step grids."""
    pngs = sorted((run / "samples").glob("epoch_[0-9][0-9][0-9].png"))
    if not pngs:
        raise SystemExit(f"{run}: no per-epoch sample grids found")
    return pngs[-1]


def save(fig: Figure, out: Path) -> Path:
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)  # pyright: ignore[reportUnknownMemberType]
    return out


def load_log(run: Path) -> DCGANLog | WGANLog | VAELog:
    """RUN/logs/losses.pth as the log class of the run's family (the folder above it).

    The classes check their keys exactly, so a WGAN file cannot be read as a DCGAN log.
    """
    path = run / "logs" / "losses.pth"
    match run.parent.name:
        case "dcgan":
            return DCGANLog.load(path)
        case "wgan" | "wgangp":  # the GP variant logs the same record
            return WGANLog.load(path)
        case "vae":
            return VAELog.load(path)
        case family:
            raise SystemExit(f"{run}: no losses.pth reader for family {family!r}")


def plot_losses(run: Path) -> Path:
    log = load_log(run)
    fig = Figure(figsize=(11, 4))
    ax_loss: Axes = fig.add_subplot(1, 2, 1)
    ax_right: Axes = fig.add_subplot(1, 2, 2)

    if isinstance(log, DCGANLog):
        _ = ax_loss.plot(
            log.step, log.loss_D, linewidth=0.8, label="loss_D (real->1 + fake->0)"
        )
        _ = ax_loss.plot(
            log.step, log.loss_G, linewidth=0.8, label="loss_G (non-saturating)"
        )
        _ = ax_right.plot(log.step, log.D_x, linewidth=0.8, label="D(x), mean")
        _ = ax_right.plot(log.step, log.D_G_z, linewidth=0.8, label="D(G(z)), mean")
        _ = ax_right.axhline(0.5, color="gray", linestyle=":", linewidth=0.8)
        _ = ax_right.set_ylabel("probability")
        _ = ax_right.set_ylim(0, 1)
        _ = ax_right.set_title("discriminator outputs (pre-update)")
    elif isinstance(log, VAELog):
        _ = ax_loss.plot(log.step, log.recon, linewidth=0.8, label="recon (train)")
        _ = ax_loss.plot(log.step, log.kl, linewidth=0.8, label="KL (train)")
        _ = ax_loss.plot(
            log.step, log.loss, linewidth=0.8, label="recon + beta KL (train)"
        )
        _ = ax_right.plot(
            log.val_epoch,
            log.val_loss,
            marker="o",
            markersize=2,
            label="val -ELBO (beta = 1)",
        )
        _ = ax_right.plot(
            log.val_epoch, log.val_recon, linewidth=0.8, label="val recon"
        )
        _ = ax_right.plot(log.val_epoch, log.val_kl, linewidth=0.8, label="val KL")
        _ = ax_right.set_xlabel("epoch")
        _ = ax_right.set_ylabel("nats per image")
        _ = ax_right.set_title("validation")
    else:
        # Critic scores are not probabilities: plot the Wasserstein estimate instead.
        _ = ax_loss.plot(
            log.step, log.loss_D, linewidth=0.8, label="loss_D (critic: fake - real)"
        )
        _ = ax_loss.plot(
            log.step, log.loss_G, linewidth=0.8, label="loss_G (-E[D(G(z))])"
        )
        _ = ax_right.plot(log.step, log.w_dist, linewidth=0.8, label="w_dist = -loss_D")
        _ = ax_right.set_ylabel("Wasserstein estimate")
        suffix = "clipped critic" if run.parent.name == "wgan" else "gradient penalty"
        _ = ax_right.set_title(f"critic estimate of W1 ({suffix})")
    _ = ax_loss.set_xlabel("step")
    _ = ax_loss.set_ylabel("loss")
    _ = ax_loss.set_title(run.name)
    _ = ax_loss.legend()
    if not isinstance(log, VAELog):
        _ = ax_right.set_xlabel("step")
    _ = ax_right.legend()

    fig.tight_layout()
    return save(fig, run / "figures" / "losses.png")


def plot_grids(runs: list[Path], out: Path) -> Path:
    fig = Figure(figsize=(5 * len(runs), 5.6))
    for i, run in enumerate(runs, start=1):
        ax: Axes = fig.add_subplot(1, len(runs), i)
        grid = final_sample(run)
        _ = ax.imshow(mpimg.imread(grid))  # pyright: ignore[reportUnknownMemberType]
        _ = ax.set_title(f"{run.name}\n{grid.name}", fontsize=9)
        _ = ax.axis("off")
    fig.tight_layout()
    return save(fig, out)


PROGRESS_STEPS = (
    "epoch_001_step_0000",
    "epoch_001_step_0100",
    "epoch_001_step_0200",
    "epoch_001",
)
PROGRESS_EPOCHS = (5, 10, 25, 50, 100)
DIAGNOSTICS = Path("outputs/diagnostics.json")


def progress_grids(run: Path) -> list[Path]:
    """Early step grids, then selected epochs that exist, then the final epoch."""
    samples = run / "samples"
    picks = [samples / f"{name}.png" for name in PROGRESS_STEPS]
    picks += [samples / f"epoch_{e:03d}.png" for e in PROGRESS_EPOCHS]
    last = final_sample(run)
    found = [p for p in picks if p.exists()]
    return found if last in found else [*found, last]


def plot_progress(run: Path, out: Path) -> Path:
    grids = progress_grids(run)
    fig = Figure(figsize=(2.6 * len(grids), 3.0))
    for i, grid in enumerate(grids, start=1):
        ax: Axes = fig.add_subplot(1, len(grids), i)
        _ = ax.imshow(mpimg.imread(grid))  # pyright: ignore[reportUnknownMemberType]
        _ = ax.set_title(
            grid.stem.replace("epoch_", "ep ").replace("_step_", " step "), fontsize=9
        )
        _ = ax.axis("off")
    fig.suptitle(f"{run.name}: the same 64 z through training", fontsize=10)
    fig.tight_layout()
    return save(fig, out)


def val_fid_curve(run: Path) -> tuple[list[int], list[float]]:
    """(epoch, FID) for val-5k records of epoch_*.pt checkpoints, sorted by epoch."""
    records = cast(
        "list[dict[str, object]]", json.loads((run / "fid.json").read_text())
    )
    points = sorted(
        (cast("int", r["epoch"]), cast("float", r["fid"]))
        for r in records
        if r["split"] == "val" and str(r["checkpoint"]).startswith("epoch_")
    )
    return [e for e, _ in points], [f for _, f in points]


def plot_fid(runs: list[Path], out: Path) -> Path:
    fig = Figure(figsize=(7, 4.2))
    ax: Axes = fig.add_subplot(1, 1, 1)
    for run in runs:
        epochs, fids = val_fid_curve(run)
        _ = ax.plot(epochs, fids, marker="o", markersize=3, linewidth=1, label=run.name)
    if DIAGNOSTICS.exists():
        floor = cast(
            "dict[str, dict[str, float]]", json.loads(DIAGNOSTICS.read_text())
        )["fid_floor"]
        _ = ax.axhline(
            floor["train5k_vs_val5k"],
            color="gray",
            linestyle=":",
            label="real-data floor (5k)",
        )
    _ = ax.set_yscale("log")
    _ = ax.set_xlabel("epoch")
    _ = ax.set_ylabel("FID, val 5k (log scale)")
    _ = ax.legend(fontsize=8)
    fig.tight_layout()
    return save(fig, out)


def export_losses(run: Path) -> Path:
    out = run / "logs" / "losses.json"
    _ = out.write_text(json.dumps(asdict(load_log(run))))
    return out


class Args(argparse.Namespace):
    """Typed arguments. Each subcommand sets the fields it uses."""

    cmd: str
    run: Path
    runs: list[Path]
    out: Path | None


def parse_args() -> Args:
    parser = argparse.ArgumentParser(
        description="Quick-look figures for DCGAN, WGAN and VAE runs."
    )
    sub = parser.add_subparsers(dest="cmd", required=True)
    p_loss = sub.add_parser("losses")
    _ = p_loss.add_argument("run", type=Path)
    p_grid = sub.add_parser("grids")
    _ = p_grid.add_argument("runs", type=Path, nargs="+")
    _ = p_grid.add_argument("-o", "--out", type=Path, required=True)
    p_prog = sub.add_parser("progress")
    _ = p_prog.add_argument("run", type=Path)
    _ = p_prog.add_argument("-o", "--out", type=Path, default=None)
    p_fid = sub.add_parser("fid")
    _ = p_fid.add_argument("runs", type=Path, nargs="+")
    _ = p_fid.add_argument("-o", "--out", type=Path, required=True)
    p_exp = sub.add_parser("export")
    _ = p_exp.add_argument("run", type=Path)
    return parser.parse_args(namespace=Args())


if __name__ == "__main__":
    args = parse_args()
    if args.cmd == "losses":
        print(plot_losses(args.run))
    elif args.cmd == "grids":
        assert args.out is not None
        print(plot_grids(args.runs, args.out))
    elif args.cmd == "progress":
        print(
            plot_progress(args.run, args.out or args.run / "figures" / "progress.png")
        )
    elif args.cmd == "fid":
        assert args.out is not None
        print(plot_fid(args.runs, args.out))
    else:
        print(export_losses(args.run))
