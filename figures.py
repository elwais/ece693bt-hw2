# pyright: reportUnknownMemberType=false
# matplotlib's keyword arguments are untyped, so every plotting call would be flagged.
# Only this rule is relaxed; all other strict checks still apply to this file.
"""Figures for DCGAN runs (Part 1 items 1-2, and the 20x comparison).

    python figures.py losses RUN_DIR
        -> RUN_DIR/figures/losses.png: loss_D and loss_G, and D_x and D_G_z, vs step

    python figures.py grids RUN_A RUN_B [...] -o OUT.png
        -> the final sample grid of each run, side by side
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # no display needed; set before pyplot is imported

import matplotlib.image as mpimg  # noqa: E402
from matplotlib.axes import Axes  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

from utils import GANLog  # noqa: E402


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


def plot_losses(run: Path) -> Path:
    log = GANLog.load(run / "logs" / "losses.pth")
    fig = Figure(figsize=(11, 4))
    ax_loss: Axes = fig.add_subplot(1, 2, 1)
    ax_prob: Axes = fig.add_subplot(1, 2, 2)

    _ = ax_loss.plot(log.step, log.loss_D, linewidth=0.8, label="loss_D (real->1 + fake->0)")
    _ = ax_loss.plot(log.step, log.loss_G, linewidth=0.8, label="loss_G (non-saturating)")
    _ = ax_loss.set_xlabel("step")
    _ = ax_loss.set_ylabel("loss")
    _ = ax_loss.set_title(run.name)
    _ = ax_loss.legend()

    _ = ax_prob.plot(log.step, log.D_x, linewidth=0.8, label="D(x), mean")
    _ = ax_prob.plot(log.step, log.D_G_z, linewidth=0.8, label="D(G(z)), mean")
    _ = ax_prob.axhline(0.5, color="gray", linestyle=":", linewidth=0.8)
    _ = ax_prob.set_xlabel("step")
    _ = ax_prob.set_ylabel("probability")
    _ = ax_prob.set_ylim(0, 1)
    _ = ax_prob.set_title("discriminator outputs (pre-update)")
    _ = ax_prob.legend()

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


class Args(argparse.Namespace):
    """Typed arguments. Each subcommand sets the fields it uses."""

    cmd: str
    run: Path
    runs: list[Path]
    out: Path


def parse_args() -> Args:
    parser = argparse.ArgumentParser(description="Figures for DCGAN runs.")
    sub = parser.add_subparsers(dest="cmd", required=True)
    p_loss = sub.add_parser("losses")
    _ = p_loss.add_argument("run", type=Path)
    p_grid = sub.add_parser("grids")
    _ = p_grid.add_argument("runs", type=Path, nargs="+")
    _ = p_grid.add_argument("-o", "--out", type=Path, required=True)
    return parser.parse_args(namespace=Args())


if __name__ == "__main__":
    args = parse_args()
    if args.cmd == "losses":
        print(plot_losses(args.run))
    else:
        print(plot_grids(args.runs, args.out))
