"""Part 3.3: sweep ten latent dimensions while keeping the other ninety fixed.

The assignment says to keep z11:100 the same and modify z0:10 (1-indexed), so
0-indexed dimensions 0..9 sweep across --values while 10..99 stay at the base
sample. Each seed draws a fresh base z ~ N(0, I) and gets its own grid image.

    python vae_traverse.py --run outputs/vae/<RUN> [--checkpoint epoch_050.pt]
        [--seeds 0 1] [--values -3 -2 -1 0 1 2 3] [--dims 0 1 2 3 4 5 6 7 8 9]

One row per varied dimension, one column per value; the value axis runs left to
right, so a row shows how that coordinate deforms the image while the other 99
stay put.
"""

import argparse
from pathlib import Path
from typing import cast

import torch
from torch import Tensor
from torchvision.utils import (
    save_image,  # pyright: ignore[reportUnknownVariableType]  # torchvision has no stubs
)

from utils import eval_mode, get_preferred_device
from vae import Z_DIM, Decoder


class Args(argparse.Namespace):
    """Typed command-line arguments."""

    run: str = ""  # outputs/vae/<RUN>; required
    checkpoint: str = "final.pt"
    seeds: tuple[int, ...] = (0,)
    values: tuple[float, ...] = (-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0)
    dims: tuple[int, ...] | None = None  # None -> 0..9, the assignment's z0:10
    out: str | None = None  # default: RUN/figures/traverse_seed{S}.png


def parse_args() -> Args:
    parser = argparse.ArgumentParser(
        description="Sweep latent dimensions of a trained VAE decoder."
    )
    _ = parser.add_argument("--run", type=str, required=True, help="outputs/vae/<RUN>")
    _ = parser.add_argument("--checkpoint", type=str, default=Args.checkpoint)
    _ = parser.add_argument("--seeds", type=int, nargs="+", default=Args.seeds)
    _ = parser.add_argument("--values", type=float, nargs="+", default=Args.values)
    _ = parser.add_argument("--dims", type=int, nargs="+", default=None)
    _ = parser.add_argument("--out", type=str, default=Args.out)
    return parser.parse_args(namespace=Args())


def load_decoder(run: Path, checkpoint: str, device: str) -> Decoder:
    """The trained decoder from RUN/checkpoints/CHECKPOINT. Key 'G' is the decoder,
    matching the VAE trainer's state() so fid.py can treat it like a generator."""
    ckpt = cast(
        "dict[str, object]",
        torch.load(
            run / "checkpoints" / checkpoint, map_location="cpu", weights_only=True
        ),
    )
    decoder = Decoder()
    decoder.load_state_dict(cast("dict[str, Tensor]", ckpt["G"]))
    return decoder.to(device)


def z_variant(z_base: Tensor, dim: int, value: float, device: torch.device) -> Tensor:
    """z_base with coordinate `dim` replaced by `value`; everything else unchanged."""
    z = z_base.clone().to(device)
    z[..., dim, 0, 0] = value
    return z


def traverse(
    decoder: Decoder,
    run: Path,
    checkpoint: str,
    seeds: list[int],
    values: list[float],
    dims: list[int],
    out: Path | None,
) -> list[Path]:
    """One grid per seed: rows are `dims`, columns are `values`."""
    written: list[Path] = []
    # nn.Module has no .device attribute; the first parameter's device is the module's.
    device = next(decoder.parameters()).device
    for seed in seeds:
        gen = torch.Generator(device=device).manual_seed(seed)
        z_base = torch.randn(1, Z_DIM, 1, 1, generator=gen, device=device)
        rows: list[Tensor] = []
        with eval_mode(decoder):
            for dim in dims:
                z = torch.cat(
                    [z_variant(z_base, dim, value, device) for value in values],
                    dim=0,
                )
                rows.append(decoder(z))
        grid = torch.cat(rows, dim=0)  # (len(dims) * len(values), 1, 28, 28)
        path = out or (run / "figures" / f"traverse_seed{seed}.png")
        if out is not None and len(seeds) > 1:  # keep the seeds apart
            path = out.with_name(f"{out.stem}_seed{seed}{out.suffix}")
        path.parent.mkdir(parents=True, exist_ok=True)

        # Columns must number the values; make the grid wider than it is tall.
        save_image(grid, path, nrow=len(values), normalize=True, value_range=(-1, 1))
        written.append(path)
    return written


if __name__ == "__main__":
    args = parse_args()
    run = Path(args.run)
    dims = list(range(10)) if args.dims is None else list(args.dims)
    device = get_preferred_device()
    decoder = load_decoder(run, args.checkpoint, device)
    paths = traverse(
        decoder,
        run,
        args.checkpoint,
        list(args.seeds),
        list(args.values),
        dims,
        None if args.out is None else Path(args.out),
    )
    print("wrote " + ", ".join(str(p) for p in paths))
