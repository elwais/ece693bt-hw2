"""Frechet Inception Distance on the classifier's features phi(x) (Part 1, item 5).

    python fid.py RUN_DIR [--checkpoint final.pt|all] [--split val|test] [--n 10000]

Loads the generator from RUN_DIR/checkpoints/CHECKPOINT and appends one record to
RUN_DIR/fid.json. `--checkpoint all` evaluates every epoch_*.pt; records already in
fid.json (same checkpoint, split, n, seed) are skipped, so re-running is cheap. Real features come from the validation set (5k, for choosing
between runs) or the test set (10k, for the reported number).

Math (B5): for real features with mean mu_r and covariance S_r, and generated
features with mean mu_g and covariance S_g,

    FID = |mu_r - mu_g|^2 + Tr(S_r) + Tr(S_g) - 2 Tr((S_r S_g)^(1/2))

Tr((S_r S_g)^(1/2)) is computed as the sum of square roots of the eigenvalues of
S_r^(1/2) S_g S_r^(1/2). That matrix is symmetric and has the same eigenvalues
as S_r S_g, so eigvalsh applies. Eigenvalues are clamped at 0, because S_g is
singular whenever n < 128.
"""

import argparse
import json
from pathlib import Path
from typing import cast

import torch
from torch import Tensor
from torch.utils.data import DataLoader

from classifier import Classifier
from data import Sample, batches, gan_loaders
from dcgan import Generator
from utils import eval_mode, get_preferred_device, load_params

CLASSIFIER_PATH = Path("outputs/classifier.pt")
Z_DIM = 100
GEN_BATCH = 500


# ---- FID math (B5) ---------------------------------------------------------

# torch.linalg has no usable type information here, like torchvision. These two
# wrappers are the typed boundary; everything else in this file is typed.


def eigh(m: Tensor) -> tuple[Tensor, Tensor]:
    """Eigenvalues (ascending) and eigenvectors (columns) of a symmetric matrix."""
    return cast("tuple[Tensor, Tensor]", torch.linalg.eigh(m))  # pyright: ignore[reportUnknownMemberType]


def eigvalsh(m: Tensor) -> Tensor:
    """Eigenvalues of a symmetric matrix."""
    return cast("Tensor", torch.linalg.eigvalsh(m))  # pyright: ignore[reportUnknownMemberType]


def mean_cov(feats: Tensor) -> tuple[Tensor, Tensor]:
    """Mean and unbiased covariance (1/(n-1)) of the rows of `feats`, in float64."""
    x = feats.to(torch.float64)
    mu = x.mean(dim=0)
    centred = x - mu
    cov = centred.T @ centred / (x.shape[0] - 1)
    return mu, cov


def sqrt_psd(m: Tensor) -> Tensor:
    """Symmetric PSD square root via eigh. Tiny negative eigenvalues are clamped to 0."""
    w, v = eigh(m)
    return (v * w.clamp(min=0).sqrt()) @ v.T


def fid_from_stats(mu_r: Tensor, cov_r: Tensor, mu_g: Tensor, cov_g: Tensor) -> float:
    """FID between two Gaussians given by their means and covariances."""
    s = sqrt_psd(cov_r)
    m = s @ cov_g @ s
    m = (m + m.T) / 2  # remove asymmetry from rounding before eigvalsh
    eig = eigvalsh(m).clamp(min=0)
    trace_sqrt = eig.sqrt().sum()
    diff = mu_r - mu_g
    fid = diff.dot(diff) + cov_r.trace() + cov_g.trace() - 2 * trace_sqrt
    return float(fid.item())


# ---- feature extraction ------------------------------------------------------


@torch.inference_mode()
def real_features(clf: Classifier, loader: DataLoader[Sample], device: str) -> Tensor:
    chunks = [clf.phi(x.to(device)).cpu() for x, _ in batches(loader)]
    return torch.cat(chunks)


def generated_features(
    G: Generator, clf: Classifier, n: int, seed: int, device: str
) -> Tensor:
    """phi(G(z)) for n seeded z. The z come from a CPU generator, so they do not depend on the device."""
    gen = torch.Generator().manual_seed(seed)
    z_all = torch.randn(n, Z_DIM, 1, 1, generator=gen)
    chunks: list[Tensor] = []
    with eval_mode(G):
        for i in range(0, n, GEN_BATCH):
            z = z_all[i : i + GEN_BATCH].to(device)
            chunks.append(clf.phi(G(z)).cpu())
    return torch.cat(chunks)


def load_classifier(device: str) -> Classifier:
    with torch.device(device):
        clf = Classifier()
    _ = load_params(clf, CLASSIFIER_PATH)
    return clf


def load_generator(path: Path, device: str) -> tuple[Generator, int]:
    """Generator weights and the epoch from a checkpoint written by dcgan.py."""
    raw = cast("dict[str, object]", torch.load(path, map_location="cpu", weights_only=True))
    state = cast("dict[str, Tensor]", raw["G"])
    epoch = cast("int", raw["epoch"])
    with torch.device(device):
        G = Generator()
    _ = G.load_state_dict(state)
    return G, epoch


# ---- command line -----------------------------------------------------------


class Args(argparse.Namespace):
    """Typed arguments. Defaults are class attributes, so read them by attribute."""

    run_dir: Path = Path()
    checkpoint: str = "final.pt"
    split: str = "val"
    n: int = 10000
    seed: int = 0


def parse_args() -> Args:
    parser = argparse.ArgumentParser(description="FID of a DCGAN checkpoint on classifier features.")
    _ = parser.add_argument("run_dir", type=Path)
    _ = parser.add_argument("--checkpoint", default=Args.checkpoint, help="a file in checkpoints/, or all")
    _ = parser.add_argument("--split", choices=["val", "test"], default=Args.split)
    _ = parser.add_argument("--n", type=int, default=Args.n)
    _ = parser.add_argument("--seed", type=int, default=Args.seed)
    return parser.parse_args(namespace=Args())


def checkpoint_names(run_dir: Path, spec: str) -> list[str]:
    """`all` means every 5-epoch checkpoint (epoch_*.pt) in order; otherwise the one name."""
    if spec != "all":
        return [spec]
    names = sorted(p.name for p in (run_dir / "checkpoints").glob("epoch_*.pt"))
    if not names:
        raise SystemExit(f"{run_dir}: no epoch_*.pt checkpoints")
    return names


def already_done(results: list[dict[str, object]], record: dict[str, object]) -> bool:
    """A record with the same checkpoint, split, n and seed is already in fid.json."""
    keys = ("checkpoint", "split", "n_generated", "seed")
    return any(all(r.get(k) == record[k] for k in keys) for r in results)


if __name__ == "__main__":
    args = parse_args()
    device = get_preferred_device()

    out = args.run_dir / "fid.json"
    results = cast("list[dict[str, object]]", json.loads(out.read_text())) if out.exists() else []
    todo = [
        name
        for name in checkpoint_names(args.run_dir, args.checkpoint)
        if not already_done(
            results, {"checkpoint": name, "split": args.split, "n_generated": args.n, "seed": args.seed}
        )
    ]
    if not todo:
        print(f"{args.run_dir.name}: nothing to do ({args.checkpoint}, {args.split}, n={args.n})")
        raise SystemExit(0)

    loaders = gan_loaders()
    real_loader = loaders.val if args.split == "val" else loaders.test
    clf = load_classifier(device)
    real = real_features(clf, real_loader, device)  # once, shared by every checkpoint
    mu_r, cov_r = mean_cov(real)

    for name in todo:
        G, epoch = load_generator(args.run_dir / "checkpoints" / name, device)
        mu_g, cov_g = mean_cov(generated_features(G, clf, args.n, args.seed, device))
        fid = fid_from_stats(mu_r, cov_r, mu_g, cov_g)
        results.append(
            {
                "checkpoint": name,
                "epoch": epoch,
                "split": args.split,
                "n_generated": args.n,
                "n_real": int(real.shape[0]),
                "seed": args.seed,
                "fid": fid,
            }
        )
        _ = out.write_text(json.dumps(results, indent=2))  # after each checkpoint: partial runs keep results
        print(f"{args.run_dir.name}  {name}  epoch {epoch}  {args.split}  n={args.n}  FID {fid:.3f}")
