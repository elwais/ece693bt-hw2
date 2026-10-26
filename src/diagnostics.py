# pyright: reportUnknownMemberType=false
# matplotlib's keyword arguments are untyped (same relaxation as figures.py).
"""Extra evidence for the Part 1 write-up. Not a deliverable by itself.

    python diagnostics.py [RUN_DIR ...]

Without RUN_DIRs, uses every run under outputs/<family>/ whose name does not end in
"_smoke" (so any family — dcgan, wgan, vae — is discovered).

1. Real-data FID floor: FID between two disjoint sets of *real* images, i.e. what a
   perfect generator would score at the same sample sizes. Two pairs:
     - train 10k vs test 10k   (reference for the reported test-10k FIDs)
     - train 5k  vs val 5k     (reference for the val-5k selection FIDs)
   Plus test first 5k vs test last 5k: MNIST's test set is two writer populations
   (the first half drawn from NIST's cleaner training writers), so this is far from 0.
   The train subsets are a seeded random draw from the 55k training split.

2. Class histogram: 10k samples from each run's final.pt, labelled by the classifier.
   A flat histogram means every digit is produced; missing or dominant digits are
   evidence of mode dropping or collapse (Part 1 item 4). Also reports the mean
   top-class probability and the share of samples the classifier is unsure about
   (top probability < 0.5), a rough "unrecognisable digit" rate.

Writes outputs/diagnostics.json and outputs/figures/class_hist.png.
"""

import json
import math
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")

import torch  # noqa: E402
from matplotlib.axes import Axes  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402
from torch import Tensor  # noqa: E402
from torch.utils.data import DataLoader, Subset  # noqa: E402

from classifier import Classifier  # noqa: E402
from data import Sample, batches, gan_loaders, get_train_val_datasets  # noqa: E402
from fid import (  # noqa: E402
    Z_DIM,
    GEN_BATCH,
    fid_from_stats,
    load_classifier,
    load_generator,
    mean_cov,
    real_features,
)
from utils import eval_mode, get_preferred_device  # noqa: E402

OUT_JSON = Path("outputs/diagnostics.json")
OUT_FIG = Path("outputs/figures/class_hist.png")
N_SAMPLES = 10_000
SEED = 0
UNSURE = 0.5  # top-class probability below this counts as "unsure"


# ---- 1. real-data FID floor -------------------------------------------------


def train_subset_loader(n: int, seed: int) -> DataLoader[Sample]:
    """A seeded random n-image subset of the 55k training split, not shuffled."""
    train_ds, _ = get_train_val_datasets("gan")
    perm = torch.randperm(len(train_ds), generator=torch.Generator().manual_seed(seed))
    idx = [int(i) for i in perm[:n]]
    return DataLoader(Subset[Sample](train_ds, idx), batch_size=500, shuffle=False)


def fid_between(a: Tensor, b: Tensor) -> float:
    mu_a, cov_a = mean_cov(a)
    mu_b, cov_b = mean_cov(b)
    return fid_from_stats(mu_a, cov_a, mu_b, cov_b)


def fid_floor(clf: Classifier, device: str) -> dict[str, float]:
    loaders = gan_loaders()
    test = real_features(clf, loaders.test, device)
    val = real_features(clf, loaders.val, device)
    train_10k = real_features(clf, train_subset_loader(10_000, SEED), device)
    return {
        "train10k_vs_test10k": fid_between(train_10k, test),
        "train5k_vs_val5k": fid_between(train_10k[:5000], val),
        "test_first5k_vs_last5k": fid_between(test[:5000], test[5000:]),
    }


# ---- 2. class histogram -------------------------------------------------------


def class_counts(probs: Tensor) -> list[int]:
    return [int(c) for c in torch.bincount(probs.argmax(dim=1), minlength=10)]


def class_stats(probs: Tensor) -> dict[str, object]:
    """Counts per digit, entropy of the class distribution, and confidence summaries."""
    top, labels = probs.max(dim=1)
    counts = torch.bincount(labels, minlength=10)
    p = counts.double() / counts.sum()
    nz = p[p > 0]
    entropy = float(-(nz * nz.log()).sum().item())
    return {
        "counts": [int(c) for c in counts],
        "entropy_ratio": entropy / math.log(10),  # 1.0 = perfectly uniform over 10 digits
        "max_over_min": float(counts.max().item()) / max(int(counts.min().item()), 1),
        "mean_top_prob": float(top.mean().item()),
        "unsure_frac": float((top < UNSURE).double().mean().item()),
    }


def classify_real_test(clf: Classifier, device: str) -> Tensor:
    chunks: list[Tensor] = []
    with eval_mode(clf):
        for x, _ in batches(gan_loaders().test):
            chunks.append(clf(x.to(device)).softmax(dim=1).cpu())
    return torch.cat(chunks)


def classify_samples(run: Path, clf: Classifier, device: str) -> tuple[Tensor, int]:
    G, epoch = load_generator(run / "checkpoints" / "final.pt", device)
    z_all = torch.randn(N_SAMPLES, Z_DIM, 1, 1, generator=torch.Generator().manual_seed(SEED))
    chunks: list[Tensor] = []
    with eval_mode(G), eval_mode(clf):
        for i in range(0, N_SAMPLES, GEN_BATCH):
            z = z_all[i : i + GEN_BATCH].to(device)
            chunks.append(clf(G(z)).softmax(dim=1).cpu())
    return torch.cat(chunks), epoch


def plot_hist(counts_by_name: dict[str, list[int]], out: Path) -> Path:
    names = list(counts_by_name)
    width = 0.8 / len(names)
    fig = Figure(figsize=(11, 4))
    ax: Axes = fig.add_subplot(1, 1, 1)
    for k, name in enumerate(names):
        counts = counts_by_name[name]
        xs = [d + (k - (len(names) - 1) / 2) * width for d in range(10)]
        _ = ax.bar(xs, [c / N_SAMPLES for c in counts], width=width, label=name)
    _ = ax.axhline(0.1, color="gray", linestyle=":", linewidth=0.8)
    _ = ax.set_xticks(range(10))
    _ = ax.set_xlabel("predicted digit")
    _ = ax.set_ylabel("share of 10k samples")
    _ = ax.set_title("Classifier labels of generated samples (final.pt) vs the real test set")
    _ = ax.legend(fontsize=8)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    return out


def default_runs() -> list[Path]:
    """Every run under outputs/, one level per family (dcgan, wgan, vae, ...)."""
    root = Path("outputs")
    if not root.is_dir():
        return []
    runs = [
        p
        for family in sorted(f for f in root.iterdir() if f.is_dir())
        for p in family.iterdir()
        if p.is_dir() and not p.name.endswith("_smoke")
    ]
    return sorted(runs)


if __name__ == "__main__":
    runs = [Path(a) for a in sys.argv[1:]] or default_runs()
    device = get_preferred_device()
    clf = load_classifier(device)

    floor = fid_floor(clf, device)
    print("real-data FID floor:", {k: round(v, 3) for k, v in floor.items()})

    real_probs = classify_real_test(clf, device)
    stats: dict[str, dict[str, object]] = {"real test": class_stats(real_probs)}
    counts: dict[str, list[int]] = {"real test": class_counts(real_probs)}
    for run in runs:
        probs, epoch = classify_samples(run, clf, device)
        stats[run.name] = {"epoch": epoch, **class_stats(probs)}
        counts[run.name] = class_counts(probs)
    for name, s in stats.items():
        print(
            f"{name:32s} counts {s['counts']}  entropy {s['entropy_ratio']:.3f}  "
            + f"max/min {s['max_over_min']:.2f}  unsure {s['unsure_frac']:.3f}"
        )

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    _ = OUT_JSON.write_text(
        json.dumps({"fid_floor": floor, "class_hist": stats, "n_samples": N_SAMPLES, "seed": SEED}, indent=2)
    )
    print(OUT_JSON, plot_hist(counts, OUT_FIG))
