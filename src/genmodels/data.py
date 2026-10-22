"""MNIST data: two scalings, one fixed 55k/5k train/validation split, and a
test loader that is only meant for final-evaluation code.

Contract (Setup_GAN.md section 2):
  * scaling "gan": ToTensor + Normalize(0.5, 0.5)  -> pixels in [-1, 1]
                   (DCGAN, WGAN, classifier, and the input to phi for FID)
  * scaling "vae": ToTensor                         -> pixels in [0, 1]
                   (VAE Bernoulli/BCE reconstruction)
  * the training set (60k) is split once into 55,000 train / 5,000 validation
    with a fixed seed; the 10,000 test images are never part of the split.
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

# Repo root is two levels above src/genmodels/data.py.
DATA_ROOT = Path(__file__).resolve().parents[2] / "data"

SPLIT_SEED = 0
N_VAL = 5000
N_TRAIN_FULL = 60000


def make_transform(scaling: str) -> transforms.Compose:
    if scaling == "gan":
        return transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
        )
    if scaling == "vae":
        return transforms.Compose([transforms.ToTensor()])
    raise ValueError(f"unknown scaling {scaling!r}; expected 'gan' or 'vae'")


def split_indices(
    n_total: int = N_TRAIN_FULL, n_val: int = N_VAL, seed: int = SPLIT_SEED
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (train_idx, val_idx). Deterministic for a given seed."""
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n_total, generator=g)
    return perm[n_val:], perm[:n_val]


def get_train_val_datasets(scaling: str, root: Path = DATA_ROOT):
    """Train (55k) and validation (5k) subsets of the MNIST training set."""
    full = datasets.MNIST(
        root=str(root), train=True, download=True, transform=make_transform(scaling)
    )
    train_idx, val_idx = split_indices(len(full))
    return Subset(full, train_idx.tolist()), Subset(full, val_idx.tolist())


def get_train_val_loaders(
    scaling: str,
    batch_size: int = 128,
    num_workers: int = 4,
    seed: int = 0,
    root: Path = DATA_ROOT,
):
    """Train loader (shuffled, seeded) and validation loader (not shuffled)."""
    train_ds, val_ds = get_train_val_datasets(scaling, root)
    g = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        generator=g,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
    )
    return train_loader, val_loader


def get_test_loader(
    scaling: str,
    batch_size: int = 128,
    num_workers: int = 4,
    root: Path = DATA_ROOT,
) -> DataLoader:
    """The 10k MNIST test set. Import and use only in final-evaluation code."""
    test_ds = datasets.MNIST(
        root=str(root), train=False, download=True, transform=make_transform(scaling)
    )
    return DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
