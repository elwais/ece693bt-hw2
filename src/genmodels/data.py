"""MNIST pixel scalings and the fixed train/validation split (Setup_GAN.md section 2).

  * "gan": ToTensor + Normalize(0.5, 0.5) -> pixels in [-1, 1]
           (matches the generator's Tanh output and the classifier input)
  * "vae": ToTensor -> pixels in [0, 1]
           (matches the Bernoulli / BCE reconstruction likelihood)
  * the 60k MNIST training set is split once into 55,000 train / 5,000
    validation with a fixed seed. The 10,000 test images are outside the split.

torchvision ships no type stubs (no py.typed), so every torchvision value is
Unknown to basedpyright. The annotations in this module are the typed boundary:
they state what torchvision is expected to return, and pyright checks the rest.
"""

from collections.abc import Callable
from pathlib import Path
from typing import cast

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms

# Repo root is two levels above src/genmodels/data.py.
DATA_ROOT = Path(__file__).resolve().parents[2] / "data"

SPLIT_SEED = 0
N_VAL = 5000
N_TRAIN_FULL = 60000

# One MNIST item after the transform: (image tensor, integer label).
Sample = tuple[torch.Tensor, int]
# A pixel scaling: PIL image in, tensor out.
Scaled = Callable[[Image.Image], torch.Tensor]


def make_transform(scaling: str) -> Scaled:
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
    """Return (train_idx, val_idx). Deterministic for a given seed.

    A dedicated torch.Generator keeps the split independent of global RNG state.
    """
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n_total, generator=g)
    return perm[n_val:], perm[:n_val]


def get_train_val_datasets(
    scaling: str, root: Path = DATA_ROOT
) -> tuple[Subset[Sample], Subset[Sample]]:
    """Train (55k) and validation (5k) subsets of the MNIST training set."""
    full: Dataset[Sample] = datasets.MNIST(
        root=str(root), train=True, download=True, transform=make_transform(scaling)
    )
    train_idx, val_idx = split_indices(len(full))
    train_list = cast(list[int], train_idx.tolist())  # index tensor -> python ints
    val_list = cast(list[int], val_idx.tolist())
    return Subset[Sample](full, train_list), Subset[Sample](full, val_list)


def get_train_val_loaders(
    scaling: str,
    batch_size: int = 128,
    num_workers: int = 4,
    seed: int = 0,
    root: Path = DATA_ROOT,
) -> tuple[DataLoader[Sample], DataLoader[Sample]]:
    """Train loader (shuffled, seeded) and validation loader (not shuffled)."""
    train_ds, val_ds = get_train_val_datasets(scaling, root)
    g = torch.Generator().manual_seed(seed)
    train_loader: DataLoader[Sample] = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        generator=g,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
    )
    val_loader: DataLoader[Sample] = DataLoader(
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
) -> DataLoader[Sample]:
    """The 10k MNIST test set. Import and use only in final-evaluation code."""
    test_ds: Dataset[Sample] = datasets.MNIST(
        root=str(root), train=False, download=True, transform=make_transform(scaling)
    )
    return DataLoader[Sample](
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
