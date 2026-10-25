"""MNIST pixel scalings and the fixed train/validation split (Setup_GAN.md section 2).

Download once, then train offline:

    python data.py

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

from collections.abc import Callable, Iterable, Iterator, Sequence
from pathlib import Path
from typing import cast

import torch
from PIL import Image
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms

from utils.records import Splits

# This file is in src/, so the repo root (which holds data/) is parents[1].
DATA_ROOT = Path(__file__).resolve().parents[1] / "data"

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


def unpack_batch(batch: object) -> tuple[torch.Tensor, torch.Tensor]:
    """Narrow one DataLoader item to (images, labels), checking types at runtime.

    DataLoader.__next__ is typed as returning Any. Accepting `object` here keeps
    that Any out of the caller's variables; the isinstance checks do the narrowing.
    """
    # default_collate returns a list [images, labels], not a tuple.
    if not isinstance(batch, (list, tuple)):
        raise TypeError(f"expected a list or tuple, got {type(batch).__name__}")
    items = cast(Sequence[object], batch)
    if len(items) != 2:
        raise TypeError(f"expected 2 items, got {len(items)}")
    images, labels = items
    if not isinstance(images, torch.Tensor) or not isinstance(labels, torch.Tensor):
        raise TypeError("batch items must be tensors")
    return images, labels


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
    # Iterating a tensor and calling int() avoids tolist(), which pyright reads as partially unknown.
    train_list = [int(i) for i in train_idx]
    val_list = [int(i) for i in val_idx]
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


def gan_loaders() -> Splits[DataLoader[Sample]]:
    """[-1, 1] scaling: DCGAN and WGAN training, and the FID real set (test)."""
    train, val = get_train_val_loaders("gan")
    return Splits(train=train, val=val, test=get_test_loader("gan"))


def vae_loaders() -> Splits[DataLoader[Sample]]:
    """[0, 1] scaling: VAE training. Its test split is for final VAE metrics only;
    FID real images always come from gan_loaders().test."""
    train, val = get_train_val_loaders("vae")
    return Splits(train=train, val=val, test=get_test_loader("vae"))


def batches(loader: DataLoader[Sample]) -> Iterator[tuple[Tensor, Tensor]]:
    """Typed (images, labels) batches. DataLoader iteration is typed Any upstream."""
    for batch in cast(Iterable[object], loader):
        yield unpack_batch(batch)


if __name__ == "__main__":
    # Download both MNIST splits into data/ (skipped if already present).
    train_set = datasets.MNIST(root=str(DATA_ROOT), train=True, download=True)
    test_set = datasets.MNIST(root=str(DATA_ROOT), train=False, download=True)
    print(f"MNIST at {DATA_ROOT}: train={len(train_set)} test={len(test_set)}")
