"""MNIST pixel scalings and the fixed train/validation split (Setup_GAN.md section 2).

  * "gan": ToTensor + Normalize(0.5, 0.5) -> pixels in [-1, 1]
           (matches the generator's Tanh output and the classifier input)
  * "vae": ToTensor -> pixels in [0, 1]
           (matches the Bernoulli / BCE reconstruction likelihood)
  * the 60k MNIST training set is split once into 55,000 train / 5,000
    validation with a fixed seed. The 10,000 test images are outside the split.
"""

from pathlib import Path

import torch
from torchvision import transforms

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
    """Return (train_idx, val_idx). Deterministic for a given seed.

    A dedicated torch.Generator keeps the split independent of global RNG state.
    """
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n_total, generator=g)
    return perm[n_val:], perm[:n_val]
