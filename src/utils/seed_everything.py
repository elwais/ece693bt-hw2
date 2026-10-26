import random

import torch


def seed_everything(seed: int) -> None:
    """Seed Python's and PyTorch's random number generators (CPU, CUDA and MPS)."""
    random.seed(seed)
    # torch.manual_seed leaves `seed` unannotated (torch 2.14), so its type is
    # partially Unknown. This is the only place that calls it.
    _ = torch.manual_seed(seed)  # pyright: ignore[reportUnknownMemberType]
