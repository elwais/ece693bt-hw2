"""Trivial but nice nonetheless "device-getter;" CUDA then MPS then CPU"""

from typing import Literal

import torch


def get_preferred_device() -> Literal["cuda", "mps", "cpu"]:
    if torch.cuda.is_available():
        return "cuda"

    if torch.mps.is_available():
        return "mps"

    return "cpu"
