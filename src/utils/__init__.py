"""Shared helpers for the assignment modules."""

from .nn_dataclass import nn_dataclass, built
from .seed_everything import seed_everything
from .eval_mode import eval_mode
from .params import load_params, save_atomic, save_params
from .get_device import get_preferred_device
from .as_tensor import as_tensor
from .records import (
    DCGANLog,
    Log,
    Metrics,
    SplitLabel,
    Splits,
    StepLog,
    VAELog,
    WGANLog,
)

__all__ = [
    "nn_dataclass",
    "built",
    "seed_everything",
    "eval_mode",
    "save_atomic",
    "save_params",
    "load_params",
    "get_preferred_device",
    "as_tensor",
    "DCGANLog",
    "Log",
    "Metrics",
    "SplitLabel",
    "Splits",
    "StepLog",
    "VAELog",
    "WGANLog",
]
