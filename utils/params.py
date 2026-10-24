from pathlib import Path
from typing import TypeVar, cast

import torch
from torch import Tensor, nn

M = TypeVar("M", bound=nn.Module)


def save_params(model: nn.Module, path: Path) -> None:
    """Write the model's parameters and buffers (its state_dict) to `path`."""
    path.parent.mkdir(parents=True, exist_ok=True)
    # Write to a temporary file, then rename it into place. If the process dies
    # mid-write, any existing file at `path` is left intact.
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(model.state_dict(), tmp)
    _ = tmp.replace(path)


def load_params(model: M, path: Path) -> M:
    """Copy the parameters saved at `path` into `model`, in place; return `model`.

    The model must have the same architecture: load_state_dict raises if any
    name or shape differs. Tensors load to the CPU first, then are copied onto
    the device the model is already on, so a file saved on a GPU loads anywhere.
    """
    # torch.load returns Any; the file holds a state_dict, and load_state_dict
    # checks every name and shape below.
    state = cast(
        "dict[str, Tensor]", torch.load(path, map_location="cpu", weights_only=True)
    )
    _ = model.load_state_dict(state)
    return model
