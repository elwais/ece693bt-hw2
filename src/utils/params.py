from pathlib import Path
from typing import cast

import torch
from torch import Tensor, nn


def save_atomic(obj: object, path: Path) -> None:
    """torch.save(obj, path) without ever leaving a half-written file at `path`.

    Writes to a temporary file, then renames it into place. If the process dies
    mid-write, any existing file at `path` is left intact.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp)
    _ = tmp.replace(path)


def save_params(model: nn.Module, path: Path) -> None:
    """Write the model's parameters and buffers (its state_dict) to `path`."""
    save_atomic(model.state_dict(), path)


def load_params[M: nn.Module](model: M, path: Path) -> M:
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
