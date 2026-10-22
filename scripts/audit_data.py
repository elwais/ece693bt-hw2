"""A2 audit drill: print what the data pipeline actually produces.

    python scripts/audit_data.py

Prints, for each scaling: one training batch's shape, dtype, and min/max.
Prints the number of rows each train and validation loader yields in one pass.
Prints split sizes, and a SHA-256 of the train and validation index arrays.
Run it twice in separate processes; the hashes must match.
The test loader is deliberately not touched.
"""

import hashlib
from collections.abc import Iterable

import torch

from genmodels.data import get_train_val_loaders, split_indices, unpack_batch


def sha(t: torch.Tensor) -> str:
    return hashlib.sha256(t.numpy().tobytes()).hexdigest()[:16]


def first_batch(batches: Iterable[object]) -> tuple[torch.Tensor, torch.Tensor]:
    return unpack_batch(next(iter(batches)))


def count_rows(batches: Iterable[object]) -> int:
    total = 0
    for batch in batches:
        xb, _ = unpack_batch(batch)
        total += int(xb.shape[0])
    return total


def main() -> None:
    for scaling in ("gan", "vae"):
        train_loader, val_loader = get_train_val_loaders(scaling, num_workers=0)
        xb, yb = first_batch(train_loader)
        print(
            f"[{scaling}] train batch x shape={tuple(xb.shape)} dtype={xb.dtype} "
            + f"min={float(xb.min()):.4f} max={float(xb.max()):.4f} "
            + f"labels={tuple(yb.shape)} label_range=({int(yb.min())},{int(yb.max())})"
        )
        xvb, _ = first_batch(val_loader)
        print(
            f"[{scaling}] val batch   x shape={tuple(xvb.shape)} "
            + f"min={float(xvb.min()):.4f} max={float(xvb.max()):.4f}"
        )
        print(
            f"[{scaling}] rows per epoch: train={count_rows(train_loader)} "
            + f"val={count_rows(val_loader)}"
        )

    tr, va = split_indices()
    print(f"split sizes: train={len(tr)} val={len(va)} sum={len(tr) + len(va)}")
    print(f"index hash: train={sha(tr)} val={sha(va)}")


if __name__ == "__main__":
    main()
