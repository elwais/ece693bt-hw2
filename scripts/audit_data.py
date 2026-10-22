"""A2 audit drill: print what the data pipeline actually produces.

    python scripts/audit_data.py

Prints, for each scaling: one training batch's shape, dtype, and min/max.
Prints the number of rows each train and validation loader yields in one pass.
Prints split sizes, and a SHA-256 of the train and validation index arrays.
Run it twice in separate processes; the hashes must match.
The test loader is deliberately not touched.
"""

import hashlib

import torch
from torch.utils.data import DataLoader

from genmodels.data import Sample, get_train_val_loaders, split_indices


def sha(t: torch.Tensor) -> str:
    return hashlib.sha256(t.numpy().tobytes()).hexdigest()[:16]


def count_rows(loader: DataLoader[Sample]) -> int:
    total = 0
    for xb, _ in loader:
        total += int(xb.shape[0])
    return total


def main() -> None:
    for scaling in ("gan", "vae"):
        train_loader, val_loader = get_train_val_loaders(scaling, num_workers=0)
        x, y = next(iter(train_loader))
        xb: torch.Tensor = x
        yb: torch.Tensor = y
        print(
            f"[{scaling}] train batch x shape={tuple(xb.shape)} dtype={xb.dtype} "
            + f"min={float(xb.min()):.4f} max={float(xb.max()):.4f} "
            + f"labels={tuple(yb.shape)} label_range=({int(yb.min())},{int(yb.max())})"
        )
        xv, _ = next(iter(val_loader))
        xvb: torch.Tensor = xv
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
