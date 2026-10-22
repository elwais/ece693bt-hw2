"""A2 audit drill: print what the data pipeline actually produces.

    python scripts/audit_data.py

Prints, for each scaling: one training batch's shape, dtype, and min/max.
Prints split sizes, and a SHA-256 of the train and validation index arrays.
Run it twice in separate processes; the hashes must match.
"""

import hashlib

import torch

from genmodels.data import get_train_val_loaders, split_indices


def sha(t: torch.Tensor) -> str:
    return hashlib.sha256(t.numpy().tobytes()).hexdigest()[:16]


def main() -> None:
    for scaling in ("gan", "vae"):
        train_loader, val_loader = get_train_val_loaders(scaling, num_workers=0)
        x, y = next(iter(train_loader))
        print(
            f"[{scaling}] train batch x shape={tuple(x.shape)} dtype={x.dtype} "
            f"min={x.min().item():.4f} max={x.max().item():.4f} "
            f"labels={tuple(y.shape)} label_range=({y.min().item()},{y.max().item()})"
        )
        xv, _ = next(iter(val_loader))
        print(
            f"[{scaling}] val batch   x shape={tuple(xv.shape)} "
            f"min={xv.min().item():.4f} max={xv.max().item():.4f}"
        )
        print(
            f"[{scaling}] sizes: train={len(train_loader.dataset)} "
            f"val={len(val_loader.dataset)}  (test loader deliberately not touched)"
        )

    tr, va = split_indices()
    print(f"split sizes: train={len(tr)} val={len(va)} sum={len(tr) + len(va)}")
    print(f"index hash: train={sha(tr)} val={sha(va)}")


if __name__ == "__main__":
    main()
