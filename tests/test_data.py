"""Data contract: scalings, split sizes, disjointness, and reproducibility."""

import numpy as np
import torch
from PIL import Image

from genmodels.data import N_VAL, make_transform, split_indices


def _as_set(idx: torch.Tensor) -> set[int]:
    return {int(i) for i in idx}


def test_split_sizes_and_disjoint():
    tr, va = split_indices()
    assert len(va) == N_VAL == 5000
    assert len(tr) == 55000
    assert len(_as_set(tr) & _as_set(va)) == 0
    assert len(_as_set(tr) | _as_set(va)) == 60000


def test_split_reproducible_and_seed_sensitive():
    a_tr, a_va = split_indices(seed=0)
    b_tr, b_va = split_indices(seed=0)
    c_tr, _ = split_indices(seed=1)
    assert torch.equal(a_tr, b_tr) and torch.equal(a_va, b_va)
    assert not torch.equal(a_va, c_tr[:N_VAL])


def test_scalings_on_a_fake_image():
    arr = np.full((28, 28), 255, dtype=np.uint8)
    arr[0, 0] = 0  # one black pixel, one white pixel
    pil = Image.fromarray(arr)
    gan = make_transform("gan")(pil)
    vae = make_transform("vae")(pil)
    assert gan.shape == vae.shape == (1, 28, 28)
    assert torch.isclose(gan.min(), torch.tensor(-1.0))
    assert torch.isclose(gan.max(), torch.tensor(1.0))
    assert torch.isclose(vae.min(), torch.tensor(0.0))
    assert torch.isclose(vae.max(), torch.tensor(1.0))
