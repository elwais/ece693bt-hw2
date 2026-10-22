"""Data contract: pixel scalings."""

import torch

from genmodels.data import make_transform


def test_scalings_on_a_fake_image():
    import numpy as np
    from PIL import Image

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
