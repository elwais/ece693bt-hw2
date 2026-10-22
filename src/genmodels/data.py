"""MNIST pixel scalings (Setup_GAN.md section 2).

  * "gan": ToTensor + Normalize(0.5, 0.5) -> pixels in [-1, 1]
           (matches the generator's Tanh output and the classifier input)
  * "vae": ToTensor -> pixels in [0, 1]
           (matches the Bernoulli / BCE reconstruction likelihood)
"""

from pathlib import Path

from torchvision import transforms

# Repo root is two levels above src/genmodels/data.py.
DATA_ROOT = Path(__file__).resolve().parents[2] / "data"


def make_transform(scaling: str) -> transforms.Compose:
    if scaling == "gan":
        return transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
        )
    if scaling == "vae":
        return transforms.Compose([transforms.ToTensor()])
    raise ValueError(f"unknown scaling {scaling!r}; expected 'gan' or 'vae'")
