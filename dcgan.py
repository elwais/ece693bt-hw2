import argparse
from typing import TypeVar

from torch import Tensor, nn
import torch

from data import batches, gan_loaders
from utils import (
    as_tensor,
    built,
    get_preferred_device,
    nn_dataclass,
    seed_everything,
)

# Initialized weights of modules do not match spec; thus we use helper functions
# to apply them without needing __post_init__ methods in classes


# Mutate module `m` to give it appropriate underlying weights via the inplace `_`-suffix
# methods
def _init_layer(m: nn.Module) -> None:
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        _ = nn.init.normal_(m.weight, 0.0, 0.02)
        if m.bias is not None:
            _ = nn.init.zeros_(m.bias)
    elif isinstance(m, nn.BatchNorm2d):
        _ = nn.init.normal_(m.weight, 1.0, 0.02)
        _ = nn.init.zeros_(m.bias)


# For making the type checker complain if inputs are not subclassed from nn.Module
M = TypeVar("M", bound=nn.Module)


# Using `_init_layer` to apply the weight initialization
def _dcgan_init(net: M) -> M:
    """Apply Setup's initialization to every layer of `net`; return `net`."""
    return net.apply(_init_layer)


@nn_dataclass
class Generator(nn.Module):
    def __call__(self, x: Tensor):
        return as_tensor(super().__call__(x))

    # Note the use of _dcgan_init to put in the appropriate weights
    network: nn.Module = built(
        lambda: _dcgan_init(
            nn.Sequential(
                # stride=1 and padding=0 are defaults for ConvTranspose2d
                # bias=False, since BatchNorm's mean substraction zeros out bias
                nn.ConvTranspose2d(
                    in_channels=100, out_channels=256, kernel_size=7, bias=False
                ),
                nn.BatchNorm2d(256),
                nn.ReLU(),
                nn.ConvTranspose2d(
                    in_channels=256,
                    out_channels=128,
                    kernel_size=4,
                    stride=2,
                    padding=1,
                    bias=False,
                ),
                nn.BatchNorm2d(128),
                nn.ReLU(),
                nn.ConvTranspose2d(
                    in_channels=128, out_channels=1, kernel_size=4, stride=2, padding=1
                ),
                nn.Tanh(),
            )
        )
    )

    def forward(self, z: Tensor):
        return as_tensor(self.network(z))


@nn_dataclass
class Discriminator(nn.Module):
    def __call__(self, x: Tensor):
        return as_tensor(super().__call__(x))

    network: nn.Module = built(
        lambda: _dcgan_init(
            nn.Sequential(
                nn.Conv2d(
                    in_channels=1,
                    out_channels=64,
                    kernel_size=4,
                    stride=2,
                    padding=1,
                    bias=True,  # No BatchNorm here, so keep bias since it has meaning
                ),
                nn.LeakyReLU(0.2),
                nn.Conv2d(
                    in_channels=64,
                    out_channels=128,
                    kernel_size=4,
                    stride=2,
                    padding=1,
                    bias=False,
                ),
                nn.BatchNorm2d(128),
                nn.LeakyReLU(0.2),
                nn.Conv2d(in_channels=128, out_channels=1, kernel_size=7),
            )
        )
    )

    def forward(self, x: Tensor):
        return as_tensor(self.network(x))


bce = nn.functional.binary_cross_entropy_with_logits


def loss_D(logits_real: Tensor, logits_fake: Tensor) -> Tensor:
    """Discriminator loss: real images labelled 1, generated images labelled 0."""
    return bce(logits_real, torch.ones_like(logits_real)) + bce(
        logits_fake, torch.zeros_like(logits_fake)
    )


def loss_G(logits_gen: Tensor) -> Tensor:
    """Non-saturating generator loss: generated images labelled 1 (real)."""
    return bce(logits_gen, torch.ones_like(logits_gen))


class Args(argparse.Namespace):
    """Typed command-line arguments; the defaults are Setup's values."""

    lr_g: float = 2e-4
    lr_d: float = 2e-4
    epochs: int = 25
    seed: int = 0


def parse_args() -> Args:
    parser = argparse.ArgumentParser(
        description="Train the DCGAN (Setup_GAN.md Task 1)."
    )
    _ = parser.add_argument("--lr-g", type=float, default=Args.lr_g)
    _ = parser.add_argument("--lr-d", type=float, default=Args.lr_d)
    _ = parser.add_argument("--epochs", type=int, default=Args.epochs)
    _ = parser.add_argument("--seed", type=int, default=Args.seed)
    return parser.parse_args(namespace=Args())


if __name__ == "__main__":
    args = parse_args()
    DEVICE = get_preferred_device()
    seed_everything(args.seed)

    loaders = gan_loaders()  # train for training; val and test later, for FID

    with torch.device(DEVICE):
        G, D = Generator(), Discriminator()
    opt_G = torch.optim.Adam(G.parameters(), lr=args.lr_g, betas=(0.5, 0.999))
    opt_D = torch.optim.Adam(D.parameters(), lr=args.lr_d, betas=(0.5, 0.999))

    step = 0  # global: counts batches across epochs, for 50-step logging
    for epoch in range(1, args.epochs + 1):
        G.train()
        D.train()

        for x, _ in batches(loaders.train):  # labels unused
            x = x.to(DEVICE)
            z = torch.randn(x.shape[0], 100, 1, 1, device=DEVICE)
            x_tilde = G(z)

            # detach() s.t. the loss does not reach G.
            opt_D.zero_grad()
            logits_real, logits_fake = D(x), D(x_tilde.detach())
            lD = loss_D(logits_real, logits_fake)
            lD.backward()  # pyright: ignore[reportUnknownMemberType]
            opt_D.step()  # pyright: ignore[reportUnknownMemberType]

            # G step: the same fakes, judged again by the updated D.
            opt_G.zero_grad()
            logits_gen = D(x_tilde)
            lG = loss_G(logits_gen)
            lG.backward()  # pyright: ignore[reportUnknownMemberType]
            opt_G.step()  # pyright: ignore[reportUnknownMemberType]

            step += 1
            # Logging every 50 steps (and epoch-1 samples every ~100) goes here.
