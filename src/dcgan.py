import argparse
from collections.abc import Iterator
from pathlib import Path
from typing import override

import torch
from torch import Tensor, nn
from torchvision.utils import (
    save_image,  # pyright: ignore[reportUnknownVariableType]  # torchvision has no stubs
)

from data import gan_loaders
from trainer import Batch, Stateful, Stats, Trainer, make_run_dir, optimizer_step
from utils import (
    DCGANLog,
    as_tensor,
    built,
    eval_mode,
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


# Using `_init_layer` to apply the weight initialization; the bound `M: nn.Module`
# makes the type checker complain if inputs are not subclassed from nn.Module
def _dcgan_init[M: nn.Module](net: M) -> M:
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
    name: str | None = None


def parse_args() -> Args:
    parser = argparse.ArgumentParser(
        description="Train the DCGAN (Setup_GAN.md Task 1)."
    )
    _ = parser.add_argument("--lr-g", type=float, default=Args.lr_g)
    _ = parser.add_argument("--lr-d", type=float, default=Args.lr_d)
    _ = parser.add_argument("--epochs", type=int, default=Args.epochs)
    _ = parser.add_argument("--seed", type=int, default=Args.seed)
    _ = parser.add_argument("--name", type=str, default=Args.name)
    return parser.parse_args(namespace=Args())


def config(args: Args) -> dict[str, float | int | str | None]:
    """The run's settings as plain values, for args.json and each checkpoint.

    vars(args) would miss the defaults: argparse skips any attribute that the
    namespace class already defines, so defaults stay as class attributes.
    """
    return {
        "lr_g": args.lr_g,
        "lr_d": args.lr_d,
        "epochs": args.epochs,
        "seed": args.seed,
        "name": args.name,
    }


def save_samples(G: Generator, z: Tensor, path: Path) -> None:
    """Save an 8x8 grid of G(z). Eval mode, so BatchNorm uses running stats and
    sampling does not change them."""
    with eval_mode(G):
        images = G(z)
    save_image(images, path, nrow=8, normalize=True, value_range=(-1, 1))


Z_DIM = 100
BETAS = (0.5, 0.999)
LOG_EVERY = 50  # steps between rows of losses.pth
SAMPLE_EVERY = 100  # steps between sample grids, in epoch 1 only
CKPT_EVERY = 5  # epochs between epoch_XXX.pt


class DCGANTrainer(Trainer):
    """Setup Task 1: per batch, one D step, then one G step on the same fake batch."""

    net_G: Generator
    net_D: Discriminator
    opt_G: torch.optim.Optimizer
    opt_D: torch.optim.Optimizer
    z_fixed: Tensor  # the 64 latent vectors of every sample grid
    config: dict[str, float | int | str | None]
    run_dir: Path
    log: DCGANLog

    def __init__(self, args: Args) -> None:
        device = get_preferred_device()
        seed_everything(args.seed)
        super().__init__(epochs=args.epochs, device=device, log_every=LOG_EVERY)

        self.config = config(args)
        self.run_dir = make_run_dir("dcgan", args.name, self.config)

        # train for training; val and test are used later, for FID
        self.train_loader = gan_loaders().train
        self.z_fixed = torch.randn(
            64, Z_DIM, 1, 1, device=self.device
        )  # drawn right after seeding

        with torch.device(self.device):
            self.net_G = Generator()
            self.net_D = Discriminator()
        self.opt_G = torch.optim.Adam(
            self.net_G.parameters(), lr=args.lr_g, betas=BETAS
        )
        self.opt_D = torch.optim.Adam(
            self.net_D.parameters(), lr=args.lr_d, betas=BETAS
        )

        self.log = DCGANLog()

    @override
    def state(self) -> dict[str, Stateful]:
        return {
            "G": self.net_G,
            "D": self.net_D,
            "opt_G": self.opt_G,
            "opt_D": self.opt_D,
        }

    @override
    def checkpoint(self) -> dict[str, object]:
        payload = super().checkpoint()
        payload["config"] = self.config
        return payload

    @override
    def train_step(self, stream: Iterator[Batch]) -> Stats:
        x = next(stream)[0].to(self.device)  # labels unused
        z = torch.randn(x.shape[0], Z_DIM, 1, 1, device=self.device)
        x_tilde = self.net_G(z)

        # detach() s.t. the loss does not reach G.
        logits_real, logits_fake = self.net_D(x), self.net_D(x_tilde.detach())
        lD = loss_D(logits_real, logits_fake)
        optimizer_step(self.opt_D, lD)

        # G step: the same fakes, judged again by the updated D.
        lG = loss_G(self.net_D(x_tilde))
        optimizer_step(self.opt_G, lG)

        if self.curr_epoch == 1 and self.step % SAMPLE_EVERY == 0:
            save_samples(
                self.net_G,
                self.z_fixed,
                self.run_dir / "samples" / f"epoch_001_step_{self.step:04d}.png",
            )

        # D_x and D_G_z come from the D step above, so they are pre-update.
        return {
            "loss_D": lD.detach(),
            "loss_G": lG.detach(),
            "D_x": torch.sigmoid(logits_real.detach()).mean(),
            "D_G_z": torch.sigmoid(logits_fake.detach()).mean(),
        }

    @override
    def log_step(self, stats: dict[str, float]) -> None:
        self.log.push(
            step=self.step,
            epoch=self.curr_epoch,
            loss_G=stats["loss_G"],
            loss_D=stats["loss_D"],
            D_x=stats["D_x"],
            D_G_z=stats["D_G_z"],
        )
        print(
            f"epoch {self.curr_epoch:3d}  step {self.step:6d}  "
            f"loss_D {stats['loss_D']:.4f}  loss_G {stats['loss_G']:.4f}  "
            f"D_x {stats['D_x']:.3f}  D_G_z {stats['D_G_z']:.3f}"
        )

    @override
    def end_epoch(self) -> None:
        save_samples(
            self.net_G,
            self.z_fixed,
            self.run_dir / "samples" / f"epoch_{self.curr_epoch:03d}.png",
        )
        self.log.save(self.run_dir / "logs" / "losses.pth")
        self.save_checkpoints(self.run_dir / "checkpoints", every=CKPT_EVERY)


if __name__ == "__main__":
    trainer = DCGANTrainer(parse_args())
    trainer.fit()
    print(f"done: {trainer.run_dir}")
