import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import TypeVar

from torch import Tensor, nn
import torch
from torchvision.utils import save_image  # pyright: ignore[reportUnknownVariableType]  # torchvision has no stubs

from data import batches, gan_loaders
from utils import (
    GANLog,
    as_tensor,
    built,
    eval_mode,
    get_preferred_device,
    nn_dataclass,
    save_atomic,
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


def new_run_dir(args: Args) -> Path:
    """outputs/dcgan/<YYYYmmdd-HHMMSS>[_<name>]/ with its subfolders."""
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run = Path("outputs") / "dcgan" / (f"{stamp}_{args.name}" if args.name else stamp)
    for sub in ("checkpoints", "samples", "logs"):
        (run / sub).mkdir(parents=True, exist_ok=False)
    return run


def save_samples(G: Generator, z: Tensor, path: Path) -> None:
    """Save an 8x8 grid of G(z). Eval mode, so BatchNorm uses running stats and
    sampling does not change them."""
    with eval_mode(G):
        images = G(z)
    save_image(images, path, nrow=8, normalize=True, value_range=(-1, 1))


if __name__ == "__main__":
    args = parse_args()
    DEVICE = get_preferred_device()
    seed_everything(args.seed)

    run = new_run_dir(args)
    run_config = config(args)
    _ = (run / "args.json").write_text(json.dumps(run_config, indent=2))

    loaders = gan_loaders()  # train for training; val and test later, for FID
    fixed_z = torch.randn(64, 100, 1, 1, device=DEVICE)  # drawn right after seeding

    with torch.device(DEVICE):
        G, D = Generator(), Discriminator()
    opt_G = torch.optim.Adam(G.parameters(), lr=args.lr_g, betas=(0.5, 0.999))
    opt_D = torch.optim.Adam(D.parameters(), lr=args.lr_d, betas=(0.5, 0.999))

    log = GANLog()
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

            # Logged values are pre-update: D_x and D_G_z come from the D step above.
            # .item() only runs on log steps, so it does not sync the device every batch.
            if step % 50 == 0:
                d_x = torch.sigmoid(logits_real).mean().item()
                d_gz = torch.sigmoid(logits_fake).mean().item()
                loss_d, loss_g = lD.item(), lG.item()
                log.push(
                    step=step,
                    epoch=epoch,
                    loss_G=loss_g,
                    loss_D=loss_d,
                    D_x=d_x,
                    D_G_z=d_gz,
                )
                print(
                    f"epoch {epoch:3d}  step {step:6d}  loss_D {loss_d:.4f}  "
                    f"loss_G {loss_g:.4f}  D_x {d_x:.3f}  D_G_z {d_gz:.3f}"
                )
            if epoch == 1 and step % 100 == 0:
                save_samples(G, fixed_z, run / "samples" / f"epoch_001_step_{step:04d}.png")
            step += 1

        save_samples(G, fixed_z, run / "samples" / f"epoch_{epoch:03d}.png")
        log.save(run / "logs" / "losses.pth")

        ckpt: dict[str, object] = {
            "G": G.state_dict(),
            "D": D.state_dict(),
            "opt_G": opt_G.state_dict(),
            "opt_D": opt_D.state_dict(),
            "epoch": epoch,
            "config": run_config,
        }
        save_atomic(ckpt, run / "checkpoints" / "last.pt")
        if epoch % 5 == 0:
            save_atomic(ckpt, run / "checkpoints" / f"epoch_{epoch:03d}.pt")
        if epoch == args.epochs:
            save_atomic(ckpt, run / "checkpoints" / "final.pt")
            print(f"done: {run}")
