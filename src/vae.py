import argparse
from collections.abc import Iterator
from pathlib import Path
from typing import TypeVar, override
from itertools import chain


from torch import Tensor, nn
import torch
from torchvision.utils import save_image  # pyright: ignore[reportUnknownVariableType]  # torchvision has no stubs

from data import vae_loaders
from trainer import Batch, Stateful, Stats, Trainer, make_run_dir, optimizer_step
from utils import (
    VAELog,
    as_tensor,
    built,
    eval_mode,
    get_preferred_device,
    nn_dataclass,
    seed_everything,
)


def _init_layer(m: nn.Module) -> None:
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        _ = nn.init.normal_(m.weight, 0.0, 0.02)
        if m.bias is not None:
            _ = nn.init.zeros_(m.bias)
    elif isinstance(m, nn.BatchNorm2d):
        _ = nn.init.normal_(m.weight, 1.0, 0.02)
        _ = nn.init.zeros_(m.bias)


M = TypeVar("M", bound=nn.Module)


def _dcgan_init(net: M) -> M:
    """Apply Setup's initialization to every layer of `net`; return `net`."""
    return net.apply(_init_layer)


# Generator architecture unchanged from DCGAN & WGAN, so we reuse it
from dcgan import Generator as Decoder

Z_DIM = 100
LOG_EVERY = 50  # steps between rows of losses.pth
CKPT_EVERY = 5  # epochs between epoch_XXX.pt


# The forward pass returns ONE tensor of 2 * Z_DIM channels, [mu | logvar],
# so a single set of weights (and one optimizer) covers both; the caller splits it.
@nn_dataclass
class Encoder(nn.Module):
    def __call__(self, x: Tensor) -> Tensor:
        return as_tensor(super().__call__(x))

    # Note the use of _dcgan_init to put in the appropriate weights
    network: nn.Module = built(
        lambda: _dcgan_init(
            nn.Sequential(
                nn.Conv2d(
                    in_channels=1,
                    out_channels=128,
                    kernel_size=4,
                    stride=2,
                    padding=1,
                    bias=False,
                ),
                nn.BatchNorm2d(128),
                nn.ReLU(),
                nn.Conv2d(
                    in_channels=128,
                    out_channels=256,
                    kernel_size=4,
                    stride=2,
                    padding=1,
                    bias=False,
                ),
                nn.BatchNorm2d(256),
                nn.ReLU(),
                nn.Conv2d(
                    in_channels=256,
                    out_channels=2 * Z_DIM,
                    kernel_size=7,
                ),
            )
        )
    )

    # (B, 1, 28, 28) -> (B, 2 * Z_DIM, 1, 1); callers do .chunk(2, dim=1) -> mu, logvar
    def forward(self, x: Tensor) -> Tensor:
        return as_tensor(self.network(x))


class Args(argparse.Namespace):
    """Typed command-line arguments; every default is the value used for the main run."""

    lr: float = 1e-3
    beta: float = 1.0  # weight on the KL term; 1.0 is the plain VAE
    epochs: int = 25
    seed: int = 0
    name: str | None = None


def parse_args() -> Args:
    parser = argparse.ArgumentParser(description="Train the VAE.")
    _ = parser.add_argument("--lr", type=float, default=Args.lr)
    _ = parser.add_argument("--beta", type=float, default=Args.beta)
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
        "lr": args.lr,
        "beta": args.beta,
        "epochs": args.epochs,
        "seed": args.seed,
        "name": args.name,
    }


def save_samples(G: Decoder, z: Tensor, path: Path) -> None:
    """Save an 8x8 grid of decoder(z) for z ~ N(0, I). Eval mode, so BatchNorm uses
    running stats and sampling does not change them."""
    with eval_mode(G):
        images = G(z)
    save_image(images, path, nrow=8, normalize=True, value_range=(-1, 1))


class VAETrainer(Trainer):
    """Part 3: one optimizer step per batch on loss = recon + beta * KL.

    `step` counts optimizer steps, so one epoch (one pass over the data) is
    len(train_loader) steps.
    """

    net_dec: Decoder
    net_enc: Encoder
    opt: torch.optim.Optimizer
    z_fixed: Tensor  # the 64 latent vectors of every sample grid
    config: dict[str, float | int | str | None]
    run_dir: Path
    beta: float
    log: VAELog

    def __init__(self, args: Args) -> None:
        device = get_preferred_device()
        seed_everything(args.seed)
        super().__init__(epochs=args.epochs, device=device, log_every=LOG_EVERY)

        self.config = config(args)
        self.beta = args.beta
        self.run_dir = make_run_dir("vae", args.name, self.config)

        # train for training; val and test are used later, for FID
        self.train_loader = vae_loaders().train
        self.z_fixed = self.noise(64)  # drawn right after seeding

        with torch.device(self.device):
            self.net_dec = Decoder()
            self.net_enc = Encoder()

        # β-VAE uses ADAM for lots of its examples, so we go with that
        self.opt = torch.optim.Adam(
            chain(self.net_dec.parameters(), self.net_enc.parameters()), lr=args.lr
        )

        self.log = VAELog()

    def noise(self, n: int) -> Tensor:
        return torch.randn(n, Z_DIM, 1, 1, device=self.device)

    @override
    def state(self) -> dict[str, Stateful]:
        return {
            "G": self.net_dec,  # key 'G' so fid.py can load the decoder as it does a generator
            "enc": self.net_enc,
            "opt": self.opt,
        }

    @override
    def checkpoint(self) -> dict[str, object]:
        payload = super().checkpoint()
        payload["config"] = self.config
        return payload

    @override
    def train_step(self, stream: Iterator[Batch]) -> Stats:
        x = next(stream)[0].to(self.device)

        # Run the image through the network, then split into 2
        mu, logvar = self.net_enc(x).chunk(2, dim=1)

        # Reparameterization is a wonderful thing
        latent_sample = torch.randn_like(logvar) * torch.exp(logvar / 2) + mu

        out = self.net_dec(latent_sample)

        # The decoder ends in Tanh, so (out + 1) / 2 maps [-1, 1] to the Bernoulli mean in [0, 1]
        recon = torch.nn.functional.binary_cross_entropy(
            (out + 1) / 2, x, reduction="sum"
        ) / x.size(0)
        kl = -0.5 * (1 + logvar - mu.square() - logvar.exp()).sum(dim=1).mean()
        loss = recon + self.beta * kl

        optimizer_step(self.opt, loss)

        return {
            "recon": recon.detach(),
            "kl": kl.detach(),
            "loss": loss.detach(),
        }

    @override
    def log_step(self, stats: dict[str, float]) -> None:
        self.log.push(
            step=self.step,
            epoch=self.curr_epoch,
            beta=self.beta,
            loss=stats["loss"],
            recon=stats["recon"],
            kl=stats["kl"],
        )
        print(
            f"epoch {self.curr_epoch:3d}  step {self.step:6d}  "
            f"loss {stats['loss']:.4f}  recon {stats['recon']:.4f}  "
            f"kl {stats['kl']:.4f}"
        )

    @override
    def end_epoch(self) -> None:
        save_samples(
            self.net_dec,
            self.z_fixed,
            self.run_dir / "samples" / f"epoch_{self.curr_epoch:03d}.png",
        )
        self.log.save(self.run_dir / "logs" / "losses.pth")
        self.save_checkpoints(self.run_dir / "checkpoints", every=CKPT_EVERY)


if __name__ == "__main__":
    trainer = VAETrainer(parse_args())
    trainer.fit()
    print(f"done: {trainer.run_dir}")
