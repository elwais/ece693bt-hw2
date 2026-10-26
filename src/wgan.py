import argparse
from collections.abc import Iterator
from pathlib import Path
from typing import TypeVar, override


from torch import Tensor, nn
import torch
from torchvision.utils import save_image  # pyright: ignore[reportUnknownVariableType]  # torchvision has no stubs

from data import gan_loaders
from trainer import Batch, Stateful, Stats, Trainer, make_run_dir, optimizer_step
from utils import (
    WGANLog,
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


# Generator architecture unchanged from DCGAN, so we reuse it
from dcgan import Generator


# Moreso "good"/"bad" instead of "real"/"fake", so not quite a Discriminator
@nn_dataclass
class Critic(nn.Module):
    def __call__(self, x: Tensor):
        return as_tensor(super().__call__(x))

    # Lipschitz constant of a network is bounded by the product of Lipschitz constants
    # of each function.
    # Any network with Lipschitz constant K can be brought back to Lipschitz-1
    # if a division by K is inserted somewhere.

    # BatchNorms have a learnable γ that can get large; must remove them
    # Also can restore the bias since it doesn't get subtracted out

    # LeakyReLU is Lipschitz-1 and convolution does not increase Lipschitz constant
    network: nn.Module = built(
        lambda: _dcgan_init(
            nn.Sequential(
                nn.Conv2d(
                    in_channels=1,
                    out_channels=64,
                    kernel_size=4,
                    stride=2,
                    padding=1,
                    bias=True,
                ),
                nn.LeakyReLU(0.2),
                nn.Conv2d(
                    in_channels=64,
                    out_channels=128,
                    kernel_size=4,
                    stride=2,
                    padding=1,
                    bias=True,
                ),
                # Drop BatchNorm; note bias has been restored
                nn.LeakyReLU(0.2),
                nn.Conv2d(in_channels=128, out_channels=1, kernel_size=7),
            )
        )
    )

    def forward(self, x: Tensor):
        return as_tensor(self.network(x))


def loss_C(logits_real: Tensor, logits_fake: Tensor) -> Tensor:
    """WGAN critic loss"""

    # critic_reward aims to be maximized by the critic
    critic_reward = logits_real.mean() - logits_fake.mean()

    # Negating it gives us the critic loss
    return -critic_reward


def loss_G(logits_gen: Tensor) -> Tensor:
    """WGAN generator loss"""
    # Generator only has control over -E[D(G(z))] and must minimize it, so
    # that is the loss as-is
    return -logits_gen.mean()


class Args(argparse.Namespace):
    """Typed command-line arguments; the defaults are the WGAN paper's learning rates."""

    # Learning rates from wgan paper
    lr_g: float = 5e-5
    lr_d: float = 5e-5
    epochs: int = 25
    seed: int = 0
    name: str | None = None
    ckpt_every: int = 5  # epochs between checkpoints/epoch_XXX.pt


def parse_args() -> Args:
    parser = argparse.ArgumentParser(description="Train the WGAN.")
    _ = parser.add_argument("--lr-g", type=float, default=Args.lr_g)
    _ = parser.add_argument("--lr-d", type=float, default=Args.lr_d)
    _ = parser.add_argument("--epochs", type=int, default=Args.epochs)
    _ = parser.add_argument("--seed", type=int, default=Args.seed)
    _ = parser.add_argument("--name", type=str, default=Args.name)
    _ = parser.add_argument("--ckpt-every", type=int, default=Args.ckpt_every)
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
        "ckpt_every": args.ckpt_every,
        "critic_steps": CRITIC_STEPS,
        "clip": CLIP,
    }


def save_samples(G: Generator, z: Tensor, path: Path) -> None:
    """Save an 8x8 grid of G(z). Eval mode, so BatchNorm uses running stats and
    sampling does not change them."""
    with eval_mode(G):
        images = G(z)
    save_image(images, path, nrow=8, normalize=True, value_range=(-1, 1))


Z_DIM = 100
CRITIC_STEPS = 5  # critic steps per generator step
CLIP = 0.01  # critic weights are clamped to [-CLIP, CLIP]; the paper chose it empirically
LOG_EVERY = 50  # steps between rows of losses.pth; a multiple of CRITIC_STEPS


class WGANTrainer(Trainer):
    """Part 2: one critic step per batch; every CRITIC_STEPS-th step also updates G.

    `step` counts critic steps, so one epoch (one pass over the data) holds
    len(train_loader) // CRITIC_STEPS generator updates.
    """

    net_G: Generator
    net_D: Critic
    opt_G: torch.optim.Optimizer
    opt_D: torch.optim.Optimizer
    z_fixed: Tensor  # the 64 latent vectors of every sample grid
    config: dict[str, float | int | str | None]
    run_dir: Path
    ckpt_every: int
    log: WGANLog

    def __init__(self, args: Args) -> None:
        # Only generator steps carry loss_G, and log_step reads it.
        assert LOG_EVERY % CRITIC_STEPS == 0, "every logged step must be a generator step"
        device = get_preferred_device()
        seed_everything(args.seed)
        super().__init__(epochs=args.epochs, device=device, log_every=LOG_EVERY)

        self.config = config(args)
        self.ckpt_every = args.ckpt_every
        self.run_dir = make_run_dir("wgan", args.name, self.config)

        # train for training; val and test are used later, for FID
        self.train_loader = gan_loaders().train
        self.z_fixed = self.noise(64)  # drawn right after seeding

        with torch.device(self.device):
            self.net_G = Generator()
            self.net_D = Critic()
        # Paper found Adam unstable and used RMSprop instead
        self.opt_G = torch.optim.RMSprop(self.net_G.parameters(), lr=args.lr_g)
        self.opt_D = torch.optim.RMSprop(self.net_D.parameters(), lr=args.lr_d)

        self.log = WGANLog()

    def noise(self, n: int) -> Tensor:
        return torch.randn(n, Z_DIM, 1, 1, device=self.device)

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

        # Critic step on this batch and fresh fakes (no graph back to G)
        with torch.no_grad():
            fake = self.net_G(self.noise(len(x)))
        loss_c = loss_C(self.net_D(x), self.net_D(fake))
        optimizer_step(self.opt_D, loss_c)
        with torch.no_grad():
            for param in self.net_D.parameters():
                _ = param.clamp_(-CLIP, CLIP)  # keeps the critic roughly Lipschitz
        stats = {"loss_D": loss_c.detach(), "w_dist": (-loss_c).detach()}

        # Every CRITIC_STEPS-th step the generator moves, on fresh z with gradients on
        if self.step % CRITIC_STEPS == 0:
            loss_g = loss_G(self.net_D(self.net_G(self.noise(len(x)))))
            optimizer_step(self.opt_G, loss_g)
            stats["loss_G"] = loss_g.detach()
        return stats

    @override
    def log_step(self, stats: dict[str, float]) -> None:
        self.log.push(
            step=self.step,
            epoch=self.curr_epoch,
            loss_G=stats["loss_G"],
            loss_D=stats["loss_D"],
            w_dist=stats["w_dist"],
        )
        print(
            f"epoch {self.curr_epoch:3d}  step {self.step:6d}  "
            f"loss_D {stats['loss_D']:.4f}  loss_G {stats['loss_G']:.4f}  "
            f"w_dist {stats['w_dist']:.4f}"
        )

    @override
    def end_epoch(self) -> None:
        save_samples(
            self.net_G,
            self.z_fixed,
            self.run_dir / "samples" / f"epoch_{self.curr_epoch:03d}.png",
        )
        self.log.save(self.run_dir / "logs" / "losses.pth")
        self.save_checkpoints(self.run_dir / "checkpoints", every=self.ckpt_every)


if __name__ == "__main__":
    trainer = WGANTrainer(parse_args())
    trainer.fit()
    print(f"done: {trainer.run_dir}")
