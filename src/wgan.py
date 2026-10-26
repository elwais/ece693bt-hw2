import argparse
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import cast, override

import torch
from torch import Tensor, nn
from torchvision.utils import (
    save_image,  # pyright: ignore[reportUnknownVariableType]  # torchvision has no stubs
)

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


def _dcgan_init[M: nn.Module](net: M) -> M:
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


def gradient_penalty(critic: nn.Module, real: Tensor, fake: Tensor) -> Tensor:
    """The WGAN-GP penalty on a batch: mean((||∇_x̂ D(x̂)||₂ − 1)²).

    x̂ = α·real + (1−α)·fake with α ~ U[0, 1] per image. Pushing the gradient
    norm toward 1 is what enforces the Lipschitz constraint, instead of the
    weight clipping of the plain WGAN. The penalty lives in the critic loss.
    """
    # One α per image, broadcast over every other dimension of `real`, whatever
    # its rank (N,C,H,W for images).
    alpha = torch.rand(len(real), *([1] * (real.ndim - 1)), device=real.device)
    interpolated = (alpha * real + (1 - alpha) * fake).requires_grad_(True)
    logits = cast(Tensor, critic(interpolated))  # nn.Module.__call__ is Any
    grads = torch.autograd.grad(
        outputs=logits,
        inputs=interpolated,
        grad_outputs=torch.ones_like(logits),
        # keep the graph: the penalty itself must backpropagate to the critic
        create_graph=True,
    )[0]
    flat: Tensor = grads.reshape(len(real), -1)
    norms: Tensor = flat.square().sum(dim=1).sqrt()
    return torch.mean((norms - 1) ** 2)


# The WGAN paper's RMSprop learning rates; the WGAN-GP variant uses its own
# Adam defaults (GP_LR, GP_BETAS) instead. Defined before Args, whose default
# gp_lambda references GP_LAMBDA.
WGAN_LR = 5e-5
GP_LR = 1e-4
GP_BETAS = (0.5, 0.9)
GP_LAMBDA = 10.0  # weight on the gradient penalty; the paper's default


class Args(argparse.Namespace):
    """Typed command-line arguments.

    The WGAN paper's defaults are the two learning rates; pass --gp for the
    WGAN-GP variant, whose own defaults are Adam 1e-4 (betas 0.5, 0.9). A None
    learning rate means "the default of the chosen variant".
    """

    lr_g: float | None = None
    lr_d: float | None = None
    epochs: int = 25
    seed: int = 0
    name: str | None = None
    ckpt_every: int = 5  # epochs between checkpoints/epoch_XXX.pt
    gp: bool = False  # train WGAN-GP (gradient penalty) instead of WGAN (clip)
    gp_lambda: float = GP_LAMBDA


def parse_args(argv: Sequence[str] | None = None) -> Args:
    parser = argparse.ArgumentParser(
        description="Train the WGAN, or the WGAN-GP with --gp."
    )
    _ = parser.add_argument(
        "--lr-g", type=float, default=Args.lr_g, help="or the variant's default"
    )
    _ = parser.add_argument(
        "--lr-d", type=float, default=Args.lr_d, help="or the variant's default"
    )
    _ = parser.add_argument("--epochs", type=int, default=Args.epochs)
    _ = parser.add_argument("--seed", type=int, default=Args.seed)
    _ = parser.add_argument("--name", type=str, default=Args.name)
    _ = parser.add_argument("--ckpt-every", type=int, default=Args.ckpt_every)
    _ = parser.add_argument(
        "--gp",
        action="store_true",
        help="WGAN-GP: Adam, no clipping, and a gradient penalty in the critic loss",
    )
    _ = parser.add_argument("--gp-lambda", type=float, default=Args.gp_lambda)
    return parser.parse_args(argv, namespace=Args())


def config(args: Args) -> dict[str, float | int | str | None]:
    """The run's settings as plain values, for args.json and each checkpoint.

    vars(args) would miss the defaults: argparse skips any attribute that the
    namespace class already defines, so defaults stay as class attributes.
    """
    lr = GP_LR if args.gp else WGAN_LR  # the chosen variant's paper default
    cfg: dict[str, float | int | str | None] = {
        "lr_g": args.lr_g if args.lr_g is not None else lr,
        "lr_d": args.lr_d if args.lr_d is not None else lr,
        "epochs": args.epochs,
        "seed": args.seed,
        "name": args.name,
        "ckpt_every": args.ckpt_every,
        "critic_steps": CRITIC_STEPS,
    }
    if args.gp:
        # WGAN-GP: no weight clipping; the penalty enforces Lipschitz
        cfg["gp"] = True
        cfg["gp_lambda"] = args.gp_lambda
    else:
        # WGAN: clipping the critic weights is the whole Lipschitz mechanism
        cfg["clip"] = CLIP
    return cfg


def save_samples(G: Generator, z: Tensor, path: Path) -> None:
    """Save an 8x8 grid of G(z). Eval mode, so BatchNorm uses running stats and
    sampling does not change them."""
    with eval_mode(G):
        images = G(z)
    save_image(images, path, nrow=8, normalize=True, value_range=(-1, 1))


Z_DIM = 100
CRITIC_STEPS = 5  # critic steps per generator step
CLIP = (
    0.01  # critic weights are clamped to [-CLIP, CLIP]; the paper chose it empirically
)
LOG_EVERY = 50  # steps between rows of losses.pth; a multiple of CRITIC_STEPS


class WGANTrainer(Trainer):
    """Part 2: one critic step per batch; every CRITIC_STEPS-th step also updates G.

    `step` counts critic steps, so one epoch (one pass over the data) holds
    len(train_loader) // CRITIC_STEPS generator updates.
    """

    # The outputs/<family>/ directory that holds this variant's run dirs.
    family: str = "wgan"

    net_G: Generator
    net_D: Critic
    opt_G: torch.optim.Optimizer
    opt_D: torch.optim.Optimizer
    z_fixed: Tensor  # the 64 latent vectors of every sample grid
    config: dict[str, float | int | str | None]
    run_dir: Path
    ckpt_every: int
    gp_lambda: float
    log: WGANLog

    def __init__(self, args: Args) -> None:
        # Only generator steps carry loss_G, and log_step reads it.
        assert LOG_EVERY % CRITIC_STEPS == 0, (
            "every logged step must be a generator step"
        )
        device = get_preferred_device()
        seed_everything(args.seed)
        super().__init__(epochs=args.epochs, device=device, log_every=LOG_EVERY)

        self.config = config(args)
        self.ckpt_every = args.ckpt_every
        self.gp_lambda = args.gp_lambda
        self.run_dir = make_run_dir(self.family, args.name, self.config)

        # train for training; val and test are used later, for FID
        self.train_loader = gan_loaders().train
        self.z_fixed = self.noise(64)  # drawn right after seeding

        with torch.device(self.device):
            self.net_G = Generator()
            self.net_D = Critic()
        self.opt_G, self.opt_D = self._make_optimizers(args)

        self.log = WGANLog()

    def _make_optimizers(
        self, args: Args
    ) -> tuple[torch.optim.Optimizer, torch.optim.Optimizer]:
        """The two optimizers. The plain WGAN uses the paper's RMSprop 5e-5
        (Adam was unstable with clipping); WGAN-GP overrides this."""
        lr_g = args.lr_g if args.lr_g is not None else WGAN_LR
        lr_d = args.lr_d if args.lr_d is not None else WGAN_LR
        return (
            torch.optim.RMSprop(self.net_G.parameters(), lr=lr_g),
            torch.optim.RMSprop(self.net_D.parameters(), lr=lr_d),
        )

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


class WGANGPTrainer(WGANTrainer):
    """The WGAN-GP variant (Gulrajani et al., 2017): the same critic and critic/
    generator step rhythm as the WGAN, but the Lipschitz constraint comes from
    a gradient penalty on the critic instead of weight clipping.

    Every deviation from WGANTrainer is marked with a "WGAN-GP deviation"
    comment below; everything else (state, checkpoints, logs, samples, and
    the epoch rhythm) is inherited unchanged.
    """

    family: str = "wgangp"  # separate run dirs from the clipped WGAN

    def _make_optimizers(
        self, args: Args
    ) -> tuple[torch.optim.Optimizer, torch.optim.Optimizer]:
        # WGAN-GP deviation #1: Adam 1e-4 with betas (0.5, 0.9), as in the GP
        # paper, instead of the clipping-WGAN's RMSprop 5e-5.
        lr_g = args.lr_g if args.lr_g is not None else GP_LR
        lr_d = args.lr_d if args.lr_d is not None else GP_LR
        return (
            torch.optim.Adam(self.net_G.parameters(), lr=lr_g, betas=GP_BETAS),
            torch.optim.Adam(self.net_D.parameters(), lr=lr_d, betas=GP_BETAS),
        )

    @override
    def train_step(self, stream: Iterator[Batch]) -> Stats:
        x = next(stream)[0].to(self.device)  # labels unused

        # Critic step on this batch and fresh fakes (no graph back to G).
        with torch.no_grad():
            fake = self.net_G(self.noise(len(x)))
        loss_c = loss_C(self.net_D(x), self.net_D(fake))
        # WGAN-GP deviation #2: no clamp_ of the critic weights. Instead the
        # gradient penalty joins the critic loss (interpolated on the fly).
        penalty = gradient_penalty(self.net_D, x, fake)
        loss_c = loss_c + self.gp_lambda * penalty
        optimizer_step(self.opt_D, loss_c)

        # w_dist stays the unpenalized distance estimate, for comparability
        # with the clipped WGAN; loss_D is what the critic actually minimized.
        stats = {
            "loss_D": loss_c.detach(),
            "w_dist": (-loss_c + self.gp_lambda * penalty.detach()).detach(),
        }

        # Every CRITIC_STEPS-th step the generator moves, on fresh z with gradients on
        if self.step % CRITIC_STEPS == 0:
            loss_g = loss_G(self.net_D(self.net_G(self.noise(len(x)))))
            optimizer_step(self.opt_G, loss_g)
            stats["loss_G"] = loss_g.detach()
        return stats


if __name__ == "__main__":
    args = parse_args()
    trainer: Trainer = WGANGPTrainer(args) if args.gp else WGANTrainer(args)
    trainer.fit()
    print(f"done: {trainer.run_dir}")
