"""WGAN-GP: the gradient penalty's math and the variant's configuration.

The penalty and config are pure and cheap to test; constructing trainers is
deliberately not tested here because it downloads MNIST (see test_data.py for
the download tests).
"""

import pytest
import torch
from torch import Tensor, nn

from wgan import GP_LAMBDA, WGAN_LR, WGANGPTrainer, config, gradient_penalty, parse_args


class _Linear(nn.Module):
    """D(x) = <w, x>; the gradient w.r.t. x is exactly w."""

    def __init__(self) -> None:
        super().__init__()
        self.w = nn.Parameter(torch.randn(8))

    def forward(self, x: Tensor) -> Tensor:
        return x @ self.w


def test_penalty_vanishes_for_unit_gradient() -> None:
    critic = _Linear()
    with torch.no_grad():
        critic.w.div_(critic.w.square().sum().sqrt())  # unit: ||∇D|| = 1
    x = torch.randn(16, 8)
    fake = torch.randn(16, 8)
    assert gradient_penalty(critic, x, fake).item() < 1e-6


def test_penalty_reports_excess_gradient() -> None:
    critic = _Linear()
    with torch.no_grad():
        critic.w.div_(critic.w.square().sum().sqrt())  # start from unit norm
        critic.w.mul_(2.0)  # ||∇D|| = 2 -> penalty = (2 - 1)^2 = 1
    x = torch.randn(16, 8)
    fake = torch.randn(16, 8)
    assert gradient_penalty(critic, x, fake).item() == pytest.approx(1.0, rel=1e-5)


def test_penalty_is_nonnegative() -> None:
    critic = _Linear()
    with torch.no_grad():
        critic.w.div_(critic.w.square().sum().sqrt())
    x = torch.randn(4, 8)
    fake = torch.randn(4, 8)
    assert gradient_penalty(critic, x, fake).item() >= 0.0


def test_config_records_the_variant() -> None:
    base = config(parse_args([]))
    assert base["clip"] == 0.01 and "gp" not in base
    assert base["lr_g"] == 5e-5 and base["lr_d"] == 5e-5

    gp = config(parse_args(["--gp"]))
    assert gp["gp"] is True and gp["gp_lambda"] == GP_LAMBDA
    assert "clip" not in gp
    # The GP paper's Adam defaults apply only when --lr-* is not given.
    assert gp["lr_g"] == 1e-4 and gp["lr_d"] == 1e-4


def test_explicit_learning_rates_win_over_gp_defaults() -> None:
    gp = config(parse_args(["--gp", "--lr-g", "5e-5", "--lr-d", "5e-5"]))
    assert gp["lr_g"] == WGAN_LR and gp["lr_d"] == WGAN_LR


def test_wgangupto_documented_family() -> None:
    # The run-directory family is what separate the two variants on disk.
    class _NoDownload(WGANGPTrainer):  # never constructed, only inspected
        pass

    assert _NoDownload.family == "wgangp"
