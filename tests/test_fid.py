"""FID math on synthetic Gaussians, checked against closed forms (no models, no data)."""

import math

import pytest
import torch

from fid import fid_from_stats, mean_cov, sqrt_psd


def _spd(dim: int, gen: torch.Generator) -> torch.Tensor:
    a = torch.randn(dim, dim, generator=gen, dtype=torch.float64)
    return a @ a.T + 1e-3 * torch.eye(dim, dtype=torch.float64)


def test_sqrt_psd_squares_back() -> None:
    gen = torch.Generator().manual_seed(0)
    m = _spd(8, gen)
    s = sqrt_psd(m)
    assert torch.allclose(s @ s, m, atol=1e-8)


def test_identical_stats_give_zero() -> None:
    feats = torch.randn(
        500, 128, generator=torch.Generator().manual_seed(1), dtype=torch.float64
    )
    mu, cov = mean_cov(feats)
    assert abs(fid_from_stats(mu, cov, mu, cov)) < 1e-6


def test_diagonal_covariances_match_closed_form() -> None:
    # Zero means and diagonal covariances a, b: the trace term is sum sqrt(a_i b_i),
    # so FID = sum (sqrt a_i - sqrt b_i)^2 exactly.
    gen = torch.Generator().manual_seed(2)
    a = torch.rand(10, generator=gen, dtype=torch.float64) + 0.1
    b = torch.rand(10, generator=gen, dtype=torch.float64) + 0.1
    zero = torch.zeros(10, dtype=torch.float64)
    got = fid_from_stats(zero, torch.diag(a), zero, torch.diag(b))
    expected = float(((a.sqrt() - b.sqrt()) ** 2).sum())
    assert got == pytest.approx(expected, rel=1e-9)


def test_mean_shift_is_detected_and_same_distribution_is_small() -> None:
    gen = torch.Generator().manual_seed(3)
    real = torch.randn(5000, 128, generator=gen, dtype=torch.float64)
    same = torch.randn(5000, 128, generator=gen, dtype=torch.float64)
    shifted = same + 3.0
    mu_r, cov_r = mean_cov(real)
    same_fid = fid_from_stats(mu_r, cov_r, *mean_cov(same))
    shift_fid = fid_from_stats(mu_r, cov_r, *mean_cov(shifted))
    # A shift of 3 in each of 128 dimensions adds |mu diff|^2 = 128 * 9 = 1152.
    assert shift_fid == pytest.approx(1152, rel=0.02)
    assert abs(same_fid) < 5


def test_rank_deficient_generated_covariance_stays_finite() -> None:
    # n = 100 generated vectors in 128 dimensions: rank <= 99, so S_g is singular.
    gen = torch.Generator().manual_seed(4)
    real = torch.randn(2000, 128, generator=gen, dtype=torch.float64)
    fake = torch.randn(100, 128, generator=gen, dtype=torch.float64)
    value = fid_from_stats(*mean_cov(real), *mean_cov(fake))
    assert not math.isnan(value) and value > -1e-6  # not NaN, not meaningfully negative


def test_injected_noise_raises_fid_monotonically() -> None:
    gen = torch.Generator().manual_seed(5)
    real = torch.randn(3000, 128, generator=gen, dtype=torch.float64)
    clean = torch.randn(3000, 128, generator=gen, dtype=torch.float64)
    noise = torch.randn(3000, 128, generator=gen, dtype=torch.float64) * 4.0
    mu_r, cov_r = mean_cov(real)

    def mixed(frac: float) -> float:
        k = int(frac * 3000)
        return fid_from_stats(mu_r, cov_r, *mean_cov(torch.cat([noise[:k], clean[k:]])))

    f0, f10, f50 = mixed(0.0), mixed(0.1), mixed(0.5)
    assert f0 < f10 < f50
