import argparse
from itertools import islice, pairwise

import pytest
import torch

from vae import (
    BETA_END,
    BETA_START,
    HOLD_EPOCHS,
    RAMP_END_EPOCH,
    SCHEDULES,
    Z_DIM,
    Decoder,
    Encoder,
    betas,
    parse_beta,
    ramp,
)

STEPS = 7  # a small fake epoch


@pytest.mark.parametrize("schedule", SCHEDULES)
def test_ramp_hits_both_endpoints(schedule: str) -> None:
    assert ramp(schedule, 0.0) == pytest.approx(BETA_START)
    assert ramp(schedule, 1.0) == pytest.approx(BETA_END)


def test_ramp_shapes_are_ordered_in_the_middle() -> None:
    # exponential is convex (below linear), logarithmic concave (above it)
    t = 0.5
    assert (
        ramp("exponential-annealing", t)
        < ramp("linear-annealing", t)
        < ramp("logarithmic-annealing", t)
    )


@pytest.mark.parametrize("schedule", SCHEDULES)
def test_schedule_holds_ramps_monotonically_then_stays_at_end(schedule: str) -> None:
    n = (RAMP_END_EPOCH + 2) * STEPS
    values = list(islice(betas(schedule, STEPS), n))
    hold = HOLD_EPOCHS * STEPS
    end = RAMP_END_EPOCH * STEPS
    assert values[:hold] == [BETA_START] * hold
    assert all(a <= b for a, b in pairwise(values))
    assert values[end - 1] == pytest.approx(BETA_END)
    assert values[end:] == [BETA_END] * (n - end)
    assert values[hold] > BETA_START  # epoch 2 starts moving


def test_constant_beta_is_constant() -> None:
    assert list(islice(betas(0.1, STEPS), 100)) == [0.1] * 100


def test_parse_beta() -> None:
    assert parse_beta("10") == 10.0
    assert parse_beta("linear-annealing") == "linear-annealing"
    with pytest.raises(argparse.ArgumentTypeError):
        _ = parse_beta("cosine-annealing")
    with pytest.raises(argparse.ArgumentTypeError):
        _ = parse_beta("-1")


def test_encoder_decoder_shapes() -> None:
    x = torch.rand(4, 1, 28, 28)
    h = Encoder()(x)
    assert h.shape == (4, 2 * Z_DIM, 1, 1)
    mu, _ = h.chunk(2, dim=1)
    assert Decoder()(mu).shape == (4, 1, 28, 28)
