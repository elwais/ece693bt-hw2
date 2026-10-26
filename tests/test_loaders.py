"""Loader sets: right scaling for each split, without touching MNIST."""

import pytest
from pytest import MonkeyPatch

import data


@pytest.fixture
def record_calls(monkeypatch: MonkeyPatch) -> list[tuple[str, str]]:
    """Replace the MNIST loaders with recorders; nothing is downloaded."""
    calls: list[tuple[str, str]] = []

    def fake_train_val(
        scaling: str, *args: object, **kwargs: object
    ) -> tuple[str, str]:
        calls.append(("train_val", scaling))
        return f"train-{scaling}", f"val-{scaling}"

    def fake_test(scaling: str, *args: object, **kwargs: object) -> str:
        calls.append(("test", scaling))
        return f"test-{scaling}"

    monkeypatch.setattr(data, "get_train_val_loaders", fake_train_val)
    monkeypatch.setattr(data, "get_test_loader", fake_test)
    return calls


def test_gan_loaders_use_gan_scaling(record_calls: list[tuple[str, str]]) -> None:
    loaders = data.gan_loaders()
    assert loaders.train == "train-gan"
    assert loaders.val == "val-gan"
    assert loaders.test == "test-gan"
    assert record_calls == [("train_val", "gan"), ("test", "gan")]


def test_vae_loaders_use_vae_scaling(record_calls: list[tuple[str, str]]) -> None:
    loaders = data.vae_loaders()
    assert (loaders.train, loaders.val, loaders.test) == (
        "train-vae",
        "val-vae",
        "test-vae",
    )
    assert record_calls == [("train_val", "vae"), ("test", "vae")]
