"""FID on real MNIST features. Skipped unless the classifier and MNIST are present."""

from pathlib import Path

import pytest
import torch

from classifier import Classifier
from data import get_test_loader, get_train_val_loaders
from fid import fid_from_stats, load_classifier, mean_cov, real_features

HAVE_DATA = Path("data/MNIST/raw").exists() and Path("outputs/classifier.pt").exists()
pytestmark = pytest.mark.skipif(
    not HAVE_DATA, reason="needs data/MNIST and outputs/classifier.pt"
)


@pytest.fixture(scope="module")
def clf() -> Classifier:
    return load_classifier("cpu")


def test_real_val_vs_real_test_is_much_smaller_than_vs_noise(clf: Classifier) -> None:
    _, val = get_train_val_loaders("gan")
    test = get_test_loader("gan")
    feats_val = mean_cov(real_features(clf, val, "cpu"))
    feats_test = mean_cov(real_features(clf, test, "cpu"))

    # Uniform noise in [-1, 1] stands in for an untrained generator.
    gen = torch.Generator().manual_seed(0)
    noise = torch.rand(2000, 1, 28, 28, generator=gen) * 2 - 1
    with torch.inference_mode():
        noise_feats = clf.phi(noise)
    feats_noise = mean_cov(noise_feats)

    real_vs_real = fid_from_stats(*feats_val, *feats_test)
    real_vs_noise = fid_from_stats(*feats_val, *feats_noise)
    assert real_vs_real < real_vs_noise / 10
