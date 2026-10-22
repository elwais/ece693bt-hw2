"""Download MNIST into data/ once, so training never needs the network.

    python scripts/prepare_data.py
"""

from torchvision import datasets

from genmodels.data import DATA_ROOT


def main() -> None:
    train = datasets.MNIST(root=str(DATA_ROOT), train=True, download=True)
    test = datasets.MNIST(root=str(DATA_ROOT), train=False, download=True)
    print(f"MNIST at {DATA_ROOT}: train={len(train)} test={len(test)}")


if __name__ == "__main__":
    main()
