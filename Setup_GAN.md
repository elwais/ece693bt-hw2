# Variational Auto-Encoders (VAE), Normalizing Flows and Generative Adversarial Networks (GAN) Homework

## Overview

This homework encourages you to use the coding agent to build VAE, NF and GANs, comparing their performance. 

## Project Structure

The repository grows in three stages — **DCGAN → VAE → Flow-VAE** — and all three share
the same data pipeline, utilities, and entry point. Only the model and its training loop change.

```
dcgan/
├── README.md
├── pyproject.toml              # package metadata + dependencies (pip install -e .)
│
├── configs/                    # one YAML per experiment; no hard-coded hyperparameters
│   ├── classifier.yaml
│   ├── dcgan.yaml
│   ├── vae.yaml
│   └── flow_vae.yaml
│
├── src/genmodels/              # importable package (the "library")
│   ├── __init__.py
│   ├── data.py                 # MNIST loaders; [-1, 1] scaling for GAN, [0, 1] for VAE
│   ├── utils.py                # seeding, device selection, checkpoint save/load, logging
│   ├── viz.py                  # sample grids, reconstructions, latent interpolation
│   ├── losses.py               # GAN BCE loss, Gaussian/Bernoulli ELBO, flow ELBO (+ log-det)
│   ├── models/
│   │   ├── classifier.py       # MNIST CNN classifier; its 128-d features φ(x) are used for FID
│   │   ├── dcgan.py            # Generator, Discriminator, weight init
│   │   ├── vae.py              # conv Encoder/Decoder, reparameterization
│   │   ├── flows.py            # flow layers (e.g. planar, radial, IAF) returning (z, log_det)
│   │   └── flow_vae.py         # VAE whose posterior q(z|x) is refined by a stack of flows
│   └── trainers/
│       ├── classifier_trainer.py  # supervised training; per-epoch train/val/test loss
│       ├── gan_trainer.py      # alternating D/G updates
│       └── vae_trainer.py      # ELBO training; used for both VAE and Flow-VAE
│
├── scripts/                    # thin command-line entry points
│   ├── prepare_data.py         # download MNIST to data/ once
│   ├── train.py                # python scripts/train.py --config configs/dcgan.yaml
│   ├── sample.py               # generate images from a saved checkpoint
│   └── evaluate.py             # test-set ELBO / NLL, sample quality, comparison figures
│
├── notebooks/                  # walkthroughs for students (import from src/, no model code here)
│   ├── 01_dcgan.ipynb
│   ├── 02_vae.ipynb
│   └── 03_flow_vae.ipynb
│
├── tests/                      # fast sanity checks (pytest)
│   ├── test_shapes.py          # every model maps inputs to the expected output shapes
│   └── test_flows.py           # log-det of each flow matches a brute-force Jacobian
│
├── data/                       # MNIST download location
└── outputs/                    # run artifacts (checkpoints, samples, logs)
    └── <model>/<YYYYmmdd-HHMMSS>/
        ├── config.yaml         # frozen copy of the config used for this run
        ├── checkpoints/
        ├── samples/
        └── logs/
```

### Design principles

- **Library vs. scripts vs. notebooks.** Model and training code lives only in `src/genmodels/`.
  Scripts and notebooks import from it, so a notebook never holds its own copy of a model.
- **Config-driven experiments.** Everything that varies between runs (latent dim, learning rate,
  epochs, number of flow layers, seed) lives in `configs/*.yaml`. Each run saves a frozen copy of
  its config next to its outputs, so results can be reproduced.
- **Share what can be shared.** The Flow-VAE is a VAE whose posterior is transformed by flows, so it
  reuses the VAE encoder/decoder and `vae_trainer.py`; it only adds `flows.py` and the
  `log_det` term in the loss. The three models can then be compared directly.
- **Reproducibility.** Fixed seeds, logged configs, and timestamped output folders.

## Setup

### 1. PyTorch environment

Requires Python 3.10+. A GPU is recommended but not required; MNIST also trains on a CPU, just slower.

```bash
# Create and activate a virtual environment in the project folder
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install --upgrade pip

# Install PyTorch + torchvision. Pick the command for your platform at
# https://pytorch.org/get-started/locally/ . For example:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128   # NVIDIA GPU (CUDA 12.8)
# pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu   # CPU only

# Install this project and the remaining dependencies
# (numpy, matplotlib, pyyaml, tqdm; dev extras: pytest, jupyter)
pip install -e ".[dev]"
```

Check the installation:

```bash
python -c "import torch; print(torch.__version__, 'CUDA:', torch.cuda.is_available())"
```

`CUDA: True` means the GPU is visible. On a CPU-only machine `False` is expected.

Once everything works, record the exact versions so the environment can be recreated:

```bash
pip freeze > requirements-lock.txt
```

### 2. MNIST dataset

MNIST has 60,000 training and 10,000 test images of handwritten digits, 28×28 grayscale
(about 50 MB). It is downloaded with `torchvision.datasets.MNIST` into `data/`:

```bash
python scripts/prepare_data.py     # downloads MNIST to data/MNIST/ (run once)
```

Downloading ahead of time means training never needs internet access (useful on clusters
where compute nodes are offline). Training scripts also download it automatically if it is missing.

How the data is used (implemented in `src/genmodels/data.py`):

| | Setting |
|---|---|
| Splits | 55,000 train / 5,000 validation (fixed seed) from the training set; the 10,000 test images are used only for final results |
| Image size | kept at 28×28, 1 channel |
| DCGAN scaling | `ToTensor()` + `Normalize((0.5,), (0.5,))` → pixels in [-1, 1], matching the generator's Tanh output |
| VAE / Flow-VAE scaling | `ToTensor()` → pixels in [0, 1], for the Bernoulli (BCE) reconstruction loss |
| VAE / Flow-VAE (optional) | dynamic binarization (resample binary pixels each batch), as in the flow-VAE papers |
| DataLoader | `batch_size=128`, `shuffle=True` for training, `num_workers=4`, `pin_memory=True` on GPU |

### 3. Sanity check

```bash
pytest -q
```

## Task 0: Training a Classification Network

Before building generative models, train a CNN classifier on MNIST. Besides being a warm-up, its
penultimate-layer activations φ(x) serve as the feature space for computing FID (Fréchet Inception
Distance, with this network in place of Inception) to compare the samples of the generative models.
Implement the model in `src/genmodels/models/classifier.py`, the training loop in
`src/genmodels/trainers/classifier_trainer.py`, and put the hyperparameters in `configs/classifier.yaml`.

### Model structure

| Layer | Operation | Output shape |
|---|---|---|
| Input | 28×28 grayscale image, scaled to [-1, 1] | 1×28×28 |
| 1 | Conv2d(1→32, kernel 3) + ReLU | 32×26×26 |
| 2 | Conv2d(32→64, kernel 3) + ReLU | 64×24×24 |
| 3 | MaxPool2d(2) | 64×12×12 |
| 4 | Dropout(0.25) | 64×12×12 |
| 5 | Flatten | 9216 |
| 6 | FC(9216→128) + ReLU ← **φ(x) for FID** | 128 |
| 7 | Dropout(0.5) | 128 |
| 8 | FC(128→10) (logits; softmax inside `nn.CrossEntropyLoss`) | 10 |

- The images use the same [-1, 1] scaling as the GAN data, so generated samples can be fed to the
  classifier (and to φ) directly.
- Extract φ(x) with `model.embed(x)` in `eval()` mode, so dropout is off.

### Training settings

| Setting | Value |
|---|---|
| Loss | cross-entropy |
| Optimizer | Adam, learning rate 1e-3 (default betas) |
| Batch size | 128 |
| Epochs | 30 |
| Data | 55,000 train / 5,000 validation / 10,000 test (same split as the generative models) |

## Task 1: Building and Training a DCGAN

Build a DCGAN (Radford et al., 2015) that generates 28×28 MNIST digits. Implement the model in
`src/genmodels/models/dcgan.py`, the training loop in `src/genmodels/trainers/gan_trainer.py`,
and put the hyperparameters in `configs/dcgan.yaml`.

### Model structure

The model structure is a DCGAN scaled down to 28×28: both networks use three convolutional
layers with no pooling and no fully connected hidden layers.

**Generator** (latent vector `z` → image)

| Layer | Operation | Output shape |
|---|---|---|
| Input | `z ~ N(0, I)`, dim 100, reshaped to 100×1×1 | 100×1×1 |
| 1 | ConvTranspose2d(100→256, kernel 7, stride 1, padding 0) + BatchNorm + ReLU | 256×7×7 |
| 2 | ConvTranspose2d(256→128, kernel 4, stride 2, padding 1) + BatchNorm + ReLU | 128×14×14 |
| 3 | ConvTranspose2d(128→1, kernel 4, stride 2, padding 1) + Tanh | 1×28×28 |

**Discriminator** (image → real/fake score)

| Layer | Operation | Output shape |
|---|---|---|
| 1 | Conv2d(1→64, kernel 4, stride 2, padding 1) + LeakyReLU(0.2), no BatchNorm | 64×14×14 |
| 2 | Conv2d(64→128, kernel 4, stride 2, padding 1) + BatchNorm + LeakyReLU(0.2) | 128×7×7 |
| 3 | Conv2d(128→1, kernel 7, stride 1, padding 0) | 1×1×1 (one logit per image) |

- The discriminator outputs a raw logit. The Sigmoid is applied inside the loss
  (`nn.BCEWithLogitsLoss`), which is numerically more stable than a Sigmoid layer followed by `nn.BCELoss`.
- Weight initialization: conv weights ~ N(0, 0.02), BatchNorm weights ~ N(1, 0.02), all biases 0.
- Real images are scaled to [-1, 1] to match the generator's Tanh output (see [MNIST dataset](#2-mnist-dataset)).

### Training settings

The learning rate is set to **2e-4** for both the generator and the discriminator.

| Setting | Value |
|---|---|
| Optimizer | Adam, betas = (0.5, 0.999), for both networks |
| Learning rate | 2e-4 (generator and discriminator) |
| Batch size | 128 |
| Latent dimension | 100 |
| Epochs | 25 |
| Updates | one discriminator step, then one generator step, per batch |
| Generator loss | non-saturating: maximize log D(G(z)), i.e. BCE against "real" labels |

### Checkpoints and logging

All files go into the run folder `outputs/dcgan/<YYYYmmdd-HHMMSS>/`.

| When | What | Where |
|---|---|---|
| Every 5 epochs | checkpoint: generator, discriminator, both optimizer states, epoch, config | `checkpoints/epoch_005.pt`, `epoch_010.pt`, … |
| End of every epoch | same checkpoint, overwritten (for resuming) | `checkpoints/last.pt` |
| End of training | final checkpoint | `checkpoints/final.pt` |
| Every epoch (and every ~100 steps in epoch 1) | 8×8 sample grid from a **fixed** set of 64 noise vectors | `samples/epoch_XXX.png` |
| Every 50 steps | generator and discriminator losses, mean D(x), mean D(G(z)) | `logs/losses.pth` |

**Note: store the losses of both the generator and the discriminator in a `.pth` file**, so they can
be viewed later in a Jupyter notebook. Save a dictionary of plain Python lists (use `.item()` on each
loss tensor) with `torch.save`, and overwrite it at the end of every epoch so it is never lost:

```python
history = {"step": [], "epoch": [], "loss_G": [], "loss_D": [], "D_x": [], "D_G_z": []}
# ... append values every 50 steps ...
torch.save(history, run_dir / "logs" / "losses.pth")
```

Viewing the losses in a notebook:

```python
import torch
import matplotlib.pyplot as plt

history = torch.load("outputs/dcgan/<run>/logs/losses.pth")
plt.plot(history["step"], history["loss_G"], label="Generator")
plt.plot(history["step"], history["loss_D"], label="Discriminator")
plt.xlabel("step"); plt.ylabel("loss"); plt.legend(); plt.show()
```

### Deliverables

_TBD_

## Training

```bash
python scripts/train.py --config configs/classifier.yaml
python scripts/train.py --config configs/dcgan.yaml
python scripts/train.py --config configs/vae.yaml
python scripts/train.py --config configs/flow_vae.yaml
```

_More details TBD._

## Results

_TBD_

## References

- Radford, Metz, Chintala. *Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks*. arXiv:1511.06434
