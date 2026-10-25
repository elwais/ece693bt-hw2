# Setup: GAN and VAE Homework on MNIST

> **Read this first.** `Assignment2_GANs.pdf` governs the deliverables. The main body of this file is the enforced spec: the model architecture, training settings, and checkpoint/logging requirements that the PDF points to, plus dataset preparation and environment. Anything not written here is not required. Project layout is free. The appendix is an optional default layout and workflow, so there is a sensible starting point.

## Overview

Assignment 2 compares a DCGAN, a WGAN and a VAE on MNIST. A small classifier's penultimate-layer features φ(x) are used to compute FID. This file has no training spec for the WGAN and VAE beyond data scaling; see the PDF.

## 1. Environment

PyTorch, ideally with GPU access. PDF Part 0 lists the options (the course GPU server, your own NVIDIA machine, or Google Colab). Installation commands are in the appendix.

## 2. MNIST data

MNIST has 60,000 training and 10,000 test images of handwritten digits, 28×28 grayscale. Download it once into `data/` before training.

| | Setting |
|---|---|
| Splits | 55,000 train / 5,000 validation (fixed seed) from the training set; the 10,000 test images are used only for final results |
| Image size | 28×28, 1 channel |
| Classifier and DCGAN scaling | `ToTensor()` + `Normalize((0.5,), (0.5,))` → pixels in [-1, 1], matching the generator's Tanh output and the classifier input |
| VAE scaling | `ToTensor()` → pixels in [0, 1], for the Bernoulli (BCE) reconstruction loss |
| Batch size | 128 |
| Shuffling | training split shuffled; validation and test not |

## 3. Task 0: Classifier (for FID features)

Train a CNN classifier on MNIST. Its penultimate-layer activations φ(x) are the feature space for FID, which compares the samples of the generative models.

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

- φ(x) is the post-ReLU output of layer 6. Expose it, e.g. via a `model.embed(x)` method, and take it in `eval()` mode so dropout is off. The name `embed` is the appendix's convention, not a requirement.

### Training settings

| Setting | Value |
|---|---|
| Loss | cross-entropy |
| Optimizer | Adam, learning rate 1e-3 (default betas) |
| Batch size | 128 |
| Epochs | 30 |
| Data | 55,000 train / 5,000 validation / 10,000 test (same split as the generative models) |
| Logging | training, validation and test loss and accuracy at the end of every epoch |

## 4. Task 1: DCGAN

Build a DCGAN (Radford et al., 2015) that generates 28×28 MNIST digits. Both networks are scaled down to 28×28: three convolutional layers each, no pooling, and no fully connected hidden layers.

### Generator

Input `z` → image.

| Layer | Operation | Output shape |
|---|---|---|
| Input | `z ~ N(0, I)`, dim 100, reshaped to 100×1×1 | 100×1×1 |
| 1 | ConvTranspose2d(100→256, kernel 7, stride 1, padding 0) + BatchNorm + ReLU | 256×7×7 |
| 2 | ConvTranspose2d(256→128, kernel 4, stride 2, padding 1) + BatchNorm + ReLU | 128×14×14 |
| 3 | ConvTranspose2d(128→1, kernel 4, stride 2, padding 1) + Tanh | 1×28×28 |

### Discriminator

Image → real/fake score.

| Layer | Operation | Output shape |
|---|---|---|
| 1 | Conv2d(1→64, kernel 4, stride 2, padding 1) + LeakyReLU(0.2), no BatchNorm | 64×14×14 |
| 2 | Conv2d(64→128, kernel 4, stride 2, padding 1) + BatchNorm + LeakyReLU(0.2) | 128×7×7 |
| 3 | Conv2d(128→1, kernel 7, stride 1, padding 0) | 1×1×1 (one logit per image) |

- The discriminator outputs a raw logit. The Sigmoid is applied inside the loss (`nn.BCEWithLogitsLoss`), which is more numerically stable than a Sigmoid layer followed by `nn.BCELoss`.
- Weight initialization: conv weights ~ N(0, 0.02), BatchNorm weights ~ N(1, 0.02), all biases 0.
- Real images are scaled to [-1, 1] to match the generator's Tanh output (§2).

### Training settings

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

**Store the losses of both the generator and the discriminator in a `.pth` file**, so they can be viewed later in a Jupyter notebook. Save a dictionary of plain Python lists (use `.item()` on each loss tensor) with `torch.save`, and overwrite it at the end of every epoch so it is never lost:

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

## 5. Parts 2 and 3: WGAN and VAE

This file has no training spec for these parts beyond the data scaling in §2 (VAE: [0, 1]). See `Assignment2_GANs.pdf`, Parts 2 and 3.

## 6. Deliverables

Not restated here. `Assignment2_GANs.pdf` defines them: Part 0 (2 items), Part 1 (5), Part 2 (3), Part 3 (5).

## References

- Radford, Metz, Chintala. *Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks*. arXiv:1511.06434

---

# Appendix: Project Structure Template and Setup

> **Optional default.** Nothing in this appendix is required. The PDF and the main body above are the spec. This flat layout is one reasonable default that stays close to the original plan. Replace any part of it with your own design.

## A.1 Project tree

One module per assignment part, at the repository root:

```
hw2/                               # repository root
├── README.md                      # how to run each part; write-up notes
├── pyproject.toml                 # project metadata, dependencies, pytest and basedpyright settings
├── requirements-lock.txt          # pip freeze of the working environment
├── .gitignore                     # venv, data, outputs, checkpoints, caches
├── Setup_GAN.md                   # this file
├── Assignment2_GANs.pdf           # the assignment (authoritative for deliverables)
├── src/
│   ├── data.py                    # MNIST download (python src/data.py); [-1,1] and [0,1] scalings; fixed 55k/5k split
│   ├── utils/                     # seeding, device choice, run-folder and args.json helpers, checkpoint save/load
│   ├── classifier.py              # Part 0: MNIST CNN with embed() = φ(x); training loop; argparse entry point
│   ├── dcgan.py                   # Part 1: Generator, Discriminator, weight init; GAN training loop; entry point
│   ├── wgan.py                    # Part 2: WGAN critic and training loop; imports Generator from dcgan.py
│   ├── vae.py                     # Part 3: conv Encoder, Decoder, reparameterization, ELBO loss; entry point
│   ├── fid.py                     # FID from φ features; imported by every part
│   ├── figures.py                 # sample grids, loss curves, latent-probe and comparison figures
│   └── audit_data.py              # prints batch shapes, value ranges, split sizes, and index hashes
├── queue.sh                       # runs a list of commands in sequence, logging each
├── tests/
│   └── test_data.py               # split sizes, scaling ranges, deterministic split (pytest)
├── results/
│   └── runs.csv                   # one row per finished run (columns in the header)
├── data/                          # MNIST download location (git-ignored)
└── outputs/                       # run artifacts (git-ignored)
    └── <model>/<YYYYmmdd-HHMMSS>/
        ├── args.json              # parsed arguments of this run (written once, never changed)
        ├── checkpoints/
        ├── samples/
        └── logs/
```

## A.2 Design principles

- **One module per part.** Each part has one file that holds its model, training loop and entry point. Empty placeholders are fine until the part is written.
- **Share what can be shared.** `wgan.py` and `vae.py` reuse `Generator` from `dcgan.py`. `fid.py` is used by every part, so all FID numbers use the same φ and scaling.
- **Hyperparameters as argparse flags.** Each flag's default is the Setup value (for example `--lr 2e-4`, `--batch-size 128`, `--epochs 25`, `--seed 0`). No hyperparameter is hard-coded in a training loop.
- **`args.json` per run.** At the start of each run, the parsed arguments are written to `outputs/<model>/<ts>/args.json`. This replaces a config file and is never edited afterwards. Checkpoints store the same dictionary under `config`, the Setup's name for that field.
- **Fixed seeds.** Split, weight initialization and data-loader shuffling each take a seed (default 0).
- **Timestamped run folders.** Each run creates `outputs/<model>/<YYYYmmdd-HHMMSS>/` at start, so runs never overwrite each other.

## A.3 Environment setup

```bash
# Create and activate a virtual environment in the project folder
python3 -m venv .venv                # or: uv venv --python 3.12 .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install --upgrade pip

# PyTorch + torchvision: pick the command for your platform at https://pytorch.org/get-started/locally/
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128   # NVIDIA GPU (CUDA 12.8)
# pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu   # CPU only
# Apple Silicon: the default macOS wheels include MPS: pip install torch torchvision

# Remaining dependencies. The project is not installed; run scripts from the repo root.
pip install numpy matplotlib pyyaml tqdm pytest
```

Check the installation:

```bash
python -c "import torch; print(torch.__version__, 'CUDA:', torch.cuda.is_available())"
# Apple Silicon:
python -c "import torch; print(torch.__version__, 'MPS:', torch.backends.mps.is_available())"
```

Once everything works, record the exact versions:

```bash
pip freeze > requirements-lock.txt
```

Recreate the environment with `pip install -r requirements-lock.txt`. The lock file is platform-specific (it pins the wheels you installed).

## A.4 Data

```bash
python src/data.py      # downloads MNIST to data/MNIST/ (once); prints 60000 / 10000
python src/audit_data.py  # optional: batch shapes, value ranges, split sizes, index hashes
```

Downloading ahead of time means training never needs internet access, which helps on clusters where compute nodes are offline.

DataLoader defaults in `data.py`: `num_workers=4`; `pin_memory=True` only when CUDA is available.

**Optional: dynamic binarization.** For a VAE you may binarize each batch by sampling `x ~ Bernoulli(x)` every time it is loaded. It is not required by the PDF. If you use it, say so in the write-up.

## A.5 Example commands

```bash
python src/classifier.py          # Part 0 with the Setup defaults
python src/dcgan.py               # Part 1 baseline (lr 2e-4 for both networks)
python src/dcgan.py --lr-d 4e-3   # Part 1 item 4: discriminator learning rate 20x
python src/audit_data.py          # data pipeline check
pytest -q                         # sanity checks
```

## A.6 Sanity checks

```bash
pytest -q
```

`tests/test_data.py` checks the split sizes (55,000 / 5,000), the two pixel ranges, and that the split is deterministic for a given seed. The test paths and import path are set in `pyproject.toml`.
