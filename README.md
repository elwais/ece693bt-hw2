# ECE-693BT Assignment 2: GANs, WGANs and VAEs on MNIST

Code for Assignment 2 (`Assignment2_GANs.pdf`). The assignment has four parts,
each with one training script:

| Part | Task | Script |
|---|---|---|
| 0 | CNN classifier on MNIST; its 128-d penultimate layer φ(x) is the feature map for FID | `src/classifier.py` |
| 1 | DCGAN: samples, discriminator loss, FID formula, ×20 discriminator lr, lr tuning | `src/dcgan.py`, `src/fid.py` |
| 2 | Wasserstein GAN: weight clipping, and the gradient-penalty variant (WGAN-GP) | `src/wgan.py` |
| 3 | VAE: encoder mirroring the generator, β and annealing schedules, latent traversal | `src/vae.py`, `src/vae_traverse.py` |

`Setup_GAN.md` is the course-provided spec for architectures, training settings,
checkpointing and logging; the scripts' defaults are its values. The written
report is a Typst document one level up (`../hw2.typ`); it reads the run
outputs listed in `outputs/manifest.json` and draws all of its plots itself.

## Results

Best FID per model family. Each model's checkpoint is the epoch with the lowest
validation FID (5k real validation images), and the test FID (10k real test
images) is reported for that same checkpoint. Lower is better.

| Model | Run | Epochs | Val FID (epoch) | Test FID |
|---|---|---|---|---|
| WGAN-GP (Adam 1e-4, λ = 10) | `wgangp_long` | 500 | 48.3 (490) | 47.9 |
| DCGAN (lr 5e-4) | `long` | 50 | 79.2 (50) | 78.3 |
| WGAN, weight clipping (RMSprop 1e-4) | `wgan_long` | 500 | 253.3 (490) | 248.2 |
| VAE (Adam 1e-3, β = 1) | `vae_long` | 300 | 253.3 | 249.2 |
| *Real vs. real: train 10k vs. test 10k* | | | | *4.950* |
| *Real vs. real: train 5k vs. val 5k* | | | | *6.029* |

The real-vs-real rows are the lowest FID a perfect generator could get at
these sample sizes (FID is biased upward at finite n). Every run, with its
hyperparameters and FIDs, is listed in `results/runs.csv`.

## Layout

```
src/
  data.py           MNIST download, pixel scalings, fixed 55k/5k train/val split
  audit_data.py     prints shapes, ranges and split hashes (sanity check)
  trainer.py        Trainer base class: the epoch/step loop the classifier and the GANs share
  classifier.py     Part 0
  dcgan.py          Part 1 (Generator and Discriminator are reused by wgan.py, vae.py, fid.py)
  wgan.py           Part 2 (--gp switches to WGAN-GP)
  vae.py            Part 3
  vae_traverse.py   Part 3.3 latent traversal grids
  fid.py            FID on φ(x) for one run's checkpoints
  diagnostics.py    real-data FID floors and class histograms of generated digits
  manifest.py       indexes every run into outputs/manifest.json for the report
  figures.py        quick-look matplotlib plots while runs train (not used by the report)
  utils/            seeding, device choice, eval-mode context, typed logs, nn_dataclass
tests/              pytest suite (pythonpath = src)
results/runs.csv    one row per run: hyperparameters and FIDs
remote.sh           offload training to a remote GPU box (rsync + tmux)
docs/               notes on the nn_dataclass helper
```

Outputs are not tracked in git; `data/` and `outputs/` are recreated by the
steps below.

## Setup

Requires Python ≥ 3.12 and [uv](https://docs.astral.sh/uv/).

```sh
uv sync --extra dev          # creates .venv from uv.lock (torch, torchvision, matplotlib, pytest, ...)
source .venv/bin/activate    # or prefix each command below with `uv run`
```

`requirements-lock.txt` pins the same environment for plain `pip` on macOS
(`pip install -r requirements-lock.txt`).

Run every command from this directory (`code/`). Paths such as `outputs/` and
`data/` are relative to it. The device is picked automatically: CUDA if
available, then Apple MPS, then CPU.

## Reproducing

### 1. Data

```sh
python src/data.py           # downloads MNIST into data/ (once); later runs train offline
python src/audit_data.py     # optional: batch shapes/ranges and the train/val split hashes
```

The 60k MNIST training images are split once into 55,000 train and 5,000
validation with a fixed seed. The 10,000 test images are used only for the
reported test FIDs. Two pixel scalings are used: `[-1, 1]` for the classifier and
the GANs (matching the generator's tanh), and `[0, 1]` for the VAE (matching its
Bernoulli/BCE likelihood).

### 2. Classifier (Part 0): required before any FID

```sh
python src/classifier.py
```

Trains for 30 epochs and keeps the epoch with the lowest validation loss in
`outputs/classifier.pt`. Per-epoch train/val/test loss and accuracy go to
`outputs/classifier_history.json`. `fid.py` and `diagnostics.py` load
`outputs/classifier.pt` for φ(x).

### 3. Generative models

Each run writes `outputs/<family>/<YYYYmmdd-HHMMSS>_<name>/`:

```
args.json              exact hyperparameters of the run
checkpoints/           epoch_XXX.pt every --ckpt-every epochs, last.pt, final.pt
samples/epoch_XXX.png  8x8 grid from 64 fixed latent vectors, every epoch
logs/losses.pth        loss history every 50 steps (losses.json: JSON copy for the report)
fid.json               appended to by fid.py
```

The commands for the reported runs follow (all use seed 0 unless `--seed` is
given). Runs with `_s1` in the name repeat a configuration with `--seed 1`.

**Part 1: DCGAN.** The defaults are Setup's: Adam (0.5, 0.999), lr 2e-4 for both
networks, batch 128, 25 epochs. Checkpoints are written every 5 epochs.

```sh
python src/dcgan.py --name baseline
python src/dcgan.py --name d5x  --lr-d 1e-3
python src/dcgan.py --name d20x --lr-d 4e-3                      # 1.4: discriminator lr x20
python src/dcgan.py --name lr1e-4 --lr-g 1e-4 --lr-d 1e-4         # 1.5: lr grid
python src/dcgan.py --name lr5e-4 --lr-g 5e-4 --lr-d 5e-4
python src/dcgan.py --name ttur   --lr-g 1e-4 --lr-d 4e-4
python src/dcgan.py --name long    --lr-g 5e-4 --lr-d 5e-4 --epochs 50   # best DCGAN
python src/dcgan.py --name long100 --lr-g 5e-4 --lr-d 5e-4 --epochs 100
python src/dcgan.py --name d20x_long --lr-d 4e-3 --epochs 50
python src/dcgan.py --name g5d2e-3 --lr-g 5e-4 --lr-d 2e-3 --epochs 50
python src/dcgan.py --name baseline_s1 --seed 1
python src/dcgan.py --name d20x_s1 --lr-d 4e-3 --seed 1
```

**Part 2: WGAN.** One epoch is 430 critic steps, which is 86 generator steps at 5
critic steps per generator step. 125 epochs therefore give the same number of
generator updates as 25 DCGAN epochs. Weight clipping uses RMSprop 5e-5 and
clipping c = 0.01 by default. `--gp` switches to Adam 1e-4 with betas (0.5, 0.9)
and a gradient penalty of weight λ = 10.

```sh
python src/wgan.py --name wgan_default --epochs 125
python src/wgan.py --name wgan_lr1e-4 --lr-g 1e-4 --lr-d 1e-4 --epochs 125
python src/wgan.py --name wgan_lr2e-4 --lr-g 2e-4 --lr-d 2e-4 --epochs 125
python src/wgan.py --name wgan_lr5e-4 --lr-g 5e-4 --lr-d 5e-4 --epochs 125
python src/wgan.py --name wgan_long   --lr-g 1e-4 --lr-d 1e-4 --epochs 500 --ckpt-every 10
python src/wgan.py --name wgangp_default --gp --epochs 125
python src/wgan.py --name wgangp_long    --gp --epochs 500 --ckpt-every 10   # best overall
# seed-1 repeats: wgan_lr5e-5_s1, wgan_lr1e-4_s1, wgan_lr2e-4_s1 (add --seed 1)
```

WGAN-GP runs are written to `outputs/wgangp/` rather than `outputs/wgan/`.

**Part 3: VAE.** Adam lr 1e-3 and β = 1 by default. `--beta` takes a constant
or a schedule name (`linear-annealing`, `exponential-annealing`,
`logarithmic-annealing`). The schedules ramp β from 0.01 to 1 over the first 10
epochs.

```sh
python src/vae.py --name vae_lr3e-4 --lr 3e-4 --epochs 50
python src/vae.py --name vae_lr1e-3 --epochs 50
python src/vae.py --name vae_b0p1 --beta 0.1  --epochs 50
python src/vae.py --name vae_b10  --beta 10   --epochs 50
python src/vae.py --name vae_lin  --beta linear-annealing      --epochs 50
python src/vae.py --name vae_exp  --beta exponential-annealing --epochs 50
python src/vae.py --name vae_log  --beta logarithmic-annealing --epochs 50
python src/vae.py --name vae_long --epochs 300 --ckpt-every 10              # best VAE
python src/vae.py --name vae_lr1e-3_s1 --epochs 50 --seed 1
# lr 2e-3 and 3e-3 diverge (log-variance overflow -> NaN); they are reported as failures

python src/vae_traverse.py --run outputs/vae/<RUN> --seeds 0 1   # 3.3: vary z_0..z_9, keep z_10..z_99
```

### 4. Evaluation

```sh
python src/fid.py RUN_DIR --checkpoint all --split val          # val-5k FID of every epoch_XXX.pt
python src/fid.py RUN_DIR --checkpoint epoch_XXX.pt --split test  # test-10k FID of the chosen one
python src/diagnostics.py      # outputs/diagnostics.json: real-data FID floors, class histograms
python src/manifest.py         # outputs/manifest.json: index of all runs for the report
```

Choose the checkpoint on the validation FID only. The test FID is computed
once, for the chosen checkpoint. `fid.py` skips records that are already in
`fid.json`, so re-running it is cheap.

### 5. Report

```sh
cd .. && typst compile hw2.typ
```

To add a run to the report's figures, re-run `src/manifest.py` and add the run's
key to the label table in `../figures/dcgan.typ`.

## Compute

Rough wall-clock times:
- Apple M3 (MPS): about 18 min per 25-epoch DCGAN run.
- One A100 slice: the 500-epoch WGAN-GP run took 25 min.

The classifier takes a few minutes on either. The long runs were trained on a
remote GPU box:

```sh
./remote.sh bootstrap                      # install uv, sync code and environment, fetch MNIST
scp outputs/classifier.pt genai:hw2/outputs/   # remote.sh never copies *.pt
./remote.sh run wgangp_long python src/wgan.py --name wgangp_long --gp --epochs 500 --ckpt-every 10
                                           # runs `uv run CMD` in tmux; log in hw2/logs/NAME.log
./remote.sh status                         # tmux sessions, GPU, last log lines
./remote.sh pull                           # copy run folders and runs.csv back
```

`HW2_HOST` (default `genai`) and `HW2_DIR` (default `hw2`) select the box and
the project directory on it.

GPU runs are not bit-reproducible. The same configuration and seed have given
validation FIDs about 30 points apart. Differences smaller than that, such as
between neighbouring WGAN learning rates, are within run-to-run noise.

## Tests

```sh
uv run --extra dev pytest
```

The suite covers the data split, loaders, the Trainer loop, typed logs, FID
math, WGAN-GP and the VAE. `tests/test_fid_real.py` needs a trained
`outputs/classifier.pt` and real MNIST data, and is skipped without them. Type checking (strict, with `Any`
flagged) is configured for basedpyright in `pyproject.toml`.
