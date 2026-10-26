"""vae_traverse.py: the z-variant helper and the grid layout, on a scratch decoder.

The traversal only reads a checkpoint file; building a random Decoder weights
needs no data, so these tests run offline.
"""

from pathlib import Path

import torch

from vae import Z_DIM, Decoder
from vae_traverse import load_decoder, traverse, z_variant


def make_run(base: Path, device: str) -> Path:
    """A run dir with a checkpoint holding a random decoder under the 'G' key."""
    run = base / "outputs" / "vae" / "20260101-000000_scratch"
    (run / "checkpoints" / "figures").mkdir(parents=True)
    decoder = Decoder().to(device)
    ckpt = {"G": decoder.state_dict(), "enc": {}, "opt": {}, "epoch": 1}
    torch.save(ckpt, run / "checkpoints" / "final.pt")
    return run


def test_z_variant_replaces_only_the_given_coordinate() -> None:
    z = torch.zeros(1, Z_DIM, 1, 1)
    changed = z_variant(z, 7, 3.5, torch.device("cpu"))
    assert changed[0, 7, 0, 0].item() == 3.5
    rest = changed.clone()
    rest[0, 7, 0, 0] = 0.0
    assert torch.equal(rest, z)  # every other coordinate untouched


def test_load_decoder_reads_the_G_key(tmp_path: Path) -> None:
    run = make_run(tmp_path, "cpu")
    decoder = load_decoder(run, "final.pt", "cpu")
    assert isinstance(decoder, Decoder)
    assert decoder(torch.randn(2, Z_DIM, 1, 1)).shape == (2, 1, 28, 28)


def test_traverse_lays_out_dim_rows_by_value_columns(tmp_path: Path) -> None:
    from PIL import Image

    run = make_run(tmp_path, "cpu")
    decoder = load_decoder(run, "final.pt", "cpu")
    out = run / "figures"
    paths = traverse(
        decoder,
        run,
        "final.pt",
        seeds=[0],
        values=[-3.0, 0.0, 3.0],
        dims=[0, 1],
        out=out / "traverse.png",
    )
    assert len(paths) == 1 and paths[0].exists()
    # Two varied dimensions -> two rows; three values -> three columns;
    # save_image pads 2 px around every cell, so width adds 4 * 2 px and
    # height adds 3 * 2 px.
    assert Image.open(paths[0]).size == (3 * 28 + 4 * 2, 2 * 28 + 3 * 2)
