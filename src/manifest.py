"""Index of finished runs, for the Typst report to read.

    python manifest.py [-o outputs/manifest.json]

Typst cannot list a directory or read .pth files, so this script does both and writes
one JSON file. The report loops over it instead of hard-coding run folders.

Scans outputs/<family>/<run>/ for every run with an args.json (smoke runs excluded) and
records, per run: its args, the files the report may use (all paths are relative to
this directory, i.e. code/), and the reported checkpoint (lowest val-5k FID over the
epoch_*.pt checkpoints, with the test-10k FID of that same checkpoint).
Also (re)writes RUN/logs/losses.json whenever losses.pth is newer or it is missing.

Schema (version 1), runs sorted by directory name (= start time):

    {"version": 1,
     "diagnostics": "outputs/diagnostics.json" | null,
     "runs": [{
        "key": "d20x",                    # args.json "name"; unique
        "family": "dcgan",                # the outputs/<family> directory
        "dir": "outputs/dcgan/2026..._d20x",
        "args": {"lr_g": 0.0002, "lr_d": 0.004, "epochs": 25, "seed": 0, "name": "d20x"},
        "files": {"args": ..., "losses": ... | null, "fid": ... | null},
        "final_grid": "outputs/.../samples/epoch_025.png",
        "samples": [{"epoch": 1, "step": 0, "path": ...}, ..., {"epoch": 25, "step": null, "path": ...}],
        "last_epoch": 25,                 # epoch of final_grid
        "final_val_fid": 128.2 | null,    # val-5k FID at last_epoch
        "reported": {"checkpoint": "epoch_025.pt", "epoch": 25,
                     "val_fid": 128.2, "test_fid": 125.8 | null} | null}]}

"samples" holds every sample grid, sorted by (epoch, step); step is null for the
end-of-epoch grid and an int (batches into epoch 1) for the early grids.
"""

import argparse
import json
import re
from pathlib import Path
from typing import TypedDict, cast

from figures import export_losses

OUTPUTS = Path("outputs")
SAMPLE = re.compile(r"epoch_(\d{3})(?:_step_(\d{4}))?\.png")


class Sample(TypedDict):
    epoch: int
    step: int | None
    path: str


class Reported(TypedDict):
    checkpoint: str
    epoch: int
    val_fid: float
    test_fid: float | None


class Files(TypedDict):
    args: str
    losses: str | None
    fid: str | None


class FidRecord(TypedDict):
    checkpoint: str
    epoch: int
    split: str
    fid: float


class Run(TypedDict):
    key: str
    family: str
    dir: str
    args: dict[str, object]
    files: Files
    final_grid: str
    samples: list[Sample]
    last_epoch: int
    final_val_fid: float | None
    reported: Reported | None


class Manifest(TypedDict):
    version: int
    diagnostics: str | None
    runs: list[Run]


def read_samples(run: Path, base: Path) -> list[Sample]:
    found: list[Sample] = []
    for png in (run / "samples").glob("*.png"):
        m = SAMPLE.fullmatch(png.name)
        if m is None:
            continue
        step = int(m[2]) if m[2] is not None else None
        found.append(
            {"epoch": int(m[1]), "step": step, "path": str(png.relative_to(base))}
        )
    # The early grids (epoch 1, step N) sort before the end-of-epoch grid of epoch 1.
    return sorted(
        found, key=lambda s: (s["epoch"], s["step"] if s["step"] is not None else 10**9)
    )


def read_fid(run: Path) -> list[FidRecord]:
    path = run / "fid.json"
    return (
        cast("list[FidRecord]", json.loads(path.read_text())) if path.exists() else []
    )


def pick_reported(records: list[FidRecord], last_epoch: int) -> Reported | None:
    """Lowest val FID over epoch_*.pt; test FID of that checkpoint (final.pt if it is the last)."""
    val = [
        r
        for r in records
        if r["split"] == "val" and r["checkpoint"].startswith("epoch_")
    ]
    if not val:
        return None
    best = min(val, key=lambda r: r["fid"])
    names = [best["checkpoint"], *(["final.pt"] if best["epoch"] == last_epoch else [])]
    test = next(
        (
            r["fid"]
            for name in names
            for r in records
            if r["split"] == "test" and r["checkpoint"] == name
        ),
        None,
    )
    return {
        "checkpoint": best["checkpoint"],
        "epoch": best["epoch"],
        "val_fid": best["fid"],
        "test_fid": test,
    }


def final_val(records: list[FidRecord], last_epoch: int) -> float | None:
    """Val FID at the last epoch, from its epoch_*.pt record or else from final.pt."""
    for r in records:
        if r["split"] == "val" and r["epoch"] == last_epoch:
            return r["fid"]
    return None


def losses_file(run: Path, base: Path) -> str | None:
    pth, js = run / "logs" / "losses.pth", run / "logs" / "losses.json"
    if not pth.exists():
        return None
    if not js.exists() or js.stat().st_mtime < pth.stat().st_mtime:
        _ = export_losses(run)
    return str(js.relative_to(base))


def read_run(run: Path, base: Path) -> Run | None:
    args_path = run / "args.json"
    if not args_path.exists():
        return None
    args = cast("dict[str, object]", json.loads(args_path.read_text()))
    if args.get("name") == "smoke" or run.name.endswith("_smoke"):
        return None
    samples = read_samples(run, base)
    finals = [s for s in samples if s["step"] is None]
    if not finals:
        raise SystemExit(f"{run}: no per-epoch sample grids found")
    last = finals[-1]
    records = read_fid(run)
    return {
        "key": str(args.get("name", run.name)),
        "family": run.parent.name,
        "dir": str(run.relative_to(base)),
        "args": args,
        "files": {
            "args": str(args_path.relative_to(base)),
            "losses": losses_file(run, base),
            "fid": str((run / "fid.json").relative_to(base)) if records else None,
        },
        "final_grid": last["path"],
        "samples": samples,
        "last_epoch": last["epoch"],
        "final_val_fid": final_val(records, last["epoch"]),
        "reported": pick_reported(records, last["epoch"]),
    }


def build(base: Path = Path(".")) -> Manifest:
    runs_dirs = sorted(
        (
            d
            for family in (base / OUTPUTS).iterdir()
            if family.is_dir()
            for d in family.iterdir()
            if d.is_dir()
        ),
        key=lambda d: d.name,
    )
    runs = [r for d in runs_dirs if (r := read_run(d, base)) is not None]
    keys = [r["key"] for r in runs]
    dupes = sorted({k for k in keys if keys.count(k) > 1})
    if dupes:
        raise SystemExit(f"duplicate run keys (args.json name): {dupes}")
    diag = base / OUTPUTS / "diagnostics.json"
    return {
        "version": 1,
        "diagnostics": str(diag.relative_to(base)) if diag.exists() else None,
        "runs": runs,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Write the run manifest for the Typst report."
    )
    _ = parser.add_argument("-o", "--out", type=Path, default=OUTPUTS / "manifest.json")
    out = cast("Path", parser.parse_args().out)
    manifest = build()
    out.parent.mkdir(parents=True, exist_ok=True)
    _ = out.write_text(json.dumps(manifest, indent=1))
    print(f"{out}: {len(manifest['runs'])} runs")
