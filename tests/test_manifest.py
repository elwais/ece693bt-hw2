import json
from pathlib import Path

import pytest

from manifest import FidRecord, build, pick_reported


def rec(checkpoint: str, epoch: int, split: str, fid: float) -> FidRecord:
    return {"checkpoint": checkpoint, "epoch": epoch, "split": split, "fid": fid}


def make_run(
    base: Path,
    dirname: str,
    name: str,
    *,
    grids: tuple[str, ...] = (
        "epoch_001_step_0000",
        "epoch_001_step_0100",
        "epoch_001",
        "epoch_002",
    ),
    fid: list[FidRecord] | None = None,
) -> Path:
    run = base / "outputs" / "dcgan" / dirname
    (run / "samples").mkdir(parents=True)
    _ = (run / "args.json").write_text(
        json.dumps({"name": name, "epochs": 2, "seed": 0})
    )
    for g in grids:
        _ = (run / "samples" / f"{g}.png").write_bytes(b"")
    if fid is not None:
        _ = (run / "fid.json").write_text(json.dumps(fid))
    return run


def test_reported_is_best_val_with_its_own_test_record() -> None:
    records = [
        rec("epoch_040.pt", 40, "val", 86.0),
        rec("epoch_045.pt", 45, "val", 84.8),
        rec("epoch_100.pt", 100, "val", 88.8),
        rec("final.pt", 100, "val", 88.8),
        rec("final.pt", 100, "test", 87.3),
        rec("epoch_045.pt", 45, "test", 82.6),
    ]
    assert pick_reported(records, last_epoch=100) == {
        "checkpoint": "epoch_045.pt",
        "epoch": 45,
        "val_fid": 84.8,
        "test_fid": 82.6,
    }


def test_reported_uses_final_pt_test_record_when_best_is_the_last_epoch() -> None:
    records = [
        rec("epoch_005.pt", 5, "val", 300.0),
        rec("epoch_010.pt", 10, "val", 200.0),
        rec("final.pt", 10, "test", 190.0),
    ]
    got = pick_reported(records, last_epoch=10)
    assert got is not None
    assert got["checkpoint"] == "epoch_010.pt"
    assert got["test_fid"] == 190.0


def test_reported_does_not_borrow_final_pt_when_best_is_not_the_last_epoch() -> None:
    records = [
        rec("epoch_005.pt", 5, "val", 100.0),
        rec("epoch_010.pt", 10, "val", 200.0),
        rec("final.pt", 10, "test", 190.0),
    ]
    got = pick_reported(records, last_epoch=10)
    assert got is not None
    assert got["epoch"] == 5
    assert got["test_fid"] is None


def test_reported_is_none_without_epoch_checkpoints() -> None:
    assert pick_reported([rec("final.pt", 10, "val", 1.0)], last_epoch=10) is None


def test_build_lists_runs_with_relative_paths_and_ordered_samples(
    tmp_path: Path,
) -> None:
    _ = make_run(
        tmp_path,
        "20260101-000000_a",
        "a",
        fid=[rec("epoch_002.pt", 2, "val", 5.0), rec("final.pt", 2, "test", 4.0)],
    )
    _ = make_run(tmp_path, "20260101-000100_b", "b")
    manifest = build(tmp_path)
    assert [r["key"] for r in manifest["runs"]] == ["a", "b"]  # sorted by start time
    a = manifest["runs"][0]
    assert a["dir"] == "outputs/dcgan/20260101-000000_a"
    assert a["final_grid"] == "outputs/dcgan/20260101-000000_a/samples/epoch_002.png"
    assert a["last_epoch"] == 2
    assert [(s["epoch"], s["step"]) for s in a["samples"]] == [
        (1, 0),
        (1, 100),
        (1, None),
        (2, None),
    ]
    assert a["reported"] == {
        "checkpoint": "epoch_002.pt",
        "epoch": 2,
        "val_fid": 5.0,
        "test_fid": 4.0,
    }
    assert a["final_val_fid"] == 5.0
    assert a["files"]["fid"] is not None
    b = manifest["runs"][1]
    assert b["reported"] is None
    assert b["files"] == {
        "args": "outputs/dcgan/20260101-000100_b/args.json",
        "losses": None,
        "fid": None,
    }
    assert manifest["diagnostics"] is None


def test_build_skips_smoke_runs(tmp_path: Path) -> None:
    _ = make_run(tmp_path, "20260101-000000_smoke", "smoke")
    _ = make_run(tmp_path, "20260101-000100_real", "real")
    assert [r["key"] for r in build(tmp_path)["runs"]] == ["real"]


def test_build_rejects_duplicate_keys(tmp_path: Path) -> None:
    _ = make_run(tmp_path, "20260101-000000_x", "same")
    _ = make_run(tmp_path, "20260101-000100_y", "same")
    with pytest.raises(SystemExit, match="duplicate run keys"):
        _ = build(tmp_path)
