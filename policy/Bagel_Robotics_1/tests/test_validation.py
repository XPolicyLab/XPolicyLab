"""Protect evidence integrity and detect numerical differences, without a GPU."""
import json
from pathlib import Path
import sys

import numpy as np
import pytest

POLICY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(POLICY))
from validation import decode, encode
from validation_compare import compare


def test_lossless_camera_capture_rejects_corruption_and_missing_arrays(tmp_path):
    frame = np.arange(180, dtype=np.uint8).reshape(6, 10, 3)
    encoded = encode({"camera": frame, "state": np.array([0.125], np.float32), "scalar": np.array(3, np.int64)}, tmp_path)
    restored = decode(encoded, tmp_path)
    np.testing.assert_array_equal(restored["camera"], frame)
    assert restored["scalar"].shape == () and restored["scalar"].item() == 3
    image_path = tmp_path / (encoded["camera"]["array"] + ".npy")
    np.save(image_path, frame + 1, allow_pickle=False)
    with pytest.raises(ValueError, match="integrity"):
        decode(encoded, tmp_path)
    image_path.unlink()
    with pytest.raises(FileNotFoundError):
        decode(encoded, tmp_path)


def test_comparison_detects_single_action_change_and_missing_events(tmp_path):
    left, right = tmp_path / "left", tmp_path / "right"
    for root in [left, right]:
        root.mkdir()
        row = {"kind": "commands", "data": encode(np.zeros((16, 16), np.float32), root / "arrays")}
        (root / "trace.jsonl").write_text(json.dumps(row) + "\n")
    assert compare(left, right)["passed"]
    changed = np.zeros((16, 16), np.float32)
    changed[7, 2] = 0.25
    row["data"] = encode(changed, right / "arrays")
    (right / "trace.jsonl").write_text(json.dumps(row) + "\n")
    report = compare(left, right)
    assert not report["passed"] and report["max_abs_stored_array_difference"] == 0.25
    (right / "trace.jsonl").write_text("")
    assert not compare(left, right)["passed"]


def test_empty_evidence_is_not_accepted(tmp_path):
    for name in ["a", "b"]:
        root = tmp_path / name
        root.mkdir()
        (root / "trace.jsonl").write_text("")
    assert not compare(tmp_path / "a", tmp_path / "b")["passed"]
