from __future__ import annotations

import subprocess
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
HELPER = ROOT / "recipes" / "common" / "eval_checkpoint.sh"


def _checkpoint(root: Path, step: int, *, complete: bool = True) -> Path:
    path = root / f"checkpoint-{step}"
    path.mkdir()
    (path / "model.safetensors").write_bytes(b"weights")
    (path / "checkpoint_manifest.json").write_text("{}")
    if complete:
        (path / ".complete").write_text("native-training-checkpoint-v2")
    return path


def test_eval_checkpoint_resolver_selects_highest_complete_bundle(
    tmp_path: Path,
) -> None:
    _checkpoint(tmp_path, 10)
    expected = _checkpoint(tmp_path, 20)
    _checkpoint(tmp_path, 30, complete=False)

    result = subprocess.run(
        [
            "bash",
            "-c",
            'source "$1"; wsp_resolve_eval_checkpoint "$2"',
            "bash",
            str(HELPER),
            str(tmp_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert result.stdout.strip() == str(expected)


def test_eval_checkpoint_resolver_preserves_explicit_path(tmp_path: Path) -> None:
    explicit = tmp_path / "external-model"
    result = subprocess.run(
        [
            "bash",
            "-c",
            'source "$1"; wsp_resolve_eval_checkpoint "$2" "$3"',
            "bash",
            str(HELPER),
            str(tmp_path),
            str(explicit),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert result.stdout.strip() == str(explicit)


def test_eval_checkpoint_resolver_accepts_only_complete_sharded_weights(
    tmp_path: Path,
) -> None:
    incomplete = _checkpoint(tmp_path, 20)
    (incomplete / "model.safetensors").unlink()
    (incomplete / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"weight": "model-00001-of-00001.safetensors"}})
    )
    expected = _checkpoint(tmp_path, 10)
    (expected / "model.safetensors").unlink()
    (expected / "model-00001-of-00001.safetensors").write_bytes(b"weights")
    (expected / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"weight": "model-00001-of-00001.safetensors"}})
    )

    result = subprocess.run(
        [
            "bash",
            "-c",
            'source "$1"; wsp_resolve_eval_checkpoint "$2"',
            "bash",
            str(HELPER),
            str(tmp_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert result.stdout.strip() == str(expected)
