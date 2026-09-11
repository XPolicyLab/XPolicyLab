"""Tests for training checkpoint resolution priority."""

from __future__ import annotations

from pathlib import Path

from omegaconf import OmegaConf

from worldscape_policy.training.checkpoint_resolve import (
    find_latest_training_checkpoint,
    is_noneish,
    resolve_training_checkpoint_sources,
)
from worldscape_policy.training.trainer import NativeTrainer


def test_is_noneish() -> None:
    assert is_noneish(None)
    assert is_noneish("")
    assert is_noneish("none")
    assert is_noneish("NONE")
    assert is_noneish("null")
    assert not is_noneish("/path/to/checkpoint")


def test_find_latest_training_checkpoint_prefers_highest_step(tmp_path: Path) -> None:
    (tmp_path / "step-1000.pt").write_bytes(b"")
    (tmp_path / "step-4000.pt").write_bytes(b"")
    (tmp_path / "step-2000.pt").write_bytes(b"")

    latest = find_latest_training_checkpoint(tmp_path)

    assert latest == tmp_path / "step-4000.pt"


def test_find_latest_training_checkpoint_supports_deepspeed_dirs(
    tmp_path: Path,
) -> None:
    step_dir = tmp_path / "step-5000"
    step_dir.mkdir()
    (step_dir / NativeTrainer.DEEPSPEED_COMPLETE).write_text(
        "deepspeed-checkpoint-v1\n"
    )
    checkpoint_dir = tmp_path / "checkpoint-6000"
    checkpoint_dir.mkdir()
    (checkpoint_dir / NativeTrainer.DEEPSPEED_COMPLETE).write_text(
        "native-training-checkpoint-v2\n"
    )

    latest = find_latest_training_checkpoint(tmp_path)

    assert latest == checkpoint_dir


def test_find_latest_training_checkpoint_ignores_invalid_marker(
    tmp_path: Path,
) -> None:
    invalid = tmp_path / "checkpoint-6000"
    invalid.mkdir()
    (invalid / NativeTrainer.DEEPSPEED_COMPLETE).write_text("incomplete\n")
    valid = tmp_path / "checkpoint-5000"
    valid.mkdir()
    (valid / NativeTrainer.DEEPSPEED_COMPLETE).write_text(
        "deepspeed-checkpoint-v2\n"
    )

    assert find_latest_training_checkpoint(tmp_path) == valid


def test_find_latest_training_checkpoint_falls_back_to_final(tmp_path: Path) -> None:
    (tmp_path / "final.pt").write_bytes(b"")

    latest = find_latest_training_checkpoint(tmp_path)

    assert latest == tmp_path / "final.pt"


def test_resolve_auto_resume_overrides_pretrained(monkeypatch) -> None:
    output_dir = Path("/tmp/wsp-resume-test")
    monkeypatch.setattr(
        "worldscape_policy.training.checkpoint_resolve.find_latest_training_checkpoint",
        lambda checkpoint_dir: output_dir / "step-3000.pt",
    )
    config = OmegaConf.create(
        {
            "training": {
                "checkpoint_dir": str(output_dir),
                "resume": None,
            },
            "model": {
                "checkpoint_dir": "/pretrained/policy",
                "initialization": "auto",
                "pretrained_action_adapter_index": 2,
            },
        }
    )

    resume_path = resolve_training_checkpoint_sources(config)

    assert resume_path == output_dir / "step-3000.pt"
    assert config.training.resume == str(output_dir / "step-3000.pt")
    assert config.model.checkpoint_dir is None
    assert config.model.initialization == "components"
    assert config.model.pretrained_action_adapter_index is None


def test_resolve_explicit_resume_overrides_pretrained() -> None:
    config = OmegaConf.create(
        {
            "training": {
                "checkpoint_dir": "/unused/output",
                "resume": "/experiment/step-3000.pt",
            },
            "model": {
                "checkpoint_dir": "/pretrained/policy",
                "initialization": "checkpoint_overlay",
                "pretrained_action_adapter_index": 4,
            },
        }
    )

    resume_path = resolve_training_checkpoint_sources(config)

    assert resume_path == Path("/experiment/step-3000.pt")
    assert config.model.checkpoint_dir is None
    assert config.model.initialization == "components"
    assert config.model.pretrained_action_adapter_index is None


def test_resolve_uses_pretrained_when_no_output_resume(monkeypatch) -> None:
    monkeypatch.setattr(
        "worldscape_policy.training.checkpoint_resolve.find_latest_training_checkpoint",
        lambda checkpoint_dir: None,
    )
    config = OmegaConf.create(
        {
            "training": {"checkpoint_dir": "/empty/output", "resume": None},
            "model": {
                "checkpoint_dir": "/pretrained/policy",
                "initialization": "auto",
            },
        }
    )

    resume_path = resolve_training_checkpoint_sources(config)

    assert resume_path is None
    assert config.model.checkpoint_dir == "/pretrained/policy"
    assert config.model.initialization == "auto"


def test_resolve_falls_back_to_components(monkeypatch) -> None:
    monkeypatch.setattr(
        "worldscape_policy.training.checkpoint_resolve.find_latest_training_checkpoint",
        lambda checkpoint_dir: None,
    )
    config = OmegaConf.create(
        {
            "training": {"checkpoint_dir": "/empty/output", "resume": None},
            "model": {"checkpoint_dir": None, "initialization": "auto"},
        }
    )

    resolve_training_checkpoint_sources(config)

    assert config.model.checkpoint_dir is None
    assert config.model.initialization == "components"
