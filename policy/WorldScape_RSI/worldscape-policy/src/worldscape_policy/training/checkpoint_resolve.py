"""Resolve training resume and policy-init checkpoint priority."""

from __future__ import annotations

import logging
import re
from pathlib import Path

from omegaconf import DictConfig, OmegaConf

LOGGER = logging.getLogger(__name__)

DEEPSPEED_COMPLETE = ".complete"
_STEP_PATTERN = re.compile(r"^(?:checkpoint|step)-(\d+)(?:\.pt)?$")
_COMPLETE_MARKER_PATTERN = re.compile(
    r"^(?:deepspeed-checkpoint-v[12]|native-training-checkpoint-v2)$"
)


def _select_component_initialization(config: DictConfig) -> None:
    if "checkpoint_dir" in config.model:
        config.model.checkpoint_dir = None
    if "initialization" in config.model:
        config.model.initialization = "components"
    if "pretrained_action_adapter_index" in config.model:
        config.model.pretrained_action_adapter_index = None


def is_noneish(value: object) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        normalized = value.strip().lower()
        return normalized in {"", "null", "none"}
    return False


def _is_training_checkpoint(path: Path) -> bool:
    if path.is_file() and path.suffix == ".pt":
        return True
    marker = path / DEEPSPEED_COMPLETE
    if not path.is_dir() or not marker.is_file():
        return False
    return _COMPLETE_MARKER_PATTERN.fullmatch(marker.read_text().strip()) is not None


def find_latest_training_checkpoint(checkpoint_dir: Path) -> Path | None:
    """Return the newest checkpoint-*/step-* or final training checkpoint."""

    if not checkpoint_dir.is_dir():
        return None

    step_candidates: list[tuple[int, Path]] = []
    final_candidate: Path | None = None

    for entry in checkpoint_dir.iterdir():
        if entry.name.startswith("."):
            continue
        if not _is_training_checkpoint(entry):
            continue

        stem = entry.name[:-3] if entry.name.endswith(".pt") else entry.name
        if stem == "final":
            final_candidate = entry
            continue

        match = _STEP_PATTERN.match(entry.name)
        if match is not None:
            step_candidates.append((int(match.group(1)), entry))

    if step_candidates:
        return max(step_candidates, key=lambda item: item[0])[1]
    return final_candidate


def resolve_training_checkpoint_sources(config: DictConfig) -> Path | None:
    """Apply resume-first checkpoint priority to ``config``.

    Priority:
    1. Explicit ``training.resume``
    2. Latest checkpoint in ``training.checkpoint_dir`` (auto-resume)
    3. ``model.checkpoint_dir`` / ``PRETRAINED_MODEL_PATH`` when not none-ish
    4. Component initialization (WAN/T5/CLIP/etc.)
    """

    training = config.training
    explicit_resume = training.get("resume")
    if not is_noneish(explicit_resume):
        resume_path = Path(str(explicit_resume))
        _select_component_initialization(config)
        LOGGER.info("Using explicit training resume checkpoint: %s", resume_path)
        return resume_path

    checkpoint_dir = Path(str(training.get("checkpoint_dir", "checkpoints")))
    latest = find_latest_training_checkpoint(checkpoint_dir)
    if latest is not None:
        training.resume = str(latest)
        _select_component_initialization(config)
        LOGGER.info(
            "Auto-resuming from latest checkpoint in %s: %s",
            checkpoint_dir,
            latest,
        )
        return latest

    pretrained = OmegaConf.select(config, "model.checkpoint_dir")
    if not is_noneish(pretrained):
        config.model.initialization = str(
            OmegaConf.select(config, "model.initialization", default="auto")
        )
        LOGGER.info("Initializing policy from pretrained checkpoint: %s", pretrained)
        return None

    _select_component_initialization(config)
    LOGGER.info(
        "No resume or pretrained checkpoint; initializing policy from base components"
    )
    return None
