"""Export resumable trainer checkpoints as evaluation-ready native bundles."""

from __future__ import annotations

import shutil
import uuid
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch
from omegaconf import DictConfig, OmegaConf

from worldscape_policy.action_space import parse_action_mode
from worldscape_policy.data.normalization import GlobalZScoreNormalizer
from worldscape_policy.checkpoint.loader import (
    save_native_checkpoint,
    source_checkpoint_fingerprint,
    validate_native_checkpoint_artifacts,
)
from worldscape_policy.checkpoint.transforms import (
    CheckpointTransformArtifact,
    EmbodimentTransform,
    TransformField,
)
from worldscape_policy.checkpoint.weights_io import (
    DEFAULT_MAX_SHARD_SIZE,
    MODEL_FILENAME,
    MODEL_INDEX_FILENAME,
    checkpoint_weight_files,
    load_checkpoint_state_dict,
)
from worldscape_policy.checkpoint.validation import (
    ConversionReport,
    ConversionResult,
    GroupCoverage,
)


def build_identity_transform_bundle(
    *,
    image_input_range: str,
    action_mode: str = "eef",
    relative_action: bool = False,
    embodiments: Mapping[str, Any] | DictConfig,
    provenance: Mapping[str, Any] | DictConfig,
) -> CheckpointTransformArtifact:
    """Build an explicit no-normalization bundle from ordered field specs."""

    parsed_action_mode = parse_action_mode(action_mode)
    raw_embodiments = _plain_mapping(embodiments, name="embodiments")
    raw_provenance = _plain_mapping(provenance, name="provenance")
    converted: dict[str, EmbodimentTransform] = {}
    for name, raw in raw_embodiments.items():
        if not isinstance(raw, Mapping):
            raise TypeError(f"embodiments.{name} must be a mapping")
        converted[str(name)] = EmbodimentTransform(
            embodiment_id=int(raw["embodiment_id"]),
            max_state_dim=int(raw["max_state_dim"]),
            max_action_dim=int(raw["max_action_dim"]),
            state_fields=_identity_fields(raw["state_fields"], prefix="state"),
            action_fields=_identity_fields(
                raw["action_fields"],
                prefix="action",
                action_mode=parsed_action_mode,
                relative_action=relative_action,
            ),
        )
    return CheckpointTransformArtifact(
        image_input_range=image_input_range,
        embodiments=converted,
        provenance=dict(raw_provenance),
    )


def build_zscore_transform_bundle(
    *,
    image_input_range: str,
    statistics_path: str,
    clip_range: Sequence[float] = (-5.0, 5.0),
    action_mode: str = "joint",
    relative_action: bool = False,
    embodiments: Mapping[str, Any] | DictConfig,
    provenance: Mapping[str, Any] | DictConfig,
) -> CheckpointTransformArtifact:
    """Build a native bundle using global z-score statistics."""

    parsed_action_mode = parse_action_mode(action_mode)
    if parsed_action_mode != "joint" or relative_action:
        raise ValueError(
            "RoboTwin z-score transforms require absolute joint action_mode"
        )
    if len(clip_range) != 2:
        raise ValueError("clip_range must contain [min, max]")
    normalizer = GlobalZScoreNormalizer(
        statistics_path,
        clip_range=(float(clip_range[0]), float(clip_range[1])),
    )
    raw_embodiments = _plain_mapping(embodiments, name="embodiments")
    raw_provenance = dict(_plain_mapping(provenance, name="provenance"))
    raw_provenance.update(
        {
            "normalization": "global_zscore",
            "statistics_path": str(normalizer.path),
            "clip_range": list(normalizer.clip_range),
            "action_mode": "joint",
            "relative_action": False,
        }
    )
    converted: dict[str, EmbodimentTransform] = {}
    for name, raw in raw_embodiments.items():
        if not isinstance(raw, Mapping):
            raise TypeError(f"embodiments.{name} must be a mapping")
        converted[str(name)] = EmbodimentTransform(
            embodiment_id=int(raw["embodiment_id"]),
            max_state_dim=int(raw["max_state_dim"]),
            max_action_dim=int(raw["max_action_dim"]),
            state_fields=_zscore_fields(
                raw["state_fields"],
                prefix="state",
                statistics=normalizer.field_statistics("state"),
            ),
            action_fields=_zscore_fields(
                raw["action_fields"],
                prefix="action",
                statistics=normalizer.field_statistics("action"),
            ),
        )
    return CheckpointTransformArtifact(
        image_input_range=image_input_range,
        embodiments=converted,
        provenance=raw_provenance,
    )


def export_training_checkpoint(
    source: str | Path,
    destination: str | Path,
    *,
    model_variant: str,
    model_config: Mapping[str, Any] | DictConfig,
    generation_config: Mapping[str, Any] | DictConfig,
    normalization: Mapping[str, Any] | DictConfig,
    transform_bundle: CheckpointTransformArtifact,
    tokenizer_source: str | Path,
    provenance: Mapping[str, Any] | DictConfig,
    git_commit: str | None = None,
    preserve_resume_state: bool | None = None,
    max_shard_size: int | str = DEFAULT_MAX_SHARD_SIZE,
) -> Path:
    """Convert a trainer checkpoint without discarding its resume state."""

    source_path = Path(source)
    destination_path = Path(destination)
    if preserve_resume_state is None:
        preserve_resume_state = (
            source_path.resolve(strict=False)
            == destination_path.resolve(strict=False)
        )
    state_dict = _load_policy_state(source_path)
    report = ConversionReport(
        source_tensors=len(state_dict),
        converted_tensors=len(state_dict),
        groups={
            "native_training": GroupCoverage(
                tensors=len(state_dict),
                parameters=sum(tensor.numel() for tensor in state_dict.values()),
                dtypes={str(tensor.dtype) for tensor in state_dict.values()},
            )
        },
    )
    result = ConversionResult(
        state_dict=state_dict,
        report=report,
        key_mapping_version="native-training-v1",
    )
    fingerprint_source = source_path
    if source_path.is_dir():
        fingerprint_source = (
            source_path
            if (source_path / MODEL_FILENAME).is_file()
            or (source_path / MODEL_INDEX_FILENAME).is_file()
            else source_path / "policy.pt"
        )
    source_hash, source_files = source_checkpoint_fingerprint(fingerprint_source)
    temporary = destination_path.with_name(
        f".{destination_path.name}.native-{uuid.uuid4().hex}.tmp"
    )
    complete_marker: str | None = None
    plain_model_config = dict(_plain_mapping(model_config, name="model_config"))
    plain_generation_config = dict(
        _plain_mapping(generation_config, name="generation_config")
    )
    try:
        save_native_checkpoint(
            result,
            temporary,
            model_variant=model_variant,
            source_checkpoint_hash=source_hash,
            source_files=source_files,
            model_config=plain_model_config,
            generation_config=plain_generation_config,
            normalization=dict(_plain_mapping(normalization, name="normalization")),
            transform_bundle=transform_bundle,
            provenance=dict(_plain_mapping(provenance, name="provenance")),
            tokenizer_source=tokenizer_source,
            git_commit=git_commit,
            max_shard_size=max_shard_size,
        )
        if destination_path.exists():
            if not destination_path.is_dir():
                raise FileExistsError(
                    f"Native export destination is not a directory: {destination_path}"
                )
            if preserve_resume_state:
                marker_path = destination_path / ".complete"
                if marker_path.is_file():
                    complete_marker = marker_path.read_text()
                    marker_path.unlink()
                _merge_bundle(temporary, destination_path)
            else:
                _replace_bundle(temporary, destination_path)
        else:
            destination_path.parent.mkdir(parents=True, exist_ok=True)
            temporary.replace(destination_path)
        validate_native_checkpoint_artifacts(destination_path)
        if complete_marker is not None:
            marker_path = destination_path / ".complete"
            marker_temporary = destination_path / f"..complete.{uuid.uuid4().hex}.tmp"
            marker_temporary.write_text(complete_marker)
            marker_temporary.replace(marker_path)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return destination_path


def _load_policy_state(source: Path) -> dict[str, torch.Tensor]:
    if source.is_dir():
        policy_file = source / "policy.pt"
        try:
            checkpoint_weight_files(source)
        except FileNotFoundError:
            has_safetensors = False
        else:
            has_safetensors = True
        if has_safetensors:
            raw = load_checkpoint_state_dict(source)
        elif policy_file.is_file():
            raw = torch.load(policy_file, map_location="cpu", weights_only=True)
        else:
            raise FileNotFoundError(
                "trainer checkpoint is missing model.safetensors/policy.pt: "
                f"{source}"
            )
    elif source.is_file():
        payload = torch.load(source, map_location="cpu", weights_only=True)
        if not isinstance(payload, Mapping) or "model" not in payload:
            raise ValueError(f"Trainer checkpoint has no model state: {source}")
        raw = payload["model"]
    else:
        raise FileNotFoundError(f"Trainer checkpoint does not exist: {source}")
    if not isinstance(raw, Mapping):
        raise TypeError("Trainer policy state must be a mapping")
    state = {
        str(key): value.detach().cpu().contiguous()
        for key, value in raw.items()
        if isinstance(value, torch.Tensor)
    }
    if len(state) != len(raw):
        raise TypeError("Trainer policy state contains non-tensor values")
    return state


def _identity_fields(
    value: Any,
    *,
    prefix: str,
    action_mode: str = "eef",
    relative_action: bool = False,
) -> tuple[TransformField, ...]:
    if isinstance(value, DictConfig):
        value = OmegaConf.to_container(value, resolve=True)
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise TypeError(f"{prefix}_fields must be an ordered sequence")
    fields: list[TransformField] = []
    for index, item in enumerate(value):
        if (
            not isinstance(item, Sequence)
            or isinstance(item, (str, bytes))
            or len(item) not in {2, 3}
        ):
            raise TypeError(
                f"{prefix}_fields[{index}] must be [key, size] or "
                "[key, size, absolute]"
            )
        key, size = str(item[0]), int(item[1])
        absolute = (
            _default_absolute_field(
                key,
                action_mode=action_mode,
                relative_action=relative_action,
            )
            if len(item) == 2
            else bool(item[2])
        )
        if not key.startswith(prefix + "."):
            raise ValueError(f"{key!r} must start with {prefix!r}")
        fields.append(
            TransformField(
                key=key,
                size=size,
                normalization=None,
                statistics={},
                per_horizon_statistics=None,
                absolute=absolute,
            )
        )
    return tuple(fields)


def _zscore_fields(
    value: Any,
    *,
    prefix: str,
    statistics: Mapping[str, Sequence[float]],
) -> tuple[TransformField, ...]:
    if isinstance(value, DictConfig):
        value = OmegaConf.to_container(value, resolve=True)
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise TypeError(f"{prefix}_fields must be an ordered sequence")
    widths = [int(item[1]) for item in value]
    expected_width = len(statistics["mean"])
    if sum(widths) != expected_width:
        raise ValueError(
            f"{prefix} field width {sum(widths)} does not match z-score "
            f"statistics width {expected_width}"
        )
    fields: list[TransformField] = []
    offset = 0
    for index, item in enumerate(value):
        if (
            not isinstance(item, Sequence)
            or isinstance(item, (str, bytes))
            or len(item) not in {2, 3}
        ):
            raise TypeError(
                f"{prefix}_fields[{index}] must be [key, size] or "
                "[key, size, absolute]"
            )
        key, size = str(item[0]), int(item[1])
        if not key.startswith(prefix + "."):
            raise ValueError(f"{key!r} must start with {prefix!r}")
        field_statistics = {
            name: [float(v) for v in values[offset : offset + size]]
            for name, values in statistics.items()
        }
        fields.append(
            TransformField(
                key=key,
                size=size,
                normalization="mean_std",
                statistics=field_statistics,
                per_horizon_statistics=None,
                absolute=True,
            )
        )
        offset += size
    return tuple(fields)


def _default_absolute_field(
    key: str,
    *,
    action_mode: str,
    relative_action: bool,
) -> bool:
    if not relative_action or not key.startswith("action."):
        return True
    if action_mode != "eef":  # pragma: no cover - guarded by parse_action_mode
        raise ValueError(f"unsupported action mode: {action_mode!r}")
    field = key.removeprefix("action.")
    if field in {"left_gripper", "right_gripper"}:
        return True
    if field in {"left_pos", "left_rot6d", "right_pos", "right_rot6d"}:
        return False
    raise ValueError(
        "relative EEF export requires decomposed position/rotation/gripper "
        f"fields, got {key!r}"
    )


def _plain_mapping(value: Any, *, name: str) -> Mapping[str, Any]:
    if isinstance(value, DictConfig):
        value = OmegaConf.to_container(value, resolve=True)
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping")
    return value


def _merge_bundle(source: Path, destination: Path) -> None:
    manifest = destination / "checkpoint_manifest.json"
    if manifest.exists():
        manifest.unlink()
    for path in (
        destination / MODEL_FILENAME,
        destination / MODEL_INDEX_FILENAME,
        *destination.glob("model-*-of-*.safetensors"),
    ):
        if path.is_file():
            path.unlink()
    entries = sorted(
        source.iterdir(),
        key=lambda entry: (entry.name == "checkpoint_manifest.json", entry.name),
    )
    for entry in entries:
        target = destination / entry.name
        if target.exists():
            if target.is_dir():
                shutil.rmtree(target)
            else:
                target.unlink()
        shutil.move(str(entry), str(target))


def _replace_bundle(source: Path, destination: Path) -> None:
    """Atomically replace a standalone bundle without retaining resume state."""
    backup = destination.with_name(
        f".{destination.name}.backup-{uuid.uuid4().hex}.tmp"
    )
    destination.replace(backup)
    try:
        source.replace(destination)
    except Exception:
        if not destination.exists():
            backup.replace(destination)
        raise
    else:
        shutil.rmtree(backup)


__all__ = [
    "build_zscore_transform_bundle",
    "build_identity_transform_bundle",
    "export_training_checkpoint",
]
