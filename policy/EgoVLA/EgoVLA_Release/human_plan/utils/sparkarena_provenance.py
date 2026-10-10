"""Checkpoint finalization and fail-closed verification for SparkArena EgoVLA.

This module is intentionally independent from the existing MANO/EE
``provenance.py`` path.  A SparkArena checkpoint is deployable only after its model
ABI, data contract, normalization statistics, prompt mapping, and immutable
artifacts have been bound by :func:`finalize_sparkarena_checkpoint`.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from ..dataset_preprocessing.sparkarena.process_data import (
    ACTION_TYPE,
    COLOR_ORDER,
    LOGICAL_DIM,
    LOGICAL_ORDER,
    MODEL_OUTPUT_DIM,
    MODEL_OUTPUT_INDICES,
    PROCESSED_IMAGE_HEIGHT,
    PROCESSED_IMAGE_WIDTH,
    PROMPT_VERSION,
    ROBOT_KEY,
    RUNTIME_CAMERA_KEY,
    TASK_INSTRUCTIONS_SHA256,
    canonical_json,
    contract_dict,
)


PROVENANCE_SCHEMA = "egovla_sparkarena_checkpoint_provenance_v2"
DATA_PROVENANCE_SCHEMA = "sparkarena-egovla-provenance-v1"
DATA_METADATA_SCHEMA = "xpolicylab-egovla-lazy-hdf5-manifest-v1"
STATS_SCHEMA = "sparkarena_joint54_stats_v1"
NORMALIZATION_NAME = "minmax_to_unit_interval_no_clip"
NORMALIZATION_EPSILON = 1.0e-6
EXPECTED_PRETRAINED_REPOSITORY = "rchal97/ego_vla_human_video_pretrained"
EXPECTED_PRETRAINED_COMMIT = "52dea0ceae8754d658d7c3f5fd8aa17c309348ea"
PROVENANCE_FILENAME = "egovla_sparkarena_provenance.json"
CONTRACT_FILENAME = "egovla_contract.json"
STATE_STATS_FILENAME = "joint_state_stats.npz"
ACTION_STATS_FILENAME = "joint_action_stats.npz"
DATA_METADATA_FILENAME = "egovla_sparkarena_data_metadata.json"
DATA_PROVENANCE_FILENAME = "egovla_sparkarena_data_provenance.json"

DATA_SOURCE_FILES = {
    STATE_STATS_FILENAME: STATE_STATS_FILENAME,
    ACTION_STATS_FILENAME: ACTION_STATS_FILENAME,
    "metadata.json": DATA_METADATA_FILENAME,
    "sparkarena_provenance.json": DATA_PROVENANCE_FILENAME,
}

MODEL_COMPONENTS = ("llm", "vision_tower", "mm_projector", "traj_decoder")
TRAINING_WORLD_SIZE = 8
OPTIMIZER_FILENAME = "optimizer.pt"
SCHEDULER_FILENAME = "scheduler.pt"
TRAINER_STATE_FILENAME = "trainer_state.json"
RNG_STATE_FILENAMES = tuple(
    f"rng_state_{rank}.pth" for rank in range(TRAINING_WORLD_SIZE)
)
TRAINING_STATE_FILENAMES = (
    OPTIMIZER_FILENAME,
    SCHEDULER_FILENAME,
    TRAINER_STATE_FILENAME,
    *RNG_STATE_FILENAMES,
)
EXPECTED_REINITIALIZED_PREFIXES = (
    "traj_decoder.decoder.proprio_projection",
    "traj_decoder.decoder.output_projection_left",
    "traj_decoder.decoder.output_projection_right",
)
DEFAULT_REUSED_PREFIXES = (
    "llm",
    "vision_tower",
    "mm_projector",
    "traj_decoder.decoder.first_norm",
    "traj_decoder.decoder.layers",
)

DEFAULT_RUN_CONTRACT: dict[str, Any] = {
    "schema": "egovla_sparkarena_run_contract_v1",
    "benchmark": "SparkArena",
    "robot_key": ROBOT_KEY,
    "action_type": ACTION_TYPE,
    "pretrained_repo": EXPECTED_PRETRAINED_REPOSITORY,
    "pretrained_commit": EXPECTED_PRETRAINED_COMMIT,
    "predict_future_step": 30,
    "history_steps": 5,
    "history_stride": 5,
    "prompt_version": PROMPT_VERSION,
    "conversation_template": "vicuna_v1",
    "runtime_camera_key": RUNTIME_CAMERA_KEY,
    "raw_input_resolution": [480, 640],
    "geometry": "full_frame_resize_480x640_to_384x384",
    "input_resolution": [PROCESSED_IMAGE_HEIGHT, PROCESSED_IMAGE_WIDTH],
    "color_order": COLOR_ORDER,
    "reverse_channel_order": False,
    "state_normalization": NORMALIZATION_NAME,
    "action_denormalization": "prediction*scale+minimum; no clipping",
    "model_abi": {
        "traj_decoder_type": "transformer_split_action_v2",
        "proprio_size": LOGICAL_DIM,
        "use_proprio": True,
        "sep_proprio": False,
        "action_output_dim": MODEL_OUTPUT_DIM,
        "model_output_indices": list(MODEL_OUTPUT_INDICES),
    },
    "reinitialized_parameter_prefixes": list(EXPECTED_REINITIALIZED_PREFIXES),
    "reused_parameter_prefixes": list(DEFAULT_REUSED_PREFIXES),
}

_SENSITIVE_KEYS = {
    "api_key",
    "wandb_api_key",
    "password",
    "secret",
    "access_token",
    "refresh_token",
}


def sha256_file(path: str | os.PathLike[str]) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise FileNotFoundError(f"required SparkArena artifact is missing: {path}") from None
    if not isinstance(value, dict):
        raise ValueError(f"JSON artifact must contain an object: {path}")
    return value


def _record(path: Path) -> dict[str, Any]:
    return {"sha256": sha256_file(path), "bytes": path.stat().st_size}


def _verify_record(path: Path, record: Mapping[str, Any]) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"checkpoint artifact is missing: {path}")
    expected_bytes = record.get("bytes")
    expected_hash = record.get("sha256")
    if not isinstance(expected_bytes, int) or expected_bytes < 0:
        raise ValueError(f"invalid byte count for checkpoint artifact: {path}")
    if not isinstance(expected_hash, str) or len(expected_hash) != 64:
        raise ValueError(f"invalid sha256 for checkpoint artifact: {path}")
    if path.stat().st_size != expected_bytes or sha256_file(path) != expected_hash:
        raise ValueError(f"checkpoint artifact hash/size mismatch: {path}")


def _atomic_bytes(destination: Path, payload: bytes) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", dir=str(destination.parent)
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_json(destination: Path, value: Mapping[str, Any]) -> None:
    _atomic_bytes(destination, canonical_json(value))


def _atomic_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", dir=str(destination.parent)
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        shutil.copyfile(source, temporary)
        with temporary.open("rb") as stream:
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def _reject_sensitive(value: Any, *, location: str = "run_contract") -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            name = str(key).lower()
            if name in _SENSITIVE_KEYS:
                raise ValueError(f"sensitive key is forbidden in {location}: {key}")
            _reject_sensitive(item, location=f"{location}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _reject_sensitive(item, location=f"{location}[{index}]")


def _json_clone(value: Mapping[str, Any]) -> dict[str, Any]:
    try:
        cloned = json.loads(json.dumps(value, ensure_ascii=False))
    except (TypeError, ValueError) as exc:
        raise TypeError("run_contract must be JSON serializable") from exc
    if not isinstance(cloned, dict):
        raise TypeError("run_contract must be a mapping")
    return cloned


def _normalise_run_contract(value: Mapping[str, Any] | None) -> dict[str, Any]:
    result = _json_clone(DEFAULT_RUN_CONTRACT)
    supplied = _json_clone(value) if value is not None else {}
    _reject_sensitive(supplied)
    result.update(supplied)

    protected = {
        key: expected
        for key, expected in DEFAULT_RUN_CONTRACT.items()
        if key not in {"reinitialized_parameter_prefixes", "reused_parameter_prefixes"}
    }
    for key, expected in protected.items():
        if canonical_json(result.get(key)) != canonical_json(expected):
            raise ValueError(
                f"SparkArena run contract field {key!r} differs from runtime ABI"
            )

    prefixes = result.get("reinitialized_parameter_prefixes")
    if not isinstance(prefixes, list) or not all(
        isinstance(item, str) and item for item in prefixes
    ):
        raise ValueError("reinitialized_parameter_prefixes must be a non-empty string list")
    for required in EXPECTED_REINITIALIZED_PREFIXES:
        if not any(
            item == required
            or item.startswith(required + ".")
            or required.startswith(item + ".")
            for item in prefixes
        ):
            raise ValueError(
                "SparkArena checkpoint did not declare reinitialization of "
                f"{required}"
            )

    reused = result.get("reused_parameter_prefixes")
    if not isinstance(reused, list) or not all(
        isinstance(item, str) and item for item in reused
    ):
        raise ValueError("reused_parameter_prefixes must be a non-empty string list")
    return result


def _scalar_text(value: np.ndarray, *, name: str, path: Path) -> str:
    array = np.asarray(value)
    if array.size != 1:
        raise ValueError(f"{path}:{name} must be a scalar string")
    item = array.reshape(()).item()
    if isinstance(item, bytes):
        item = item.decode("utf-8")
    if not isinstance(item, str):
        raise ValueError(f"{path}:{name} must be a scalar string")
    return item


def load_normalization_stats(
    path: str | os.PathLike[str], *, expected_source: str
) -> dict[str, Any]:
    """Load and strictly validate one 54-D train-split stats artifact."""

    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(
            f"SparkArena normalization statistics are missing: {source}"
        )
    required = {
        "schema",
        "minimum",
        "maximum",
        "mean",
        "std",
        "scale",
        "count",
        "order",
        "source",
        "normalization",
        "epsilon",
    }
    with np.load(source, allow_pickle=False) as archive:
        missing = sorted(required.difference(archive.files))
        if missing:
            raise ValueError(f"{source} is missing stats fields: {missing}")
        arrays = {
            name: np.asarray(archive[name], dtype=np.float32).reshape(-1)
            for name in ("minimum", "maximum", "mean", "std", "scale")
        }
        stats_schema = _scalar_text(archive["schema"], name="schema", path=source)
        count_array = np.asarray(archive["count"])
        epsilon_array = np.asarray(archive["epsilon"])
        order_array = np.asarray(archive["order"])
        stats_source = _scalar_text(archive["source"], name="source", path=source)
        normalization = _scalar_text(
            archive["normalization"], name="normalization", path=source
        )

    for name, array in arrays.items():
        if array.shape != (LOGICAL_DIM,) or not np.isfinite(array).all():
            raise ValueError(f"{source}:{name} must be finite and exactly ({LOGICAL_DIM},)")
    if np.any(arrays["minimum"] > arrays["maximum"]):
        raise ValueError(f"{source} contains minimum greater than maximum")
    expected_scale = np.maximum(
        arrays["maximum"] - arrays["minimum"], NORMALIZATION_EPSILON
    )
    if np.any(arrays["scale"] <= 0.0) or not np.allclose(
        arrays["scale"], expected_scale, rtol=1e-6, atol=1e-7
    ):
        raise ValueError(f"{source}:scale does not equal max(maximum-minimum, 1e-6)")
    if np.any(arrays["std"] < 0.0):
        raise ValueError(f"{source}:std contains negative values")
    if count_array.size != 1 or int(count_array.reshape(()).item()) <= 0:
        raise ValueError(f"{source}:count must be a positive scalar")
    if epsilon_array.size != 1 or not np.isclose(
        float(epsilon_array.reshape(()).item()),
        NORMALIZATION_EPSILON,
        rtol=0.0,
        atol=1.0e-12,
    ):
        raise ValueError(
            f"{source}:epsilon must be exactly {NORMALIZATION_EPSILON}"
        )
    order = tuple(str(item) for item in order_array.reshape(-1).tolist())
    if order != LOGICAL_ORDER:
        raise ValueError(f"{source}:order differs from the shared 54-D logical order")
    if stats_source != expected_source:
        raise ValueError(
            f"{source}:source must be {expected_source!r}, got {stats_source!r}"
        )
    if normalization != NORMALIZATION_NAME:
        raise ValueError(
            f"{source}:normalization must be {NORMALIZATION_NAME!r}"
        )
    if stats_schema != STATS_SCHEMA:
        raise ValueError(f"{source}:schema must be {STATS_SCHEMA!r}")
    return {
        **arrays,
        "schema": stats_schema,
        "count": int(count_array.reshape(()).item()),
        "epsilon": float(epsilon_array.reshape(()).item()),
        "order": order,
        "source": stats_source,
        "normalization": normalization,
        "path": source,
    }


def _model_inventory(root: Path) -> dict[str, dict[str, Any]]:
    if not (root / "config.json").is_file():
        raise FileNotFoundError(f"model config is missing: {root / 'config.json'}")
    files = [root / "config.json"]
    for component in MODEL_COMPONENTS:
        directory = root / component
        if not directory.is_dir():
            raise FileNotFoundError(f"model component is missing: {directory}")
        component_files = sorted(path for path in directory.rglob("*") if path.is_file())
        if not component_files:
            raise ValueError(f"model component is empty: {directory}")
        if not any(
            path.suffix.lower() in {".safetensors", ".bin", ".pt", ".pth"}
            for path in component_files
        ):
            raise ValueError(f"model component has no weight artifact: {directory}")
        files.extend(component_files)
    return {
        path.relative_to(root).as_posix(): _record(path)
        for path in sorted(set(files))
    }


def _checkpoint_step(root: Path) -> int:
    prefix = "checkpoint-"
    if not root.name.startswith(prefix) or not root.name[len(prefix) :].isdigit():
        raise ValueError(
            "SparkArena resumable checkpoints must be named checkpoint-<global_step>: "
            f"{root}"
        )
    step = int(root.name[len(prefix) :])
    if step <= 0:
        raise ValueError(f"SparkArena checkpoint step must be positive: {root}")
    return step


def _training_state_inventory(root: Path) -> tuple[int, dict[str, dict[str, Any]]]:
    """Bind the HF state required to resume all eight ranks of this run."""

    step = _checkpoint_step(root)
    paths = [root / name for name in TRAINING_STATE_FILENAMES]
    missing = [path.name for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "SparkArena checkpoint is not resumable; missing Trainer state: "
            + ", ".join(missing)
        )
    empty = [path.name for path in paths if path.stat().st_size <= 0]
    if empty:
        raise ValueError(
            "SparkArena checkpoint contains empty Trainer state: " + ", ".join(empty)
        )

    trainer_state = _json(root / TRAINER_STATE_FILENAME)
    state_step = trainer_state.get("global_step")
    if not isinstance(state_step, int) or isinstance(state_step, bool) or state_step != step:
        raise ValueError(
            f"{TRAINER_STATE_FILENAME} global_step={state_step!r} does not match {root.name}"
        )
    return step, {path.name: _record(path) for path in paths}

def _validate_model_abi(root: Path) -> dict[str, Any]:
    config = _json(root / "config.json")
    expected = {
        "traj_decoder_type": "transformer_split_action_v2",
        "proprio_size": LOGICAL_DIM,
        "use_proprio": True,
        "sep_proprio": False,
        "sep_query_token": True,
        "action_output_dim": MODEL_OUTPUT_DIM,
        "egovla_joint_schema": "egovla_sparkarena_raw_joint_model_v1",
        "egovla_benchmark": "SparkArena",
        "egovla_robot_key": ROBOT_KEY,
    }
    mismatches = [
        f"{key}={config.get(key)!r}, expected {value!r}"
        for key, value in expected.items()
        if config.get(key) != value
    ]
    if mismatches:
        raise ValueError(
            "SparkArena checkpoint model ABI mismatch: " + "; ".join(mismatches)
        )

    vision = _json(root / "vision_tower" / "config.json")
    if vision.get("image_size") != PROCESSED_IMAGE_HEIGHT:
        raise ValueError("SparkArena checkpoint vision tower must use image_size=384")
    if int(vision.get("num_channels", 3)) != 3:
        raise ValueError(
            "SparkArena checkpoint vision tower must consume exactly three RGB channels"
        )
    return config


def _verify_data_provenance(data_dir: Path, provenance: Mapping[str, Any]) -> None:
    if provenance.get("format") != DATA_PROVENANCE_SCHEMA:
        raise ValueError(
            "data provenance format differs from the strict SparkArena converter"
        )
    if provenance.get("prompt_mapping_sha256") != TASK_INSTRUCTIONS_SHA256:
        raise ValueError("data provenance prompt mapping differs from runtime")
    artifacts = provenance.get("artifact_sha256")
    if not isinstance(artifacts, Mapping):
        raise ValueError("data provenance must contain artifact_sha256")
    # A provenance document cannot contain its own content hash.  It binds
    # every source artifact; the checkpoint sidecar separately binds the
    # provenance document itself after copying it.
    for name in (
        "train.jsonl",
        "val.jsonl",
        STATE_STATS_FILENAME,
        ACTION_STATS_FILENAME,
        "metadata.json",
    ):
        path = data_dir / name
        digest = artifacts.get(name)
        if not isinstance(digest, str) or len(digest) != 64:
            raise ValueError(f"data provenance does not bind {name}")
        if not path.is_file() or sha256_file(path) != digest:
            raise ValueError(f"data artifact hash mismatch: {path}")

    metadata = _json(data_dir / "metadata.json")
    if metadata.get("format") != DATA_METADATA_SCHEMA:
        raise ValueError("unsupported SparkArena manifest format")
    expected_contract = contract_dict()
    expected_fields = {
        "benchmark": "SparkArena",
        "robot": ROBOT_KEY,
        "state_dim": LOGICAL_DIM,
        "action_dim": LOGICAL_DIM,
        "image_preprocessing": expected_contract["geometry"],
    }
    mismatches = [
        key
        for key, expected in expected_fields.items()
        if canonical_json(metadata.get(key)) != canonical_json(expected)
    ]
    if mismatches:
        raise ValueError(
            "SparkArena metadata differs from runtime contract: "
            + ", ".join(mismatches)
        )
    action_policy = metadata.get("action_policy")
    if not isinstance(action_policy, Mapping):
        raise ValueError("SparkArena metadata is missing action_policy")
    if (
        action_policy.get("temporal_shift") != 0
        or "state[t+1]" not in action_policy.get("forbidden_sources", [])
        or "/mano/action" not in action_policy.get("forbidden_sources", [])
    ):
        raise ValueError("SparkArena metadata does not enforce original action[t]")


def _checkpoint_root(value: str | os.PathLike[str]) -> Path:
    root = Path(value).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"SparkArena checkpoint directory does not exist: {root}")
    if (root / PROVENANCE_FILENAME).is_file():
        return root
    candidates = sorted(
        path
        for path in root.glob("checkpoint-*")
        if path.is_dir() and (path / PROVENANCE_FILENAME).is_file()
    )
    if len(candidates) != 1:
        raise ValueError(
            "SparkArena checkpoint selector must identify exactly one finalized checkpoint; "
            f"found {len(candidates)} below {root}"
        )
    return candidates[0]


def finalize_sparkarena_checkpoint(
    checkpoint: str | os.PathLike[str],
    data_dir: str | os.PathLike[str],
    pretrained_path: str | os.PathLike[str],
    run_contract: Mapping[str, Any] | None = None,
) -> Path:
    """Finalize one SparkArena checkpoint with immutable sidecars.

    The provenance file is published last and acts as the completion marker.
    Repeated calls are idempotent: an already finalized checkpoint is verified
    and returned without rewriting any artifact.
    """

    root = Path(checkpoint).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"SparkArena checkpoint directory does not exist: {root}")
    if (root / PROVENANCE_FILENAME).exists():
        verify_sparkarena_checkpoint(root)
        return root

    source = Path(data_dir).expanduser().resolve()
    pretrained = Path(pretrained_path).expanduser().resolve()
    if not source.is_dir():
        raise FileNotFoundError(f"SparkArena data directory does not exist: {source}")
    if not pretrained.is_dir():
        raise FileNotFoundError(f"pretrained model directory does not exist: {pretrained}")

    config = _validate_model_abi(root)
    checkpoint_step, training_state_artifacts = _training_state_inventory(root)
    state_stats = load_normalization_stats(
        source / STATE_STATS_FILENAME, expected_source="/state/*"
    )
    action_stats = load_normalization_stats(
        source / ACTION_STATS_FILENAME, expected_source="/action/*"
    )
    del state_stats, action_stats

    data_provenance = _json(source / "sparkarena_provenance.json")
    _verify_data_provenance(source, data_provenance)
    _json(source / "metadata.json")
    normalized_run_contract = _normalise_run_contract(run_contract)
    expected_contract = contract_dict()
    contract_hash = hashlib.sha256(canonical_json(expected_contract)).hexdigest()

    _atomic_json(root / CONTRACT_FILENAME, expected_contract)
    for source_name, destination_name in DATA_SOURCE_FILES.items():
        _atomic_copy(source / source_name, root / destination_name)

    copied_artifacts: dict[str, dict[str, Any]] = {
        CONTRACT_FILENAME: {
            **_record(root / CONTRACT_FILENAME),
            "role": "sparkarena_runtime_contract",
        }
    }
    for source_name, destination_name in DATA_SOURCE_FILES.items():
        copied_artifacts[destination_name] = {
            **_record(root / destination_name),
            "role": f"copied_from_data_dir:{source_name}",
        }

    sidecar = {
        "schema": PROVENANCE_SCHEMA,
        "contract": expected_contract,
        "contract_sha256": contract_hash,
        "robot_key": ROBOT_KEY,
        "action_type": ACTION_TYPE,
        "logical_action_dim": LOGICAL_DIM,
        "model_output_dim": MODEL_OUTPUT_DIM,
        "model_output_indices": list(MODEL_OUTPUT_INDICES),
        "task_instructions_sha256": TASK_INSTRUCTIONS_SHA256,
        "run_contract": normalized_run_contract,
        "data": {
            "source_dir": str(source),
            "source_provenance_sha256": sha256_file(
                source / "sparkarena_provenance.json"
            ),
            "artifacts": copied_artifacts,
        },
        "pretrained": {
            "resolved_path": str(pretrained),
            "repository": EXPECTED_PRETRAINED_REPOSITORY,
            "commit": EXPECTED_PRETRAINED_COMMIT,
            "model_artifacts": _model_inventory(pretrained),
            "reinitialized_parameter_prefixes": normalized_run_contract[
                "reinitialized_parameter_prefixes"
            ],
            "reused_parameter_prefixes": normalized_run_contract[
                "reused_parameter_prefixes"
            ],
        },
        "checkpoint": {
            "global_step": checkpoint_step,
            "world_size": TRAINING_WORLD_SIZE,
            "config_abi": {
                key: config[key]
                for key in (
                    "traj_decoder_type",
                    "proprio_size",
                    "use_proprio",
                    "sep_proprio",
                    "action_output_dim",
                )
            },
            "model_artifacts": _model_inventory(root),
            "training_state_artifacts": training_state_artifacts,
        },
    }
    _reject_sensitive(sidecar, location="checkpoint_provenance")
    _atomic_json(root / PROVENANCE_FILENAME, sidecar)
    return root


def verify_sparkarena_checkpoint(
    checkpoint: str | os.PathLike[str],
) -> dict[str, Any]:
    """Verify and load a finalized SparkArena raw-joint checkpoint.

    No fallback to an EE/MANO or un-finalized checkpoint is permitted.
    """

    root = _checkpoint_root(checkpoint)
    provenance = _json(root / PROVENANCE_FILENAME)
    _reject_sensitive(provenance, location="checkpoint_provenance")
    if provenance.get("schema") != PROVENANCE_SCHEMA:
        raise ValueError(
            "unsupported SparkArena checkpoint provenance schema: "
            f"{provenance.get('schema')!r}"
        )

    expected_contract = contract_dict()
    expected_hash = hashlib.sha256(canonical_json(expected_contract)).hexdigest()
    if provenance.get("contract_sha256") != expected_hash:
        raise ValueError("SparkArena checkpoint contract hash differs from runtime contract")
    if canonical_json(provenance.get("contract")) != canonical_json(expected_contract):
        raise ValueError("SparkArena checkpoint embeds a different runtime contract")
    contract = _json(root / CONTRACT_FILENAME)
    if canonical_json(contract) != canonical_json(expected_contract):
        raise ValueError("egovla_contract.json differs from the shared SparkArena contract")
    if provenance.get("task_instructions_sha256") != TASK_INSTRUCTIONS_SHA256:
        raise ValueError("checkpoint task instruction mapping hash differs from runtime")
    if provenance.get("robot_key") != ROBOT_KEY or provenance.get("action_type") != ACTION_TYPE:
        raise ValueError(
            "checkpoint robot/action contract is not tianji_marvin_wuji/joint"
        )
    if provenance.get("model_output_indices") != list(MODEL_OUTPUT_INDICES):
        raise ValueError(
            "checkpoint decoder output indices differ from process_data contract"
        )

    run_contract = _normalise_run_contract(provenance.get("run_contract"))
    pretrained = provenance.get("pretrained")
    if not isinstance(pretrained, Mapping):
        raise ValueError("checkpoint provenance is missing pretrained lineage")
    if (
        pretrained.get("repository") != EXPECTED_PRETRAINED_REPOSITORY
        or pretrained.get("commit") != EXPECTED_PRETRAINED_COMMIT
    ):
        raise ValueError("checkpoint provenance has an unpinned pretrained lineage")
    prefixes = pretrained.get("reinitialized_parameter_prefixes")
    if prefixes != run_contract["reinitialized_parameter_prefixes"]:
        raise ValueError("checkpoint reinitialized prefixes differ across sidecar sections")
    model_artifacts = pretrained.get("model_artifacts")
    if not isinstance(model_artifacts, Mapping) or not model_artifacts:
        raise ValueError("checkpoint provenance does not bind pretrained artifacts")
    for relative, record in model_artifacts.items():
        if not isinstance(relative, str) or not isinstance(record, Mapping):
            raise ValueError("invalid pretrained artifact inventory entry")
        digest = record.get("sha256")
        size = record.get("bytes")
        if not isinstance(digest, str) or len(digest) != 64 or not isinstance(size, int):
            raise ValueError("invalid pretrained artifact hash record")

    data = provenance.get("data")
    if not isinstance(data, Mapping) or not isinstance(data.get("artifacts"), Mapping):
        raise ValueError("checkpoint provenance is missing copied data artifacts")
    copied_artifacts = data["artifacts"]
    required_copied = {CONTRACT_FILENAME, *DATA_SOURCE_FILES.values()}
    for name in required_copied:
        record = copied_artifacts.get(name)
        if not isinstance(record, Mapping):
            raise ValueError(f"checkpoint provenance does not bind copied artifact {name}")
        _verify_record(root / name, record)

    config = _validate_model_abi(root)
    checkpoint_section = provenance.get("checkpoint")
    if not isinstance(checkpoint_section, Mapping) or not isinstance(
        checkpoint_section.get("model_artifacts"), Mapping
    ):
        raise ValueError("checkpoint provenance does not bind model artifacts")
    checkpoint_step = _checkpoint_step(root)
    expected_training_names = set(TRAINING_STATE_FILENAMES)
    if (
        checkpoint_section.get("global_step") != checkpoint_step
        or checkpoint_section.get("world_size") != TRAINING_WORLD_SIZE
    ):
        raise ValueError("checkpoint provenance has an incompatible step/world size")
    recorded_training_state = checkpoint_section.get("training_state_artifacts")
    if not isinstance(recorded_training_state, Mapping):
        raise ValueError("checkpoint provenance does not bind Trainer resume state")
    if set(recorded_training_state) != expected_training_names:
        raise ValueError("checkpoint Trainer state artifact set differs from provenance")
    # Validate the HF layout and trainer_state/global-step relation before hashing
    # every artifact.  This rejects pre-v2, model-only completion markers.
    current_step, current_training_state = _training_state_inventory(root)
    if current_step != checkpoint_step:
        raise ValueError("checkpoint Trainer state step differs from its directory")
    for relative, record in recorded_training_state.items():
        if not isinstance(relative, str) or not isinstance(record, Mapping):
            raise ValueError(f"invalid checkpoint Trainer state record: {relative}")
        if dict(record) != current_training_state[relative]:
            raise ValueError(f"checkpoint Trainer state hash/size mismatch: {root / relative}")
    recorded_model = checkpoint_section["model_artifacts"]
    current_model = _model_inventory(root)
    if set(recorded_model) != set(current_model):
        raise ValueError("checkpoint model artifact set differs from provenance")
    for relative, record in recorded_model.items():
        if not isinstance(record, Mapping):
            raise ValueError(f"invalid checkpoint model artifact record: {relative}")
        _verify_record(root / relative, record)

    state_stats = load_normalization_stats(
        root / STATE_STATS_FILENAME, expected_source="/state/*"
    )
    action_stats = load_normalization_stats(
        root / ACTION_STATS_FILENAME, expected_source="/action/*"
    )
    data_provenance = _json(root / DATA_PROVENANCE_FILENAME)
    if data.get("source_provenance_sha256") != sha256_file(
        root / DATA_PROVENANCE_FILENAME
    ):
        raise ValueError("copied data provenance hash differs from checkpoint sidecar")
    if data_provenance.get("format") != DATA_PROVENANCE_SCHEMA:
        raise ValueError("copied data provenance has a different format")
    if data_provenance.get("prompt_mapping_sha256") != TASK_INSTRUCTIONS_SHA256:
        raise ValueError("copied data provenance has a different prompt mapping")
    metadata = _json(root / DATA_METADATA_FILENAME)
    if (
        metadata.get("format") != DATA_METADATA_SCHEMA
        or metadata.get("benchmark") != "SparkArena"
        or metadata.get("robot") != ROBOT_KEY
        or metadata.get("state_dim") != LOGICAL_DIM
        or metadata.get("action_dim") != LOGICAL_DIM
        or canonical_json(metadata.get("image_preprocessing"))
        != canonical_json(expected_contract["geometry"])
    ):
        raise ValueError("copied SparkArena metadata differs from runtime contract")

    return {
        "root": root,
        "provenance": provenance,
        "contract": contract,
        "run_contract": run_contract,
        "config": config,
        "state_stats": state_stats,
        "action_stats": action_stats,
    }


__all__ = [
    "ACTION_STATS_FILENAME",
    "CONTRACT_FILENAME",
    "DATA_METADATA_FILENAME",
    "DATA_PROVENANCE_SCHEMA",
    "DATA_PROVENANCE_FILENAME",
    "EXPECTED_PRETRAINED_COMMIT",
    "EXPECTED_PRETRAINED_REPOSITORY",
    "NORMALIZATION_NAME",
    "NORMALIZATION_EPSILON",
    "PROVENANCE_FILENAME",
    "PROVENANCE_SCHEMA",
    "RNG_STATE_FILENAMES",
    "SCHEDULER_FILENAME",
    "STATS_SCHEMA",
    "STATE_STATS_FILENAME",
    "TRAINER_STATE_FILENAME",
    "TRAINING_STATE_FILENAMES",
    "TRAINING_WORLD_SIZE",
    "OPTIMIZER_FILENAME",
    "finalize_sparkarena_checkpoint",
    "load_normalization_stats",
    "sha256_file",
    "verify_sparkarena_checkpoint",
]
