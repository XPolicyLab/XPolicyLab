"""Strict raw-joint training entry point for EgoVLA on SparkArena.

This adapter deliberately leaves the released EE/MANO entry points untouched.
It transfers the compatible vision, language, projector, and trajectory
Transformer weights, but gives the Tianji Marvin Wuji policy a new 54-D
proprioception projection and two 27-D joint output projections.

Labels are supplied by ``dataset_sparkarena`` and originate exclusively from
the four HDF5 ``action/*`` joint datasets.  No next-state or MANO field can
become an action target.
"""

from __future__ import annotations

import functools
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Mapping


TRAIN_DIR = Path(__file__).resolve().parent
RELEASE_DIR = Path(__file__).resolve().parents[2]
POLICY_DIR = RELEASE_DIR.parent
XPL_ROOT = POLICY_DIR.parents[1]
DATA_PREP_DIR = RELEASE_DIR / "human_plan" / "dataset_preprocessing" / "sparkarena"
UTILS_DIR = RELEASE_DIR / "human_plan" / "utils"
UPSTREAM_ROOT = Path(
    os.environ.get("EGOVLA_UPSTREAM_ROOT", RELEASE_DIR)
).expanduser().resolve()
for path in (TRAIN_DIR, DATA_PREP_DIR, UTILS_DIR, RELEASE_DIR, POLICY_DIR, XPL_ROOT, UPSTREAM_ROOT / "VILA", UPSTREAM_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

try:
    from human_plan.dataset_preprocessing.sparkarena.process_data import (
        LOGICAL_DIM,
        MODEL_OUTPUT_DIM,
        MODEL_OUTPUT_INDICES,
        TASK_INSTRUCTIONS_SHA256,
        contract_dict,
    )
except ImportError:  # Executed as ``python train_sparkarena_entry.py``.
    from process_data import (
        LOGICAL_DIM,
        MODEL_OUTPUT_DIM,
        MODEL_OUTPUT_INDICES,
        TASK_INSTRUCTIONS_SHA256,
        contract_dict,
    )


JOINT_CONFIG_SCHEMA = "egovla_sparkarena_raw_joint_model_v1"
RUN_CONTRACT_SCHEMA = "egovla_sparkarena_run_contract_v1"
TRANSFER_SCHEMA = "egovla_sparkarena_raw_joint_transfer_v1"
EXPECTED_PRETRAINED_REPO = "rchal97/ego_vla_human_video_pretrained"
EXPECTED_PRETRAINED_COMMIT = "52dea0ceae8754d658d7c3f5fd8aa17c309348ea"
EXPECTED_TRANSFER_MISMATCHED_KEYS = (
    "decoder.proprio_projection.0.weight",
    "decoder.output_projection_left.2.weight",
    "decoder.output_projection_left.2.bias",
    "decoder.output_projection_right.2.weight",
    "decoder.output_projection_right.2.bias",
)
REINITIALIZED_PREFIXES = (
    "traj_decoder.decoder.proprio_projection",
    "traj_decoder.decoder.output_projection_left",
    "traj_decoder.decoder.output_projection_right",
)


def _under_module_prefix(name: str, prefixes: tuple[str, ...]) -> bool:
    """Match a module itself or its children, never a same-prefix sibling."""

    return any(name == prefix or name.startswith(prefix + ".") for prefix in prefixes)


def _canonical_json(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    """Create deterministic metadata, refusing semantic replacement."""

    payload = _canonical_json(dict(value))
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != payload:
            raise FileExistsError(f"refusing to replace different metadata: {path}")
        return
    with tempfile.NamedTemporaryFile(
        mode="wb", dir=path.parent, prefix=f".{path.name}.", delete=False
    ) as stream:
        temporary = Path(stream.name)
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _rank_zero() -> bool:
    return int(os.environ.get("RANK", "0")) == 0


def _run_contract_path() -> Path:
    output = os.environ.get("EGOVLA_OUTPUT_DIR")
    if not output:
        raise ValueError("EGOVLA_OUTPUT_DIR must be exported by train.sh")
    return Path(output).expanduser().resolve() / "egovla_sparkarena_run_contract.json"


def _run_contract_with_transfer() -> dict[str, Any]:
    run_contract = json.loads(_run_contract_path().read_text(encoding="utf-8"))
    transfer_path = (
        _run_contract_path().parent / "egovla_sparkarena_transfer_manifest.json"
    )
    if not transfer_path.is_file():
        raise FileNotFoundError(f"joint transfer manifest is missing: {transfer_path}")
    run_contract["transfer"] = json.loads(transfer_path.read_text(encoding="utf-8"))
    return run_contract


def _write_run_contract() -> None:
    if not _rank_zero():
        return
    data_dir = Path(os.environ["EGOVLA_DATA_DIR"]).expanduser().resolve()
    pretrained = Path(os.environ["EGOVLA_PRETRAINED_PATH"]).expanduser().resolve()
    source_files = (
        DATA_PREP_DIR / "process_data.py",
        RELEASE_DIR / "VILA" / "llava" / "data" / "dataset_sparkarena.py",
        TRAIN_DIR / "train_sparkarena_entry.py",
        POLICY_DIR / "train.sh",
        UPSTREAM_ROOT / "human_plan" / "preprocessing" / "prompting_format.py",
        UPSTREAM_ROOT / "human_plan" / "preprocessing" / "preprocessing.py",
        UPSTREAM_ROOT / "VILA" / "llava" / "conversation.py",
        UPSTREAM_ROOT
        / "VILA"
        / "llava"
        / "model"
        / "ego_vla_decoder"
        / "transformer.py",
    )
    missing = [str(path) for path in source_files if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"SparkArena training source is incomplete: {missing}")
    value = {
        "schema": RUN_CONTRACT_SCHEMA,
        "run_name": os.environ["RUN_NAME"],
        "benchmark": "SparkArena",
        "robot_key": "tianji_marvin_wuji",
        "action_type": "joint",
        "label_source": [
            "action/left_arm_joint_states",
            "action/left_ee_joint_states",
            "action/right_arm_joint_states",
            "action/right_ee_joint_states",
        ],
        "state_source": [
            "state/left_arm_joint_states",
            "state/left_ee_joint_states",
            "state/right_arm_joint_states",
            "state/right_ee_joint_states",
        ],
        "target_alignment": "observation[t] -> action[t]; future k -> action[t+k]",
        "forbid_next_state_as_action": True,
        "data_dir": str(data_dir),
        "pretrained_path": str(pretrained),
        "pretrained_repo": EXPECTED_PRETRAINED_REPO,
        "pretrained_commit": EXPECTED_PRETRAINED_COMMIT,
        "global_batch_size": int(os.environ.get("EGOVLA_GLOBAL_BATCH_SIZE", "64")),
        "per_device_batch_size": int(os.environ.get("EGOVLA_PER_DEVICE_BATCH", "4")),
        "gradient_accumulation_steps": int(os.environ.get("EGOVLA_GRAD_ACCUM", "2")),
        "world_size": 8,
        "max_steps": int(os.environ.get("EGOVLA_MAX_STEPS", "80000")),
        "save_steps": int(os.environ.get("EGOVLA_SAVE_STEPS", "10000")),
        "seed": int(os.environ["EGOVLA_SEED"]),
        "predict_future_step": 30,
        "history_steps": 5,
        "history_stride": 5,
        "prompt_version": "v0",
        "conversation_template": "vicuna_v1",
        "runtime_camera_key": "cam_head",
        "raw_input_resolution": [480, 640],
        "geometry": "full_frame_resize_480x640_to_384x384",
        "input_resolution": [384, 384],
        "reverse_channel_order": False,
        "state_normalization": "minmax_to_unit_interval_no_clip",
        "action_denormalization": "prediction*scale+minimum; no clipping",
        "loss": "mean absolute error over exactly 54 normalized raw-action joints",
        "model_output_dim": MODEL_OUTPUT_DIM,
        "model_output_indices": list(MODEL_OUTPUT_INDICES),
        "reinitialized_parameter_prefixes": list(REINITIALIZED_PREFIXES),
        "prompt_instruction_sha256": TASK_INSTRUCTIONS_SHA256,
        "camera": contract_dict()["camera_mode"],
        "color_order": contract_dict()["color_order"],
        "processed_image_hw": contract_dict()["processed_image_hw"],
        "source_sha256": {
            path.name: _sha256_file(path) for path in source_files
        },
    }
    _atomic_json(_run_contract_path(), value)


def _patch_optional_attention() -> None:
    try:
        from .train_entry import _patch_optional_attention as patch
    except ImportError:
        from train_entry import _patch_optional_attention as patch
    patch()


def _patch_training_config(train_module: Any) -> None:
    """Force and serialize the raw-joint ABI before model construction."""

    import llava.train.utils as train_utils

    original = getattr(
        train_utils.prepare_config_for_training,
        "_egovla_joint_original",
        train_utils.prepare_config_for_training,
    )

    @functools.wraps(original)
    def strict_prepare(config: Any, model_args: Any, training_args: Any, data_args: Any) -> None:
        was_joint = getattr(config, "egovla_joint_schema", None) == JOINT_CONFIG_SCHEMA
        if was_joint:
            if list(getattr(config, "egovla_model_output_indices", [])) != list(
                MODEL_OUTPUT_INDICES
            ):
                raise ValueError("resume checkpoint has a different joint output mapping")
            if int(getattr(config, "proprio_size", -1)) != LOGICAL_DIM:
                raise ValueError(
                    f"resume checkpoint does not have {LOGICAL_DIM}-D joint proprioception"
                )

        if int(model_args.proprio_size) != LOGICAL_DIM:
            raise ValueError(f"joint training requires --proprio_size {LOGICAL_DIM}")
        if bool(model_args.sep_proprio):
            raise ValueError("joint training requires --sep_proprio False")
        if not bool(model_args.use_proprio):
            raise ValueError("joint training requires --use_proprio True")
        if not bool(model_args.sep_query_token):
            raise ValueError("joint training requires --sep_query_token True")
        if int(model_args.traj_action_output_dim) != MODEL_OUTPUT_DIM:
            raise ValueError(
                f"split joint decoder must retain its physical {MODEL_OUTPUT_DIM}-D output"
            )

        original(config, model_args, training_args, data_args)
        config.proprio_size = LOGICAL_DIM
        config.use_proprio = True
        config.sep_proprio = False
        config.sep_query_token = True
        config.action_output_dim = MODEL_OUTPUT_DIM
        config.traj_decoder_type = "transformer_split_action_v2"
        config.egovla_joint_schema = JOINT_CONFIG_SCHEMA
        config.egovla_action_type = "joint"
        config.egovla_benchmark = "SparkArena"
        config.egovla_robot_key = "tianji_marvin_wuji"
        config.egovla_action_source = "action/* joint datasets"
        config.egovla_state_source = "state/* joint datasets"
        config.egovla_target_alignment = "observation[t] -> action[t]"
        config.egovla_logical_action_dim = LOGICAL_DIM
        config.egovla_model_output_indices = list(MODEL_OUTPUT_INDICES)
        config.egovla_reinitialized_parameter_prefixes = list(REINITIALIZED_PREFIXES)
        config.egovla_initialize_joint_io = not was_joint
        config.egovla_prompt_instruction_sha256 = TASK_INSTRUCTIONS_SHA256

    strict_prepare._egovla_joint_original = original
    train_utils.prepare_config_for_training = strict_prepare
    # The release imports the function into this module's globals.
    train_module.prepare_config_for_training = strict_prepare


def _torch_dtype(torch: Any, value: Any) -> Any:
    name = str(value).replace("torch.", "")
    dtype = getattr(torch, name, None)
    if dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise ValueError(f"unsupported model dtype: {value!r}")
    return dtype


def _reset_linear_modules(module: Any) -> None:
    import torch

    for child in module.modules():
        if isinstance(child, torch.nn.Linear):
            child.reset_parameters()


def _loading_key(item: Any) -> str:
    if isinstance(item, str):
        return item
    if isinstance(item, (tuple, list)) and item:
        return str(item[0])
    raise ValueError(f"invalid from_pretrained loading-info item: {item!r}")


def _validate_loading_info(info: Mapping[str, Any], *, initial_transfer: bool) -> dict[str, Any]:
    """Fail closed unless only the known 54-D joint IO tensors mismatch."""

    expected = set(EXPECTED_TRANSFER_MISMATCHED_KEYS if initial_transfer else ())
    mismatched = {_loading_key(item) for item in info.get("mismatched_keys", ())}
    missing = {_loading_key(item) for item in info.get("missing_keys", ())}
    unexpected = {_loading_key(item) for item in info.get("unexpected_keys", ())}
    errors = [str(item) for item in info.get("error_msgs", ()) if str(item)]
    # Transformers versions differ on whether a shape-mismatched tensor is
    # repeated in missing_keys.  It is allowed there only when whitelisted.
    unapproved_missing = missing.difference(expected)
    if mismatched != expected or unapproved_missing or unexpected or errors:
        raise ValueError(
            "unexpected pretrained trajectory-decoder loading result: "
            f"mismatched={sorted(mismatched)}, missing={sorted(missing)}, "
            f"unexpected={sorted(unexpected)}, errors={errors}"
        )
    return {
        "mismatched_keys": sorted(mismatched),
        "missing_keys": sorted(missing),
        "unexpected_keys": sorted(unexpected),
        "error_messages": errors,
    }


def _write_transfer_manifest(
    decoder: Any,
    source: Path,
    reinitialized: bool,
    loading_info: Mapping[str, Any],
) -> None:
    if not _rank_zero():
        return
    names = sorted(decoder.state_dict().keys())
    local_prefixes = tuple(prefix.removeprefix("traj_decoder.") for prefix in REINITIALIZED_PREFIXES)
    reset_names = [name for name in names if _under_module_prefix(name, local_prefixes)]
    loaded_names = [name for name in names if name not in set(reset_names)]
    value = {
        "schema": TRANSFER_SCHEMA,
        "source_traj_decoder": str(source.resolve()),
        "source_model_sha256": _sha256_file(source / "model.safetensors"),
        "initial_transfer": bool(reinitialized),
        "loaded_tensor_names": loaded_names,
        "reinitialized_tensor_names": reset_names,
        "reinitialized_parameter_prefixes": list(REINITIALIZED_PREFIXES),
        "from_pretrained_loading_info": dict(loading_info),
        "note": (
            "All non-listed trajectory tensors were loaded from the pretrained decoder. "
            "The 54-D proprio and both 27-D joint projections were reset because "
            "the pretrained tensors encode incompatible EE/MANO semantics."
        ),
    }
    path = _run_contract_path().parent / "egovla_sparkarena_transfer_manifest.json"
    if path.exists() and not reinitialized:
        return
    _atomic_json(path, value)


def _patch_decoder_builder() -> None:
    """Transfer compatible decoder tensors and reset incompatible joint IO."""

    import torch
    import llava.model.ego_vla_decoder.builder as decoder_builder
    import llava.model.llava_arch as llava_arch
    from llava.model.ego_vla_decoder.traj_decoder import TrajDecoder

    original = getattr(
        decoder_builder.build_traj_decoder,
        "_egovla_joint_original",
        decoder_builder.build_traj_decoder,
    )

    def build_joint(model_type_or_path: str, config: Any) -> Any:
        if getattr(config, "egovla_joint_schema", None) != JOINT_CONFIG_SCHEMA:
            return original(model_type_or_path, config)
        if not model_type_or_path:
            raise ValueError("joint training requires a trajectory decoder checkpoint")
        source = Path(model_type_or_path).expanduser()
        if not source.is_dir() or not (source / "model.safetensors").is_file():
            raise FileNotFoundError(f"trajectory decoder checkpoint is incomplete: {source}")

        pinned_pretrained = (
            Path(os.environ["EGOVLA_PRETRAINED_PATH"]).expanduser().resolve()
            / "traj_decoder"
        )
        initial_transfer = source.resolve() == pinned_pretrained
        if not initial_transfer:
            output_root = Path(os.environ["EGOVLA_OUTPUT_DIR"]).expanduser().resolve()
            checkpoint_root = source.resolve().parent
            if checkpoint_root.parent != output_root or not checkpoint_root.name.startswith(
                "checkpoint-"
            ):
                raise ValueError(
                    "joint decoder source must be either the pinned pretrained decoder "
                    f"or a checkpoint under {output_root}; got {source}"
                )
            if getattr(config, "egovla_joint_schema", None) != JOINT_CONFIG_SCHEMA:
                raise ValueError("resume decoder requires a raw-joint checkpoint config")
        config.egovla_initialize_joint_io = initial_transfer
        decoder, raw_loading_info = TrajDecoder.from_pretrained(
            str(source),
            config,
            torch_dtype=_torch_dtype(torch, config.model_dtype),
            ignore_mismatched_sizes=initial_transfer,
            output_loading_info=True,
        )
        loading_info = _validate_loading_info(
            raw_loading_info, initial_transfer=initial_transfer
        )
        body = decoder.decoder
        if initial_transfer:
            _reset_linear_modules(body.proprio_projection)
            _reset_linear_modules(body.output_projection_left)
            _reset_linear_modules(body.output_projection_right)

        if body.proprio_projection[0].in_features != LOGICAL_DIM:
            raise ValueError(
                f"joint decoder did not construct a {LOGICAL_DIM}-D proprio projection"
            )
        branch_dim = MODEL_OUTPUT_DIM // 2
        if body.output_projection_left[-1].out_features != branch_dim:
            raise ValueError(f"left split decoder branch is not {branch_dim}-D")
        if body.output_projection_right[-1].out_features != branch_dim:
            raise ValueError(f"right split decoder branch is not {branch_dim}-D")
        _write_transfer_manifest(decoder, source, initial_transfer, loading_info)
        print(
            "[egovla-sparkarena] loaded trajectory Transformer from",
            source,
            "; reset joint IO =",
            initial_transfer,
            flush=True,
        )
        return decoder

    build_joint._egovla_joint_original = original
    decoder_builder.build_traj_decoder = build_joint
    # llava_arch binds the function at import time.
    llava_arch.build_traj_decoder = build_joint


def _global_masked_means(*pairs: tuple[Any, Any]) -> tuple[Any, ...]:
    """Compute true element means while compensating for DDP gradient averaging."""

    import torch
    import torch.distributed as dist

    local_sums = [values[mask].float().sum() for values, mask in pairs]
    local_counts = [mask.sum().to(dtype=torch.float32) for _, mask in pairs]
    if any(float(count.detach().cpu()) <= 0 for count in local_counts):
        raise ValueError("joint loss component contains no valid /action targets")
    if not dist.is_available() or not dist.is_initialized():
        return tuple(total / count for total, count in zip(local_sums, local_counts))

    statistics = torch.stack(
        [item for pair in zip(
            [total.detach() for total in local_sums],
            [count.detach() for count in local_counts],
        ) for item in pair]
    )
    dist.all_reduce(statistics, op=dist.ReduceOp.SUM)
    world_size = float(dist.get_world_size())
    results = []
    for index, local_sum in enumerate(local_sums):
        global_sum = statistics[index * 2]
        global_count = statistics[index * 2 + 1]
        if float(global_count.cpu()) <= 0:
            raise ValueError("distributed joint loss component has no valid targets")
        gradient_value = local_sum * (world_size / global_count)
        global_value = global_sum / global_count
        # All ranks report the same global value, while the gradient term is
        # scaled so DDP's subsequent rank average yields global sum/count.
        results.append(gradient_value + (global_value - gradient_value.detach()))
    return tuple(results)


def joint_l1_losses(prediction: Any, raw_labels: Any, raw_masks: Any) -> tuple[Any, Any, Any]:
    """Return total, arm, and hand L1 losses for the exact 54-D contract."""

    import torch

    if prediction.ndim != 2 or prediction.shape[-1] != MODEL_OUTPUT_DIM:
        raise ValueError(f"joint prediction must be [N,{MODEL_OUTPUT_DIM}], got {prediction.shape}")
    if raw_labels.ndim != 2 or raw_labels.shape[-1] != LOGICAL_DIM:
        raise ValueError(
            f"raw joint labels must be [N,{LOGICAL_DIM}], got {raw_labels.shape}"
        )
    if raw_masks.shape != raw_labels.shape:
        raise ValueError("raw joint label/mask shapes differ")
    if prediction.shape[0] != raw_labels.shape[0]:
        raise ValueError(
            f"joint prediction/label row mismatch: {prediction.shape[0]} vs {raw_labels.shape[0]}"
        )

    indices = torch.as_tensor(MODEL_OUTPUT_INDICES, device=prediction.device)
    selected = prediction.index_select(-1, indices)
    target = raw_labels.to(dtype=selected.dtype)
    mask = raw_masks.to(dtype=torch.bool)
    error = torch.abs(selected - target)

    arm_positions = torch.as_tensor(
        list(range(0, 7)) + list(range(27, 34)), device=prediction.device
    )
    hand_positions = torch.as_tensor(
        list(range(7, 27)) + list(range(34, 54)), device=prediction.device
    )
    arm_error = error.index_select(-1, arm_positions)
    arm_mask = mask.index_select(-1, arm_positions)
    hand_error = error.index_select(-1, hand_positions)
    hand_mask = mask.index_select(-1, hand_positions)
    total, arm, hand = _global_masked_means(
        (error, mask),
        (arm_error, arm_mask),
        (hand_error, hand_mask),
    )
    return total, arm, hand


def _patch_resume_selector(train_module: Any) -> None:
    """Resume only the newest fully finalized checkpoint.

    The released selector trusts the numerically newest ``checkpoint-*``
    directory, even if a process died while writing it.  Rank zero verifies all
    candidates and atomically quarantines incomplete directories so a later HF
    save cannot collide with stale files at the same global step.
    """

    import torch.distributed as dist

    original = getattr(
        train_module.get_checkpoint_path,
        "_egovla_sparkarena_original",
        train_module.get_checkpoint_path,
    )

    def quarantine(path: Path) -> Path:
        base = path.parent / f".incomplete-{path.name}"
        destination = base
        suffix = 1
        while destination.exists():
            destination = path.parent / f"{base.name}-{suffix}"
            suffix += 1
        path.replace(destination)
        return destination

    def strict(output_dir: str, checkpoint_prefix: str = "checkpoint") -> tuple[str | None, bool]:
        root = Path(output_dir).expanduser().resolve()
        resume_requested = os.environ.get("EGOVLA_RESUME", "0") == "1"

        candidates: list[tuple[int, Path]] = []
        if root.is_dir():
            prefix = f"{checkpoint_prefix}-"
            for path in root.iterdir():
                suffix = path.name[len(prefix) :] if path.name.startswith(prefix) else ""
                if path.is_dir() and suffix.isdigit():
                    candidates.append((int(suffix), path.resolve()))
        if not candidates:
            if resume_requested:
                raise RuntimeError(
                    "EGOVLA_RESUME=1 was requested but no checkpoint-* directory exists "
                    f"under {root}"
                )
            if (root / "config.json").is_file():
                return str(root), False
            return original(output_dir, checkpoint_prefix)
        if not resume_requested:
            raise RuntimeError(
                f"found checkpoint directories under {root}, but EGOVLA_RESUME=1 was not set"
            )

        distributed = dist.is_available() and dist.is_initialized()
        configured_world = int(os.environ.get("WORLD_SIZE", "1"))
        if configured_world > 1 and not distributed:
            raise RuntimeError(
                "distributed process group must be initialized before SparkArena resume selection"
            )
        rank = dist.get_rank() if distributed else 0
        payload: list[Any] = [None, None, []]
        if rank == 0:
            try:
                from XPolicyLab.policy.EgoVLA.EgoVLA_Release.human_plan.utils.sparkarena_provenance import (
                    verify_sparkarena_checkpoint,
                )

                selected: Path | None = None
                invalid: list[tuple[Path, str]] = []
                for _, path in sorted(candidates, reverse=True):
                    try:
                        verify_sparkarena_checkpoint(path)
                    except Exception as exc:
                        invalid.append((path, f"{type(exc).__name__}: {exc}"))
                    else:
                        selected = path
                        break
                if selected is None:
                    details = "; ".join(f"{path.name}: {reason}" for path, reason in invalid)
                    raise RuntimeError(
                        "no fully finalized SparkArena checkpoint is available for resume; "
                        + details
                    )
                # Only the rejected newer candidates can collide with future
                # global-step saves.  Every move is same-filesystem and fully
                # recoverable by renaming the directory back.
                quarantined = []
                for path, reason in invalid:
                    destination = quarantine(path)
                    quarantined.append(f"{path.name}->{destination.name} ({reason})")
                payload[0] = str(selected)
                payload[2] = quarantined
            except Exception as exc:
                payload[1] = f"{type(exc).__name__}: {exc}"
        if distributed:
            dist.broadcast_object_list(payload, src=0)
        if payload[1] is not None:
            raise RuntimeError(f"SparkArena resume selection failed: {payload[1]}")
        if rank == 0:
            for item in payload[2]:
                print(f"[egovla-sparkarena] quarantined incomplete checkpoint: {item}", flush=True)
            print(
                f"[egovla-sparkarena] resuming newest fully verified checkpoint: {payload[0]}",
                flush=True,
            )
        return str(payload[0]), True

    strict._egovla_sparkarena_original = original
    train_module.get_checkpoint_path = strict


def _patch_checkpoint_finalizer_callback(train_module: Any) -> None:
    """Publish the provenance completion marker after all HF state is durable."""

    import torch.distributed as dist

    original = train_module.AutoResumeCallback
    if getattr(original, "_egovla_sparkarena_finalizer", False):
        return

    class SparkArenaFinalizeAfterSave(original):
        _egovla_sparkarena_finalizer = True

        def on_save(self: Any, args: Any, state: Any, control: Any, **kwargs: Any) -> Any:
            inherited = super().on_save(args, state, control, **kwargs)
            if inherited is not None:
                control = inherited

            distributed = dist.is_available() and dist.is_initialized()
            if distributed:
                # HF writes one rng_state_<rank>.pth per process.  Do not let
                # rank zero publish the completion marker until every file is closed.
                dist.barrier()

            status: list[str | None] = [None]
            rank = dist.get_rank() if distributed else 0
            checkpoint = (
                Path(args.output_dir).expanduser().resolve()
                / f"checkpoint-{int(state.global_step)}"
            )
            if rank == 0:
                try:
                    from XPolicyLab.policy.EgoVLA.EgoVLA_Release.human_plan.utils.sparkarena_provenance import (
                        finalize_sparkarena_checkpoint,
                    )

                    finalize_sparkarena_checkpoint(
                        checkpoint,
                        os.environ["EGOVLA_DATA_DIR"],
                        os.environ["EGOVLA_PRETRAINED_PATH"],
                        run_contract=_run_contract_with_transfer(),
                    )
                except Exception as exc:
                    status[0] = f"{type(exc).__name__}: {exc}"
            if distributed:
                dist.broadcast_object_list(status, src=0)
            if status[0] is not None:
                raise RuntimeError(
                    f"failed to finalize complete SparkArena checkpoint {checkpoint}: {status[0]}"
                )
            if rank == 0:
                print(
                    "[egovla-sparkarena] finalized model + optimizer/scheduler/"
                    f"trainer/RNG checkpoint-{int(state.global_step)}",
                    flush=True,
                )
            return control

    train_module.AutoResumeCallback = SparkArenaFinalizeAfterSave


def _patch_smoke_stop_callback(train_module: Any) -> None:
    """Allow an offline smoke run to stop only after a complete checkpoint save."""

    value = os.environ.get("EGOVLA_SMOKE_STOP_AFTER_SAVE_STEP")
    if value is None:
        return
    stop_step = int(value)
    if stop_step <= 0:
        raise ValueError("EGOVLA_SMOKE_STOP_AFTER_SAVE_STEP must be positive")
    original = train_module.AutoResumeCallback

    class SparkArenaSmokeStopAfterSave(original):
        def on_save(self: Any, args: Any, state: Any, control: Any, **kwargs: Any) -> Any:
            # The parent callback first finalizes and hash-binds all HF resume
            # state.  Smoke termination is only allowed after that succeeds.
            inherited = super().on_save(args, state, control, **kwargs)
            if inherited is not None:
                control = inherited
            if int(state.global_step) >= stop_step:
                if state.is_world_process_zero:
                    print(
                        "[egovla-sparkarena] smoke stopping after complete "
                        f"checkpoint-{state.global_step}",
                        flush=True,
                    )
                control.should_training_stop = True
            return control

    train_module.AutoResumeCallback = SparkArenaSmokeStopAfterSave


def _patch_split_checkpoint_resume() -> None:
    """Accept verified VILA split-component checkpoints during Trainer resume.

    The upstream model constructor has already loaded llm, vision tower,
    projector, and trajectory decoder from the selected checkpoint.  The base
    Transformers loader only recognizes a monolithic root weight file, so for
    this verified joint ABI we skip that duplicate model load while leaving
    Trainer's optimizer, scheduler, state, scaler, and RNG restoration intact.
    """

    import torch.distributed as dist
    from llava.train.llava_trainer import LLaVATrainer

    original = getattr(
        LLaVATrainer._load_from_checkpoint,
        "_egovla_joint_original",
        LLaVATrainer._load_from_checkpoint,
    )
    original_rng = getattr(
        LLaVATrainer._load_rng_state,
        "_egovla_joint_original",
        LLaVATrainer._load_rng_state,
    )

    def compatible(self: Any, resume_from_checkpoint: Any, model: Any = None) -> Any:
        root = Path(str(resume_from_checkpoint)).expanduser().resolve()
        if (root / "egovla_sparkarena_provenance.json").is_file():
            status: list[str | None] = [None]
            should_verify = not dist.is_available() or not dist.is_initialized() or dist.get_rank() == 0
            if should_verify:
                try:
                    from XPolicyLab.policy.EgoVLA.EgoVLA_Release.human_plan.utils.sparkarena_provenance import (
                        verify_sparkarena_checkpoint,
                    )

                    verify_sparkarena_checkpoint(root)
                except Exception as exc:  # Broadcast a deterministic failure to every rank.
                    status[0] = f"{type(exc).__name__}: {exc}"
            if dist.is_available() and dist.is_initialized():
                dist.broadcast_object_list(status, src=0)
            if status[0] is not None:
                raise ValueError(
                    f"refusing invalid joint split checkpoint {root}: {status[0]}"
                )
            if not getattr(self.model.config, "egovla_joint_schema", None) == JOINT_CONFIG_SCHEMA:
                raise ValueError("split checkpoint was loaded into a non-joint model")
            if self.is_world_process_zero():
                print(
                    "[egovla-sparkarena] verified split model components; "
                    f"Trainer will restore optimizer/scheduler/state/RNG from {root}",
                    flush=True,
                )
            self._egovla_joint_verified_resume = str(root)
            return None
        return original(self, resume_from_checkpoint, model=model)

    def compatible_rng(self: Any, checkpoint: Any) -> Any:
        root = Path(str(checkpoint)).expanduser().resolve()
        if getattr(self, "_egovla_joint_verified_resume", None) == str(root):
            # PyTorch 2.6 changed torch.load's default to weights_only=True,
            # but the Transformers RNG file legitimately contains Python and
            # NumPy RNG tuples.  This checkpoint was fully hash-verified above.
            import torch
            from unittest import mock

            torch_load = torch.load

            def trusted_load(*args: Any, **kwargs: Any) -> Any:
                kwargs["weights_only"] = False
                return torch_load(*args, **kwargs)

            with mock.patch.object(torch, "load", new=trusted_load):
                return original_rng(self, checkpoint)
        return original_rng(self, checkpoint)

    compatible._egovla_joint_original = original
    compatible_rng._egovla_joint_original = original_rng
    LLaVATrainer._load_from_checkpoint = compatible
    LLaVATrainer._load_rng_state = compatible_rng


def _patch_joint_forward() -> None:
    """Replace the released EE/MANO loss with strict raw-action joint L1."""

    import torch
    from llava.model.language_model.llava_llama import (
        HOILMOutputWithPast,
        LlavaLlamaModel,
    )

    original = getattr(LlavaLlamaModel.forward, "_egovla_joint_original", LlavaLlamaModel.forward)

    @functools.wraps(original)
    def joint_forward(self: Any, *args: Any, **kwargs: Any) -> Any:
        if getattr(self.config, "egovla_joint_schema", None) != JOINT_CONFIG_SCHEMA:
            return original(self, *args, **kwargs)
        raw_labels = kwargs.get("raw_action_labels")
        raw_masks = kwargs.get("raw_action_masks")
        # Let the release perform multimodal/token/decoder forward, but never
        # let its EE/MANO slicing interpret our joint labels.
        kwargs["raw_action_labels"] = None
        kwargs["raw_action_masks"] = None
        kwargs["raw_ee_movement_masks"] = None
        outputs = original(self, *args, **kwargs)
        if raw_labels is None:
            return outputs
        if raw_masks is None:
            raise ValueError("joint raw_action_labels require raw_action_masks")
        prediction = outputs.prediction
        total, arm, hand = joint_l1_losses(prediction, raw_labels, raw_masks)
        zero = total.detach() * torch.zeros((), device=total.device, dtype=total.dtype)
        return HOILMOutputWithPast(
            loss=total,
            recon_loss=total,
            ee_l2_loss=arm,
            ee_2d_l2_loss=zero,
            ee_rot_loss=zero,
            hand_l2_loss=hand,
            hand_kp_loss=zero,
            kl_loss=zero,
            logits=outputs.logits,
            prediction=prediction,
            raw_action_labels=raw_labels,
            raw_action_masks=raw_masks,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    joint_forward._egovla_joint_original = original
    LlavaLlamaModel.forward = joint_forward


def _patch_save_pretrained() -> None:
    """Use the modern save signature without publishing an early completion marker.

    HF calls ``save_pretrained`` before it writes optimizer, scheduler, Trainer,
    and per-rank RNG state.  Finalization therefore belongs exclusively to the
    post-save Trainer callback installed by
    :func:`_patch_checkpoint_finalizer_callback`.
    """

    from llava.model.llava_arch import LlavaMetaModel

    original = getattr(
        LlavaMetaModel.save_pretrained,
        "_egovla_joint_original",
        getattr(LlavaMetaModel.save_pretrained, "_egovla_original_save_pretrained", LlavaMetaModel.save_pretrained),
    )

    def compatible(
        self: Any,
        output_dir: str | os.PathLike[str],
        state_dict: Any = None,
        safe_serialization: bool = True,
        **_: Any,
    ) -> Any:
        del safe_serialization
        return original(self, output_dir, state_dict=state_dict)

    compatible._egovla_joint_original = original
    LlavaMetaModel.save_pretrained = compatible


def _register_datasets() -> None:
    from llava.data import builder

    if builder.DATASETS is None:
        builder.DATASETS = {}
    target = "llava.data.dataset_sparkarena.EgoVLASparkArenaHDF5Dataset"
    builder.DATASETS["sparkarena_egovla_train"] = {
        "_target_": target,
        "split": "train",
    }
    builder.DATASETS["sparkarena_egovla_val"] = {
        "_target_": target,
        "split": "val",
    }


def main() -> None:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size != 8:
        raise ValueError(f"SparkArena EgoVLA training requires WORLD_SIZE=8, got {world_size}")
    try:
        from human_plan.utils.compat import install_optional_compat
    except ImportError:
        from compat import install_optional_compat

    install_optional_compat()
    import human_plan.vila_train.train as train_module

    _write_run_contract()
    _patch_optional_attention()
    _patch_training_config(train_module)
    _patch_decoder_builder()
    _patch_joint_forward()
    _patch_save_pretrained()
    _patch_resume_selector(train_module)
    _patch_split_checkpoint_resume()
    _patch_checkpoint_finalizer_callback(train_module)
    _patch_smoke_stop_callback(train_module)
    _register_datasets()
    train_module.train()


if __name__ == "__main__":
    from unittest import mock

    try:
        from human_plan.utils.compat import install_optional_compat
    except ImportError:
        from compat import install_optional_compat

    install_optional_compat()
    from llava.train.transformer_normalize_monkey_patch import patched_normalize

    def _batch_len(self: Any) -> int:
        return len(self.batch_sampler)

    def _batch_iter(self: Any) -> Any:
        return self.batch_sampler.__iter__()

    with mock.patch(
        "transformers.image_processing_utils.normalize", new=patched_normalize
    ), mock.patch(
        "accelerate.data_loader.BatchSamplerShard.__len__", new=_batch_len
    ), mock.patch(
        "accelerate.data_loader.BatchSamplerShard.__iter__", new=_batch_iter
    ):
        main()
