"""Policy server wrapper.

Encapsulates a :class:`~cogwam.models.base.CogWAMBase` instance plus a
:class:`PolicyNormProcessor` that reuses the *training-time*
``ComposedModalityTransform`` for action un-normalization (no hand-rolled
math). The websocket server therefore returns already-unnormalized actions.

Client-side responsibilities that REMAIN on the client:
  - environment-specific adapters (composite stitching, gripper packing)
  - chunk-cache scheduling (the execution horizon is shorter than the model's
    predicted action chunk) and all per-environment semantic memory state.
    The server is stateless.

Exposed API:
  - ``metadata`` (dict, sent at handshake): the full inference ABI the eval
    client validates itself against.
  - ``predict_action(examples, unnorm_key=None, **kwargs)`` returns
    ``{"actions": np.ndarray[B, T, action_dim]}`` plus ``planner_text`` and the
    three aligned event-memory lists.
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import yaml
from omegaconf import OmegaConf

from cogwam.data.dataset import _append_state_norm_if_needed, _drop_video_transforms
from cogwam.data.lerobot.schema import DatasetMetadata, StateActionMetadata
from cogwam.data.lerobot.transform.base import ComposedModalityTransform
from cogwam.data.robodojo import DATASET_NAMED_MIXTURES, ROBOT_TYPE_CONFIG_MAP
from cogwam.models.base import FRAMEWORK_NAME, CogWAMBase, build_model, load_state_dict_file
from cogwam.training.config import apply_config_compat, dict_to_namespace, read_mode_config

logger = logging.getLogger(__name__)

MODEL_WEIGHTS_FILENAME = "model.safetensors"
DATASET_STATISTICS_FILENAME = "dataset_statistics.json"
MANIFEST_FILENAME = "artifact_manifest.json"
INFERENCE_CONFIG_FILENAME = "inference_config.yaml"

_CHUNK = 8 * 1024 * 1024


# ---------------------------------------------------------------------------
# Released artifact directory
# ---------------------------------------------------------------------------


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(_CHUNK):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_json(payload: Any) -> str:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def validate_artifact_directory(artifact_dir: Path) -> dict:
    """Verify an artifact produced by ``tools/convert_checkpoint.py``.

    The manifest records a sha256 for every released file. Checking it before
    loading turns a truncated or half-synced 7.8 GiB transfer into a one-second
    failure instead of a silently wrong policy.
    """

    manifest_path = artifact_dir / MANIFEST_FILENAME
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Artifact directory is missing {MANIFEST_FILENAME}: {artifact_dir}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise TypeError(f"{manifest_path} must contain a mapping")

    recorded_self = manifest.pop("manifest_sha256", None)
    if recorded_self is None:
        raise ValueError(f"{manifest_path} has no manifest_sha256")
    actual_self = _sha256_json(manifest)
    if actual_self != recorded_self:
        raise ValueError(
            f"{manifest_path} is self-inconsistent: manifest_sha256={recorded_self}, recomputed={actual_self}"
        )
    manifest["manifest_sha256"] = recorded_self

    files = manifest.get("files") or {}
    for name in (MODEL_WEIGHTS_FILENAME, DATASET_STATISTICS_FILENAME):
        entry = files.get(name)
        if not isinstance(entry, dict) or "sha256" not in entry:
            raise ValueError(f"{manifest_path} does not record a sha256 for {name}")
        path = artifact_dir / name
        if not path.is_file():
            raise FileNotFoundError(f"Artifact directory is missing {name}: {artifact_dir}")
        actual = _sha256_file(path)
        if actual != entry["sha256"]:
            raise ValueError(f"{path} sha256 mismatch: manifest={entry['sha256']}, actual={actual}")
        logging.info("Artifact %s sha256 OK (%s)", name, actual)

    framework = ((manifest.get("model") or {}).get("framework")) or ""
    if str(framework) != FRAMEWORK_NAME:
        raise ValueError(f"{manifest_path} declares framework={framework!r}; this server only serves {FRAMEWORK_NAME!r}")
    return manifest


def _load_yaml_config(path: Path) -> dict:
    ocfg = OmegaConf.load(str(path))
    # Normalise legacy / pre-v0.21 configs to current schema (idempotent).
    apply_config_compat(ocfg)
    config = OmegaConf.to_container(ocfg, resolve=True)
    if not isinstance(config, dict):
        raise TypeError(f"Config must be a mapping, got {type(config).__name__}: {path}")
    return config


# ---------------------------------------------------------------------------
# Inference ABI fields read from the saved config
# ---------------------------------------------------------------------------


def find_run_file(checkpoint: Path | str, name: str) -> Path:
    """Find a run artifact above a checkpoint path."""

    checkpoint = Path(checkpoint)
    for parent in checkpoint.parents:
        candidate = parent / name
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"Could not find {name} above checkpoint {checkpoint}")


def load_checkpoint_contract_config(
    checkpoint: Path | str,
    *,
    accessed_config: Optional[dict] = None,
) -> Tuple[dict, Path]:
    """Load the best config snapshot for ABI checks.

    ``config.yaml`` is intentionally an accessed-only snapshot, so runtime
    contract fields can be absent even though they are present in the complete
    ``config.full.yaml`` saved beside it. Model construction must keep using
    the accessed snapshot for backwards compatibility, while deployment code
    may use the complete snapshot to recover input/output ABI facts such as
    whether the checkpoint was conditioned on proprioception.
    """

    accessed_path = find_run_file(checkpoint, "config.yaml")
    full_path = accessed_path.with_name("config.full.yaml")
    selected_path = full_path if full_path.is_file() else accessed_path

    if selected_path == accessed_path and accessed_config is not None:
        config = accessed_config
    else:
        with selected_path.open("r", encoding="utf-8") as handle:
            config = yaml.safe_load(handle)

    if not isinstance(config, dict):
        raise TypeError(f"Checkpoint config must be a mapping, got {type(config).__name__}: {selected_path}")
    return config, selected_path


def _truthy(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() not in {"", "0", "false", "none", "no"}
    return bool(value)


def resolve_config_expects_state(config: dict) -> Tuple[bool, str]:
    """Read the training YAML's state switch for inference.

    Do not infer this ABI from run names or dataset markers: an absent field
    keeps the historical no-state behavior.
    """

    vla_cfg = (config.get("datasets") or {}).get("vla_data") or {}
    if "include_state" in vla_cfg:
        enabled = _truthy(vla_cfg["include_state"])
        return enabled, f"datasets.vla_data.include_state={enabled}"

    return False, "datasets.vla_data.include_state missing; default=False"


# ---------------------------------------------------------------------------
# Training-time normalization, rebuilt from dataset_statistics.json
# ---------------------------------------------------------------------------


def _resolve_robot_type(
    model_cfg: dict,
    unnorm_key: Optional[str] = None,
) -> str:
    """Look up the training robot_type from the saved cfg.

    ``cfg.datasets.vla_data.data_mix`` is a key in ``DATASET_NAMED_MIXTURES``
    whose value is a list of ``(dataset_name, weight, robot_type)`` tuples.

    When a data_mix contains entries from multiple robot types, ``unnorm_key``
    is used to identify which embodiment is requested. In those mixtures the
    ``robot_type`` field of each entry **matches** the top-level key in
    ``dataset_statistics.json``, so ``unnorm_key`` serves as the selector.
    """
    try:
        data_mix = model_cfg["datasets"]["vla_data"]["data_mix"]
    except (KeyError, TypeError) as e:
        raise KeyError(
            "ckpt config.yaml is missing `datasets.vla_data.data_mix`; cannot resolve training-time robot_type."
        ) from e

    if data_mix not in DATASET_NAMED_MIXTURES:
        raise KeyError(
            f"data_mix={data_mix!r} not in DATASET_NAMED_MIXTURES (available: {sorted(DATASET_NAMED_MIXTURES.keys())})."
        )

    mixture = DATASET_NAMED_MIXTURES[data_mix]
    robot_types = sorted({entry[2] for entry in mixture})

    if len(robot_types) == 1:
        return robot_types[0]

    # Multiple robot types in the mixture. Use unnorm_key as a direct selector:
    # for multi-robot mixtures the dataset_statistics.json top-level keys equal
    # the robot_type values.
    if unnorm_key is not None and unnorm_key in robot_types:
        return unnorm_key

    raise ValueError(
        f"data_mix={data_mix!r} contains multiple robot_types {robot_types}. "
        f"Pass `unnorm_key` matching one of them to disambiguate (e.g. unnorm_key={robot_types[0]!r})."
    )


def _infer_key_dims(
    data_config: Any,
    combined_stats: Dict[str, Any],
    modality_keys: Sequence[str],
    modality: str,
) -> Dict[str, int]:
    """Compute per-key dimensions for splitting combined stats arrays.

    Lookup priority:
      1. ``data_config.<modality>_key_dims`` — explicit dict classvar on the DataConfig
         (required for DataConfigs with non-uniform or multi-dim keys).
      2. Infer uniformly from stats array length: if ``D_total / n_keys`` is an
         integer, use that as the uniform per-key dim.
      3. Fall back to dim=1 when no stats are available (empty combined dict).
    """
    attr = f"{modality}_key_dims"
    if hasattr(data_config, attr):
        return dict(getattr(data_config, attr))

    combined = combined_stats.get(modality, {})
    stat_arr = next((v for k, v in combined.items() if k != "mask"), None)
    n_keys = len(modality_keys)
    if stat_arr is not None and n_keys > 0:
        d_total = len(stat_arr)
        if d_total == n_keys:
            return {k: 1 for k in modality_keys}
        elif n_keys > 0 and d_total % n_keys == 0:
            dim = d_total // n_keys
            return {k: dim for k in modality_keys}
        else:
            raise ValueError(
                f"Cannot infer per-key dims for modality={modality!r}: "
                f"D_total={d_total} is not evenly divisible by n_keys={n_keys}. "
                f"Add `{attr} = {{key: dim, ...}}` to the DataConfig (keys={list(modality_keys)})."
            )
    return {k: 1 for k in modality_keys}


def _build_dataset_metadata(
    stats_for_key: Dict[str, Any],
    embodiment_tag: Any,
    action_keys: Sequence[str],
    state_keys: Sequence[str],
    action_key_dims: Optional[Dict[str, int]] = None,
    state_key_dims: Optional[Dict[str, int]] = None,
) -> DatasetMetadata:
    """Convert the *combined* stats arrays from ``dataset_statistics.json`` back
    into a per-subkey :class:`DatasetMetadata` matching what the training
    pipeline produced.

    The saved ``dataset_statistics.json`` stores stats as flat arrays of length
    ``D = sum(per_key_dims)``. For each ``"action.<sub>"`` key we slice out
    ``dim_k`` elements starting at the current cursor and store them under
    ``statistics.action.<sub> = {"min": [v0..v_{k-1}], ...}``.
    """
    if action_key_dims is None:
        action_key_dims = {k: 1 for k in action_keys}
    if state_key_dims is None:
        state_key_dims = {k: 1 for k in state_keys}

    def _split_combined(
        combined: Dict[str, Sequence[float]],
        keys: Sequence[str],
        key_dims: Dict[str, int],
    ):
        """Split combined arrays into per-subkey dicts using per-key dims.

        ``combined`` looks like ``{"min": [..D..], "max": [..D..], "mask": [..D..], ...}``.
        Returns ``(stats_per_subkey, meta_per_subkey)``.
        """
        stats_per_subkey: Dict[str, Dict[str, List[float]]] = {}
        meta_per_subkey: Dict[str, StateActionMetadata] = {}
        cursor = 0
        for full_key in keys:
            subkey = full_key.split(".", 1)[1]
            dim_k = key_dims.get(full_key, 1)
            per_key: Dict[str, List[float]] = {}
            for stat_name, arr in combined.items():
                if stat_name == "mask":
                    continue
                end = cursor + dim_k
                if end > len(arr):
                    # Saved combined array shorter than expected (truncated
                    # pad channels etc.). Skip this stat field silently.
                    continue
                per_key[stat_name] = [float(v) for v in arr[cursor:end]]
            stats_per_subkey[subkey] = per_key
            meta_per_subkey[subkey] = StateActionMetadata(
                absolute=True,
                rotation_type=None,
                shape=(dim_k,),
                continuous=True,
            )
            cursor += dim_k
        return stats_per_subkey, meta_per_subkey

    action_combined = stats_for_key.get("action", {})
    state_combined = stats_for_key.get("state", {})

    action_stats, action_meta = _split_combined(action_combined, action_keys, action_key_dims)
    state_stats, state_meta = _split_combined(state_combined, state_keys, state_key_dims)

    # Pydantic accepts dict input with field validators
    return DatasetMetadata.model_validate(
        {
            "statistics": {
                "state": state_stats,
                "action": action_stats,
            },
            "modalities": {
                "video": {},
                "state": state_meta,
                "action": action_meta,
            },
            "embodiment_tag": embodiment_tag.value if hasattr(embodiment_tag, "value") else embodiment_tag,
        }
    )


class PolicyNormProcessor:
    """Server-side normalization helper backed by training-time transforms.

    This class replaces the hand-rolled un-normalization math that otherwise
    ends up duplicated in every eval client. It rebuilds the *exact*
    ``ComposedModalityTransform`` used at training time:

      1. Resolve ``data_mix`` -> ``robot_type`` via ``DATASET_NAMED_MIXTURES``
         -> fetch the ``DataConfig`` from ``ROBOT_TYPE_CONFIG_MAP``.
      2. Build the transform pipeline via ``data_config.transform()``.
      3. Reconstruct a ``DatasetMetadata`` from the saved
         ``dataset_statistics.json`` (which stores **combined** per-modality
         arrays of length ``D``) by splitting it into per-key entries.
      4. ``set_metadata(...)`` binds the metadata into every transform.

    There is no second source of truth for normalization math.
    """

    def __init__(self, model_cfg: dict, norm_stats: dict, unnorm_key: Optional[str] = None) -> None:
        self._model_cfg = model_cfg
        self._norm_stats = norm_stats

        # 3-early) Pick the requested unnorm_key (or auto-select) BEFORE
        # resolving robot_type so we can use it as a hint for multi-robot mixtures.
        if unnorm_key is None:
            if len(norm_stats) == 1:
                unnorm_key = next(iter(norm_stats.keys()))
            # else: defer error to step 3 below after robot_type resolution attempt
        elif unnorm_key not in norm_stats:
            raise KeyError(f"unnorm_key={unnorm_key!r} not in {list(norm_stats.keys())}")
        self._unnorm_key = unnorm_key  # may still be None for multi-key case

        # 1) Resolve which DataConfig was used at training.
        robot_type = _resolve_robot_type(model_cfg, unnorm_key=unnorm_key)
        if robot_type not in ROBOT_TYPE_CONFIG_MAP:
            raise KeyError(
                f"robot_type={robot_type!r} not in ROBOT_TYPE_CONFIG_MAP "
                f"(available: {sorted(ROBOT_TYPE_CONFIG_MAP.keys())})."
            )
        self._data_config = ROBOT_TYPE_CONFIG_MAP[robot_type]
        self._action_keys: List[str] = list(self._data_config.action_keys)
        self._state_keys: List[str] = list(getattr(self._data_config, "state_keys", []))

        # 2) Build training-time transform pipeline.
        transform = self._data_config.transform()
        if not isinstance(transform, ComposedModalityTransform):
            transform = ComposedModalityTransform(transforms=[transform])
        self._transform = transform

        # 3) Pick the requested unnorm_key (finalize; error if still None here).
        if self._unnorm_key is None:
            raise ValueError(
                f"Multiple unnorm_keys in dataset_statistics.json: {list(norm_stats.keys())}. "
                "Pass unnorm_key explicitly."
            )
        unnorm_key = self._unnorm_key

        # 4) Resolve per-key dims (handles multi-d action/state keys).
        stats_for_unnorm = norm_stats[unnorm_key]
        self._action_key_dims: Dict[str, int] = _infer_key_dims(
            self._data_config, stats_for_unnorm, self._action_keys, "action"
        )
        self._state_key_dims: Dict[str, int] = _infer_key_dims(
            self._data_config, stats_for_unnorm, self._state_keys, "state"
        )

        # 5) Build & bind metadata.
        ds_meta = _build_dataset_metadata(
            stats_for_key=stats_for_unnorm,
            embodiment_tag=self._data_config.embodiment_tag,
            action_keys=self._action_keys,
            state_keys=self._state_keys,
            action_key_dims=self._action_key_dims,
            state_key_dims=self._state_key_dims,
        )
        self._transform.set_metadata(ds_meta)
        self._transform.eval()  # mark transforms as eval-mode

        logger.info(
            "PolicyNormProcessor ready: robot_type=%s, unnorm_key=%s, action_keys=%s (dims=%s), state_keys=%s",
            robot_type,
            unnorm_key,
            self._action_keys,
            [self._action_key_dims[k] for k in self._action_keys],
            self._state_keys,
        )

    @property
    def action_keys(self) -> List[str]:
        return list(self._action_keys)

    @property
    def state_keys(self) -> List[str]:
        return list(self._state_keys)

    @property
    def unnorm_key(self) -> str:
        return self._unnorm_key

    @property
    def available_unnorm_keys(self) -> List[str]:
        return list(self._norm_stats.keys())

    @property
    def transform(self) -> ComposedModalityTransform:
        return self._transform

    def unapply_actions(self, normalized_actions: np.ndarray) -> np.ndarray:
        """Invert action normalization using the training-time pipeline.

        Args:
            normalized_actions: shape ``(T, D)`` where
                ``D == sum(action_key_dims.values())``.

        Returns:
            ``(T, D)`` un-normalized actions in env coordinates.
        """
        normalized_actions = np.asarray(normalized_actions)
        assert normalized_actions.ndim == 2, f"Expected (T, D); got shape {normalized_actions.shape}"

        # Split (T, D) into per-key {full_key: torch.Tensor[T, dim_k]}.
        data: Dict[str, torch.Tensor] = {}
        cursor = 0
        for full_key in self._action_keys:
            dim_k = self._action_key_dims.get(full_key, 1)
            slice_ = normalized_actions[..., cursor : cursor + dim_k]
            data[full_key] = torch.as_tensor(slice_, dtype=torch.float32)
            cursor += dim_k

        if cursor != normalized_actions.shape[-1]:
            raise ValueError(
                f"Sum of per-key dims ({cursor}) != action_dim ({normalized_actions.shape[-1]}). "
                f"action_keys={self._action_keys}, action_key_dims={self._action_key_dims}"
            )

        out = self._transform.unapply(data)

        parts: List[np.ndarray] = []
        for full_key in self._action_keys:
            value = out[full_key]
            if isinstance(value, torch.Tensor):
                value = value.detach().cpu().numpy()
            parts.append(np.asarray(value))
        return np.concatenate(parts, axis=-1)


# ---------------------------------------------------------------------------
# Server-side policy
# ---------------------------------------------------------------------------


def _build_framework(model_cfg: dict, weights: Dict[str, torch.Tensor]) -> CogWAMBase:
    """Construct the framework from its saved config and load weights strictly."""

    framework = build_model(dict_to_namespace(model_cfg))
    model_keys = set(framework.state_dict().keys())
    checkpoint_keys = set(weights.keys())
    try:
        framework.load_state_dict(weights, strict=True)
    except RuntimeError:
        # Key mismatch must be fatal, but report both directions first: a bare
        # torch message truncates the lists that identify which module moved.
        common_keys = model_keys & checkpoint_keys
        missing_keys = model_keys - common_keys
        unexpected_keys = checkpoint_keys - common_keys
        if missing_keys:
            logger.warning("Missing keys in state_dict: %s", sorted(missing_keys))
        if unexpected_keys:
            logger.warning("Unexpected keys in state_dict: %s", sorted(unexpected_keys))
        raise
    return framework


class PolicyServerWrapper:
    """Wraps a :class:`CogWAMBase` for use as a websocket-server policy."""

    def __init__(
        self,
        ckpt_path: Optional[str] = None,
        artifact_dir: Optional[str] = None,
        device: str = "cuda",
        use_bf16: bool = False,
        unnorm_key: Optional[str] = None,
        dino_stats_path: Optional[str] = None,
    ) -> None:
        if (ckpt_path is None) == (artifact_dir is None):
            raise ValueError("Pass exactly one of ckpt_path (training run) or artifact_dir (released artifact)")

        if artifact_dir is not None:
            source = Path(artifact_dir).expanduser().resolve()
            if not source.is_dir():
                raise FileNotFoundError(f"Artifact directory does not exist: {source}")
            self._artifact_manifest = validate_artifact_directory(source)
            config_path = source / INFERENCE_CONFIG_FILENAME
            if not config_path.is_file():
                raise FileNotFoundError(
                    f"Artifact directory is missing {INFERENCE_CONFIG_FILENAME}: {source}. "
                    "Re-run tools/convert_checkpoint.py with --config."
                )
            model_cfg = _load_yaml_config(config_path)
            norm_stats = json.loads((source / DATASET_STATISTICS_FILENAME).read_text(encoding="utf-8"))
            # A released artifact carries exactly one config, so the accessed
            # and complete snapshots cannot diverge.
            contract_cfg, contract_cfg_path = model_cfg, config_path
            weights_path = source / MODEL_WEIGHTS_FILENAME
        else:
            source = Path(ckpt_path)
            self._artifact_manifest = None
            # ``config.yaml`` remains authoritative for construction;
            # ``config.full.yaml`` supplies strict runtime ABI fields.
            model_cfg, norm_stats = read_mode_config(str(source))
            contract_cfg, contract_cfg_path = load_checkpoint_contract_config(
                str(source),
                accessed_config=model_cfg,
            )
            weights_path = source

        self._ckpt_path = str(source)

        logging.info("PolicyServerWrapper: loading framework from %s", self._ckpt_path)
        framework = _build_framework(model_cfg, load_state_dict_file(weights_path))

        if dino_stats_path:
            framework.set_dino_stats(str(dino_stats_path))
            logging.info("PolicyServerWrapper: loaded online-DINO stats from %s", dino_stats_path)
        if use_bf16:
            framework = framework.to(torch.bfloat16)
        framework = framework.to(device).eval()
        self._framework = framework

        # Co-located metadata.
        self._model_cfg = model_cfg
        self._contract_cfg_path = str(contract_cfg_path)
        contract_framework_cfg = contract_cfg.get("framework") or {}
        accessed_framework_cfg = model_cfg.get("framework") or {}
        self._framework_name = str(contract_framework_cfg.get("name", accessed_framework_cfg.get("name", "")))
        qwen_interface = getattr(framework, "qwen_vl_interface", None)
        self._qwen_attn_implementation = (
            str(getattr(qwen_interface, "attn_implementation", "")).lower() if qwen_interface is not None else ""
        ) or None
        planner_cfg = (contract_cfg.get("framework") or {}).get("planner") or {}
        text_cfg = planner_cfg.get("text_supervision") or {}
        self._text_planning_enabled = bool(
            getattr(
                framework,
                "text_planning_enabled",
                text_cfg.get("enabled", False),
            )
        )
        self._event_memory_enabled = bool(
            getattr(
                framework,
                "event_memory_enabled",
                str(text_cfg.get("mode", "")).lower() == "event_driven_memory_ntp",
            )
        )
        self._planner_text_cache_supported = bool(
            self._text_planning_enabled
            and not self._event_memory_enabled
            and hasattr(framework, "_cached_or_generated_planner_hidden")
        )
        self._event_semantic_fields = dict(getattr(framework, "event_semantic_fields", {}) or {})
        event_data_cfg = dict(getattr(framework, "event_data_config", {}) or {})
        self._event_empty_memory = str(event_data_cfg.get("empty_memory", "None."))
        self._event_empty_cached_subtask = str(event_data_cfg.get("empty_cached_subtask", "None."))
        self._event_semantic_offset = int(event_data_cfg.get("semantic_offset", -10))
        self._event_replan_interval = int(event_data_cfg.get("replan_interval", 10))
        if self._text_planning_enabled and not self._event_memory_enabled and not self._planner_text_cache_supported:
            raise RuntimeError("Text-planning checkpoint does not support cached planner text inference")
        contract_vla_cfg = (contract_cfg.get("datasets") or {}).get("vla_data") or {}
        self._image_layout = str(contract_vla_cfg.get("image_layout", "separate_views"))
        self._obs_image_size = list(contract_vla_cfg.get("obs_image_size", []) or [])
        self._composite_view_key = contract_vla_cfg.get("composite_view_key")
        self._composite_source_view_keys = list(contract_vla_cfg.get("composite_source_view_keys", []) or [])
        # Per-camera planner stream. Present only when the checkpoint was
        # trained with vlm_view_source=separate_views; the client uses it to
        # decide whether to pay the extra wire cost of sending per-view
        # pixels alongside the composite the DINO branch still needs.
        self._vlm_view_size = list(contract_vla_cfg.get("vlm_view_size", []) or [])
        self._state_normalizer: Optional[ComposedModalityTransform] = None
        self._state_keys: List[str] = []
        self._state_key_dims: Dict[str, int] = {}
        self._state_total_dim = 0
        self._expects_state, self._state_contract_source = resolve_config_expects_state(contract_cfg)
        logging.info(
            "PolicyServerWrapper: expects_state=%s (%s; config=%s)",
            self._expects_state,
            self._state_contract_source,
            self._contract_cfg_path,
        )
        self._sync_framework_state_contract(framework)

        # action_chunk_size = future_action_window_size + 1 (matches old client).
        action_model_cfg = model_cfg["framework"]["action_model"]

        if "action_horizon" in action_model_cfg:
            self._action_chunk_size = int(action_model_cfg["action_horizon"])
        elif "future_action_window_size" in action_model_cfg:
            self._action_chunk_size = int(action_model_cfg["future_action_window_size"]) + 1
        else:
            raise ValueError(
                f"PolicyServerWrapper: no action_horizon or future_action_window_size found "
                f"in model config for {self._ckpt_path}"
            )
        # Cache of PolicyNormProcessor instances per unnorm_key.
        # For single-dataset ckpts unnorm_key is auto-selected; for multi-dataset
        # ckpts clients must pass unnorm_key per request.
        self._default_unnorm_key = unnorm_key
        self._norm_stats = norm_stats
        self._norm_processors: Dict[str, PolicyNormProcessor] = {}

        # Peek at available keys without building a full processor.
        self._available_unnorm_keys: List[str] = list(norm_stats.keys())

        # Eagerly build when unambiguous; defer for multi-key / no explicit key.
        if unnorm_key is not None or len(self._available_unnorm_keys) == 1:
            default_proc = self._get_processor(unnorm_key)
            self._default_unnorm_key = default_proc.unnorm_key
            logging.info(
                "PolicyServerWrapper ready: action_chunk_size=%d, default_unnorm_key=%s, "
                "available_unnorm_keys=%s, action_keys=%s, state_keys=%s",
                self._action_chunk_size,
                default_proc.unnorm_key,
                default_proc.available_unnorm_keys,
                default_proc.action_keys,
                default_proc.state_keys,
            )
        else:
            logging.info(
                "PolicyServerWrapper ready (multi-key): action_chunk_size=%d, "
                "available_unnorm_keys=%s — clients must pass unnorm_key per request.",
                self._action_chunk_size,
                self._available_unnorm_keys,
            )

        if self._expects_state:
            self._build_state_normalizer(contract_cfg, norm_stats)

    @classmethod
    def _config_expects_state(cls, model_cfg: dict) -> bool:
        """Compatibility shim for callers/tests that only have a config dict."""

        return resolve_config_expects_state(model_cfg)[0]

    def _sync_framework_state_contract(self, framework: CogWAMBase) -> None:
        """Make the loaded model consume exactly the checkpoint-declared state ABI.

        Model construction intentionally uses the compact accessed-only config,
        while the complete checkpoint contract may come from ``config.full.yaml``.
        The framework checks ``framework.config`` again during every forward, so
        an old compact config that omitted ``include_state`` must be synchronized
        after construction. This changes only runtime input routing, not model
        shape.
        """

        source = self._state_contract_source
        if not source.startswith("datasets.vla_data.include_state"):
            return
        uses_action_state = getattr(framework, "_uses_action_state", None)
        if not callable(uses_action_state) or bool(uses_action_state()) == self._expects_state:
            return

        datasets_cfg = getattr(getattr(framework, "config", None), "datasets", None)
        vla_cfg = getattr(datasets_cfg, "vla_data", None) if datasets_cfg is not None else None
        if vla_cfg is None:
            raise RuntimeError(
                "PolicyServerWrapper: checkpoint declares a state ABI, but the loaded framework has no "
                "datasets.vla_data config to enforce it"
            )
        try:
            vla_cfg["include_state"] = self._expects_state
        except (TypeError, KeyError):
            vla_cfg.include_state = self._expects_state
        if bool(uses_action_state()) != self._expects_state:
            raise RuntimeError("PolicyServerWrapper: failed to synchronize the framework's include_state contract")
        logging.info(
            "PolicyServerWrapper: synchronized framework include_state=%s from %s",
            self._expects_state,
            self._contract_cfg_path,
        )

    def _build_state_normalizer(self, model_cfg: dict, norm_stats: dict) -> None:
        unnorm_key = self._default_unnorm_key
        if unnorm_key is None:
            if len(norm_stats) == 1:
                unnorm_key = next(iter(norm_stats.keys()))
            else:
                logging.warning(
                    "PolicyServerWrapper: model expects state but multiple unnorm keys exist (%s); "
                    "state will be forwarded without normalization until unnorm_key is specified.",
                    list(norm_stats.keys()),
                )
                return

        robot_type = _resolve_robot_type(model_cfg, unnorm_key=unnorm_key)
        data_config = ROBOT_TYPE_CONFIG_MAP[robot_type]
        state_keys = list(getattr(data_config, "state_keys", []))
        if not state_keys:
            logging.info("PolicyServerWrapper: model expects state but data_config has no state keys.")
            return

        action_keys = list(data_config.action_keys)
        stats_for_key = norm_stats[unnorm_key]
        state_key_dims = _infer_key_dims(data_config, stats_for_key, state_keys, "state")
        action_key_dims = _infer_key_dims(data_config, stats_for_key, action_keys, "action")
        ds_meta = _build_dataset_metadata(
            stats_for_key=stats_for_key,
            embodiment_tag=data_config.embodiment_tag,
            action_keys=action_keys,
            state_keys=state_keys,
            action_key_dims=action_key_dims,
            state_key_dims=state_key_dims,
        )

        transform = _drop_video_transforms(data_config.transform())
        # Training appends q99/binary state normalization on top of the
        # action-only transform. Mirror that here so include_state=true
        # checkpoints receive normalized proprioception at inference.
        vla_cfg = (model_cfg.get("datasets") or {}).get("vla_data") or {}
        state_norm_modes = vla_cfg.get("state_norm_modes", None)
        transform = _append_state_norm_if_needed(
            transform,
            state_keys,
            state_norm_modes,
        )
        if not isinstance(transform, ComposedModalityTransform):
            transform = ComposedModalityTransform(transforms=[transform])
        transform.set_metadata(ds_meta)
        transform.eval()

        self._state_normalizer = transform
        self._state_keys = state_keys
        self._state_key_dims = state_key_dims
        self._state_total_dim = sum(int(state_key_dims.get(key, 1)) for key in state_keys)
        logging.info(
            "PolicyServerWrapper: state normalizer ready (robot_type=%s, unnorm_key=%s, keys=%s, total_dim=%d)",
            robot_type,
            unnorm_key,
            state_keys,
            self._state_total_dim,
        )

    def _normalize_state(self, raw_state: Any) -> np.ndarray:
        arr = np.asarray(raw_state, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr[None, :]
        total = self._state_total_dim
        if total <= 0:
            return arr
        if arr.shape[-1] != total:
            raise ValueError(
                f"PolicyServerWrapper: checkpoint expects a {total}-D state, got shape {arr.shape}. "
                "Refusing to pad or truncate state because that would break train/inference consistency."
            )

        data: Dict[str, np.ndarray] = {}
        cursor = 0
        for key in self._state_keys:
            dim = int(self._state_key_dims.get(key, 1))
            data[key] = np.ascontiguousarray(arr[..., cursor : cursor + dim], dtype=np.float32)
            cursor += dim

        out = self._state_normalizer.apply(data)
        parts: List[np.ndarray] = []
        for key in self._state_keys:
            value = out[key]
            if torch.is_tensor(value):
                value = value.detach().cpu().numpy()
            parts.append(np.asarray(value, dtype=np.float32))
        return np.concatenate(parts, axis=-1)

    def _prepare_examples(self, examples: List[dict]) -> List[dict]:
        prepared_examples = []
        for example in examples:
            if not isinstance(example, dict):
                prepared_examples.append(example)
                continue
            prepared = dict(example)
            if not self._expects_state:
                prepared.pop("state", None)
            else:
                if prepared.get("state") is None:
                    raise ValueError(
                        "PolicyServerWrapper: this checkpoint was trained with include_state=true, "
                        "but the inference request did not provide state."
                    )
                if self._state_normalizer is None:
                    raise RuntimeError(
                        "PolicyServerWrapper: this checkpoint requires state, but its training-time "
                        "state normalizer could not be constructed."
                    )
                prepared["state"] = self._normalize_state(prepared["state"])
            prepared_examples.append(prepared)
        return prepared_examples

    def _get_processor(self, unnorm_key: Optional[str]) -> PolicyNormProcessor:
        cache_key = unnorm_key if unnorm_key is not None else "__default__"
        if cache_key not in self._norm_processors:
            self._norm_processors[cache_key] = PolicyNormProcessor(
                self._model_cfg,
                self._norm_stats,
                unnorm_key=unnorm_key,
            )
        return self._norm_processors[cache_key]

    @property
    def metadata(self) -> Dict[str, Any]:
        """Model-invariant metadata; sent to the client at websocket handshake.

        The client treats every field here as a hard ABI, not a hint: it aborts
        the rollout when its own configuration disagrees.
        """
        physical = getattr(self._framework, "action_model", None)
        qwen_interface = getattr(self._framework, "qwen_vl_interface", None)
        base = {
            "env": "cogwam_policy_server",
            "ckpt_path": self._ckpt_path,
            "action_chunk_size": self._action_chunk_size,
            "available_unnorm_keys": self._available_unnorm_keys,
            "default_unnorm_key": self._default_unnorm_key,
            "expects_state": self._expects_state,
            "state_contract_source": self._state_contract_source,
            "contract_config": self._contract_cfg_path,
            "framework_name": self._framework_name,
            "image_layout": self._image_layout,
            "obs_image_size": list(self._obs_image_size or []),
            "composite_view_key": self._composite_view_key,
            "composite_source_view_keys": self._composite_source_view_keys,
            "vlm_view_size": list(self._vlm_view_size or []),
            "text_planning_enabled": self._text_planning_enabled,
            "planner_text_cache_supported": self._planner_text_cache_supported,
            "event_memory_enabled": bool(self._event_memory_enabled),
            "event_semantic_fields": dict(self._event_semantic_fields or {}),
            "event_empty_memory": str(self._event_empty_memory),
            "event_empty_cached_subtask": str(self._event_empty_cached_subtask),
            "event_semantic_offset": int(self._event_semantic_offset),
            "event_replan_interval": int(self._event_replan_interval),
            "qwen_attn_implementation": self._qwen_attn_implementation,
            "qwen35_causal_conv1d_backend": getattr(qwen_interface, "causal_conv1d_backend", None),
            "qwen35_fla_backend": getattr(qwen_interface, "fla_backend", None),
            "qwen35_attn_implementation_source": getattr(qwen_interface, "attn_implementation_source", None),
            "planner_query_mask_contract": getattr(self._framework, "planner_query_mask_contract", None),
            "mot_contract_version": getattr(physical, "checkpoint_contract_version", None),
            "mot_interaction_mode": getattr(physical, "interaction_mode", None),
            "mot_layerwise_planner_coupling": bool(getattr(physical, "layerwise_planner_coupling", False)),
            "mot_num_inference_timesteps": int(getattr(physical, "inference_steps", 0) or 0),
            "mot_action_prediction_type": getattr(physical, "action_prediction_type", None),
            "mot_action_velocity_target": getattr(physical, "action_velocity_target", None),
            "mot_rtc_supported": bool(getattr(physical, "rtc_guidance_supported", False)),
            "mot_multires_world_input": bool(getattr(physical, "multires_world_input", False)),
            "mot_current_dino_tokens": getattr(physical, "current_world_tokens", None),
            "mot_future_dino_tokens": getattr(physical, "future_world_tokens", None)
            or getattr(physical, "world_tokens", None),
            "mot_world_dino_tokens": getattr(physical, "world_tokens", None),
        }
        if self._artifact_manifest is not None:
            files = self._artifact_manifest.get("files") or {}
            base["artifact_manifest_sha256"] = self._artifact_manifest.get("manifest_sha256")
            base["artifact_weights_sha256"] = (files.get(MODEL_WEIGHTS_FILENAME) or {}).get("sha256")
        # Enrich with per-embodiment keys when a default processor already exists.
        if self._default_unnorm_key is not None:
            proc = self._get_processor(self._default_unnorm_key)
            base["action_keys"] = proc.action_keys
            base["state_keys"] = proc.state_keys
        return base

    def predict_action(
        self,
        examples: List[dict],
        unnorm_key: Optional[str] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Run the framework, then un-normalize via training-time transforms.

        Args:
            examples: list of dicts (each with ``image`` / ``lang`` / optional ``state``).
            unnorm_key: dataset key for un-normalization stats. ``None`` -->
                use the wrapper's default (auto-picked at startup).
            **kwargs: forwarded to the framework's ``predict_action``
                (``do_sample``, ``use_ddim``, ``num_ddim_steps``, ...).

        Returns:
            Un-normalized ``actions``, the generated ``planner_text``, and the
            three aligned event-memory decision lists.
        """
        examples = self._prepare_examples(examples)
        effective_key = unnorm_key if unnorm_key is not None else self._default_unnorm_key
        if effective_key is None:
            if len(self._available_unnorm_keys) == 1:
                effective_key = self._available_unnorm_keys[0]
            else:
                raise ValueError(
                    f"predict_action: unnorm_key not specified and no default set. "
                    f"Pass one of {self._available_unnorm_keys}."
                )
        proc = self._get_processor(effective_key)

        out = self._framework.predict_action(examples=examples, **kwargs)
        normalized = np.asarray(out["normalized_actions"])  # (B, T, D)

        unnorm = np.stack(
            [proc.unapply_actions(normalized[b]) for b in range(normalized.shape[0])],
            axis=0,
        )
        # RTC feeds the old chunk's tail back into the flow sampler in the
        # training-time normalized coordinate system. Clients that do not use
        # RTC keep consuming only the unnormalized ``actions`` field.
        result = {"actions": unnorm, "normalized_actions": normalized}
        # The unified text planner exposes its low-frequency AR plan alongside
        # the action chunk.
        if "planner_text" in out:
            result["planner_text"] = out["planner_text"]
        if "planner_text_refreshed" in out:
            result["planner_text_refreshed"] = out["planner_text_refreshed"]
        for key in (
            "semantic_decision",
            "semantic_memory_add",
            "semantic_current_subtask",
        ):
            if key in out:
                result[key] = out[key]
        return result


__all__ = [
    "PolicyNormProcessor",
    "PolicyServerWrapper",
    "load_checkpoint_contract_config",
    "resolve_config_expects_state",
    "validate_artifact_directory",
]
