from __future__ import annotations

import copy
import gc
import json
import logging
import os
import sys
from collections import deque
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

from XPolicyLab.model_template import ModelTemplate
from XPolicyLab.utils.checkpoint_resolver import candidate_checkpoint_roots
from XPolicyLab.utils.process_data import (
    get_robot_action_dim_info,
    pack_robot_state,
    unpack_robot_state,
)


_CUR_DIR = Path(__file__).resolve().parent
# XPolicyLab is imported as a package from its parent workspace.
_XPOLICYLAB_PARENT = _CUR_DIR.parents[2]
_CHECKPOINTS_DIR = _CUR_DIR / "checkpoints"


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() in {"none", "null"}:
        return None
    return text


def _as_bool(value: Any, *, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off", "", "none", "null"}:
        return False
    raise ValueError(f"Expected a boolean-like value, got {value!r}.")


def _normalize_prompt(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="ignore")
    elif isinstance(value, np.ndarray) and value.ndim == 0:
        value = value.item()
    elif isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, (list, tuple)):
        for item in value:
            prompt = _normalize_prompt(item)
            if prompt:
                return prompt
        return None
    text = str(value).strip()
    return text or None


def _resolve_prompt(observation: Mapping[str, Any], default_prompt: str) -> str:
    for key in ("instruction", "instructions", "prompt", "task", "language_instruction"):
        prompt = _normalize_prompt(observation.get(key))
        if prompt:
            return prompt
    fallback = _normalize_prompt(default_prompt)
    if not fallback:
        raise ValueError("No language instruction was found in the observation or deploy config.")
    return fallback


def _extract_image(observation: Mapping[str, Any], candidate_names: Iterable[str]) -> Any:
    vision = observation.get("vision", {})
    if not isinstance(vision, Mapping):
        raise ValueError("obs['vision'] must be a camera mapping.")
    for name in candidate_names:
        if name not in vision:
            continue
        value = vision[name]
        if isinstance(value, Mapping):
            for image_key in ("color", "rgb"):
                if image_key in value:
                    return value[image_key]
        else:
            return value
    raise KeyError(f"Could not find a camera image among {list(candidate_names)}.")


def _ensure_hwc_rgb_uint8(image: Any) -> np.ndarray:
    """Standardize an already-decoded RGB image without decoding or swapping channels."""
    array = np.asarray(image)
    if array.ndim != 3:
        raise ValueError(f"Expected a 3-D image array, got shape {array.shape}.")
    if array.shape[-1] not in (1, 3) and array.shape[0] in (1, 3):
        array = np.transpose(array, (1, 2, 0))
    if array.shape[-1] == 1:
        array = np.repeat(array, 3, axis=-1)
    if array.shape[-1] != 3:
        raise ValueError(f"Expected an RGB image, got shape {array.shape}.")

    if np.issubdtype(array.dtype, np.floating):
        finite_max = float(np.nanmax(array)) if array.size else 0.0
        if finite_max <= 1.0:
            array = array * 255.0
        array = np.clip(array, 0.0, 255.0).astype(np.uint8)
    elif array.dtype != np.uint8:
        array = np.clip(array, 0, 255).astype(np.uint8)
    # Server-decoded buffers can be read-only views. WSA's tensor path expects
    # a contiguous writable array, so take one copy here.
    return np.ascontiguousarray(array).copy()


def _encode_observation(
    observation: Mapping[str, Any],
    action_type: str,
    robot_action_dim_info: dict[str, list[int]],
    default_prompt: str,
) -> dict[str, Any]:
    if "images" in observation and not isinstance(observation.get("state"), Mapping):
        images = observation["images"]
        return {
            "images": {
                "cam_high": _ensure_hwc_rgb_uint8(images["cam_high"]),
                "cam_left_wrist": _ensure_hwc_rgb_uint8(images["cam_left_wrist"]),
                "cam_right_wrist": _ensure_hwc_rgb_uint8(images["cam_right_wrist"]),
            },
            "state": np.asarray(observation["state"], dtype=np.float32),
            "prompt": _resolve_prompt(observation, default_prompt),
        }

    images = {
        "cam_high": _ensure_hwc_rgb_uint8(
            _extract_image(observation, ("cam_head", "cam_high", "head_camera", "top_camera"))
        ),
        "cam_left_wrist": _ensure_hwc_rgb_uint8(
            _extract_image(observation, ("cam_left_wrist", "left_camera", "left_wrist", "wrist_left"))
        ),
        "cam_right_wrist": _ensure_hwc_rgb_uint8(
            _extract_image(observation, ("cam_right_wrist", "right_camera", "right_wrist", "wrist_right"))
        ),
    }
    state = pack_robot_state(
        dict(observation), action_type, robot_action_dim_info, source_type="obs"
    ).astype(np.float32)
    return {
        "images": images,
        "state": state,
        "prompt": _resolve_prompt(observation, default_prompt),
    }


def _latest_pretrained_dir(root: Path) -> Path | None:
    if (root / "config.json").is_file():
        return root
    if (root / "pretrained_model" / "config.json").is_file():
        return root / "pretrained_model"

    candidates = list(root.glob("**/pretrained_model/config.json")) if root.is_dir() else []
    if not candidates:
        return None

    def sort_key(config_path: Path) -> tuple[int, float]:
        digits = "".join(ch for ch in config_path.parent.parent.name if ch.isdigit())
        try:
            modified = config_path.stat().st_mtime
        except OSError:
            modified = 0.0
        return (int(digits) if digits else -1, modified)

    return max(candidates, key=sort_key).parent


class Model(ModelTemplate):
    """XPolicyLab adapter for WSA-Base and WSA-Large RoboTwin checkpoints."""

    def __init__(self, model_cfg):
        self.model_cfg = dict(model_cfg)
        self.action_type = str(self.model_cfg.get("action_type") or "joint")
        if self.action_type != "joint":
            raise ValueError("WSA RoboTwin checkpoints support only action_type='joint'.")

        self.env_cfg_type = _optional_str(self.model_cfg.get("env_cfg_type"))
        if self.env_cfg_type is None:
            raise ValueError("env_cfg_type is required by the WSA adapter.")
        if self.env_cfg_type != "aloha_agilex":
            raise ValueError(
                "Released WSA RoboTwin checkpoints support only "
                f"env_cfg_type='aloha_agilex', got {self.env_cfg_type!r}."
            )
        self.robot_action_dim_info = self._load_robot_action_dim_info(self.env_cfg_type)
        if len(self.robot_action_dim_info["arm_dim"]) != 2:
            raise ValueError(
                "Released WSA RoboTwin checkpoints require a dual-arm robot; "
                f"{self.env_cfg_type!r} resolves to {self.robot_action_dim_info!r}."
            )
        self.action_dim = sum(self.robot_action_dim_info["arm_dim"]) + sum(
            self.robot_action_dim_info["ee_dim"]
        )

        self.task_name = str(self.model_cfg.get("task_name") or "default_task")
        self.default_prompt = str(self.model_cfg.get("default_prompt") or self.task_name)
        self.stats_key = str(self.model_cfg.get("stats_key") or "aloha")
        self.action_mode = str(self.model_cfg.get("action_mode") or "delta").lower()
        if self.action_mode not in {"delta", "abs"}:
            raise ValueError(f"action_mode must be 'delta' or 'abs', got {self.action_mode!r}.")

        self.image_history_interval = int(self.model_cfg.get("image_history_interval") or 15)
        self.resize_size = int(self.model_cfg.get("resize_size") or 224)
        self.seed = int(self.model_cfg.get("seed") or 0)
        self._load_task_overrides()

        self.wsa_root = self._resolve_wsa_root()
        self.allow_dummy_policy = _as_bool(self.model_cfg.get("allow_dummy_policy"), default=False)
        self.policy = None
        self.model = None
        self.policy_is_large = False
        self.input_transforms = None
        self.unnormalize_fn = None
        self.policy_config = None
        self.large_state_normalizer = None

        self._latest_obs_by_env: dict[int, dict[str, Any]] = {}
        self._history_by_env: dict[int, dict[str, deque[np.ndarray]]] = {}
        self._latest_env_idx_list: list[int] = [0]

        if self.allow_dummy_policy:
            print("[WSA] allow_dummy_policy=true; checkpoint loading skipped (protocol debug only).")
            return

        self._import_wsa_runtime()
        self.device = self._resolve_device(str(self.model_cfg.get("device") or "cuda"))
        self.dtype = self._resolve_dtype(str(self.model_cfg.get("dtype") or "bfloat16"))
        self.ckpt_dir = self._resolve_checkpoint_dir()
        self._build_policy_and_transforms()
        self.model = self.policy
        print(
            f"[WSA] initialized type={self.policy_config.type}, task={self.task_name}, "
            f"env_cfg={self.env_cfg_type}, action_mode={self.action_mode}, checkpoint={self.ckpt_dir}"
        )

    def _load_robot_action_dim_info(self, env_cfg_type: str) -> dict[str, list[int]]:
        try:
            return get_robot_action_dim_info(env_cfg_type)
        except (FileNotFoundError, KeyError, OSError):
            # Standalone-checkout fallback, matching get_action_dim.sh semantics.
            # Normal RoboTwin/RoboDojo workspaces use the shared helper above.
            fallback = _XPOLICYLAB_PARENT / "XPolicyLab" / "utils" / "robot" / "_robot_info.json"
            if fallback.is_file():
                payload = json.loads(fallback.read_text(encoding="utf-8"))
                if env_cfg_type in payload:
                    return payload[env_cfg_type]
            raise

    def _resolve_wsa_root(self) -> Path:
        configured = _optional_str(self.model_cfg.get("wsa_root")) or os.environ.get("WSA_ROOT")
        root = Path(configured).expanduser() if configured else _CUR_DIR / "WSA"
        if not root.is_absolute():
            root = _CUR_DIR / root
        return root.resolve()

    def _load_task_overrides(self) -> None:
        config_value = _optional_str(self.model_cfg.get("eval_config_path"))
        if config_value:
            config_path = Path(config_value).expanduser()
            if not config_path.is_absolute():
                config_path = _CUR_DIR / config_path
        else:
            config_path = _CUR_DIR / "WSA" / "configs" / "robotwin_eval_config.yaml"

        task_config: dict[str, Any] = {}
        if config_path.is_file():
            import yaml

            payload = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
            task_config = payload.get(self.task_name, {}) or {}

        explicit_horizon = self.model_cfg.get("infer_horizon")
        if _optional_str(explicit_horizon) is None:
            explicit_horizon = task_config.get("infer_horizon", 16)
        self.infer_horizon = int(explicit_horizon)
        if self.infer_horizon <= 0:
            raise ValueError(f"infer_horizon must be positive, got {self.infer_horizon}.")

        explicit_binarize = self.model_cfg.get("binarize_gripper")
        if _optional_str(explicit_binarize) is None:
            explicit_binarize = task_config.get("binarize_gripper", False)
        self.binarize_gripper = _as_bool(explicit_binarize, default=False)

    def _import_wsa_runtime(self) -> None:
        src_root = self.wsa_root / "src"
        if not (src_root / "lerobot").is_dir():
            raise FileNotFoundError(
                f"WSA source tree not found at {self.wsa_root}. Set wsa_root/WSA_ROOT to a WSA checkout."
            )
        for path in (str(_XPOLICYLAB_PARENT), str(src_root), str(self.wsa_root)):
            if path not in sys.path:
                sys.path.insert(0, path)

        import torch
        import torch.nn.functional as torch_f
        from huggingface_hub import snapshot_download
        from lerobot.configs.policies import PreTrainedConfig
        from lerobot.configs.train import TrainPipelineConfig
        from lerobot.datasets.utils import load_json
        from lerobot.policies.factory import get_policy_class
        from lerobot.policies.names import is_wsa_base, is_wsa_large
        from lerobot.policies.WSA_Base.transform_wsa_base import (
            Qwen3_VLProcessorTransformFn as WSABaseProcessorTransformFn,
        )
        from lerobot.policies.WSA_Large.core.data.lerobot.utils.normalizer import (
            SingleFieldLinearNormalizer,
            load_dataset_stats_from_json,
        )
        from lerobot.policies.WSA_Large.dataset_wsa_large import (
            resolve_wsa_large_concat_layout,
            resolve_wsa_large_video_size,
        )
        from lerobot.policies.WSA_Large.stats_adapter import ensure_wsa_large_stats_format
        from lerobot.policies.WSA_Large.text_cache import build_wsa_large_prompt
        from lerobot.transforms.core import (
            NormalizeTransformFn,
            RemapImageKeyTransformFn,
            ResizeImagesWithPadFn,
            UnNormalizeTransformFn,
            compose,
        )
        from lerobot.utils.constants import OBS_IMAGES

        self.torch = torch
        self.torch_f = torch_f
        self.snapshot_download = snapshot_download
        self.PreTrainedConfig = PreTrainedConfig
        self.TrainPipelineConfig = TrainPipelineConfig
        self.load_json = load_json
        self.get_policy_class = get_policy_class
        self.is_wsa_base = is_wsa_base
        self.is_wsa_large = is_wsa_large
        self.WSABaseProcessorTransformFn = WSABaseProcessorTransformFn
        self.SingleFieldLinearNormalizer = SingleFieldLinearNormalizer
        self.load_dataset_stats_from_json = load_dataset_stats_from_json
        self.resolve_wsa_large_concat_layout = resolve_wsa_large_concat_layout
        self.resolve_wsa_large_video_size = resolve_wsa_large_video_size
        self.ensure_wsa_large_stats_format = ensure_wsa_large_stats_format
        self.build_wsa_large_prompt = build_wsa_large_prompt
        self.NormalizeTransformFn = NormalizeTransformFn
        self.RemapImageKeyTransformFn = RemapImageKeyTransformFn
        self.ResizeImagesWithPadFn = ResizeImagesWithPadFn
        self.UnNormalizeTransformFn = UnNormalizeTransformFn
        self.compose = compose
        self.OBS_IMAGES = OBS_IMAGES

    def _resolve_device(self, requested: str):
        if requested == "auto":
            requested = "cuda" if self.torch.cuda.is_available() else "cpu"
        device = self.torch.device(requested)
        if device.type == "cuda" and not self.torch.cuda.is_available():
            logging.warning("CUDA is unavailable; falling back to CPU for WSA.")
            device = self.torch.device("cpu")
        return device

    def _resolve_dtype(self, requested: str):
        normalized = requested.lower()
        mapping = {
            "float32": self.torch.float32,
            "float16": self.torch.float16,
            "bfloat16": self.torch.bfloat16,
        }
        if normalized not in mapping:
            raise ValueError(f"Unsupported WSA dtype {requested!r}.")
        if self.device.type == "cpu" and normalized != "float32":
            return self.torch.float32
        return mapping[normalized]

    def _resolve_checkpoint_dir(self) -> Path:
        candidates = candidate_checkpoint_roots(
            self.model_cfg,
            _CHECKPOINTS_DIR,
            policy_dir=_CUR_DIR,
            explicit_keys=("checkpoint_path", "ckpt_path", "model_path"),
        )
        for candidate in candidates:
            resolved = _latest_pretrained_dir(candidate)
            if resolved is not None:
                return resolved.resolve()

        source = None
        for key in ("checkpoint_path", "ckpt_path", "model_path", "ckpt_name"):
            source = _optional_str(self.model_cfg.get(key))
            if source:
                break
        if source and source.count("/") == 1 and "\\" not in source and not source.startswith("."):
            downloaded = Path(self.snapshot_download(repo_id=source))
            resolved = _latest_pretrained_dir(downloaded)
            if resolved is not None:
                return resolved.resolve()
        checked = "\n  ".join(str(path) for path in candidates) or "<no local candidates>"
        raise FileNotFoundError(
            "Could not resolve a WSA checkpoint. Pass a local checkpoint path, a standard "
            "XPolicyLab run name, or a Hugging Face repo id such as "
            f"'zaleni/WSA-Base-RoboTwin'. Checked:\n  {checked}"
        )

    def _apply_config_overrides(self, config) -> None:
        override_keys = {
            "qwen3_vl_pretrained_path": "qwen3_vl_pretrained_path",
            "qwen3_vl_processor_path": "qwen3_vl_processor_path",
            "cosmos_tokenizer_path_or_name": "cosmos_tokenizer_path_or_name",
            "da3_model_path_or_name": "da3_model_path_or_name",
            "da3_code_root": "da3_code_root",
            "wsa_large_model_id": "model_id",
            "wsa_large_tokenizer_model_id": "tokenizer_model_id",
            "wsa_large_action_dit_pretrained_path": "action_dit_pretrained_path",
            "wsa_large_future_3d_pretrained_path": "future_3d_pretrained_path",
        }
        for cfg_key, attr_name in override_keys.items():
            value = _optional_str(self.model_cfg.get(cfg_key))
            if value is not None and hasattr(config, attr_name):
                setattr(config, attr_name, value)

        if hasattr(config, "lambda_3d") and _as_bool(
            self.model_cfg.get("disable_3d_teacher_for_eval"), default=True
        ):
            config.lambda_3d = 0.0
        if self.is_wsa_large(config.type):
            bool_overrides = {
                "wsa_large_load_text_encoder": ("load_text_encoder", True),
                "wsa_large_redirect_common_files": ("redirect_common_files", True),
                "wsa_large_skip_dit_load_from_pretrain": ("skip_dit_load_from_pretrain", True),
            }
            for cfg_key, (attr_name, default) in bool_overrides.items():
                if hasattr(config, attr_name):
                    setattr(config, attr_name, _as_bool(self.model_cfg.get(cfg_key), default=default))
            if hasattr(config, "dtype"):
                config.dtype = str(self.model_cfg.get("dtype") or "bfloat16")
        inference_steps = self.model_cfg.get("num_inference_steps")
        if _optional_str(inference_steps) is not None and hasattr(config, "num_inference_steps"):
            config.num_inference_steps = int(inference_steps)

    def _checkpoint_load_device(self, config) -> str:
        """Choose where checkpoint tensors are materialized during model loading.

        LeRobot's ``from_pretrained`` passes ``config.device`` to safetensors.
        Loading directly on CUDA therefore holds both the constructed WSA model
        and a full CUDA state dict at once. Stage WSA-Base on CPU by default and
        perform one final dtype/device transfer after the state dict is consumed.
        """
        stage_on_cpu = _as_bool(
            self.model_cfg.get("load_checkpoint_on_cpu"), default=True
        )
        if self.is_wsa_base(config.type) and self.device.type == "cuda" and stage_on_cpu:
            return "cpu"
        return str(self.device)

    def _release_checkpoint_load_cache(self) -> None:
        # The temporary safetensors state dict is out of scope once
        # ``from_pretrained`` returns. Force collection before returning unused
        # cached blocks to CUDA so nvidia-smi reflects the resident model rather
        # than the checkpoint-loading peak.
        gc.collect()
        if self.device.type == "cuda":
            self.torch.cuda.empty_cache()

    def _log_cuda_model_memory(self) -> None:
        if self.device.type != "cuda":
            return
        parameter_bytes = sum(
            parameter.numel() * parameter.element_size()
            for parameter in self.policy.parameters()
        )
        buffer_bytes = sum(
            buffer.numel() * buffer.element_size()
            for buffer in self.policy.buffers()
        )
        allocated = self.torch.cuda.memory_allocated(self.device)
        reserved = self.torch.cuda.memory_reserved(self.device)
        gib = 1024**3
        print(
            "[WSA] CUDA memory after load: "
            f"parameters+buffers={(parameter_bytes + buffer_bytes) / gib:.2f} GiB, "
            f"allocated={allocated / gib:.2f} GiB, reserved={reserved / gib:.2f} GiB"
        )

    def _build_policy_and_transforms(self) -> None:
        config = self.PreTrainedConfig.from_pretrained(self.ckpt_dir)
        if not (self.is_wsa_base(config.type) or self.is_wsa_large(config.type)):
            raise ValueError(f"Expected a WSA-Base or WSA-Large checkpoint, got type={config.type!r}.")
        self._apply_config_overrides(config)
        self.policy_is_large = self.is_wsa_large(config.type)

        target_device = str(self.device)
        load_device = self._checkpoint_load_device(config)
        config.device = load_device
        if load_device != target_device:
            logging.info(
                "Staging WSA-Base checkpoint on CPU before one-time transfer to %s.",
                target_device,
            )

        policy_config = config
        if self.policy_is_large and hasattr(config, "action_stats_path"):
            policy_config = copy.deepcopy(config)
            policy_config.action_stats_path = None
        policy_cls = self.get_policy_class(config.type)
        self.policy = policy_cls.from_pretrained(
            config=policy_config, pretrained_name_or_path=self.ckpt_dir
        )
        if self.is_wsa_base(config.type) and hasattr(self.policy, "model"):
            setattr(
                self.policy.model,
                "omit_visual_tokens_in_causal_inference",
                _as_bool(self.model_cfg.get("omit_visual_tokens_in_causal_inference"), default=True),
            )
        self.policy.to(device=self.device, dtype=self.dtype).eval()
        # ``config.device`` is also consulted by a few upstream utilities. Keep
        # both the source config and the policy-owned config truthful after CPU
        # checkpoint staging.
        config.device = target_device
        self.policy.config.device = target_device
        self._release_checkpoint_load_cache()
        self._log_cuda_model_memory()
        self.policy_config = config

        output_features = getattr(config, "output_features", {})
        action_feature = output_features.get("action") if isinstance(output_features, Mapping) else None
        action_shape = getattr(action_feature, "shape", None)
        checkpoint_action_dim = int(
            getattr(config, "action_dim", 0) or (action_shape[0] if action_shape else 0)
        )
        if checkpoint_action_dim and checkpoint_action_dim < self.action_dim:
            raise ValueError(
                f"Checkpoint action dimension {checkpoint_action_dim} is smaller than the robot dimension "
                f"{self.action_dim} resolved from env_cfg_type={self.env_cfg_type!r}."
            )

        if self.policy_is_large:
            self._configure_large_runtime()
        else:
            self._configure_base_runtime()

    def _configure_base_runtime(self) -> None:
        stats = self.load_json(self.ckpt_dir / "stats.json")[self.stats_key]
        stat_keys = ("min", "max", "mean", "std")
        state_stats = {
            "observation.state": {
                key: np.asarray(stats["observation.state"][key]) for key in stat_keys
            }
        }
        action_stats = {
            "action": {key: np.asarray(stats["action"][key]) for key in stat_keys}
        }
        self.unnormalize_fn = self.UnNormalizeTransformFn(
            selected_keys=["action"], mode="mean_std", norm_stats=action_stats
        )
        image_keys = [f"{self.OBS_IMAGES}.image{i}" for i in range(3)]
        processor_path = (
            _optional_str(self.model_cfg.get("qwen3_vl_processor_path"))
            or getattr(self.policy_config, "qwen3_vl_processor_path", None)
            or getattr(self.policy_config, "qwen3_vl_pretrained_path", None)
        )
        if processor_path is None:
            raise ValueError("Could not resolve the Qwen3-VL processor used by WSA-Base.")
        self.input_transforms = self.compose(
            [
                self.ResizeImagesWithPadFn(height=self.resize_size, width=self.resize_size),
                self.RemapImageKeyTransformFn(mapping={key: key for key in image_keys}),
                self.WSABaseProcessorTransformFn(
                    pretrained_model_name_or_path=processor_path,
                    max_length=int(getattr(self.policy_config, "tokenizer_max_length", 48)),
                ),
                self.NormalizeTransformFn(
                    selected_keys=["observation.state"], norm_stats=state_stats
                ),
            ]
        )

    def _candidate_large_stats_paths(self) -> list[Path]:
        candidates: list[Path] = []
        explicit = _optional_str(self.model_cfg.get("wsa_large_stats_path"))
        if explicit:
            raw = Path(explicit).expanduser()
            candidates.extend([raw, _CUR_DIR / raw, self.ckpt_dir / raw, self.wsa_root / raw])
        candidates.append(self.ckpt_dir / "stats.json")
        try:
            train_config = self.TrainPipelineConfig.from_pretrained(self.ckpt_dir)
        except Exception:
            train_config = None
        dataset_config = getattr(train_config, "dataset", None) if train_config is not None else None
        normalization_path = getattr(dataset_config, "normalization_stats_path", None)
        if normalization_path:
            raw = Path(str(normalization_path)).expanduser()
            candidates.extend([raw, self.ckpt_dir / raw, self.wsa_root / raw])
        external_root = getattr(dataset_config, "external_stats_root", None)
        if external_root:
            raw = Path(str(external_root)).expanduser()
            for root in (raw, self.ckpt_dir / raw, self.wsa_root / raw):
                candidates.append(root / self.stats_key / self.action_mode / "stats.json")

        unique: list[Path] = []
        seen: set[str] = set()
        for candidate in candidates:
            try:
                candidate = candidate.resolve()
            except OSError:
                pass
            if str(candidate) not in seen:
                seen.add(str(candidate))
                unique.append(candidate)
        return unique

    def _load_large_stats(self, *, require_state: bool) -> dict[str, Any] | None:
        shape_meta = {
            "action": [{"key": "default", "raw_shape": self.action_dim, "shape": self.action_dim}],
            "state": [{"key": "default", "raw_shape": self.action_dim, "shape": self.action_dim}],
        }
        for stats_path in self._candidate_large_stats_paths():
            if not stats_path.is_file():
                continue
            raw = self.load_dataset_stats_from_json(str(stats_path))
            if not any(key in raw for key in ("WSA_Large", "wsa_large", "wsa_large_policy")):
                keyed = raw.get(self.stats_key)
                if isinstance(keyed, dict):
                    raw = keyed
            try:
                payload = self.ensure_wsa_large_stats_format(
                    raw, shape_meta=shape_meta, require_state=require_state
                )
            except Exception as exc:
                logging.warning("Skipping unusable WSA-Large stats %s: %s", stats_path, exc)
                continue
            logging.info("Loaded WSA-Large runtime stats from %s", stats_path)
            return payload
        return None

    def _configure_large_runtime(self) -> None:
        action_stats = self._load_large_stats(require_state=False)
        if not getattr(self.policy, "_action_denorm_specs", None):
            if action_stats is None:
                raise FileNotFoundError(
                    "WSA-Large action normalization stats are required. Put stats.json in the "
                    "checkpoint or set wsa_large_stats_path/WSA_LARGE_STATS_PATH."
                )
            self.policy.set_action_postprocess_from_stats(action_stats)

        state_payload = self._load_large_stats(require_state=True)
        if state_payload is None:
            raise FileNotFoundError(
                "WSA-Large state normalization stats are required. Put stats.json in the "
                "checkpoint or set wsa_large_stats_path/WSA_LARGE_STATS_PATH."
            )
        state_stats = state_payload.get("state", {})
        state_key = str(self.model_cfg.get("wsa_large_state_key") or "default")
        if state_key not in state_stats:
            if len(state_stats) == 1:
                state_key = next(iter(state_stats))
            else:
                raise KeyError(
                    f"WSA-Large state stats key {state_key!r} not found; available={list(state_stats)}."
                )
        selected = {
            key.removeprefix("global_"): value
            for key, value in state_stats[state_key].items()
            if key.startswith("global_")
        }
        mode = str(getattr(self.policy_config, "action_norm_default_mode", "z-score"))
        exception_mode = getattr(self.policy_config, "action_norm_exception_mode", None) or {}
        mode = exception_mode.get("state", {}).get(state_key, mode)
        self.large_state_normalizer = self.SingleFieldLinearNormalizer(stats=selected, mode=mode)

    def reset(self):
        if self.policy is not None:
            self.policy.reset()
        self._latest_obs_by_env = {}
        self._history_by_env = {}
        self._latest_env_idx_list = [0]

    def update_obs(self, obs):
        self.update_obs_batch([obs])

    def update_obs_batch(self, obs_list):
        if isinstance(obs_list, Mapping):
            obs_list = [obs_list]
        if not obs_list:
            raise ValueError("update_obs_batch received no observations.")
        self._latest_env_idx_list = []
        for index, observation in enumerate(obs_list):
            env_idx = int(observation.get("env_idx", index))
            encoded = _encode_observation(
                observation,
                self.action_type,
                self.robot_action_dim_info,
                self.default_prompt,
            )
            self._latest_obs_by_env[env_idx] = encoded
            self._latest_env_idx_list.append(env_idx)
            history = self._history_by_env.setdefault(
                env_idx,
                {
                    name: deque(maxlen=self.image_history_interval + 1)
                    for name in ("cam_high", "cam_left_wrist", "cam_right_wrist")
                },
            )
            for name, image in encoded["images"].items():
                history[name].append(image)

    def _gripper_slices(self) -> list[slice]:
        slices: list[slice] = []
        offset = 0
        for arm_dim, ee_dim in zip(
            self.robot_action_dim_info["arm_dim"], self.robot_action_dim_info["ee_dim"]
        ):
            offset += arm_dim
            slices.append(slice(offset, offset + ee_dim))
            offset += ee_dim
        return slices

    def _postprocess_actions(self, actions, state: np.ndarray) -> np.ndarray:
        actions = self.torch.as_tensor(actions)[..., : self.action_dim]
        if actions.shape[-1] != self.action_dim:
            raise RuntimeError(
                f"WSA returned action shape {tuple(actions.shape)}, expected last dim {self.action_dim}."
            )
        if self.action_mode == "delta":
            initial = self.torch.as_tensor(state, device=actions.device, dtype=actions.dtype).clone()
            for gripper_slice in self._gripper_slices():
                initial[gripper_slice] = 0.0
            actions = actions + initial
        result = actions.detach().cpu().to(self.torch.float32).numpy()
        if self.binarize_gripper:
            for gripper_slice in self._gripper_slices():
                result[..., gripper_slice] = (result[..., gripper_slice] >= 0.5).astype(np.float32)
        return result

    def _infer_base(self, env_idx: int, observation: dict[str, Any]):
        history = self._history_by_env[env_idx]
        image_tensors = []
        for name in ("cam_high", "cam_left_wrist", "cam_right_wrist"):
            frames = history[name]
            past_index = max(len(frames) - self.image_history_interval - 1, 0)
            stacked = np.stack((frames[past_index], frames[-1]), axis=0)
            tensor = self.torch.as_tensor(stacked).contiguous().to(self.dtype) / 255.0
            image_tensors.append(tensor.permute(0, 3, 1, 2))

        state = self.torch.from_numpy(observation["state"]).float()
        sample = {
            f"{self.OBS_IMAGES}.image0": image_tensors[0],
            f"{self.OBS_IMAGES}.image1": image_tensors[1],
            f"{self.OBS_IMAGES}.image2": image_tensors[2],
            "observation.state": state,
            "task": observation["prompt"],
        }
        sample = self.input_transforms(sample)
        inputs: dict[str, Any] = {}
        for key, value in sample.items():
            if key == "task":
                inputs[key] = [value]
            elif isinstance(value, self.torch.Tensor):
                value = value.unsqueeze(0)
                if value.dtype.is_floating_point:
                    value = value.to(device=self.device, dtype=self.dtype)
                else:
                    value = value.to(device=self.device)
                inputs[key] = value
            else:
                inputs[key] = value
        for index in range(3):
            inputs[f"{self.OBS_IMAGES}.image{index}_mask"] = self.torch.tensor(
                [True], device=self.device
            )

        with self.torch.inference_mode():
            action_pred, _ = self.policy.predict_action_chunk(inputs, decode_image=False)
        action_pred = action_pred[0, : self.infer_horizon, : self.action_dim]
        action_pred = self.unnormalize_fn({"action": action_pred})["action"]
        return self._postprocess_actions(action_pred, observation["state"])

    def _large_image_tensor(self, image: np.ndarray):
        tensor = self.torch.as_tensor(image).permute(2, 0, 1).contiguous()
        return tensor.to(self.torch.float32) / 255.0

    def _resize_large_view(self, video, size: tuple[int, int]):
        try:
            return self.torch_f.interpolate(
                video, size=size, mode="bilinear", align_corners=False, antialias=True
            )
        except TypeError:
            return self.torch_f.interpolate(
                video, size=size, mode="bilinear", align_corners=False
            )

    def _concat_large_views(self, video, target_size: tuple[int, int], layout: str):
        camera_count = int(video.shape[0])
        height, width = target_size
        if layout == "single":
            if camera_count != 1:
                raise ValueError(f"single layout requires one camera, got {camera_count}.")
            return self._resize_large_view(video[0], target_size)
        if layout == "horizontal":
            if width % camera_count:
                raise ValueError("WSA-Large horizontal layout width must divide by camera count.")
            return self.torch.cat(
                [self._resize_large_view(view, (height, width // camera_count)) for view in video],
                dim=-1,
            )
        if layout == "vertical":
            if height % camera_count:
                raise ValueError("WSA-Large vertical layout height must divide by camera count.")
            return self.torch.cat(
                [self._resize_large_view(view, (height // camera_count, width)) for view in video],
                dim=-2,
            )
        if layout == "robotwin":
            if camera_count != 3 or height % 3 or width % 2:
                raise ValueError(
                    "WSA-Large robotwin layout requires 3 cameras, height divisible by 3, "
                    "and width divisible by 2."
                )
            bottom_height = height // 3
            top = self._resize_large_view(video[0], (height - bottom_height, width))
            left = self._resize_large_view(video[1], (bottom_height, width // 2))
            right = self._resize_large_view(video[2], (bottom_height, width // 2))
            return self.torch.cat((top, self.torch.cat((left, right), dim=-1)), dim=-2)
        raise ValueError(f"Unsupported WSA-Large camera layout {layout!r}.")

    def _infer_large(self, observation: dict[str, Any]):
        images = observation["images"]
        video = self.torch.stack(
            [
                self._large_image_tensor(images["cam_high"]),
                self._large_image_tensor(images["cam_left_wrist"]),
                self._large_image_tensor(images["cam_right_wrist"]),
            ],
            dim=0,
        ).unsqueeze(1)
        configured_size = (
            int(self.model_cfg.get("wsa_large_video_height") or 384),
            int(self.model_cfg.get("wsa_large_video_width") or 320),
        )
        standardize = _as_bool(
            self.model_cfg.get("wsa_large_standardize_video_size_by_cameras"), default=True
        )
        target_size = self.resolve_wsa_large_video_size(3, configured_size, standardize)
        layout = self.resolve_wsa_large_concat_layout(
            3, str(self.model_cfg.get("wsa_large_concat_multi_camera") or "robotwin")
        )
        input_image = (self._concat_large_views(video, target_size, layout) - 0.5) / 0.5

        state = self.torch.from_numpy(observation["state"]).to(self.torch.float32)
        proprio = state.detach().cpu().flatten()
        if self.large_state_normalizer is not None:
            proprio = self.large_state_normalizer.forward(proprio)
        target_dim = int(getattr(self.policy_config, "proprio_dim", self.action_dim))
        if proprio.numel() < target_dim:
            proprio = self.torch_f.pad(proprio, (0, target_dim - proprio.numel()))
        elif proprio.numel() > target_dim:
            proprio = proprio[:target_dim]
        inputs = {
            "input_image": input_image,
            "proprio": proprio.unsqueeze(0),
            "prompt": [self.build_wsa_large_prompt(observation["prompt"])],
        }
        with self.torch.inference_mode():
            action_pred = self.policy.predict_action_chunk(inputs, seed=self.seed)
        action_pred = action_pred[0, : self.infer_horizon, : self.action_dim]
        return self._postprocess_actions(action_pred, observation["state"])

    def _dummy_actions(self, observation: dict[str, Any]) -> np.ndarray:
        # Hold the current joint configuration; this is protocol testing only.
        return np.repeat(
            np.asarray(observation["state"], dtype=np.float32)[None],
            repeats=min(self.infer_horizon, 4),
            axis=0,
        )

    def infer(self, env_idx: int) -> np.ndarray:
        observation = self._latest_obs_by_env.get(env_idx)
        if observation is None:
            raise AssertionError(f"No observation stored for env_idx={env_idx}; call update_obs first.")
        if self.allow_dummy_policy:
            return self._dummy_actions(observation)
        if self.policy_is_large:
            return self._infer_large(observation)
        return self._infer_base(env_idx, observation)

    def get_action(self):
        env_idx = self._latest_env_idx_list[0]
        return self.get_action_batch([env_idx])[0]

    def get_action_batch(self, env_idx_list=None):
        if env_idx_list is None:
            env_idx_list = self._latest_env_idx_list
        elif isinstance(env_idx_list, (int, np.integer)):
            env_idx_list = [int(env_idx_list)]
        else:
            env_idx_list = list(env_idx_list)
        if not env_idx_list:
            raise ValueError("get_action_batch received an empty env_idx_list.")
        chunks = []
        for env_idx in env_idx_list:
            raw_actions = self.infer(int(env_idx))
            chunks.append(
                unpack_robot_state(
                    raw_actions,
                    self.action_type,
                    self.robot_action_dim_info,
                    source_type="obs",
                )
            )
        return chunks
