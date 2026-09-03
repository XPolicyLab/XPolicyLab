"""Direct Hugging Face MolmoAct2 inference adapter for XPolicyLab."""

from __future__ import annotations

import contextlib
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import numpy as np
import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

_POLICY_DIR = Path(__file__).resolve().parent
_IMPORTABLE_ROOT = _POLICY_DIR.parents[2]
_CHECKPOINTS_DIR = _POLICY_DIR / "checkpoints"

if str(_IMPORTABLE_ROOT) not in sys.path:
    sys.path.insert(0, str(_IMPORTABLE_ROOT))

from XPolicyLab.model_template import ModelTemplate
from XPolicyLab.utils.checkpoint_resolver import candidate_checkpoint_roots
from XPolicyLab.utils.process_data import (
    get_robot_action_dim_info,
    pack_robot_state,
    unpack_robot_state,
)

_DEFAULT_REPO_ID = "hqfang/MolmoAct2-RoboDojo"
_DEFAULT_REVISION = "68964756dbfe5b455e6b4e4aa571199aa17d087c"
_HF_REQUIRED_FILES = ("config.json", "processor_config.json", "norm_stats.json")
_COMMIT_PATTERN = re.compile(r"^[0-9a-f]{40}$")

_CAMERA_CANDIDATES: dict[str, tuple[str, ...]] = {
    "cam_high": ("cam_high", "cam_head", "head_camera", "top_camera"),
    "cam_left_wrist": ("cam_left_wrist", "left_camera", "left_wrist", "wrist_left"),
    "cam_right_wrist": ("cam_right_wrist", "right_camera", "right_wrist", "wrist_right"),
}

Action = dict[str, np.ndarray]


@dataclass(frozen=True)
class CheckpointSource:
    """Resolved local snapshot or immutable Hugging Face Hub reference."""

    pretrained_name_or_path: str
    revision: str | None
    is_local: bool
    display_name: str


def _clean_text(value: Any) -> str | None:
    """Return a stripped string for scalar config values."""

    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _is_complete_hf_snapshot(path: Path) -> bool:
    """Check the minimum files needed by the public Transformers loader."""

    if not path.is_dir():
        return False
    if not all((path / filename).is_file() for filename in _HF_REQUIRED_FILES):
        return False
    return (path / "model.safetensors").is_file() or (
        path / "model.safetensors.index.json"
    ).is_file()


def _parse_hf_reference(
    value: str,
    *,
    configured_repo_id: str,
    default_revision: str,
) -> tuple[str, str] | None:
    """Parse the configured repo, an ``hf://`` URI, or a Hub web URL."""

    text = value.strip()
    revision: str | None = None

    if text.startswith("hf://"):
        text = text.removeprefix("hf://")
        if "@" in text:
            text, revision = text.rsplit("@", 1)
    elif text.startswith("https://huggingface.co/") or text.startswith("http://huggingface.co/"):
        parsed = urlparse(text)
        parts = [part for part in parsed.path.split("/") if part]
        if len(parts) < 2:
            raise ValueError(f"Invalid Hugging Face checkpoint URL: {value!r}")
        text = "/".join(parts[:2])
        if len(parts) >= 4 and parts[2] in {"tree", "resolve"}:
            revision = parts[3]
    elif text != configured_repo_id:
        return None

    if text.count("/") != 1:
        raise ValueError(f"Expected a Hugging Face repo id in 'owner/name' form, got {text!r}")
    resolved_revision = _clean_text(revision) or default_revision
    if not _COMMIT_PATTERN.fullmatch(resolved_revision):
        raise ValueError(
            "MolmoAct2 Hub inference requires an immutable 40-character commit revision, "
            f"got {resolved_revision!r}."
        )
    return text, resolved_revision


def resolve_checkpoint_source(model_cfg: dict[str, Any]) -> CheckpointSource:
    """Resolve local checkpoints through XPolicyLab, then accept pinned Hub references."""

    local_candidates = candidate_checkpoint_roots(
        model_cfg,
        _CHECKPOINTS_DIR,
        policy_dir=_POLICY_DIR,
        explicit_keys=("pretrained_path", "model_path", "checkpoint_path"),
    )
    for candidate in local_candidates:
        if not candidate.exists():
            continue
        if not _is_complete_hf_snapshot(candidate):
            raise FileNotFoundError(
                f"Resolved checkpoint exists but is not a complete Transformers snapshot: {candidate}"
            )
        return CheckpointSource(
            pretrained_name_or_path=str(candidate),
            revision=None,
            is_local=True,
            display_name=str(candidate),
        )

    configured_repo_id = str(model_cfg.get("hf_repo_id") or _DEFAULT_REPO_ID)
    default_revision = str(model_cfg.get("hf_revision") or _DEFAULT_REVISION)
    reference_candidates = (
        _clean_text(model_cfg.get("ckpt_name")),
        _clean_text(model_cfg.get("hf_repo_id")),
    )
    for reference in reference_candidates:
        if reference is None:
            continue
        parsed = _parse_hf_reference(
            reference,
            configured_repo_id=configured_repo_id,
            default_revision=default_revision,
        )
        if parsed is None:
            continue
        repo_id, revision = parsed
        return CheckpointSource(
            pretrained_name_or_path=repo_id,
            revision=revision,
            is_local=False,
            display_name=f"{repo_id}@{revision}",
        )

    checked = "\n  ".join(str(path) for path in local_candidates) or "<none>"
    raise FileNotFoundError(
        "Could not resolve a MolmoAct2 checkpoint. Pass the configured Hugging Face repo id, "
        "an hf://owner/name@commit reference, a huggingface.co URL, or a complete local snapshot. "
        f"\nLocal candidates checked:\n  {checked}"
    )


def _extract_image(observation: dict[str, Any], candidate_names: tuple[str, ...]) -> np.ndarray:
    """Read an already-decoded RGB image from an XPolicyLab observation."""

    vision = observation.get("vision")
    if not isinstance(vision, dict):
        raise KeyError("Observation is missing the 'vision' mapping")
    for candidate_name in candidate_names:
        if candidate_name not in vision:
            continue
        value = vision[candidate_name]
        if isinstance(value, dict):
            for image_key in ("color", "rgb"):
                if image_key in value:
                    return np.asarray(value[image_key])
        return np.asarray(value)
    raise KeyError(f"Could not find any image for candidates: {candidate_names}")


def _as_rgb_pil(image: np.ndarray) -> Image.Image:
    """Convert an already-decoded RGB HWC/CHW array to an RGB PIL image."""

    array = np.asarray(image)
    if array.ndim != 3:
        raise ValueError(f"Expected an RGB image with three dimensions, got shape {array.shape}")
    if array.shape[0] in (1, 3) and array.shape[-1] not in (1, 3):
        array = np.moveaxis(array, 0, -1)
    if array.shape[-1] != 3:
        raise ValueError(f"Expected three RGB channels, got shape {array.shape}")
    if np.issubdtype(array.dtype, np.floating):
        maximum = float(np.nanmax(array)) if array.size else 0.0
        scale = 255.0 if maximum <= 1.0 else 1.0
        array = np.clip(array * scale, 0.0, 255.0).astype(np.uint8)
    elif array.dtype != np.uint8:
        array = np.clip(array, 0, 255).astype(np.uint8)
    return Image.fromarray(np.ascontiguousarray(array))


def _normalize_instruction(value: Any) -> str | None:
    """Normalize scalar, numpy, or list instruction representations."""

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
            instruction = _normalize_instruction(item)
            if instruction is not None:
                return instruction
        return None
    if isinstance(value, str):
        value = value.strip()
        return value or None
    return str(value)


def _resolve_instruction(observation: dict[str, Any], fallback: str) -> str:
    """Prefer the task/layout-specific instruction emitted by the environment."""

    for key in ("instruction", "instructions", "prompt", "task", "language_instruction"):
        instruction = _normalize_instruction(observation.get(key))
        if instruction is not None:
            return instruction
    instruction = _normalize_instruction(fallback)
    if instruction is None:
        raise ValueError("No valid language instruction is available for MolmoAct2 inference")
    return instruction


def _resolve_dtype(name: str) -> torch.dtype:
    """Map the documented deploy value to a supported inference dtype."""

    normalized = name.strip().lower()
    dtypes = {
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    if normalized not in dtypes:
        raise ValueError(f"Unsupported MolmoAct2 dtype {name!r}; use bfloat16 or float32")
    return dtypes[normalized]


class Model(ModelTemplate):
    """XPolicyLab model contract backed by the checkpoint's public HF API."""

    def __init__(self, model_cfg: dict[str, Any]) -> None:
        super().__init__()
        self.model_cfg = dict(model_cfg)
        self.action_type = str(self.model_cfg.get("action_type") or "joint")
        if self.action_type != "joint":
            raise ValueError("MolmoAct2-RoboDojo supports only action_type='joint'")

        env_cfg_type = _clean_text(self.model_cfg.get("env_cfg_type"))
        if env_cfg_type is None:
            raise ValueError("env_cfg_type is required")
        self.robot_action_dim_info = get_robot_action_dim_info(env_cfg_type)
        self.action_dim = sum(self.robot_action_dim_info["arm_dim"]) + sum(
            self.robot_action_dim_info["ee_dim"]
        )
        self.default_instruction = str(
            self.model_cfg.get("prompt") or self.model_cfg.get("task_name") or ""
        )
        self.norm_tag = str(self.model_cfg.get("norm_tag") or "robodojo")
        self.inference_action_mode = str(
            self.model_cfg.get("inference_action_mode") or "continuous"
        )
        if self.inference_action_mode != "continuous":
            raise ValueError("MolmoAct2-RoboDojo supports only continuous inference")
        self.expected_action_representation = str(
            self.model_cfg.get("expected_action_representation") or "absolute"
        )
        if self.expected_action_representation != "absolute":
            raise ValueError("MolmoAct2-RoboDojo supports only absolute action output")
        self.num_steps = int(self.model_cfg.get("num_steps", 10))
        self.actions_per_chunk = int(self.model_cfg.get("actions_per_chunk", 25))
        self.enable_cuda_graph = bool(self.model_cfg.get("enable_cuda_graph", False))
        self.seed = int(self.model_cfg.get("seed", 0))
        self.device = torch.device(str(self.model_cfg.get("device") or "cuda"))
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("MolmoAct2 was configured for CUDA, but CUDA is unavailable")
        self.dtype = _resolve_dtype(str(self.model_cfg.get("dtype") or "bfloat16"))

        self.checkpoint = resolve_checkpoint_source(self.model_cfg)
        common_kwargs: dict[str, Any] = {
            "trust_remote_code": True,
            "local_files_only": self.checkpoint.is_local,
        }
        if self.checkpoint.revision is not None:
            common_kwargs["revision"] = self.checkpoint.revision
            common_kwargs["code_revision"] = self.checkpoint.revision

        self.processor = AutoProcessor.from_pretrained(
            self.checkpoint.pretrained_name_or_path,
            **common_kwargs,
        )
        loaded_model = AutoModelForImageTextToText.from_pretrained(
            self.checkpoint.pretrained_name_or_path,
            dtype=self.dtype,
            **common_kwargs,
        )
        self.model = loaded_model.to(self.device).eval()
        self._validate_checkpoint_contract()

        resolved_commit = _clean_text(getattr(self.model.config, "_commit_hash", None))
        if (
            self.checkpoint.revision is not None
            and resolved_commit is not None
            and resolved_commit != self.checkpoint.revision
        ):
            raise RuntimeError(
                "Resolved checkpoint revision mismatch: "
                f"expected={self.checkpoint.revision} actual={resolved_commit}"
            )
        self.resolved_commit = resolved_commit or self.checkpoint.revision or "local-snapshot"
        self.latest_observations: dict[int, dict[str, Any]] = {}
        self.generators: dict[int, torch.Generator] = {}
        self.seen_instructions: set[str] = set()
        print(
            "[MolmoAct2] loaded "
            f"checkpoint={self.checkpoint.display_name} resolved_commit={self.resolved_commit} "
            f"dtype={self.dtype} device={self.device} action_dim={self.action_dim}",
            flush=True,
        )

    def _validate_checkpoint_contract(self) -> None:
        """Fail before rollout when normalization or action metadata is incompatible."""

        robot_stats = self.model._get_robot_stats()
        norm_tag = robot_stats.validate_tag(self.norm_tag)
        representation = robot_stats.validate_expected_action_representation(
            norm_tag,
            self.expected_action_representation,
        )
        checkpoint_action_dim = robot_stats.get_action_dim(norm_tag)
        checkpoint_state_dim = robot_stats.get_state_dim(norm_tag)
        checkpoint_horizon = robot_stats.get_action_horizon(norm_tag)
        if checkpoint_action_dim != self.action_dim:
            raise ValueError(
                f"Checkpoint action dim {checkpoint_action_dim} does not match robot dim {self.action_dim}"
            )
        if checkpoint_state_dim != self.action_dim:
            raise ValueError(
                f"Checkpoint state dim {checkpoint_state_dim} does not match robot dim {self.action_dim}"
            )
        if checkpoint_horizon is None or checkpoint_horizon < self.actions_per_chunk:
            raise ValueError(
                f"Checkpoint horizon {checkpoint_horizon} is shorter than requested chunk "
                f"{self.actions_per_chunk}"
            )
        if representation != "absolute":
            raise ValueError(f"Expected absolute actions, got {representation!r}")
        if self.num_steps < 1 or self.actions_per_chunk < 1:
            raise ValueError("num_steps and actions_per_chunk must both be positive")

    def _generator_for_env(self, env_idx: int) -> torch.Generator:
        """Return an order-independent deterministic RNG stream for one environment."""

        if env_idx not in self.generators:
            generator = torch.Generator(device=self.device)
            generator.manual_seed((self.seed + env_idx * 1_000_003) % (2**63 - 1))
            self.generators[env_idx] = generator
        return self.generators[env_idx]

    def update_obs(self, obs: dict[str, Any]) -> None:
        self.latest_observations = {0: obs}

    def update_obs_batch(self, obs_list: list[dict[str, Any]]) -> None:
        self.latest_observations = {
            int(observation.get("env_idx", index)): observation
            for index, observation in enumerate(obs_list)
        }

    def _autocast_context(self) -> contextlib.AbstractContextManager[Any]:
        """Use the BF16 CUDA path documented by the checkpoint model card."""

        if self.device.type == "cuda" and self.dtype == torch.bfloat16:
            return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        return contextlib.nullcontext()

    def _predict(self, observation: dict[str, Any], env_idx: int) -> list[Action]:
        """Predict one de-normalized absolute action chunk."""

        images = [
            _as_rgb_pil(_extract_image(observation, _CAMERA_CANDIDATES[camera]))
            for camera in ("cam_high", "cam_left_wrist", "cam_right_wrist")
        ]
        state = pack_robot_state(
            observation,
            self.action_type,
            self.robot_action_dim_info,
            source_type="obs",
        ).astype(np.float32)
        if state.shape != (self.action_dim,):
            raise ValueError(f"Expected robot state shape {(self.action_dim,)}, got {state.shape}")
        instruction = _resolve_instruction(observation, self.default_instruction)
        if instruction not in self.seen_instructions:
            self.seen_instructions.add(instruction)
            print(f"[MolmoAct2] instruction={instruction!r}", flush=True)

        with torch.inference_mode(), self._autocast_context():
            output = self.model.predict_action(
                processor=self.processor,
                images=images,
                task=instruction,
                state=state,
                norm_tag=self.norm_tag,
                expected_action_representation=self.expected_action_representation,
                inference_action_mode=self.inference_action_mode,
                enable_depth_reasoning=False,
                num_steps=self.num_steps,
                n_action_steps=self.actions_per_chunk,
                generator=self._generator_for_env(env_idx),
                normalize_language=True,
                enable_cuda_graph=self.enable_cuda_graph,
            )
        raw_actions = output.actions
        if torch.is_tensor(raw_actions):
            raw_actions = raw_actions.detach().to(device="cpu", dtype=torch.float32).numpy()  # [1,T,D] or [T,D]
        actions = np.asarray(raw_actions, dtype=np.float32)
        if actions.ndim == 3 and actions.shape[0] == 1:
            actions = actions[0]
        expected_shape = (self.actions_per_chunk, self.action_dim)
        if actions.shape != expected_shape:
            raise ValueError(f"Expected action shape {expected_shape}, got {actions.shape}")
        if not bool(np.isfinite(actions).all()):
            raise ValueError("MolmoAct2 returned non-finite actions")

        unpacked = unpack_robot_state(
            actions,
            self.action_type,
            self.robot_action_dim_info,
            source_type="obs",
        )
        if not isinstance(unpacked, list):
            raise TypeError("Expected a list of unpacked action dictionaries")
        return unpacked

    def get_action(self, **kwargs: Any) -> list[Action]:
        del kwargs
        if 0 not in self.latest_observations:
            raise RuntimeError("get_action called before update_obs")
        return self._predict(self.latest_observations[0], env_idx=0)

    def get_action_batch(
        self,
        env_idx_list: list[int] | None = None,
        **kwargs: Any,
    ) -> list[list[Action]]:
        del kwargs
        if env_idx_list is None:
            env_idx_list = sorted(self.latest_observations)
        normalized_indices = [int(env_idx) for env_idx in env_idx_list]
        missing_indices = [
            env_idx for env_idx in normalized_indices if env_idx not in self.latest_observations
        ]
        if missing_indices:
            raise KeyError(f"Missing observations for env indices: {missing_indices}")
        return [
            self._predict(self.latest_observations[env_idx], env_idx=env_idx)
            for env_idx in normalized_indices
        ]

    def reset(self) -> None:
        self.latest_observations = {}
        self.generators = {}
        self.seen_instructions = set()
