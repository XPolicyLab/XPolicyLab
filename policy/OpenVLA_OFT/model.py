import logging
import os
import re
import sys

# OpenVLA constants are selected from sys.argv at import time; force ALOHA for XPolicyLab eval.
if "aloha" not in " ".join(sys.argv).lower():
    sys.argv.append("--aloha")

import numpy as np
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image

from .openvla_oft.prismatic.vla.constants import NUM_ACTIONS_CHUNK, PROPRIO_DIM
from .openvla_oft.experiments.robot.openvla_utils import (
    get_vla,
    get_processor,
    get_action_head,
    get_proprio_projector,
    get_vla_action,
)

from XPolicyLab.model_template import ModelTemplate
from XPolicyLab.utils.process_data import (
    decode_image_bit,
    get_robot_action_dim_info,
    pack_robot_state,
    unpack_robot_state,
)


_POLICY_DIR = Path(__file__).resolve().parent
_CHECKPOINTS_DIR = _POLICY_DIR / "checkpoints"
_ALOHA_PREPROCESS_SIZE = 256

_LOGGER = logging.getLogger("OpenVLA_OFT")

# Training saves loadable checkpoints to sibling dirs named `<run_id>--<step>_chkpt`
# (see openvla_oft/vla-scripts/finetune.py:save_training_checkpoint). The bare
# `<run_id>/` dir usually only carries dataset_statistics.json.
_CHKPT_STEP_RE = re.compile(r"--(\d+)_chkpt$")

# config.json is the reliable marker that a dir holds real fine-tune weights
# (the merged model); a `<run_id>/` dir with only dataset_statistics.json is not.
_WEIGHTS_MARKER = "config.json"
_STATS_FILE = "dataset_statistics.json"


def _extract_step_number(value: Any) -> int | None:
    digits = "".join(ch for ch in str(value) if ch.isdigit())
    return int(digits) if digits else None


def _chkpt_step(path: Path) -> int | None:
    match = _CHKPT_STEP_RE.search(path.name)
    return int(match.group(1)) if match else None


def _has_weights(path: Path) -> bool:
    return (path / _WEIGHTS_MARKER).is_file()


def _iter_weight_dirs(checkpoint_root: Path) -> list[Path]:
    """Return directories under (and including) checkpoint_root that hold real
    fine-tune weights (a config.json), preferring immediate children but falling
    back to a bounded recursive search."""
    found: list[Path] = []
    if _has_weights(checkpoint_root):
        found.append(checkpoint_root)
    for child in sorted(checkpoint_root.iterdir()):
        if child.is_dir() and _has_weights(child):
            found.append(child)
    if found:
        return found
    # Fallback: weights may live deeper than the immediate children.
    seen: set[Path] = set()
    for config_path in sorted(checkpoint_root.rglob(_WEIGHTS_MARKER)):
        parent = config_path.parent
        if parent not in seen:
            seen.add(parent)
            found.append(parent)
    return found


def _select_weight_dir(checkpoint_root: Path, target_step: int | None) -> Path | None:
    """Pick the weights dir under checkpoint_root, honoring checkpoint_num (step)
    selection across multiple `--<step>_chkpt` dirs, else the latest step."""
    weight_dirs = _iter_weight_dirs(checkpoint_root)
    if not weight_dirs:
        return None

    stepped = [(d, _chkpt_step(d)) for d in weight_dirs]
    chkpt_dirs = [(d, step) for d, step in stepped if step is not None]
    if chkpt_dirs:
        if target_step is not None:
            for directory, step in chkpt_dirs:
                if step == target_step:
                    return directory
            available = sorted(step for _, step in chkpt_dirs)
            _LOGGER.warning(
                "[OpenVLA_OFT] checkpoint_num=%s has no matching --%s_chkpt under %s; "
                "available steps=%s. Using the latest step instead.",
                target_step,
                target_step,
                checkpoint_root,
                available,
            )
        return max(chkpt_dirs, key=lambda item: item[1])[0]

    # No step-tagged dirs (e.g. an explicit merged/exported checkpoint dir).
    return weight_dirs[0]


def _resolve_dataset_stats_path(finetune_dir: Path | None) -> Path | None:
    """Locate dataset_statistics.json. The `--<step>_chkpt` weights dir carries a
    copy, but fall back to the run root / sibling run dir where training also
    writes it (finetune.py saves it to run_root_dir)."""
    if finetune_dir is None:
        return None
    direct = finetune_dir / _STATS_FILE
    if direct.is_file():
        return direct
    parent = finetune_dir.parent
    if parent != finetune_dir:
        root_stats = parent / _STATS_FILE
        if root_stats.is_file():
            return root_stats
        for sibling in sorted(parent.iterdir()):
            if sibling.is_dir():
                candidate = sibling / _STATS_FILE
                if candidate.is_file():
                    return candidate
    return None


def _build_ckpt_setting(model_cfg: dict[str, Any]) -> str | None:
    # ckpt_name is the full run directory name under checkpoints/; an explicit
    # ckpt_setting override still takes priority.
    if model_cfg.get("ckpt_setting"):
        return str(model_cfg["ckpt_setting"])
    ckpt_name = model_cfg.get("ckpt_name")
    if ckpt_name is None:
        return None
    return str(ckpt_name)


def _build_tfds_dataset_name(model_cfg: dict[str, Any]) -> str | None:
    # An explicit tfds_dataset_name override still takes priority; otherwise
    # derive from ckpt_name (the full run directory name).
    if model_cfg.get("tfds_dataset_name"):
        return str(model_cfg["tfds_dataset_name"])
    ckpt_name = model_cfg.get("ckpt_name")
    if ckpt_name is None:
        return None
    return f"aloha_{ckpt_name}"


def _resolve_unnorm_key(model_cfg: dict[str, Any], norm_stats: dict | None = None) -> str:
    explicit_key = model_cfg.get("unnorm_key")
    if explicit_key:
        return str(explicit_key)

    candidates = []
    tfds_dataset_name = _build_tfds_dataset_name(model_cfg)
    if tfds_dataset_name:
        candidates.append(tfds_dataset_name)

    if norm_stats:
        for key in candidates:
            if key in norm_stats:
                return key
        if len(norm_stats) == 1:
            return next(iter(norm_stats))
        raise ValueError(
            "Could not resolve unnorm_key. Set deploy.yml unnorm_key or tfds_dataset_name. "
            f"Available keys: {list(norm_stats.keys())}"
        )

    if tfds_dataset_name:
        return tfds_dataset_name
    raise ValueError("unnorm_key, tfds_dataset_name, or ckpt_name is required for OpenVLA_OFT eval.")


def _resolve_finetune_dir(model_cfg: dict[str, Any]) -> Path | None:
    ckpt_setting = _build_ckpt_setting(model_cfg)
    ckpt_name = model_cfg.get("ckpt_name")
    candidates: list[str] = []
    for value in (ckpt_name, ckpt_setting):
        if value and str(value) not in candidates:
            candidates.append(str(value))
    if not candidates:
        return None

    checkpoint_num = model_cfg.get("checkpoint_num")
    target_step = _extract_step_number(checkpoint_num) if checkpoint_num not in (None, "") else None

    existing_roots: list[Path] = []
    for name in candidates:
        checkpoint_root = _resolve_checkpoint_root(name)
        if not checkpoint_root.is_dir():
            continue
        existing_roots.append(checkpoint_root)
        resolved = _select_weight_dir(checkpoint_root, target_step)
        if resolved is not None:
            return resolved

    if existing_roots:
        raise FileNotFoundError(
            f"Found OpenVLA_OFT checkpoint root but no fine-tune weights ({_WEIGHTS_MARKER}) "
            f"under: {existing_roots[0]}. Expected a merged checkpoint directory such as "
            "`<run_id>--<step>_chkpt`."
        )
    if candidates:
        checked = [_resolve_checkpoint_root(name) for name in candidates]
        raise FileNotFoundError(
            "Could not find OpenVLA_OFT checkpoint. Pass the full run directory name "
            f"under checkpoints/ or a valid path. Checked: {[str(path) for path in checked]}"
        )
    return None


def _resolve_policy_path(value: str | None) -> Path | None:
    if not value:
        return None
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = (_POLICY_DIR / path).resolve()
    else:
        path = path.resolve()
    return path


def _resolve_checkpoint_root(name: str) -> Path:
    path = Path(name).expanduser()
    if path.is_absolute():
        return path.resolve()
    if path.parent != Path("."):
        return (_POLICY_DIR / path).resolve()
    return (_CHECKPOINTS_DIR / path).resolve()


def _resolve_checkpoint_path(model_cfg: dict[str, Any]) -> str:
    base_model_path = model_cfg.get("base_model_path")
    explicit_checkpoint = model_cfg.get("checkpoint_path") or model_cfg.get("model_path")
    if explicit_checkpoint:
        path = Path(explicit_checkpoint).expanduser().resolve()
        if (path / "config.json").exists():
            return str(path)

    finetune_dir = _resolve_finetune_dir(model_cfg)
    if finetune_dir is not None and (finetune_dir / "config.json").exists():
        return str(finetune_dir)

    resolved_base = _resolve_policy_path(base_model_path)
    if resolved_base is not None and (resolved_base / "config.json").exists():
        return str(resolved_base)

    ckpt_setting = _build_ckpt_setting(model_cfg)
    if ckpt_setting:
        checkpoint_root = _resolve_checkpoint_root(ckpt_setting)
        return str(checkpoint_root)

    ckpt_name = model_cfg.get("ckpt_name")
    if ckpt_name:
        checkpoint_root = _resolve_checkpoint_root(str(ckpt_name))
        return str(checkpoint_root)

    if explicit_checkpoint:
        return str(Path(explicit_checkpoint).expanduser().resolve())
    raise ValueError("ckpt_name, base_model_path, or checkpoint_path is required for OpenVLA_OFT.")

@dataclass
class InferenceConfig:
    pretrained_checkpoint: str
    use_l1_regression: bool = True
    use_diffusion: bool = False
    use_film: bool = True
    use_proprio: bool = True
    load_in_8bit: bool = False
    load_in_4bit: bool = False
    num_images_in_input: int = 3
    center_crop: bool = True
    unnorm_key: str = ""
    num_open_loop_steps: int = NUM_ACTIONS_CHUNK
    lora_rank: int = 32

def extract_image(observation, candidate_names):
    vision = observation.get("vision", {})
    for candidate_name in candidate_names:
        if candidate_name not in vision:
            continue
        image = vision[candidate_name]
        if isinstance(image, dict):
            for image_key in ("color", "rgb"):
                if image_key in image:
                    return image[image_key]
        else:
            return image
    raise KeyError(f"Could not find any image for candidates: {candidate_names}")


def ensure_hwc_uint8(image):
    if isinstance(image, (bytes, bytearray, memoryview)):
        image = decode_image_bit(np.frombuffer(bytes(image), dtype=np.uint8))

    image = np.asarray(image)
    if image.ndim == 1 and image.dtype == np.uint8:
        image = decode_image_bit(image)

    if image.ndim != 3:
        raise ValueError(f"Expected image ndim=3, got shape {image.shape}")

    if np.issubdtype(image.dtype, np.floating):
        image = np.clip(image, 0.0, 1.0)
        image = (image * 255.0).astype(np.uint8)
    elif image.dtype != np.uint8:
        image = image.astype(np.uint8)

    if image.shape[-1] in (1, 3):
        return image
    if image.shape[0] in (1, 3):
        return np.transpose(image, (1, 2, 0))
    raise ValueError(f"Unsupported image shape: {image.shape}")


def resize_image_for_aloha_preprocessing(image: np.ndarray) -> np.ndarray:
    """Match preprocess_split_aloha_data.py: 480x640 -> 256x256 BICUBIC before RLDS resize."""
    if image.shape[0] == _ALOHA_PREPROCESS_SIZE and image.shape[1] == _ALOHA_PREPROCESS_SIZE:
        return image
    return np.array(
        Image.fromarray(image).resize(
            (_ALOHA_PREPROCESS_SIZE, _ALOHA_PREPROCESS_SIZE),
            resample=Image.BICUBIC,
        )
    )


def prepare_vla_image(image) -> np.ndarray:
    return resize_image_for_aloha_preprocessing(ensure_hwc_uint8(image))


def extract_prompt(observation, default_prompt):
    for key in ("instruction", "instructions", "prompt", "task_instruction"):
        value = observation.get(key)
        if value is None:
            continue
        if isinstance(value, (list, tuple)):
            value = value[0] if value else None
        if value is None:
            continue
        if hasattr(value, "item"):
            value = value.item()
        if isinstance(value, bytes):
            value = value.decode("utf-8", errors="replace")
        text = str(value).strip()
        if text:
            return text
    return default_prompt


def encode_obs(observation, action_type, robot_action_dim_info, default_prompt):
    if "images" in observation and "state" in observation:
        state = np.asarray(observation["state"], dtype=np.float32)
        prompt = extract_prompt(observation, default_prompt)
        return {
            "full_image": prepare_vla_image(observation["images"]["cam_high"]),
            "left_wrist_image": prepare_vla_image(observation["images"]["cam_left_wrist"]),
            "right_wrist_image": prepare_vla_image(observation["images"]["cam_right_wrist"]),
            "state": state,
            "instruction": prompt,
        }

    if robot_action_dim_info is None:
        raise ValueError("env_cfg is required when encoding raw environment observations.")

    images = {
        "cam_high": prepare_vla_image(
            extract_image(observation, ["cam_high", "cam_head", "head_camera", "top_camera"])
        ),
        "cam_left_wrist": prepare_vla_image(
            extract_image(observation, ["cam_left_wrist", "left_camera", "left_wrist", "wrist_left"])
        ),
        "cam_right_wrist": prepare_vla_image(
            extract_image(observation, ["cam_right_wrist", "right_camera", "right_wrist", "wrist_right"])
        ),
    }
    state = pack_robot_state(observation, action_type, robot_action_dim_info, source_type="obs").astype(np.float32)
    prompt = extract_prompt(observation, default_prompt)
    return {
        "full_image": images["cam_high"],
        "left_wrist_image": images["cam_left_wrist"],
        "right_wrist_image": images["cam_right_wrist"],
        "state": state,
        "instruction": prompt,
    }

class Model(ModelTemplate):
    def __init__(self, model_cfg):
        self._finetune_dir = _resolve_finetune_dir(model_cfg)
        self._dataset_stats_path = _resolve_dataset_stats_path(self._finetune_dir)
        self._dataset_stats: dict | None = None
        if self._dataset_stats_path is not None:
            import json

            with open(self._dataset_stats_path, "r", encoding="utf-8") as f:
                self._dataset_stats = json.load(f)

        self.cfg = self.get_model(model_cfg)

        self.vla = get_vla(self.cfg)
        if self._dataset_stats_path is not None:
            from .openvla_oft.experiments.robot.openvla_utils import _load_dataset_stats

            _load_dataset_stats(self.vla, str(self._dataset_stats_path.parent))
        self.processor = get_processor(self.cfg)
        self.action_head = None
        if self.cfg.use_l1_regression or self.cfg.use_diffusion:
            self.action_head = get_action_head(self.cfg, self.vla.llm_dim)
        self.proprio_projector = None
        if self.cfg.use_proprio:
            self.proprio_projector = get_proprio_projector(
                self.cfg, self.vla.llm_dim, PROPRIO_DIM
            )
        
        self.task_name = model_cfg["task_name"]
        self.action_type = model_cfg.get("action_type", "joint")
        self.default_prompt = model_cfg.get("prompt", self.task_name)
        env_cfg = model_cfg.get("env_cfg") or model_cfg.get("env_cfg_type")
        self.robot_action_dim_info = (
            get_robot_action_dim_info(env_cfg) if env_cfg is not None else None
        )
        self.observation_window: dict[str, Any] | None = None
        self._latest_env_idx_list: list[int] = [0]

    def update_obs(self, obs):
        self.update_obs_batch([obs])

    def update_obs_batch(self, obs_list):
        self._latest_env_idx_list = [obs.get("env_idx", index) for index, obs in enumerate(obs_list)]
        encoded_obs_list = [
            encode_obs(obs, self.action_type, self.robot_action_dim_info, self.default_prompt) for obs in obs_list
        ]
        self.observation_window = encoded_obs_list

    def infer(self, observation: dict):
        actions = get_vla_action(
            cfg=self.cfg,
            vla=self.vla,
            processor=self.processor,
            obs=observation,
            task_label=observation["instruction"],
            action_head=self.action_head,
            proprio_projector=self.proprio_projector,
            use_film=self.cfg.use_film,
        )
        return actions
    
    def get_action(self, **kwargs):
        action_list = self.get_action_batch(env_idx_list=[self._latest_env_idx_list[0]], **kwargs)
        return action_list[0]

    def get_action_batch(self, env_idx_list=None, **kwargs):
        if self.observation_window is None:
            raise AssertionError("update_obs or update_obs_batch first!")

        env_idx_list = env_idx_list or self._latest_env_idx_list

        action_list = []

        for batch_index, _ in enumerate(env_idx_list):
            action_chunk = self.infer(self.observation_window[batch_index])
            if self.robot_action_dim_info is None:
                action_list.append(action_chunk)
            else:
                action_list.append(
                    unpack_robot_state(
                        action_chunk,
                        self.action_type,
                        self.robot_action_dim_info,
                        source_type="obs",
                    )
                )
        
        return action_list
    
    def reset(self):
        return
    # TODO
    def get_model(self, model_cfg: dict[str, Any]):
        finetune_dir = _resolve_finetune_dir(model_cfg)
        has_finetune_weights = finetune_dir is not None and (finetune_dir / "config.json").exists()
        use_film = bool(model_cfg.get("use_film", True))
        use_l1_regression = bool(model_cfg.get("use_l1_regression", True))
        use_proprio = bool(model_cfg.get("use_proprio", True))
        if not has_finetune_weights:
            use_film = False
            use_l1_regression = False
            use_proprio = False

        unnorm_key = _resolve_unnorm_key(model_cfg, self._dataset_stats)

        config_args = {
            "pretrained_checkpoint": _resolve_checkpoint_path(model_cfg),
            "use_l1_regression": use_l1_regression,
            "use_diffusion": model_cfg.get("use_diffusion", False),
            "use_film": use_film,
            "use_proprio": use_proprio,
            "load_in_8bit": model_cfg.get("load_in_8bit", False),
            "load_in_4bit": model_cfg.get("load_in_4bit", False),
            "num_images_in_input": model_cfg.get("num_images_in_input", 3),
            "center_crop": model_cfg.get("center_crop", True),
            "unnorm_key": unnorm_key,
            "num_open_loop_steps": model_cfg.get("num_open_loop_steps", NUM_ACTIONS_CHUNK),
            "lora_rank": model_cfg.get("lora_rank", 32),
        }

        cfg = InferenceConfig(**config_args)
        return cfg