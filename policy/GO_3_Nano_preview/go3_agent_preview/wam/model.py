from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import cv2
import numpy as np

from XPolicyLab.model_template import ModelTemplate
from XPolicyLab.utils.process_data import get_robot_action_dim_info, pack_robot_state, unpack_robot_state

from .backend_client import GeActBackendClient
from .ttc_config import TTCConfig
from .memory_config import MemoryConfig, ObservationMemory


CAMERA_ORDER = ("cam_head", "cam_left_wrist", "cam_right_wrist")
ROBOT_ARM_DIM = 6
GRIPPER_DIM = 1
ACTION_DIM = 2 * (ROBOT_ARM_DIM + GRIPPER_DIM)
NATIVE_ARX_ROBOTS = frozenset(("sl_agilex", "sl_acone", "sl_arxvr"))
ROBODOJO_ROBOTS = frozenset(("robodojo", "robodojo_real"))
NATIVE_14D_ROBOTS = NATIVE_ARX_ROBOTS | ROBODOJO_ROBOTS


def _positive_int(value: Any, name: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise ValueError(f"{name} must be positive, got {parsed}")
    return parsed


def _positive_float(value: Any, name: str) -> float:
    parsed = float(value)
    if not np.isfinite(parsed) or parsed <= 0:
        raise ValueError(f"{name} must be a positive finite number, got {value!r}")
    return parsed


class Model(ModelTemplate):
    def __init__(self, model_cfg: Mapping[str, Any], client: Any | None = None) -> None:
        self.memory_config = MemoryConfig.from_mapping(model_cfg)
        self._memory = ObservationMemory(self.memory_config.n_previous) if self.memory_config.enabled else None
        self._memory_metadata_checked = False
        self._check_memory_metadata = "memory_enabled" in model_cfg or self.memory_config.enabled
        self.ttc_config = TTCConfig.from_mapping(model_cfg)
        self._ttc_metadata_checked = False
        self._check_ttc_metadata = "ttc_enabled" in model_cfg
        self.camera_names = tuple(model_cfg.get("camera_names", CAMERA_ORDER))
        if self.camera_names != CAMERA_ORDER:
            raise ValueError(
                f"GEAct requires camera order {CAMERA_ORDER}, got {self.camera_names}"
            )
        self.image_height, self.image_width = self._image_size(model_cfg)
        self.action_horizon = _positive_int(
            model_cfg.get("action_horizon", 30), "action_horizon"
        )
        self.num_ae_steps = _positive_int(
            model_cfg.get("num_ae_steps", 5), "num_ae_steps"
        )
        self.action_type = str(model_cfg.get("action_type", "joint"))
        if self.action_type != "joint":
            raise ValueError("GEAct RoboDojo adapter supports action_type='joint' only")
        self.geact_robot_name = str(
            model_cfg.get("geact_robot_name", "sl_arxvr")
        ).strip().lower()
        if self.geact_robot_name not in NATIVE_14D_ROBOTS:
            raise ValueError(
                "GEAct RoboDojo adapter requires a supported native 14D embodiment, got "
                f"{self.geact_robot_name!r}; expected one of {sorted(NATIVE_14D_ROBOTS)}"
            )

        self.robot_action_dim_info = get_robot_action_dim_info(model_cfg["env_cfg_type"])
        actual_dim = sum(self.robot_action_dim_info["arm_dim"]) + sum(
            self.robot_action_dim_info["ee_dim"]
        )
        if actual_dim != ACTION_DIM:
            raise ValueError(
                f"GEAct requires the 14D arx_x5 action layout, got {actual_dim}D"
            )

        host = str(model_cfg.get("geact_server_host", "127.0.0.1"))
        port = int(model_cfg.get("geact_server_port", 8001))
        if not 1 <= port <= 65535:
            raise ValueError(f"geact_server_port must be in [1, 65535], got {port}")
        self.client = client or GeActBackendClient(
            f"ws://{host}:{port}",
            connect_timeout_s=_positive_float(
                model_cfg.get("geact_connect_timeout_s", 30),
                "geact_connect_timeout_s",
            ),
            request_timeout_s=_positive_float(
                model_cfg.get("geact_request_timeout_s", 600),
                "geact_request_timeout_s",
            ),
        )
        self.bbox_enabled = bool(model_cfg.get("bbox_enabled", False))
        self.bbox = None
        if self.bbox_enabled:
            from .bbox_conditioning import BBoxConditioner
            self.bbox = BBoxConditioner(model_cfg)
        self._bbox_metadata_checked = False
        self._robot_metadata_checked = False
        self._pending_by_env: dict[int, tuple[np.ndarray, np.ndarray, str]] = {}
        self._latest_env_idx_list: list[int] = []
        print(
            "[GEAct] proxy ready "
            f"backend=ws://{host}:{port} image={self.image_height}x{self.image_width} "
            f"robot={self.geact_robot_name} horizon={self.action_horizon} "
            f"ae_steps={self.num_ae_steps} memory={self.memory_config.enabled} n_previous={self.memory_config.n_previous}"
        )

    @staticmethod
    def _image_size(model_cfg: Mapping[str, Any]) -> tuple[int, int]:
        image_size = model_cfg.get("image_size", [256, 384])
        if (
            not isinstance(image_size, Sequence)
            or isinstance(image_size, str)
            or len(image_size) != 2
        ):
            raise ValueError(f"image_size must be [height, width], got {image_size!r}")
        return (
            _positive_int(image_size[0], "image_size[0]"),
            _positive_int(image_size[1], "image_size[1]"),
        )

    def _encode_image(self, value: Any, camera_name: str) -> np.ndarray:
        image = np.asarray(value)
        if image.ndim == 2:
            image = np.repeat(image[..., None], 3, axis=-1)
        if image.ndim != 3:
            raise ValueError(f"{camera_name}.color must be HWC, got {image.shape}")
        if image.shape[-1] == 4:
            image = image[..., :3]
        elif image.shape[-1] != 3 and image.shape[0] == 3:
            image = image.transpose(1, 2, 0)
        if image.shape[-1] != 3 or image.size == 0:
            raise ValueError(f"{camera_name}.color must contain non-empty RGB data")
        if image.shape[:2] != (self.image_height, self.image_width):
            interpolation = (
                cv2.INTER_AREA
                if image.shape[0] > self.image_height or image.shape[1] > self.image_width
                else cv2.INTER_LINEAR
            )
            image = cv2.resize(
                image, (self.image_width, self.image_height), interpolation=interpolation
            )
        if not np.isfinite(image).all():
            raise ValueError(f"{camera_name}.color contains NaN or infinity")
        return np.ascontiguousarray(image, dtype=np.float32)

    @staticmethod
    def _prompt(observation: Mapping[str, Any]) -> str:
        prompt = observation.get("instruction")
        if not prompt:
            prompts = observation.get("instructions")
            if isinstance(prompts, Sequence) and not isinstance(prompts, str) and prompts:
                prompt = prompts[0]
        if not prompt:
            raise ValueError("RoboDojo observation did not provide an instruction")
        return str(prompt)

    def _prepare_observation(
        self, observation: Mapping[str, Any]
    ) -> tuple[np.ndarray, np.ndarray, str]:
        vision = observation.get("vision")
        if not isinstance(vision, Mapping):
            raise KeyError("observation must contain a mapping field 'vision'")
        views = []
        for camera_name in self.camera_names:
            camera = vision.get(camera_name)
            if not isinstance(camera, Mapping) or "color" not in camera:
                raise KeyError(f"observation is missing vision.{camera_name}.color")
            views.append(self._encode_image(camera["color"], camera_name))

        state = pack_robot_state(
            observation,
            self.action_type,
            self.robot_action_dim_info,
            source_type="obs",
        )
        state = np.asarray(state, dtype=np.float32).reshape(-1)
        if state.size != ACTION_DIM or not np.isfinite(state).all():
            raise ValueError(f"expected finite {ACTION_DIM}D robot state, got {state.shape}")
        return (
            np.stack(views, axis=0)[None, ...],
            state.reshape(1, 1, ACTION_DIM),
            self._prompt(observation),
        )

    def update_obs(self, observation: Mapping[str, Any]) -> None:
        self.update_obs_batch([observation])

    def update_obs_batch(self, obs_list: Sequence[Mapping[str, Any]]) -> None:
        env_indices = [int(obs.get("env_idx", i)) for i, obs in enumerate(obs_list)]
        if len(set(env_indices)) != len(env_indices):
            raise ValueError("observation batch contains duplicate env_idx values")
        prepared = {
            env_idx: self._prepare_observation(obs)
            for env_idx, obs in zip(env_indices, obs_list)
        }
        if self._memory is not None:
            for env_idx, (frame, state, prompt) in prepared.items():
                self._memory.append(env_idx, frame[0])
        self._pending_by_env.update(prepared)
        self._latest_env_idx_list = env_indices

    @staticmethod
    def _values(result: Mapping[str, Any], key: str, *, scalar_series: bool = False) -> np.ndarray:
        value = result.get(key)
        if isinstance(value, Mapping):
            value = value.get("values")
        array = np.asarray(value, dtype=np.float32)
        if array.ndim == 1:
            array = array[:, None] if scalar_series else array[None, :]
        if array.ndim != 2 or not np.isfinite(array).all():
            raise ValueError(f"result.{key} must be a finite 2D array, got {array.shape}")
        return array

    def _infer_action_chunk(
        self, payload: tuple[np.ndarray, np.ndarray, str], env_idx: int = 0
    ) -> list[dict[str, np.ndarray]]:
        obs, state, prompt = payload
        if self._check_memory_metadata and not self._memory_metadata_checked:
            self.memory_config.validate_backend(self.client.metadata())
            self._memory_metadata_checked = True
        if self._memory is not None:
            obs = self._memory.sample(env_idx)
        if self._check_ttc_metadata and not self._ttc_metadata_checked:
            self.ttc_config.validate_backend(self.client.metadata())
            self._ttc_metadata_checked = True
        if self.geact_robot_name in ROBODOJO_ROBOTS and not self._robot_metadata_checked:
            metadata = self.client.metadata()
            if metadata.get("robot_name") != self.geact_robot_name:
                raise ValueError(
                    f"GEAct adapter robot={self.geact_robot_name!r} does not match "
                    f"backend robot={metadata.get('robot_name')!r}; start the backend "
                    f"with --robot-name {self.geact_robot_name}"
                )
            self._robot_metadata_checked = True
        extra = {}
        if self.bbox is not None:
            if not self._bbox_metadata_checked:
                metadata = self.client.metadata()
                if not metadata.get("bbox_enabled") or metadata.get("vlm_fixed_sequence_length") != 384:
                    raise ValueError("bbox adapter requires a bbox-enabled backend with VLM length 384")
                if metadata.get("bbox_input_mode") != "red_reference":
                    raise ValueError("bbox backend must use red_reference inputs")
                self._bbox_metadata_checked = True
            extra = self.bbox.prepare(obs[0, 0], prompt, env_idx)
        response = self.client.infer(
            obs=obs, state=state, prompt=prompt, num_ae_steps=self.num_ae_steps, **extra
        )
        result = response.get("result")
        if not isinstance(result, Mapping):
            raise ValueError("GEAct backend response is missing result")
        left_arm = self._values(result, "left_arm")
        right_arm = self._values(result, "right_arm")
        left_eff = self._values(result, "left_effector", scalar_series=True)
        right_eff = self._values(result, "right_effector", scalar_series=True)
        lengths = {len(left_arm), len(right_arm), len(left_eff), len(right_eff)}
        if len(lengths) != 1 or not lengths or min(lengths) <= 0:
            raise ValueError("GEAct backend returned inconsistent or empty action chunks")
        if left_arm.shape[1] != ROBOT_ARM_DIM or right_arm.shape[1] != ROBOT_ARM_DIM:
            raise ValueError("GEAct backend must return exactly six arm joints per side")
        left_eff = np.clip(left_eff[:, :1], 0.0, 1.0)
        right_eff = np.clip(right_eff[:, :1], 0.0, 1.0)
        packed = np.concatenate(
            [left_arm, left_eff, right_arm, right_eff], axis=-1
        )
        return unpack_robot_state(
            packed[: self.action_horizon],
            self.action_type,
            self.robot_action_dim_info,
            source_type="obs",
        )

    def get_action(self) -> list[dict[str, np.ndarray]]:
        if not self._latest_env_idx_list:
            raise RuntimeError("get_action called before update_obs")
        return self.get_action_batch([self._latest_env_idx_list[0]])[0]

    def get_action_batch(
        self, env_idx_list: Sequence[int] | None = None
    ) -> list[list[dict[str, np.ndarray]]]:
        indices = (
            list(self._latest_env_idx_list)
            if env_idx_list is None
            else [int(env_idx) for env_idx in env_idx_list]
        )
        missing = [env_idx for env_idx in indices if env_idx not in self._pending_by_env]
        if missing:
            raise RuntimeError(f"get_action_batch called without observations for env_idx={missing}")
        return [self._infer_action_chunk(self._pending_by_env[env_idx], env_idx) for env_idx in indices]

    def reset(self) -> None:
        if self._memory is not None:
            self._memory.clear()
        if self.bbox is not None:
            self.bbox.reset()
        self._pending_by_env.clear()
        self._latest_env_idx_list = []

    def drop_env(self, env_idx: int) -> None:
        if self._memory is not None or self.bbox is not None:
            raise RuntimeError("local skill switch reset requires memory-free, bbox-free WAM")
        self._pending_by_env.pop(int(env_idx), None)
        self._latest_env_idx_list = [i for i in self._latest_env_idx_list if i != int(env_idx)]

    def close(self) -> None:
        close = getattr(self.client, "close", None)
        if callable(close):
            close()
