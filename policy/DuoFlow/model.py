"""DuoFlow 的 XPolicyLab 接口；仅使用标准公共观测。"""

from pathlib import Path
import sys

import numpy as np

POLICY_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(POLICY_DIR.parents[2]))

from XPolicyLab.model_template import ModelTemplate
from XPolicyLab.utils.checkpoint_resolver import resolve_checkpoint_root
from XPolicyLab.utils.process_data import (
    get_robot_action_dim_info,
    pack_robot_state,
    unpack_robot_state,
)


class Model(ModelTemplate):
    def __init__(self, model_cfg, engine=None):
        super().__init__()
        self.cfg = dict(model_cfg)
        if self.cfg.get("action_type") != "joint":
            raise ValueError("DuoFlow supports the Aloha-AgileX joint checkpoint only")
        self.dim_info = get_robot_action_dim_info(self.cfg["env_cfg_type"])
        if list(self.dim_info["arm_dim"]) != [6, 6] or list(self.dim_info["ee_dim"]) != [1, 1]:
            raise ValueError("The released checkpoint requires two 6-joint arms and two grippers")
        self.cameras = self.cfg.get("camera_map") or {
            "cam_high": "cam_head",
            "cam_left_wrist": "cam_left_wrist",
            "cam_right_wrist": "cam_right_wrist",
        }
        if set(self.cameras) != {"cam_high", "cam_left_wrist", "cam_right_wrist"}:
            raise ValueError("camera_map must map all three native cameras")
        if engine is None:
            from .inference import DuoFlowInference

            checkpoint_root = resolve_checkpoint_root(
                self.cfg, POLICY_DIR / "checkpoints", policy_dir=POLICY_DIR
            )
            adapter = checkpoint_root / "duoflow_adapter.safetensors"
            self.engine = DuoFlowInference(adapter)
        else:
            self.engine = engine
        self.model = self.engine
        self.reset()

    def reset(self):
        self._observations = {}
        self._env_ids = []
        self._last_env = None
        self._last_instruction = None
        self._reset_pending = True

    def prepare_case(self, case_meta=None):
        self.reset()

    def update_obs(self, obs):
        self.update_obs_batch([obs])

    def update_obs_batch(self, obs_list):
        self._observations = {}
        self._env_ids = []
        for index, obs in enumerate(obs_list):
            env_id = int(obs.get("env_idx", index))
            if env_id in self._observations:
                raise ValueError(f"duplicate environment index: {env_id}")
            self._observations[env_id] = obs
            self._env_ids.append(env_id)

    def _predict(self, env_id):
        obs = self._observations[env_id]
        instruction = obs.get("instruction", obs.get("instructions"))
        if isinstance(instruction, (list, tuple)) and len(instruction) == 1:
            instruction = instruction[0]
        if not isinstance(instruction, str) or not instruction.strip():
            raise ValueError("The environment must provide one selected, non-empty instruction")
        state = np.asarray(pack_robot_state(obs, "joint", self.dim_info), dtype=np.float32)
        if state.shape != (14,) or not np.isfinite(state).all():
            raise ValueError("Expected one finite 14D joint state")
        images = {}
        for native_name, standard_name in self.cameras.items():
            # 官方 websocket 服务已解码；这里不做解码或 RGB/BGR 交换。
            rgb = np.asarray(obs["vision"][standard_name]["color"])
            if rgb.ndim != 3 or rgb.shape[-1] != 3 or rgb.dtype != np.uint8:
                raise ValueError(f"{standard_name}.color must be decoded HWC uint8 RGB")
            images[native_name] = rgb
        reset = self._reset_pending or env_id != self._last_env or instruction != self._last_instruction
        actions = np.asarray(self.engine.predict(state, images, instruction, reset=reset), dtype=np.float32)
        if actions.shape != (50, 14) or not np.isfinite(actions).all():
            raise ValueError(f"Native inference must return finite [50,14], got {actions.shape}")
        self._last_env, self._last_instruction = env_id, instruction
        self._reset_pending = False
        # 原生预测 50 步，标准 deploy 循环只接收待执行的前 30 步。
        return unpack_robot_state(actions[:30], "joint", self.dim_info, source_type="obs")

    def get_action(self):
        if not self._env_ids:
            raise RuntimeError("update_obs must precede get_action")
        return self._predict(self._env_ids[0])

    def get_action_batch(self, env_idx_list=None):
        ids = self._env_ids if env_idx_list is None else env_idx_list
        if isinstance(ids, (int, np.integer)):
            ids = [int(ids)]
        return [self._predict(int(env_id)) for env_id in ids]
