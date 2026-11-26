"""XPolicyLab adapter for SLA_Memory, SLA with history frames (RoboDojo simulation and real robots).

Besides the three current views the model sees history frames from the head camera: the episode's first frame
and one frame per second over the last 8 seconds. It owns one history per environment, so ``update_obs`` must be
called after every executed control step, not only when a new action chunk is requested, and ``reset`` must be
called between episodes.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np
from openpi.policies import policy_config as _policy_config
from openpi.policies.robodojo_memory import RobodojoMemoryBuffer
from openpi.training import config as _config

from XPolicyLab.model_template import ModelTemplate
from XPolicyLab.utils.checkpoint_resolver import resolve_checkpoint_root
from XPolicyLab.utils.process_data import get_robot_action_dim_info
from XPolicyLab.utils.process_data import pack_robot_state
from XPolicyLab.utils.process_data import unpack_robot_state

_POLICY_DIR = Path(__file__).resolve().parent


class Model(ModelTemplate):
    def __init__(self, model_cfg: dict):
        super().__init__()
        if model_cfg.get("action_type") != "ee":
            raise ValueError(f"SLA_Memory predicts end-effector actions; action_type must be 'ee', got {model_cfg.get('action_type')!r}")
        self._robot_action_dim_info = get_robot_action_dim_info(model_cfg["env_cfg_type"])

        train_config_name = model_cfg["train_config_name"]
        train_config = _config.get_config(train_config_name)
        data = train_config.data
        if isinstance(data, _config.MixedLeRobotDataConfig):
            # Inference uses the source recorded on this robot. All sources share the transforms and the norm
            # stats; they only differ in the frame rate the history frames are spaced by.
            sources = {source.name: source for source in data.sources}
            source = sources.get(f"real_{model_cfg['env_cfg_type']}")
            if source is None:
                raise ValueError(
                    f"{train_config_name} has no data source for env_cfg_type={model_cfg['env_cfg_type']!r}; "
                    f"expected one of {sorted(sources)}"
                )
            data = dataclasses.replace(
                source.data_factory, assets=data.assets, use_quantile_norm=data.use_quantile_norm
            )

        # The history buffer rebuilds the training layout: the episode's first frame, then one frame per second.
        extra_frames = data.base_config.extra_frames
        self._memory_fps = -extra_frames[-1]
        self._history_seconds = len(extra_frames) - 1
        if extra_frames != (0, *range(-self._history_seconds * self._memory_fps, 0, self._memory_fps)):
            raise ValueError(f"{train_config_name}: unsupported history frames {extra_frames}")

        # Must match training: the history frames go ahead of (true) or after (false) the current views.
        extra_frames_first = model_cfg.get("extra_frames_first", True)
        if not isinstance(extra_frames_first, bool):
            raise ValueError(f"extra_frames_first must be true or false, got {extra_frames_first!r}")
        model = dataclasses.replace(
            train_config.model,
            extra_frames_first=extra_frames_first,
            attention_impl=model_cfg.get("attention_impl", "xla"),
        )
        train_config = dataclasses.replace(train_config, data=data, model=model)

        # train.sh writes checkpoints/<train config>/<run>/<step>/; ckpt_name may also be a run or step directory.
        checkpoint_dir = resolve_checkpoint_root(
            model_cfg, _POLICY_DIR / "checkpoints" / train_config_name, policy_dir=_POLICY_DIR
        )
        if not (checkpoint_dir / "params").is_dir():
            steps = [path for path in checkpoint_dir.iterdir() if path.name.isdigit() and (path / "params").is_dir()]
            if not steps:
                raise FileNotFoundError(f"No checkpoint step with params/ under {checkpoint_dir}")
            checkpoint_dir = max(steps, key=lambda path: int(path.name))
        print(f"[SLA_Memory] config={train_config_name} checkpoint={checkpoint_dir} history_fps={self._memory_fps}")

        self._policy = _policy_config.create_trained_policy(train_config, checkpoint_dir)
        self.model = self._policy
        self._history_by_env: dict[int, RobodojoMemoryBuffer] = {}
        self._last_env_idx: int | None = None

    def update_obs(self, obs: dict) -> None:
        index = int(obs.get("env_idx", 0))
        history = self._history_by_env.get(index)
        if history is None:
            history = RobodojoMemoryBuffer(fps=self._memory_fps, history_seconds=self._history_seconds)
            self._history_by_env[index] = history
        history.update(self._to_openpi_observation(obs))
        self._last_env_idx = index

    def update_obs_batch(self, obs_list: list[dict]) -> None:
        for obs in obs_list:
            self.update_obs(obs)

    def get_action(self) -> list[dict[str, np.ndarray]]:
        if self._last_env_idx is None:
            raise ValueError("no observation has been provided")
        return self.get_action_batch([self._last_env_idx])[0]

    def get_action_batch(self, env_idx_list=None) -> list[list[dict[str, np.ndarray]]]:
        indices = (
            sorted(self._history_by_env)
            if env_idx_list is None
            else [int(index) for index in env_idx_list]
        )
        if not indices:
            raise ValueError("no observation has been provided")

        chunks = []
        for index in indices:
            history = self._history_by_env.get(index)
            if history is None:
                raise KeyError(f"no observation for env_idx={index}")
            result = self._policy.infer(history.build_observation())
            if "actions" not in result:
                raise ValueError("OpenPI response is missing 'actions'")
            chunks.append(self._from_openpi_actions(result["actions"]))
        return chunks

    def reset(self) -> None:
        self._history_by_env.clear()
        self._last_env_idx = None
        self._policy.reset()

    def on_trial_end(self, payload=None) -> dict[str, bool]:
        """Prevent history from leaking into the next XPolicyLab trial."""
        del payload
        self.reset()
        return {"memory_reset": True}

    def _to_openpi_observation(self, obs: dict) -> dict:
        vision = obs.get("vision", {})
        images = {}
        for source, target in (
            ("cam_head", "cam_high"),
            ("cam_left_wrist", "cam_left_wrist"),
            ("cam_right_wrist", "cam_right_wrist"),
        ):
            if source not in vision or "color" not in vision[source]:
                raise ValueError(f"missing vision/{source}/color")
            image = np.asarray(vision[source]["color"])
            if image.ndim != 3 or image.shape[-1] != 3:
                raise ValueError(
                    f"vision/{source}/color must have shape [H,W,3], got {image.shape}"
                )
            images[target] = image

        # Per arm [xyz, qw, qx, qy, qz, gripper], left arm first: the layout the model was trained on.
        state = pack_robot_state(obs, "ee", self._robot_action_dim_info).astype(np.float32)
        if not np.isfinite(state).all():
            raise ValueError("state must be finite")
        instruction = obs.get("instruction", obs.get("instructions", ""))
        if isinstance(instruction, (list, tuple)):
            instruction = instruction[0] if instruction else ""
        return {
            "images": images,
            "state": state,
            "prompt": str(instruction),
        }

    def _from_openpi_actions(self, actions) -> list[dict[str, np.ndarray]]:
        values = np.asarray(actions, dtype=np.float32)
        if values.ndim != 2:
            raise ValueError(f"OpenPI returned expected (T, D), got {values.shape}")
        if not np.isfinite(values).all():
            raise ValueError("OpenPI returned non-finite actions")
        return unpack_robot_state(values, "ee", self._robot_action_dim_info)
