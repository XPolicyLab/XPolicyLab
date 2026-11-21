"""Native observation clock and joint-action boundary for local inference.

Soma owns history selection, image preprocessing, sampling and normalization.
This boundary collects every native observation between action queries.
It does not decode or recolor images.
"""

from pathlib import Path

import numpy as np
from XPolicyLab.utils.process_data import pack_robot_state, unpack_robot_state

VIEW_MAP = {
    "cam_head": "cam_high",
    "cam_left_wrist": "cam_left_wrist",
    "cam_right_wrist": "cam_right_wrist",
}
# Checkpoint contract; Model validates it against XPolicyLab's robot registry.
ACTION_DIMS = {"arm_dim": (6, 6), "ee_dim": (1, 1)}


class NativePolicy:
    """One local inference instance and one episode's observation state."""

    def __init__(self, inference, *, action_horizon=32, query_seed_base=42):
        if type(action_horizon) is not int or action_horizon < 1:
            raise ValueError("action_horizon must be a positive integer")
        if query_seed_base is not None and (type(query_seed_base) is not int or query_seed_base < 0):
            raise ValueError("query_seed_base must be a nonnegative integer or None")
        self.inference = inference
        self.action_horizon = action_horizon
        self.query_seed_base = query_seed_base
        self._clear()

    def _clear(self):
        self._step = None
        self._last_query_step = None
        self._query_index = 0
        self._pending = {name: [] for name in VIEW_MAP.values()}
        self._images = None
        self._state = None
        self._instruction = None

    def reset(self):
        self.inference.reset()
        self._clear()

    def update_obs(self, obs):
        step = obs.get("observation_step")
        if type(step) is not int or step < 0:
            raise ValueError("observation_step must be a nonnegative native action count")
        expected = 0 if self._step is None else self._step + 1
        if step != self._step and step != expected:
            raise ValueError("Every native observation is required, starting at zero after reset")
        images = {}
        for source, target in VIEW_MAP.items():
            image = np.asarray(obs["vision"][source]["color"])
            if image.dtype != np.uint8 or image.ndim != 3 or image.shape[-1] != 3:
                raise ValueError(f"{source} must be an already decoded uint8 HWC RGB array")
            images[target] = image.copy()
        state = pack_robot_state(obs, "joint", ACTION_DIMS).astype(np.float32, copy=False)
        if state.shape != (14,) or not np.isfinite(state).all():
            raise ValueError("Invalid state fields: expected one finite joint14 state")
        instruction = obs.get("instruction", obs.get("instructions"))
        if not isinstance(instruction, str) or not instruction.strip():
            raise ValueError("A nonempty instruction is required")
        # Repeated reads before inference replace the current frame, not its clock.
        for name, image in images.items():
            if step == self._step:
                if self._pending[name]:
                    self._pending[name][-1] = image
            else:
                self._pending[name].append(image)
        self._step = step
        self._images = images
        self._state = state
        self._instruction = instruction

    def get_action(self):
        if self._images is None or not all(self._pending.values()):
            raise RuntimeError("New native observations are required before an action query")
        if self._last_query_step is not None and self._step - self._last_query_step != self.action_horizon:
            raise ValueError(f"Execute the complete {self.action_horizon}-action chunk before querying again")
        result = self.inference.infer(
            self._images,
            self._instruction,
            seed=None if self.query_seed_base is None else self.query_seed_base + self._query_index,
            state=self._state,
            video_views=[np.stack(self._pending[name]) for name in VIEW_MAP.values()],
        )
        actions = result["action"]
        if (not isinstance(actions, np.ndarray) or actions.shape != (self.action_horizon, 14)
                or actions.dtype.kind != "f" or not np.isfinite(actions).all()):
            raise ValueError(f"Inference must return {self.action_horizon} finite absolute joint14 actions")
        output = unpack_robot_state(actions.copy(), "joint", ACTION_DIMS)
        for frames in self._pending.values():
            frames.clear()
        self._last_query_step = self._step
        self._query_index += 1
        return output

    def close(self):
        self.inference.close()


def load_policy(bundle, *, device="cuda", policy_config=None,
                checkpoint_file="step-5000.safetensors",
                model_config_file="config.yaml",
                action_stats_file="action_stats.json", query_seed_base=42):
    """Load downloaded policy weights and separately supplied frozen encoders."""
    from soma.inference import build_policy
    from soma.inference.config import EmbodiedConfig
    from dataclasses import replace

    bundle = Path(bundle).expanduser().resolve()
    checkpoint = bundle / Path(checkpoint_file).expanduser()
    model_config = bundle / Path(model_config_file).expanduser()
    action_stats = bundle / Path(action_stats_file).expanduser()
    for path in (checkpoint, model_config, action_stats):
        if not path.is_file():
            raise FileNotFoundError(path)
    config_path = Path(policy_config).expanduser() if policy_config else Path(__file__).with_name("config.py")
    cfg = EmbodiedConfig.load(str(config_path), checkpoint=str(checkpoint))
    cfg.load_model_config(str(model_config))
    profile = cfg.policy.action.profile
    if profile is None or profile.normalization is None:
        raise ValueError("RoboDojo requires an action profile with normalization")
    cfg.policy.action.profile = replace(profile, normalization=replace(
        profile.normalization, stats=str(action_stats),
    ))
    horizon = (cfg.policy.observation.num_frames - 1) * cfg.policy.action.video_action_freq
    policy = NativePolicy(None, action_horizon=horizon, query_seed_base=query_seed_base)
    policy.inference = build_policy(cfg, device=device)
    return policy
