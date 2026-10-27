"""XPolicyLab Model contract for the released dual-arm ALOHA joint policy.

No transport or image decoding is implemented here: XPolicyLab's official
server decodes images and serializes calls. Independent environments share
the immutable network/JIT but never a random key or cached previous plan.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import copy
from pathlib import Path

import numpy as np

from XPolicyLab.model_template import ModelTemplate
from XPolicyLab.utils.checkpoint_resolver import resolve_checkpoint_root


CAMERAS = {
    "cam_high": "cam_head",
    "cam_left_wrist": "cam_left_wrist",
    "cam_right_wrist": "cam_right_wrist",
}


def state_fields(dimensions):
    # The released checkpoint is ALOHA-specific; metadata controls all slicing.
    if dimensions != {"arm_dim": [6, 6], "ee_dim": [1, 1]}:
        raise ValueError("This checkpoint requires the ALOHA dual-arm joint layout")
    return tuple(
        field
        for prefix, arm, ee in zip(
            ("left_", "right_"),
            dimensions["arm_dim"],
            dimensions["ee_dim"],
            strict=True,
        )
        for field in (
            (prefix + "arm_joint_state", arm),
            (prefix + "ee_joint_state", ee),
        )
    )


def observation_from_xpolicylab(observation, fields):
    """Convert decoded standard XPolicyLab RGB/state fields without rescaling."""
    if not isinstance(observation, Mapping):
        raise TypeError("Expected one XPolicyLab observation dictionary")
    prompt = observation.get("instruction") or observation.get("instructions")
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("A nonempty instruction string is required")
    state = observation["state"]
    pieces = []
    for field, dimension in fields:
        value = np.asarray(state[field], dtype=np.float32)
        if value.shape != (dimension,) or not np.isfinite(value).all():
            raise ValueError(f"Expected finite {field}[{dimension}], got {value.shape}")
        pieces.append(value)
    images = {}
    for target, source in CAMERAS.items():
        image = np.asarray(observation["vision"][source]["color"])
        if image.ndim != 3 or image.shape[-1] != 3 or image.dtype != np.uint8:
            raise ValueError(
                f"Expected decoded HWC RGB uint8 for {source}; decoding belongs to XPolicyLab"
            )
        if not image.shape[0] or not image.shape[1]:
            raise ValueError(f"Empty image: {source}")
        images[target] = np.ascontiguousarray(image.transpose(2, 0, 1))
    return {"images": images, "state": np.concatenate(pieces), "prompt": prompt}


def actions_to_xpolicylab(commands, execute_k, fields):
    """Preserve joint/gripper coordinates and emit standard action dictionaries."""
    commands = np.asarray(commands, dtype=np.float32)
    width = sum(size for _, size in fields)
    if commands.shape != (execute_k, width) or not np.isfinite(commands).all():
        raise ValueError(
            f"Expected finite action chunk [{execute_k},{width}], got {commands.shape}"
        )
    return [
        {
            field: row[start : start + size].copy()
            for (field, size), start in zip(
                fields, np.cumsum([0] + [size for _, size in fields[:-1]]), strict=True
            )
        }
        for row in commands
    ]


def _env_id(value):
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 0:
        raise ValueError(f"env_idx must be a nonnegative integer, got {value!r}")
    return int(value)


class Model(ModelTemplate):
    """XPolicyLab adapter for the published, unmodified WorldLAPolicy.

    One server serves one evaluation stream (single environment or explicit
    batches). All K returned commands must run before the next get_action.
    reset() clears the whole stream between episodes/batch rounds.
    """

    def __init__(self, model_cfg):
        super().__init__()
        self.model_cfg = dict(model_cfg)
        self.action_type = self.model_cfg.get("action_type")
        self.env_cfg_type = self.model_cfg.get("env_cfg_type")
        if self.env_cfg_type != "arx_x5":
            raise ValueError("WorldLA release supports env_cfg_type=arx_x5 only")
        if self.model_cfg.get("action_type", "joint") != "joint":
            raise ValueError("WorldLA supports ALOHA joint actions only, not EE poses")
        policy_dir = Path(__file__).resolve().parent
        model_dir = resolve_checkpoint_root(self.model_cfg, policy_dir / "checkpoints")
        if (
            not (model_dir / "config.json").is_file()
            and (model_dir / "original/config.json").is_file()
        ):
            model_dir = model_dir / "original"
        self.initial_seed = int(self.model_cfg.get("inference_seed", 20260901))
        if not 0 <= self.initial_seed < 2**32:
            raise ValueError("inference_seed must fit uint32")
        weight = self.model_cfg.get("old_plan_weight")
        if weight is not None and not 0 <= float(weight) <= 1:
            raise ValueError("old_plan_weight must lie in [0,1]")
        from worldla.policy import WorldLAPolicy

        self.policy = WorldLAPolicy(
            Path(model_dir).expanduser().resolve(), seed=self.initial_seed
        )
        # Load CUDA/JAX before the shared helper imports OpenCV/HDF5.
        from XPolicyLab.utils.process_data import get_robot_action_dim_info

        self.robot_action_dim_info = get_robot_action_dim_info(self.env_cfg_type)
        self.fields = state_fields(self.robot_action_dim_info)
        self.action_dim = sum(size for _, size in self.fields)
        self.model = self.policy.model
        # Never mutate a release file: deployment overrides are process-local.
        self.policy.config = copy.deepcopy(self.policy.config)
        if weight is not None:
            self.policy.config["deployment"]["old_plan_weight"] = float(weight)
        self.execute_k = int(self.policy.config["deployment"]["execute_k"])
        if self.execute_k != 10:
            raise ValueError(
                "This adapter is validated for the released execute_k=10 protocol"
            )
        self.episode_index = -1
        self._sessions = {}
        self._observations = {}
        self._order = []
        self._consumed = set()
        self.case_meta = None
        print(
            f'[worldla/xpolicylab] Full checkpoint: {self.policy.root}; '
            f'execute_k={self.execute_k}; old_plan_weight='
            f'{self.policy.config["deployment"]["old_plan_weight"]}',
            flush=True,
        )

    def reset(self):
        """Called with no payload by XPolicyLab before each episode/round."""
        self.episode_index += 1
        self._sessions.clear()
        self._observations.clear()
        self._order = []
        self._consumed.clear()

    def prepare_case(self, case_meta=None):
        # Metadata is not used to replace the observation's instruction/state.
        if case_meta is not None and not isinstance(case_meta, Mapping):
            raise TypeError("case_meta must be a dictionary")
        self.case_meta = None if case_meta is None else dict(case_meta)

    def on_trial_end(self, result=None):
        self._observations.clear()
        self._sessions.clear()
        self._order = []
        self._consumed.clear()

    def update_obs(self, obs):
        self.update_obs_batch([obs])

    def update_obs_batch(self, obs_list):
        if (
            not isinstance(obs_list, Sequence)
            or isinstance(obs_list, (str, bytes))
            or not obs_list
        ):
            raise ValueError("update_obs_batch needs a nonempty list of observations")
        encoded, order = {}, []
        for index, observation in enumerate(obs_list):
            if not isinstance(observation, Mapping):
                raise TypeError("Each batch observation must be a dictionary")
            env_idx = _env_id(observation.get("env_idx", index))
            if env_idx in encoded:
                raise ValueError(f"Duplicate env_idx in observation batch: {env_idx}")
            encoded[env_idx] = observation_from_xpolicylab(observation, self.fields)
            order.append(env_idx)
        # Preserve sessions of other active environments when the batch shrinks.
        self._observations, self._order = encoded, order
        self._consumed.difference_update(order)

    def _get_actions(self, env_idx):
        if env_idx not in self._observations:
            raise ValueError(
                f"No observation for env_idx={env_idx}; update observations first"
            )
        if env_idx in self._consumed:
            raise ValueError(
                "Acquire a new observation after executing the returned action chunk"
            )
        if env_idx not in self._sessions:
            # WorldLAPolicy stores only RNG and previous_plan per episode; all
            # network parameters, input/output transforms and compiled sampler
            # are shared read-only. copy.copy does not duplicate GPU weights.
            session = copy.copy(self.policy)
            episode = max(0, self.episode_index)
            seed = (self.initial_seed + 1009 * episode + 1000003 * env_idx) % 2**32
            session.reset(seed=seed)
            self._sessions[env_idx] = session
        commands = self._sessions[env_idx].plan(self._observations[env_idx])
        result = actions_to_xpolicylab(commands, self.execute_k, self.fields)
        self._consumed.add(env_idx)
        return result

    def get_action(self):
        if len(self._order) != 1:
            raise ValueError(
                "get_action needs exactly one observation; use get_action_batch for batches"
            )
        return self._get_actions(self._order[0])

    def get_action_batch(self, env_idx_list=None):
        """Serial inference with isolated state; not a vectorized throughput claim."""
        order = (
            self._order if env_idx_list is None else [_env_id(i) for i in env_idx_list]
        )
        if not order or len(set(order)) != len(order):
            raise ValueError(
                "get_action_batch needs nonempty, distinct environment IDs"
            )
        # Validate the whole selection before mutating any plan/RNG state.
        if any(i not in self._observations or i in self._consumed for i in order):
            raise ValueError("Every selected environment needs a fresh observation")
        return [self._get_actions(i) for i in order]
