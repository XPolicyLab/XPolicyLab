"""RAP-WAM: retained v58_light checkpoint and historical RoboDojo inference."""
from pathlib import Path
import numpy as np
import torch
from scipy.signal import savgol_filter
from XPolicyLab.model_template import ModelTemplate
from XPolicyLab.utils.process_data import get_robot_action_dim_info, pack_robot_state, unpack_robot_state
from .checkpoint import load_policy
from .language import LanguageContext
from .runtime import historical_module
from .common import CAMERAS, build_tshape, instruction_text


class Model(ModelTemplate):
    def __init__(self, model_cfg):
        super().__init__()
        self.cfg = dict(model_cfg)
        if (self.cfg.get("bench_name"), self.cfg.get("env_cfg_type"), self.cfg.get("action_type")) != ("RoboDojo", "arx_x5", "joint"):
            raise ValueError("Supported profile: RoboDojo / arx_x5 / joint")
        self.action_type = "joint"
        self.dim_info = get_robot_action_dim_info(self.cfg["env_cfg_type"])
        self.action_dim = sum(self.dim_info["arm_dim"]) + sum(self.dim_info["ee_dim"])
        self.policy, tasks, self.bundle_root, wan = load_policy(self.cfg, Path(__file__).resolve().parent, self.action_dim)
        self.action_horizon = int(self.policy.action_horizon)
        profile = {"replan_steps": 8, "savgol_window": 9, "savgol_polyorder": 2, "boundary_blend_steps": 4}
        for name, expected in profile.items():
            if int(self.cfg.get(name, expected)) != expected:
                raise ValueError(f"Historical profile requires {name}={expected}")
            setattr(self, name, expected)
        self.lang = LanguageContext(tasks, wan, self.cfg.get("tokenizer_path"))
        self.rgb_processor = historical_module("psi_policy.dataset.preprocessors.rgb").RGBPreprocessor(image_size=(384, 320))
        self.state_processor = historical_module("psi_policy.dataset.preprocessors.low_dim").LowDimPreprocessor()
        # Preserve the historical policy RNG. The benchmark seed is not a request-level noise seed.
        self.reset()

    def _process_obs(self, obs):
        rgb = build_tshape(*(obs["vision"][name]["color"] for name in CAMERAS))
        state = np.asarray(pack_robot_state(obs, self.action_type, self.dim_info, source_type="obs"), dtype=np.float32).reshape(-1)
        if state.shape != (self.action_dim,) or not np.isfinite(state).all():
            raise ValueError("Expected one finite robot joint-state vector")
        return {"rgb_head": self.rgb_processor.process(rgb[None], key="rgb_head", obs_shape=(3, 384, 320)),
                "robot_state": self.state_processor.process(state[None], key="robot_state", obs_shape=(self.action_dim,))}

    def update_obs(self, obs):
        self.update_obs_batch([obs])

    def update_obs_batch(self, obs_list):
        if not obs_list:
            raise ValueError("Observation batch must not be empty")
        indices = [int(obs.get("env_idx", i)) for i, obs in enumerate(obs_list)]
        if len(set(indices)) != len(indices):
            raise ValueError("Duplicate environment indices")
        for idx, obs in zip(indices, obs_list):
            self._obs_processed_batch[idx] = self._process_obs(obs)
            self._instr_batch[idx] = instruction_text(obs)
        self._latest_env_idx_list = indices

    @torch.no_grad()
    def _infer_one(self, obs_proc, instruction):
        batched = {key: value.unsqueeze(0).to(self.policy.device) for key, value in obs_proc.items()}
        idx = self.lang.resolve_index(instruction)
        original = None
        if idx is not None:
            self.policy._cur_task_index = torch.tensor([idx], device=self.policy.device)
        else:
            dtype = self.policy.model.text_embedding[0].weight.dtype
            context = self.lang.encode_zero_shot(instruction, self.policy.device, dtype)
            self.policy._cur_task_index = None
            original = self.policy._build_context_list
            self.policy._build_context_list = lambda batch_size: [context for _ in range(batch_size)]
        try:
            out = self.policy.predict_action(batched)
        finally:
            self.policy._cur_task_index = None
            if original is not None:
                self.policy._build_context_list = original
        actions = out["action_pred"]
        if tuple(actions.shape) != (1, self.action_horizon, self.action_dim) or not torch.isfinite(actions).all():
            raise ValueError("Invalid action prediction")
        return actions[0].detach().float().cpu().numpy()

    def _postprocess(self, actions, env_idx):
        # Match the original float64 SG computation, then return float32 joint targets.
        actions = savgol_filter(actions.astype(np.float64), window_length=self.savgol_window,
                                polyorder=self.savgol_polyorder, axis=0).astype(np.float32)
        smoothed = actions.copy()
        previous = self._prev_chunk.get(env_idx)
        if previous is not None:
            count = min(self.boundary_blend_steps, len(actions), max(0, len(previous) - self.replan_steps))
            for i in range(count):
                alpha = float(i + 1) / float(count + 1)
                actions[i] = alpha * actions[i] + (1.0 - alpha) * previous[self.replan_steps + i]
        # Retain pre-blend full horizon, so the next call uses the predicted continuation.
        self._prev_chunk[env_idx] = smoothed
        return [unpack_robot_state(actions[t], self.action_type, self.dim_info, source_type="obs")
                for t in range(self.replan_steps)]

    def get_action(self):
        if not self._latest_env_idx_list:
            raise ValueError("Call update_obs before get_action")
        return self.get_action_batch([self._latest_env_idx_list[0]])[0]

    def get_action_batch(self, env_idx_list=None):
        indices = self._latest_env_idx_list if env_idx_list is None else [int(i) for i in env_idx_list]
        if not indices or any(i not in self._obs_processed_batch for i in indices):
            raise ValueError("Missing observations for requested environments")
        # Original deployment inferred serially. Do not change its per-request RNG order.
        return [self._postprocess(self._infer_one(self._obs_processed_batch[i], self._instr_batch[i]), i) for i in indices]

    def reset(self):
        self._obs_processed_batch = {}
        self._instr_batch = {}
        self._prev_chunk = {}
        self._latest_env_idx_list = []
