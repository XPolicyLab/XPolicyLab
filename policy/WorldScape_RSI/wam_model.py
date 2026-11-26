"""Public XPolicyLab model entrypoint for RoboDojo Flash and Memory."""
import json

from XPolicyLab.model_template import ModelTemplate


class Model(ModelTemplate):
    def __init__(self, model_cfg):
        from .xpolicylab_adapter.runtime_config import validate_config
        model_cfg = validate_config(model_cfg)
        self.runtime_profile = "robodojo"
        if (model_cfg.get("multi_gpu") or {}).get("enabled", False):
            from .xpolicylab_adapter.multi_gpu import MultiGpuModel
            self.backend = MultiGpuModel(model_cfg)
        else:
            from .xpolicylab_adapter.robodojo_model import RoboDojoModel
            self.backend = RoboDojoModel(model_cfg)
        self.model_cfg = self.backend.model_cfg
        print("[WorldScape_RSI-runtime] " + json.dumps({
            **self.get_runtime_info(),
            "source": str(getattr(self.backend, "worldscape_root", None)
                          or getattr(self.backend, "upstream_root", "dummy")),
            "checkpoint": str(getattr(self.backend, "checkpoint", None)
                              or getattr(self.backend, "checkpoint_path", None)),
            "mode": getattr(self.backend.mode, "value", None),
            "input_color_order": self.model_cfg.get("input_color_order", "rgb"),
            "gripper_discretization": self.model_cfg.get("gripper_discretization", {"enabled": False}),
            "action_horizon": self.backend.action_horizon,
            "replan_steps": getattr(self.backend, "action_steps", None)
                            or getattr(self.backend, "replan_steps", None),
            "wam_history_mode": self.model_cfg.get("wam_history_mode", "periodic"),
            "history_window_chunks": getattr(self.backend, "task_window_chunks", None),
            "use_causal_memory": self.model_cfg.get("use_causal_memory", True),
            "joint_action_representation": getattr(self.backend, "joint_action_representation", "absolute"),
            "constrain_action_padding": self.model_cfg.get(
                "constrain_action_padding", getattr(self.backend, "constrain_action_padding", False),
            ),
        }, sort_keys=True))

    def __getattr__(self, name):
        backend = self.__dict__.get("backend")
        if backend is None:
            raise AttributeError(name)
        return getattr(backend, name)

    def get_runtime_info(self):
        return {
            "runtime_profile": self.runtime_profile,
            "supports_batch": self.backend.supports_batch,
            "requires_explicit_commit": self.backend.requires_explicit_commit,
            "multi_gpu": bool((self.model_cfg.get("multi_gpu") or {}).get("enabled", False)),
        }

    def close(self):
        close = getattr(self.backend, "close", None)
        if close is not None:
            close()

    def update_obs(self, obs):
        return self.backend.update_obs(obs)

    def update_obs_batch(self, obs_list):
        return self.backend.update_obs_batch(obs_list)

    def get_action(self):
        return self.backend.get_action()

    def get_action_batch(self, env_idx_list=None):
        return self.backend.get_action_batch(env_idx_list)

    def reset(self):
        return self.backend.reset()

    def prepare_case(self, case_meta=None):
        return self.backend.prepare_case(case_meta)

    def on_trial_end(self, result=None):
        return self.backend.on_trial_end(result)

    def commit_action(self, execution=None):
        return self.backend.commit_action(execution)

    def discard_action(self, execution=None):
        return self.backend.discard_action(execution)
