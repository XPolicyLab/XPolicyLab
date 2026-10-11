"""Public XPolicyLab model entrypoint for the two checkpoint contracts."""
import json

from XPolicyLab.model_template import ModelTemplate


class Model(ModelTemplate):
    def __init__(self, model_cfg):
        from .xpolicylab_adapter.runtime_config import validate_config
        model_cfg = validate_config(model_cfg)
        self.runtime_profile = model_cfg.get("runtime_profile", "robotwin2")
        if self.runtime_profile == "robotwin2":
            from .xpolicylab_adapter.robotwin_model import RoboTwin2Model
            self.backend = RoboTwin2Model(model_cfg)
        elif self.runtime_profile == "robodojo":
            if (model_cfg.get("multi_gpu") or {}).get("enabled", False):
                from .xpolicylab_adapter.multi_gpu import MultiGpuModel
                self.backend = MultiGpuModel(model_cfg)
            else:
                from .xpolicylab_adapter.robodojo_model import RoboDojoModel
                self.backend = RoboDojoModel(model_cfg)
        else:
            raise ValueError("runtime_profile must be robotwin2 or robodojo")
        self.model_cfg = self.backend.model_cfg
        print("[WorldScape_Policy_2-runtime] " + json.dumps({
            **self.get_runtime_info(),
            "source": str(getattr(self.backend, "worldscape_root", None)
                          or getattr(self.backend, "upstream_root", "dummy")),
            "checkpoint": str(getattr(self.backend, "checkpoint", None)
                              or getattr(self.backend, "checkpoint_path", None)),
            "mode": getattr(self.backend.mode, "value", None),
            "input_color_order": self.model_cfg.get("input_color_order", "rgb"),
            "action_horizon": self.backend.action_horizon,
            "replan_steps": getattr(self.backend, "action_steps", None)
                            or getattr(self.backend, "replan_steps", None),
            "diffusion_concat_size": [self.model_cfg.get("diffusion_concat_height", 384),
                                      self.model_cfg.get("diffusion_concat_width", 320)],
            "diffusion_concat_view_order": self.model_cfg.get("diffusion_concat_view_order", [0, 1, 2]),
            "single_window_reference": self.model_cfg.get("single_window_reference", "latest"),
            "wam_history_mode": self.model_cfg.get("wam_history_mode", "periodic"),
            "history_window_chunks": getattr(self.backend, "task_window_chunks", None),
            "use_causal_memory": self.model_cfg.get("use_causal_memory", True),
            "joint_action_representation": getattr(self.backend, "joint_action_representation", "absolute"),
            "constrain_action_padding": self.model_cfg.get(
                "constrain_action_padding", getattr(self.backend, "constrain_action_padding", False),
            ),
            "interactive_qwen": (self.model_cfg.get("interactive_qwen") or {}).get("enabled", False),
            "press_by_number_prompt": (self.model_cfg.get("press_by_number_prompt") or {}).get("enabled", False),
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
        if not self.backend.requires_explicit_commit:
            raise NotImplementedError("robotwin2 commits through post-action observations")
        return self.backend.commit_action(execution)

    def discard_action(self, execution=None):
        if not self.backend.requires_explicit_commit:
            raise NotImplementedError("robotwin2 discards through its native trial/reset lifecycle")
        return self.backend.discard_action(execution)
