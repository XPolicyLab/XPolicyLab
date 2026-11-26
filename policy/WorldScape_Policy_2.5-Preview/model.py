"""RoboDojo policy entrypoint."""
import json

from XPolicyLab.model_template import ModelTemplate


class Model(ModelTemplate):
    def __init__(self, model_cfg):
        from .xpolicylab_adapter.runtime_config import validate_config
        model_cfg = validate_config(model_cfg)
        self.runtime_profile = "robodojo"
        from .xpolicylab_adapter.robodojo_memory import RoboDojoMemoryPolicy
        self.backend = RoboDojoMemoryPolicy(model_cfg)
        self.model_cfg = self.backend.model_cfg
        print("[WorldScape_Policy_2.5-Preview-runtime] " + json.dumps({
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
            "observation_interval": getattr(self.backend, "observation_interval", None),
            "diffusion_concat_size": [self.model_cfg.get("diffusion_concat_height", 384),
                                      self.model_cfg.get("diffusion_concat_width", 320)],
            "diffusion_concat_view_order": self.model_cfg.get("diffusion_concat_view_order", [0, 1, 2]),
            "single_window_reference": self.model_cfg.get("single_window_reference", "latest"),
            "wam_history_mode": self.model_cfg.get("wam_history_mode", "periodic"),
            "memory_reset_chunks": self.model_cfg.get("memory_reset_chunks", 0),
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
        policy = getattr(self.backend, "policy", None)
        router = getattr(policy, "condition_router", None)
        text_encoder = getattr(getattr(router, "interactive", None), "t5", None)
        text_encoder_name = type(text_encoder).__name__ if text_encoder is not None else None
        return {
            "runtime_profile": self.runtime_profile,
            "checkpoint_format": self.model_cfg.get("checkpoint_format", "native"),
            "history_memory": self.model_cfg.get("history_memory", False),
            "vlm_memory": self.model_cfg.get("vlm_memory", False),
            "vlm_conditioning": self.model_cfg.get("vlm_conditioning"),
            "text_encoder": text_encoder_name,
            "text_encoding": {"T5InstructionEncoder": "online",
                              "PrecomputedT5InstructionEncoder": "cached"}.get(text_encoder_name),
            "vlm_input_color_order": self.model_cfg.get("vlm_input_color_order", "rgb"),
            "history_capture_mode": self.model_cfg.get("history_capture_mode", "dense"),
            "checkpoint": str(getattr(self.backend, "checkpoint", None)
                              or getattr(self.backend, "checkpoint_path", None)),
            "action_horizon": self.backend.action_horizon,
            "replan_steps": getattr(self.backend, "action_steps", None)
                            or getattr(self.backend, "replan_steps", None),
            "inference_precision": self.model_cfg.get("inference_precision", "native"),
            "supports_batch": self.backend.supports_batch,
            "requires_explicit_commit": self.backend.requires_explicit_commit,
            "observation_interval": getattr(self.backend, "observation_interval", None),
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
