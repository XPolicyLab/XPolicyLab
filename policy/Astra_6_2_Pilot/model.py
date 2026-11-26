"""RGB and proprioception to short ARX X5 joint-space trajectories."""
import base64
import json

import numpy as np
from openai import OpenAI

from XPolicyLab.model_template import ModelTemplate
from XPolicyLab.policy.GPT_6_Astra_Direct_EEF.inspect.embodiment_docs import ARX_X5_JOINT_DOCS
from XPolicyLab.utils.process_data import encode_image_bit, get_robot_action_dim_info, get_robot_state_schema


class Model(ModelTemplate):
    def __init__(self, model_cfg):
        super().__init__()
        if (model_cfg["env_cfg_type"], model_cfg["action_type"]) != ("arx_x5", "joint"):
            raise ValueError("The pilot requires arx_x5 with joint actions.")
        if model_cfg.get("eval_batch"):
            raise ValueError("Use single-episode evaluation for this pilot.")
        self.widths = get_robot_state_schema("joint", get_robot_action_dim_info("arx_x5"))
        self.client = OpenAI(timeout=180, max_retries=0)
        self.reset()

    def reset(self):
        self.obs = None
        self.note = ""

    def update_obs(self, obs):
        self.obs = obs

    def get_action(self):
        if self.obs is None:
            raise RuntimeError("Missing observation after reset")
        obs = self.obs
        state = {key: np.asarray(obs["state"][key], dtype=float).reshape(width)
                 for key, width in self.widths.items()}
        context = {
            "goal": obs.get("instruction") or obs.get("instructions"),
            "official_description": (obs.get("extra_info") or {}).get("detailed_instruction"),
            "joint_positions": {key: values.tolist() for key, values in state.items()},
            "previous_note": self.note,
        }
        parts = [{"type": "input_text", "text": json.dumps(context, allow_nan=False)}]
        for name in ("cam_head", "cam_left_wrist", "cam_right_wrist"):
            view = obs["vision"].get(name, {})
            if view.get("color") is None:
                continue
            encoded = base64.b64encode(encode_image_bit(view["color"])).decode("ascii")
            parts.extend([{"type": "input_text", "text": name},
                          {"type": "input_image", "image_url": "data:image/jpeg;base64," + encoded}])
        if len(parts) == 1:
            raise ValueError("At least one RGB observation is required.")
        properties = {key: {"type": "array", "items": {"type": "number"},
                            "minItems": width, "maxItems": width}
                      for key, width in self.widths.items()}
        properties["note"] = {"type": "string"}
        reply = self.client.responses.create(
            model="gpt-6.2-astra", store=False, reasoning={"effort": "high"},
            instructions=ARX_X5_JOINT_DOCS + "\nChoose the next absolute joint and gripper "
            "targets for the given goal. Use the observed values for stationary joints. "
            "Arm moves are limited to 0.25 radians per decision. Write a brief note "
            "about the intended motion and what to verify on the next observation.",
            input=[{"role": "user", "content": parts}],
            text={"format": {"type": "json_schema", "name": "motion", "strict": True,
                             "schema": {"type": "object", "properties": properties,
                                        "required": list(properties), "additionalProperties": False}}},
        )
        target = json.loads(reply.output_text)
        delta = {}
        for key, current in state.items():
            desired = np.asarray(target[key], dtype=float)
            if desired.shape != current.shape or not np.isfinite(desired).all():
                raise ValueError(f"Invalid target for {key}")
            if key.endswith("_arm_joint_state"):
                delta[key] = np.clip(desired - current, -0.25, 0.25)
            else:
                delta[key] = np.clip(desired, 0, 1) - current
        self.note = target["note"][:500]
        return [{key: (current + delta[key] * fraction).astype(np.float32)
                 for key, current in state.items()} for fraction in np.linspace(0.125, 1, 8)]

    def update_obs_batch(self, obs_list):
        raise NotImplementedError("Single-episode evaluation only")

    def get_action_batch(self, env_idx_list=None):
        raise NotImplementedError("Single-episode evaluation only")
