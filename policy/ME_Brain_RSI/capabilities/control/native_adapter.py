"""Transport-only wrapper around the unmodified upstream raw Model."""
import numpy as np
from coap.contracts import restrict_observation

def validate_joint_chunk(chunk):
    expected = {f"{arm}_{kind}" for arm in ("left", "right")
                for kind in ("arm_joint_state", "ee_joint_state")}
    if not isinstance(chunk, (list, tuple)) or not chunk:
        raise ValueError("Expected nonempty upstream joint action chunk")
    for action in chunk:
        if set(action) != expected:
            raise ValueError("PhysicalRSI joint action keys mismatch")
        for key, value in action.items():
            array = np.asarray(value, dtype=float)
            n = 6 if "arm_joint" in key else 1
            if array.shape != (n,) or not np.isfinite(array).all():
                raise ValueError("Invalid PhysicalRSI joint action: " + key)
            if n == 1 and not 0 <= array[0] <= 1:
                raise ValueError("Invalid PhysicalRSI gripper")
    return chunk

class Model:
    def __init__(self, config):
        if (config.get("task_name") != "plug_in_charger" or
                config.get("action_type") != "joint" or not config.get("eval_batch") or
                config.get("env_cfg_type") != "arx_x5"):
            raise ValueError("PhysicalRSI charger requires arx_x5/joint/batch")
        from rdj_rgb_adapters.plug_in_charger import Model as RawModel
        self.raw = RawModel(config)

    def reset(self):
        self.raw.reset()

    def update_obs(self, obs):
        self.raw.update_obs(restrict_observation(obs))

    def update_obs_batch(self, observations):
        self.raw.update_obs_batch([
            {**restrict_observation(obs), "env_idx": int(obs.get("env_idx", 0))}
            for obs in observations])

    def get_action(self):
        return validate_joint_chunk(self.raw.get_action())

    def get_action_batch(self, env_idx_list=None):
        return [validate_joint_chunk(chunk) for chunk in self.raw.get_action_batch(env_idx_list)]

    def prepare_case(self, case_meta=None):
        # Episode metadata is deliberately not passed into the policy.
        self.raw.reset()

    def on_trial_end(self, result=None):
        # Outcome/reward is deliberately not passed into the policy.
        self.raw.reset()
