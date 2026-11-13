import base64
import io
import json
from pathlib import Path
import time
import numpy as np
from PIL import Image
import httpx
from XPolicyLab.model_template import ModelTemplate
from XPolicyLab.utils.process_data import get_robot_action_dim_info


class Model(ModelTemplate):
    def __init__(self, config):
        super().__init__()
        if (config["bench_name"], config["env_cfg_type"], config["action_type"]) != ("RoboDojo", "arx_x5", "joint") or config["eval_batch"]:
            raise ValueError("ActWeave supports RoboDojo/arx_x5/joint with eval_batch=false")
        runtime = json.loads(Path(config["ckpt_name"]).read_text())
        self.http = httpx.Client(base_url="http://127.0.0.1:" + str(runtime["ports"]["bridge"]), timeout=30)
        dimensions = get_robot_action_dim_info(config["env_cfg_type"])
        self.action_dimensions = {
            f"{side}_{kind}_joint_state": dimensions[field][index]
            for index, side in enumerate(['left', 'right'])
            for kind, field in [('arm', 'arm_dim'), ('ee', 'ee_dim')]
        }
        self.config = config
        self.final_observation = None

    def post(self, endpoint, value):
        response = self.http.post(endpoint, json=value)
        response.raise_for_status()
        return response.json()

    def reset(self):
        self.final_observation = None
        return self.post("/reset", {"task": self.config["task_name"]})

    def encode_obs(self, obs):
        images = {}
        for name, camera in obs["vision"].items():
            data = io.BytesIO()
            Image.fromarray(np.asarray(camera["color"], dtype=np.uint8)).save(data, format="PNG", compress_level=1)
            images[name] = base64.b64encode(data.getvalue()).decode()
        state = {key: np.asarray(value).tolist() for key, value in obs["state"].items()}
        meta = {key: obs[key] for key in ['step', 'step_limit']}
        return {"instruction": obs["instruction"], "images": images, "state": state, "env_idx": int(obs["env_idx"]), **meta}

    def update_obs(self, obs):
        return self.post("/observation", self.encode_obs(obs))

    def poll_action(self):
        result = self.post("/action", {})
        if result["status"] == "action":
            action = result["action"]
            if set(action) != set(self.action_dimensions):
                raise ValueError("Runtime action keys do not match the configured robot")
            action = {key: np.asarray(value, dtype=np.float32) for key, value in action.items()}
            for key, dimension in self.action_dimensions.items():
                if action[key].shape != (dimension,):
                    raise ValueError(f"Runtime action {key} has shape {action[key].shape}, expected {(dimension,)}")
            result["action"] = action
        return result

    def get_action(self):
        while True:
            result = self.poll_action()
            if result["status"] == "action":
                return [result["action"]]
            if result["status"] == "finished":
                return []
            time.sleep(0.1)

    def update_obs_batch(self, obs_list):
        raise NotImplementedError("ActWeave supports one environment per policy instance; use eval_batch=false")

    def get_action_batch(self, env_idx_list=None):
        raise NotImplementedError("ActWeave supports one environment per policy instance; use eval_batch=false")

    def cache_final_observation(self, obs):
        self.final_observation = self.encode_obs(obs)

    def on_trial_end(self, result=None):
        if self.final_observation is None:
            raise RuntimeError("Trial end requires the final public observation")
        return self.post("/end", {**result, "final_observation": self.final_observation})
