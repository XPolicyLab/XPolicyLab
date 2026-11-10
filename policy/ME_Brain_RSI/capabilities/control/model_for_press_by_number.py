import json
from pathlib import Path
from uuid import uuid4
from PIL import Image
from XPolicyLab.model_template import ModelTemplate
from XPolicyLab.utils.process_data import get_robot_action_dim_info
from b88r42wq.policy import Policy
from b88r42wq.contracts import restrict_observation


class Model(ModelTemplate):
    def __init__(self, model_cfg):
        if model_cfg["env_cfg_type"]!="arx_x5" or model_cfg["action_type"]!="ee":
            raise ValueError("Baseline supports arx_x5/ee only")
        dims=get_robot_action_dim_info(model_cfg["env_cfg_type"])
        if len(dims["arm_dim"])!=2 or dims["ee_dim"]!=[1,1]:
            raise ValueError("Incompatible robot schema")
        self.config=model_cfg
        self.calibration=json.loads(Path(model_cfg["calibration_path"]).read_text())
        self.trace_path=Path(model_cfg["trace_path"])
        self.trace_path.parent.mkdir(parents=True,exist_ok=True)
        self.reset()

    def reset(self):
        self.policies={}
        self.observations={}
        self.episode_token=uuid4().hex[:12]
        self.saved_first_rgb=False

    def update_obs(self, obs):
        self.observations[0]=restrict_observation(obs)
        if not self.saved_first_rgb:
            directory=self.trace_path.parent/"first_rgb"
            directory.mkdir(exist_ok=True)
            for camera,frame in self.observations[0]["vision"].items():
                Image.fromarray(frame["color"]).save(directory/f"{self.episode_token}-{camera}.png")
            (directory/f"{self.episode_token}-robot.json").write_text(json.dumps(
                {key:value.tolist() for key,value in self.observations[0]["state"].items()},indent=2))
            self.saved_first_rgb=True

    def update_obs_batch(self, obs_list):
        for obs in obs_list:
            self.observations[int(obs.get("env_idx",0))]=restrict_observation(obs)

    def _action(self, idx):
        if idx not in self.policies:
            self.policies[idx]=Policy(self.config["task_name"],self.calibration)
        policy=self.policies[idx]
        action=policy.act(self.observations[idx])
        with self.trace_path.open("a") as stream:
            stream.write(json.dumps({"env_idx":idx,**policy.diagnostics()})+"\n")
        return [action]

    def get_action(self):
        return self._action(0)

    def get_action_batch(self, env_idx_list=None):
        return [self._action(i) for i in (env_idx_list if env_idx_list is not None else sorted(self.observations))]
