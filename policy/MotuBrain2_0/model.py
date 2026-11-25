"""XPolicyLab adapter that loads the downloaded policy on the evaluator's GPU."""

from pathlib import Path

from XPolicyLab.model_template import ModelTemplate
from XPolicyLab.utils.checkpoint_resolver import resolve_checkpoint_root
from XPolicyLab.utils.process_data import get_robot_action_dim_info

from motubrain_robodojo import load_policy


class Model(ModelTemplate):
    def __init__(self, model_cfg):
        super().__init__()
        setting = (model_cfg.get("bench_name"), model_cfg.get("env_cfg_type"), model_cfg.get("action_type"))
        if setting not in {("RoboDojo", "arx_x5", "joint"), ("RoboTwin", "aloha_agilex", "ee")}:
            raise ValueError(f"Unsupported benchmark, robot and action type: {setting}")
        if model_cfg.get("eval_batch"):
            raise ValueError("This direct inference release supports one environment per process")
        dims = get_robot_action_dim_info(model_cfg["env_cfg_type"])
        if tuple(dims["arm_dim"]) != (6, 6) or tuple(dims["ee_dim"]) != (1, 1):
            raise ValueError("Expected two 6-DOF arms with scalar grippers")
        policy_dir = Path(__file__).resolve().parent
        root = resolve_checkpoint_root(model_cfg, policy_dir / "checkpoints", policy_dir=policy_dir)
        options = {name: model_cfg[name] for name in (
            "checkpoint_file", "model_config_file", "action_stats_file", "query_seed_base",
        ) if name in model_cfg}
        if model_cfg.get("policy_config"):
            options["policy_config"] = str((policy_dir / Path(model_cfg["policy_config"]).expanduser()).resolve())
        self.policy = load_policy(root, device="cuda", action_type=setting[2], **options)
        self.batch_size = 1

    def update_obs(self, obs):
        self.policy.update_obs(obs)

    def get_action(self):
        return self.policy.get_action()

    def reset(self):
        self.policy.reset()

    def update_obs_batch(self, obs_list):
        raise NotImplementedError("Use eval_batch=false; batch inference is not part of this release")

    def get_action_batch(self, env_idx_list=None):
        raise NotImplementedError("Use eval_batch=false; batch inference is not part of this release")

    def close(self):
        self.policy.close()
