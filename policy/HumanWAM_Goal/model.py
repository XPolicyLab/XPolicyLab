"""Thin ModelTemplate boundary; the released HumanWAM-Goal package owns all physical transforms."""

import atexit
import json
from pathlib import Path

import numpy as np
from XPolicyLab.model_template import ModelTemplate
from XPolicyLab.utils.checkpoint_resolver import resolve_checkpoint_root
from XPolicyLab.utils.process_data import get_robot_action_dim_info


POLICY_DIR = Path(__file__).resolve().parent
ACTION_KEYS = ("left_arm_joint_state", "left_ee_joint_state", "right_arm_joint_state", "right_ee_joint_state")


def verify_runtime(config, output):
    """Require the pinned cuBLASLt (preloaded by launch.py) before any weights load."""
    import torch
    from goalwam.robotwin import cublaslt_runtime

    # Checks the library actually mapped into this process: a missing preload,
    # another binary or a second loaded copy fails startup.
    runtime = dict(
        torch=torch.__version__, cuda=torch.version.cuda, cublaslt=cublaslt_runtime(config["cublaslt_sha256"])
    )
    output.mkdir(parents=True, exist_ok=True)
    (output / "runtime.json").write_text(json.dumps(runtime, indent=2) + "\n")
    return runtime


def build_engine(config, checkpoint):
    """goalwam.GoalWAM plus the RoboTwin session of the released HumanWAM-Goal code."""
    output = Path(config["output"]).resolve()
    verify_runtime(config, output)
    # Delayed imports keep installation/profile checks independent of CUDA.
    from goalwam import GoalWAM
    from goalwam.robotwin import OfficialGoalCache, RoboTwinSession

    goals = None
    if config.get("eval_env_type") != "debug":
        goals = OfficialGoalCache(
            config["official_goal_cache"], config["official_goal_cache_sha256"], config["robotwin_profile"]
        )
    model = GoalWAM(checkpoint, backend="native", cuda_graph=False)
    try:
        # Debug mode exercises transport, observation conversion, inference and
        # action packing with the current head image as the goal: XPolicyLab's
        # random debug observations match no cached scene. Simulator runs always
        # use the released goal of the exact native case.
        return RoboTwinSession(model, output / "session", goals)
    except BaseException:
        model.close()
        raise


def debug_observation(obs):
    """XPolicyLab's debug client sends ee_pose = ones(7); HumanWAM-Goal requires unit wxyz quaternions.

    Debug mode only: normalize the synthetic quaternions so the transport, conversion,
    inference and packing path can run. Simulator observations are never altered.
    """
    state = dict(obs["state"])
    for side in ("left", "right"):
        pose = np.asarray(state[f"{side}_ee_pose"], dtype=np.float64).copy()
        norm = np.linalg.norm(pose[3:7])
        pose[3:7] = pose[3:7] / norm if norm > 0 else (1.0, 0.0, 0.0, 0.0)
        state[f"{side}_ee_pose"] = pose
    return dict(obs, state=state)


class Model(ModelTemplate):
    def __init__(self, model_cfg):
        super().__init__()
        self.model_cfg = dict(model_cfg)
        if (
            model_cfg.get("policy_name") != "HumanWAM_Goal"
            or model_cfg.get("bench_name") != "RoboTwin"
            or model_cfg.get("protocol") != "ws"
            or model_cfg.get("action_type") != "joint"
            or model_cfg.get("eval_batch") is not False
            or model_cfg.get("seed") != 0
        ):
            raise ValueError("HumanWAM-Goal requires RoboTwin/ws/joint, seed=0 and eval_batch=false")
        required = ["output", "cublaslt_sha256"]
        if model_cfg.get("eval_env_type") != "debug":
            required += ["official_goal_cache", "official_goal_cache_sha256", "robotwin_profile"]
        for key in required:
            if not isinstance(model_cfg.get(key), str) or not model_cfg[key].strip():
                raise ValueError("Missing deployment setting: " + key + " (use eval.sh or launch.py)")
        if model_cfg.get("env_cfg_type") not in {"arx_x5", "aloha_agilex"}:
            raise ValueError("Unsupported wire configuration for the Aloha-AgileX checkpoint")
        self.dimensions = get_robot_action_dim_info(model_cfg["env_cfg_type"])
        aloha = get_robot_action_dim_info("aloha_agilex")
        if self.dimensions != aloha or len(aloha["arm_dim"]) != 2 or len(aloha["ee_dim"]) != 2:
            raise ValueError("Wire dimensions differ from the official Aloha-AgileX metadata")
        checkpoint = resolve_checkpoint_root(model_cfg, POLICY_DIR / "checkpoints", policy_dir=POLICY_DIR)
        self.model = build_engine(model_cfg, checkpoint)
        self.debug = model_cfg.get("eval_env_type") == "debug"
        self._closed = False
        atexit.register(self.close)

    def prepare_case(self, case_meta=None):
        return self.model.prepare_case(case_meta)

    def validate_cases(self, request):
        return self.model.validate_cases(request)

    def update_obs(self, obs):
        # Images have already been decoded by XPolicyLab. Never decode/swap here.
        if self.debug:
            obs = debug_observation(obs)
        return self.model.update_obs(obs)

    def get_action(self):
        commands = np.asarray(self.model.get_action())
        widths = [n for pair in zip(self.dimensions["arm_dim"], self.dimensions["ee_dim"]) for n in pair]
        if (
            commands.ndim != 2
            or commands.shape[1] != sum(widths)
            or not len(commands)
            or not np.isfinite(commands).all()
        ):
            raise ValueError("The HumanWAM-Goal session returned an invalid native joint command chunk")
        boundaries = np.cumsum([0, *widths])
        # Packing only: values are already absolute native joints and openness.
        # No normalization, sign conversion, geometry or controller logic here.
        return [
            {key: row[start:stop].copy() for key, start, stop in zip(ACTION_KEYS, boundaries[:-1], boundaries[1:])}
            for row in commands
        ]

    def update_obs_batch(self, obs_list):
        if len(obs_list) != 1 or obs_list[0].get("env_idx", 0) != 0:
            raise ValueError("HumanWAM-Goal supports one active environment per policy process")
        return [self.update_obs(obs_list[0])]

    def get_action_batch(self, env_idx_list=None):
        if env_idx_list is not None and list(env_idx_list) != [0]:
            raise ValueError("HumanWAM-Goal supports only env_idx=0")
        return [self.get_action()]

    def reset(self):
        return self.model.reset()

    def on_trial_end(self, result=None):
        return self.model.on_trial_end(result)

    def close(self):
        if not self._closed:
            self._closed = True
            self.model.model.close()
