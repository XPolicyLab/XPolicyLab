"""Local PolicyLab bridge for RoboDojo's native task evaluators."""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys

from policylab.api import InvalidAction


class _NoPolicyServer:
    """EvalEnv only needs a policy reset; PolicyLab supplies actions directly."""

    def __init__(self, **kwargs):
        pass

    def call(self, **kwargs):
        if kwargs.get("func_name") != "reset":
            raise RuntimeError("Unexpected native policy call")


class Adapter:
    def __init__(self, environment_dir, options):
        self.root = Path(environment_dir).resolve()
        os.chdir(self.root)
        sys.path[:0] = [str(Path(__file__).resolve().parent), str(self.root), str(self.root / "XPolicyLab")]

        from isaaclab.app import AppLauncher

        parser = argparse.ArgumentParser()
        AppLauncher.add_app_launcher_args(parser)
        args = parser.parse_args([
            "--headless", "--enable_cameras", "--device", f"cuda:{int(options.get('gpu', 0))}",
            "--kit_args", " --enable isaacsim.replicator.behavior --enable isaacsim.sensors.camera",
        ])
        if (self.root / "env/camera_manager/capture/render_sync.py").is_file():
            from env.camera_manager.capture.render_sync import add_zero_delay_kit_args
            add_zero_delay_kit_args(args)
        self.app = AppLauncher(args).app

        from omegaconf import OmegaConf
        from env.global_configs import BENCHMARK, ENV_CONFIG_PATH
        from task.RoboDojo import task_registry
        from utils.load_file import load_yaml
        from utils.pipeline_utils import process_config, process_randomization
        from src.eval_client import eval_env

        task = options.get("task", "organize_table")
        available = {p.stem for p in (self.root / "task/RoboDojo/config").glob("*.yml")
                     if not p.stem.startswith("_") and not p.stem.endswith("_random")}
        if task not in available:
            raise ValueError("Unsupported standard RoboDojo task")
        self.task = task
        base = load_yaml(str(self.root / "env_cfg/arx_x5.yml"))
        base.update(task_name=task, num_envs=1, device_id=int(options.get("gpu", 0)),
                    eval_batch=False, policy_name="demo_policy", additional_info="policylab",
                    seed=0, eval_num=1, physx_monitor_enabled=False)
        cfg = OmegaConf.create({
            "sim": load_yaml(str(self.root / "env_cfg/sim" / (base["config"]["sim"] + ".yml"))),
            "scene": load_yaml(str(self.root / "env_cfg/scene" / (base["config"]["scene"] + ".yml"))),
            "camera": load_yaml(str(self.root / "env_cfg/camera" / (base["config"]["camera"] + ".yml"))),
            "robot": load_yaml(str(self.root / "env_cfg/robot" / (base["config"]["robot"] + ".yml"))),
            "task_env": load_yaml(task_registry.task_config_path(str(self.root / "task" / BENCHMARK / "config"), task)),
            "eval_cfg": base,
            "deploy_cfg": {"policy_name": "demo_policy", "host": "localhost", "port": 1},
        })
        cfg.sim.scene.num_envs = 1
        cfg.sim.seed = [0]
        cfg = process_randomization(cfg)
        cfg, _ = process_config(cfg, task_name=task)
        OmegaConf.update(cfg, "camera.default_frequency", base["observation"]["collect_freq"], force_add=True)
        eval_env.WsModelClient = _NoPolicyServer
        self.env = eval_env.create_eval_env(cfg, self.app)
        self.cfg = cfg
        self.action_mode = options.get("action_mode", "joint")
        if self.action_mode not in ("joint", "eef"):
            raise ValueError("Unknown action mode")
        if self.action_mode == "eef":
            from eef_kinematics import DualKinematics
            self.kinematics = DualKinematics(self.env)
        self.record_dir = None
        self.observation_dir = None
        self.frame = 0
        self.finalized = False

    def reset(self, episode, record_dir, observation_dir):
        if episode.get("task") != self.task or episode.get("robot") != "arx_x5":
            raise ValueError("Unexpected RoboDojo task or robot")
        if getattr(self, "physics_log", None) is not None:
            self.physics_log.close()
        self.record_dir = Path(record_dir)
        self.observation_dir = Path(observation_dir)
        self.record_dir.mkdir(parents=True, exist_ok=True)
        self.observation_dir.mkdir(parents=True, exist_ok=True)
        self.frame = 0
        self.finalized = False
        self.env._stream_dir = str(self.record_dir / "_stream")
        self.env.eval_cfg["seed"] = int(episode["seed"])
        self.env.seed_manager.init_eval()
        layout = int(episode["layout_id"])
        layout_path = Path(self.env.seed_manager.seed_info[layout]["scene_layout"])
        layout_sha256 = hashlib.sha256(layout_path.read_bytes()).hexdigest()
        self.env.reset(seed=[layout])
        self.env.run_reward()
        if hasattr(self.env, "get_score"):
            self.env.get_score()
        from omegaconf import OmegaConf
        OmegaConf.save(self.cfg, self.record_dir / "native_config.yaml")
        (self.record_dir / "native_episode.json").write_text(json.dumps({
            "episode": episode,
            "layout_sha256": layout_sha256,
            "physics_dt": float(self.env.sim.physics_dt),
            "physics_steps_per_action": int(self.env.obs_manager.collect_interval),
        }) + "\n")
        self.physics_start = int(self.env.sim._sim_step_counter)
        self.physics_log = (self.record_dir / "physics.jsonl").open("w", buffering=1)
        return self._transition()

    def _transition(self, applied_action=None):
        import numpy as np
        from PIL import Image

        raw = self.env.get_obs()
        pictures = {}
        for camera, fields in raw["vision"].items():
            name = f"{self.frame:05d}_{camera}.png"
            path = self.observation_dir / name
            Image.fromarray(np.asarray(fields["color"], dtype=np.uint8)).save(path)
            pictures[camera] = f"obs://{name}"
        state = {key: np.asarray(value).tolist() for key, value in raw["state"].items()}
        done = bool(self.env.end_flag[0])
        observation = {"instruction": raw["instruction"], "state": state,
                       "cameras": pictures, "control_step": int(self.env.take_action_cnt[0]),
                       "control_frequency_hz": int(raw["additional_info"]["frequency"])}
        self.frame += 1
        result = {"observation": observation, "done": done, "metrics": {}}
        if applied_action is not None:
            result["applied_action"] = applied_action
        if done:
            result["metrics"] = self._native_result()
        return result

    def _native_result(self):
        if not self.finalized:
            success = bool(self.env.success[0])
            score = 1.0 if success else (float(self.env.reward_manager.get_score()[0]) / 100.0
                                         if hasattr(self.env, "get_score") else 0.0)
            self.env.save_video(0, str(self.record_dir / "episode_0000000.mp4"),
                                "success" if success else "fail")
            self.result = {"success": success, "score": score,
                           "layout_id": int(self.env.env_seeds[0])}
            self.finalized = True
        return self.result

    def step(self, action):
        if self.action_mode == "eef":
            return self._step_eef(action)
        return self._step_joint(action)

    def _step_eef(self, action):
        import numpy as np
        from scipy.spatial.transform import Rotation

        if not isinstance(action, dict) or set(action) != {"mode", "target", "reason", "assessment"} or action["mode"] != "eef":
            raise InvalidAction("Provide mode=eef, dual-arm target, current-state assessment and decision reason")
        assessment = action["assessment"]
        if (not isinstance(assessment, dict) or set(assessment) != {"current_subgoal"}
                or not isinstance(assessment["current_subgoal"], str)
                or not assessment["current_subgoal"].strip()):
            raise InvalidAction("assessment.current_subgoal is required")
        if not isinstance(action["reason"], str) or not action["reason"].strip():
            raise InvalidAction("Decision reason is required")
        targets = action["target"]
        if not isinstance(targets, dict) or set(targets) != {"left", "right"}:
            raise InvalidAction("EEF target needs left and right arms")
        state = self.env.get_obs()["state"]
        for arm in ("left", "right"):
            target = targets[arm]
            if not isinstance(target, dict) or set(target) != {"position", "quaternion_wxyz", "gripper_closed"}:
                raise InvalidAction(f"{arm} target needs position, quaternion_wxyz and gripper_closed")
            try:
                p, q = np.asarray(target["position"], float), np.asarray(target["quaternion_wxyz"], float)
            except (TypeError, ValueError) as error:
                raise InvalidAction("Numeric EEF position and quaternion required") from error
            if p.shape != (3,) or q.shape != (4,) or not np.isfinite(np.r_[p, q]).all():
                raise InvalidAction("Finite EEF position and quaternion required")
            if abs(np.linalg.norm(q) - 1) > 1e-4 or type(target["gripper_closed"]) is not bool:
                raise InvalidAction("Unit quaternion and explicit boolean gripper required")
            old = np.asarray(state[f"{arm}_ee_pose"], float)
            if np.linalg.norm(p - old[:3]) > .05 + 1e-9:
                raise InvalidAction("EEF target exceeds 5 cm")
            turn = Rotation.from_quat(q[[1, 2, 3, 0]]) * Rotation.from_quat(old[[4, 5, 6, 3]]).inv()
            if turn.magnitude() > .35 + 1e-9:
                raise InvalidAction("EEF target exceeds 0.35 rad")
        proposal = self.kinematics.target(targets)
        values = proposal["action"]
        command = {"left_arm_joint_state": values[:6], "left_ee_joint_state": values[6:7],
                   "right_arm_joint_state": values[7:13], "right_ee_joint_state": values[13:14]}
        result = self._step_joint(command)
        result["applied_action"] = {"joint_command": command, "ik_diagnostics": proposal["diagnostics"]}
        return result

    def _step_joint(self, action):
        names = ("left_arm_joint_state", "left_ee_joint_state", "right_arm_joint_state", "right_ee_joint_state")
        dims = (6, 1, 6, 1)
        if not isinstance(action, dict) or set(action) != set(names):
            raise InvalidAction("Provide both six-joint arm targets and both single-value gripper targets")
        for name, dim in zip(names, dims):
            values = action[name]
            if not isinstance(values, list) or len(values) != dim or any(type(x) not in (int, float) or not math.isfinite(x) for x in values):
                raise InvalidAction(f"{name} must contain {dim} finite numbers")
        for name in ("left_ee_joint_state", "right_ee_joint_state"):
            if not 0 <= action[name][0] <= 1:
                raise InvalidAction(f"{name} must be in [0, 1]")
        import numpy as np

        native_step = self.env.step

        def recorded_step(env_idx_list, decimation=1):
            if decimation != 1:
                raise RuntimeError("Unsupported native physics decimation")
            native_step(env_idx_list=env_idx_list, decimation=decimation)
            controls = self.env.robot_manager.control_manager.prev_control[0]
            applied = {key: {field: np.asarray(value).tolist() for field, value in command.items()}
                       for key, command in controls.items()}
            measured = {}
            manager = self.env.robot_manager
            for robot in manager.robot_list:
                if robot.type != "target":
                    continue
                measured[manager.process_name(robot.arm_name)] = manager.get_joint(robot, [0])[0].tolist()
                measured[manager.process_name(robot.gripper_name)] = manager.get_end_effector_real_val(robot, [0])[0].tolist()
            physics_step = int(self.env.sim._sim_step_counter) - self.physics_start
            self.physics_log.write(json.dumps({
                "control_step": int(self.env.take_action_cnt[0]),
                "physics_step": physics_step,
                "simulation_time_s": physics_step * float(self.env.sim.physics_dt),
                "applied_control": applied,
                "measured_joint_state": measured,
            }) + "\n")

        self.env.step = recorded_step
        try:
            self.env.take_action(action)
        finally:
            self.env.step = native_step
        return self._transition()

    def finish(self):
        if not self.env.end_flag[0]:
            reward = self.env.reward_manager.get_reward(final_check=True)[0]
            self.env.success[0] = bool(reward > 1 - 1e-3)
            self.env.end_flag[0] = True
        return self._transition()

    def close(self):
        if getattr(self, "physics_log", None) is not None:
            self.physics_log.close()
        if getattr(self, "env", None) is not None:
            self.env.close()
        if getattr(self, "app", None) is not None:
            self.app.close()
