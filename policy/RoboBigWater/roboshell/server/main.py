"""robo-server, direct mode: the RoboDojo environment runs in this process.

One process serves one task. The simulator only advances while a command runs.
Every motion goes through the official take_action(), so step counting and the
success check are the official ones. Used for development and for experiments
that need the truth (replay, counterfactuals); the leaderboard path is bridge.py.
"""

import argparse
import json
import os
import queue
import threading
import time
import traceback
from concurrent.futures import Future

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(allow_abbrev=False)
parser.add_argument("--task", required=True)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--device_id", type=int, default=0)
parser.add_argument("--host", default="127.0.0.1", help="address of the agent interface")
parser.add_argument("--port", type=int, default=28700)
parser.add_argument("--admin-port", type=int, default=28701)
parser.add_argument("--run-root", required=True, help="one sub-directory per episode is created here")
parser.add_argument("--budget", type=int, default=60, help="commands per episode, 0 = unlimited")
parser.add_argument("--video-every", type=int, default=5, help="record a video frame every N action steps")
AppLauncher.add_app_launcher_args(parser)
ARGS = parser.parse_args()
os.environ.setdefault("ROBODOJO_RUN_ID", time.strftime("%Y-%m-%d_%H-%M-%S"))
os.makedirs(ARGS.run_root, exist_ok=True)
os.chdir(ARGS.run_root)  # the official environment writes below ./eval_result

APP = AppLauncher(ARGS).app

import cv2
import numpy as np

from roboshell.contract import v0 as C
from roboshell.server import geometry as geo
from roboshell.server.core import CAMERA_NAMES, SUCCESS_THRESHOLD, Episode, log
from roboshell.server.envsetup import build_env, start_episode
from roboshell.server.http_api import dispatch, start_http
from roboshell.server.tools import load_tools, schema

# camera frame of the simulator (x right, y up, z backward) to the OpenCV one (x right, y down, z forward)
USD_TO_OPENCV = np.diag([1.0, -1.0, -1.0, 1.0])


class DirectExecutor:
    """Drives the official environment in-process."""

    mode = "direct"

    def __init__(self, env):
        self.env = env
        manager = env.robot_manager
        self.robots = {r.arm_name.split("_")[0]: r for r in manager.robot_list if r.type == "target"}
        self.manager = manager
        self.episode = None

    def arm_tags(self):
        return list(self.robots)

    def dt(self):
        return float(self.env.dt)

    def step_lim(self):
        return int(self.env.step_lim)

    def steps(self):
        return int(self.env.take_action_cnt[0])

    def planner(self, tag):
        return self.manager.planner[self.robots[tag].robot_name]

    def robot(self, tag):
        return self.robots[tag]

    def limits(self, tag):
        key = self.manager.robot_key[self.manager.robot_list.index(self.robots[tag])]
        limits = key.data.soft_joint_pos_limits[0][self.robots[tag].arm_joint_indices].cpu().numpy()
        return np.stack([limits[:, 0], limits[:, 1]]).astype(float)

    def joints(self, tag):
        return self.manager.get_joint(self.robots[tag], env_idx_list=[0])[0]

    def ee(self, tag):
        return geo.pose_to_matrix(self.manager.get_real_endpose(self.robots[tag], env_idx_list=[0])[0])

    def gripper_command(self, tag):
        return 1.0  # both grippers start open

    def gripper_measured(self, tag):
        robot = self.robots[tag]
        low, high = robot.gripper_scale
        value = float(self.manager.get_end_effector_real_val(robot, env_idx_list=[0])[0][0])
        value = (value - low) / (high - low)
        if robot.gripper_move["sign"] != 1:
            value = 1.0 - value
        return float(np.clip(value, 0.0, 1.0))

    def end_reason(self):
        if self.env.success[0]:
            return "auto_success"
        if self.steps() >= self.step_lim():
            return "sim_time"
        return "early_fail"

    def run_chunk(self, chunk):
        for step in chunk:
            action = {}
            for tag, (joints, gripper) in step.items():
                robot = self.robots[tag]
                action[self.manager.process_name(robot.arm_name)] = [float(v) for v in joints]
                action[self.manager.process_name(robot.gripper_name)] = [float(gripper)]
            self.env.take_action(action)
            if ARGS.video_every > 0 and self.steps() % ARGS.video_every == 0 and not self.env.end_flag[0]:
                self.env.get_obs()
            if self.env.end_flag[0]:
                return False
        return True

    def observe(self):
        obs = self.env.get_obs()
        png, cameras, depth = {}, {}, {}
        for source in CAMERA_NAMES:
            camera = obs["vision"][source]
            color = np.ascontiguousarray(np.asarray(camera["color"])[:, :, :3]).astype(np.uint8)
            ok, encoded = cv2.imencode(".png", cv2.cvtColor(color, cv2.COLOR_RGB2BGR))
            png[source] = encoded.tobytes()
            if "depth" in camera:
                depth[source] = np.asarray(camera["depth"], dtype=np.float32)
            cameras[source] = {
                "intrinsics": np.asarray(camera["intrinsic_matrix"], dtype=float),
                "extrinsics_world": np.asarray(camera["extrinsic_matrix"], dtype=float) @ USD_TO_OPENCV,
                "size": [color.shape[1], color.shape[0]],
            }
        return {"png": png, "cameras": cameras, "depth": depth}

    def success_now(self):
        reward = float(self.env.reward_manager.get_reward(final_check=True)[0])
        return bool(reward > SUCCESS_THRESHOLD)

    def progress_score(self, success):
        if success:
            return 100.0
        if hasattr(self.env, "get_score"):
            try:
                return float(self.env.reward_manager.get_score()[0])
            except Exception as error:  # the progress score must never break the episode
                log(f"progress score failed: {error}")
        return None

    def private_note(self):
        """Server-side only, written to commands.jsonl for failure analysis: measured gripper openings and
        the true object poses after the command. The agent never sees this."""
        objects = {}
        for key, value in self.object_poses().items():
            objects[key] = [round(float(v), 4) for v in value[:3]]
        return {"gripper_measured": {tag: round(self.gripper_measured(tag), 3) for tag in self.robots}, "objects": objects}

    def object_poses(self):
        layout = self.env.scene_manager.layout_manager
        labels = {}
        for records in layout.object_records_by_type.values():
            for record in records.layout_records_by_env[0]:
                if record.get("label") is not None:
                    labels[record["inst_name"]] = record["label"]
        poses = {}
        for index, (name, kind) in enumerate(layout.instance_type_by_env[0].items()):
            if kind not in ("rigid", "articulation"):
                continue
            try:
                position, rotation = layout.get_instance_pose(env_idx=0, inst_name=name)
                poses[labels.get(name, f"unlabelled{index}")] = np.concatenate([
                    np.asarray(position.detach().cpu() if hasattr(position, "detach") else position, dtype=float).reshape(-1)[:3],
                    np.asarray(rotation.detach().cpu() if hasattr(rotation, "detach") else rotation, dtype=float).reshape(-1)[:4],
                ])
            except Exception as error:
                log(f"pose of {name} not recorded: {error}")
        return poses

    def final_state(self, episode):
        """Joints of both arms and the pose of every object: used to check replays."""
        state = {}
        for tag, arm in episode.arms.items():
            state[f"arm_{tag}"] = np.append(arm.joints(), self.gripper_measured(tag))
        for label, pose in self.object_poses().items():
            state[f"object_{label}"] = pose
        return state

    def on_finish(self, episode, success):
        try:
            self.env.end_flag[0] = True
            self.env.save_video(0, os.path.join(episode.directory, "episode.mp4"), "success" if success else "fail")
        except Exception as error:
            log(f"video not saved: {error}")


class Server:
    def __init__(self):
        self.requests = queue.Queue()
        self.env = None
        self.executor = None
        self.episode = None
        self.episode_started = False
        self.registry = load_tools(ARGS.task)
        if self.registry:
            log(f"tools for {ARGS.task}: {sorted(self.registry)}")

    def tool_schema(self):
        return schema(self.registry)

    def submit(self, kind, payload):
        future = Future()
        self.requests.put((kind, payload, future))
        return future.result()

    def obs_cache(self):
        return self.episode.obs_cache if self.episode is not None else {}

    def build(self):
        started = time.time()
        self.env = build_env(APP, ARGS.task, ARGS.seed, device_id=ARGS.device_id)
        self.executor = DirectExecutor(self.env)
        log(f"environment built in {time.time() - started:.0f} s")

    def reset(self, payload):
        layout = int(payload["layout"])
        if self.episode is not None and not self.episode.over:
            self.episode.finish("infra")
        name = f"{ARGS.task}_s{ARGS.seed}_l{layout}_{time.strftime('%Y%m%dT%H%M%S')}"
        directory = os.path.join(ARGS.run_root, name)
        os.makedirs(directory, exist_ok=True)
        started = time.time()
        if self.episode_started:
            self.env.close()  # the official loop closes the environment between two resets
        self.episode = None
        try:
            start_episode(self.env, layout)
        except Exception as error:
            if type(error).__name__ != "UnStableError":
                raise
            log(f"layout {layout} is unstable")  # the official evaluation skips such layouts
            self.episode_started = True
            return {"error": "unstable_layout", "layout": layout}
        self.episode_started = True
        episode = Episode(self.executor, ARGS.task, ARGS.seed, layout, directory, ARGS.budget, self.registry)
        episode.refresh_obs()
        episode.instruction = str(self.env.get_obs()["instruction"])
        self.episode = episode
        log(f"episode {name} ready in {time.time() - started:.0f} s")
        return {"instruction": episode.instruction, "episode": name, "directory": directory}

    def replay(self, payload):
        """Reset the same layout, feed the recorded joint targets, compare the final state with the recorded one."""
        source = payload["directory"]
        with open(os.path.join(source, "result.json")) as handle:
            recorded = json.load(handle)
        targets = np.load(os.path.join(source, "targets.npz"), allow_pickle=False)
        reference = dict(np.load(os.path.join(source, "final_state.npz"), allow_pickle=False))
        started = self.reset({"layout": recorded["layout"]})
        if "error" in started:
            return started
        episode = self.episode
        order = [str(tag) for tag in targets["arms"]]
        width = targets["targets"].shape[1] // len(order)
        rows = targets["targets"]
        sequences = {tag: rows[:, index * width : (index + 1) * width - 1] for index, tag in enumerate(order)}
        grippers = {tag: rows[:, (index + 1) * width - 1] for index, tag in enumerate(order)}
        alive = True
        for row in range(len(rows)):
            for tag in order:
                episode.arms[tag].gripper_target = float(grippers[tag][row])
            alive = episode.run({tag: sequences[tag][row : row + 1] for tag in order})
            if not alive:
                break
        extra = payload.get("then")
        if extra and not episode.over:
            episode.execute(dict(extra))  # counterfactual: one more command after the recorded ones
        state = self.executor.final_state(episode)
        if not episode.over:
            episode.finish("done", success=self.executor.success_now())
        differences = {}
        for key, value in reference.items():
            if key not in state:
                differences[key] = None
                continue
            other = state[key]
            if key.startswith("object_"):
                angle = geo.angle_between_deg(geo.pose_to_matrix(value)[:3, :3], geo.pose_to_matrix(other)[:3, :3])
                differences[key] = {"pos_m": float(np.linalg.norm(value[:3] - other[:3])), "rot_deg": angle}
            else:
                differences[key] = {"joint_rad": float(np.abs(value[:-1] - other[:-1]).max())}
        report = {
            "source": source,
            "replay": episode.directory,
            "then": extra,
            "steps_recorded": int(recorded["action_steps"]),
            "steps_replayed": episode.steps,
            "success_recorded": recorded["success_official"],
            "success_replayed": episode.result["success_official"],
            "progress_score_replayed": episode.result.get("progress_score"),
            "max_object_pos_diff_m": max([d["pos_m"] for d in differences.values() if d and "pos_m" in d], default=0.0),
            "max_joint_diff_rad": max([d["joint_rad"] for d in differences.values() if d and "joint_rad" in d], default=0.0),
            "differences": differences,
        }
        with open(os.path.join(episode.directory, "replay_report.json"), "w") as handle:
            json.dump(report, handle, indent=1)
        return report

    def handle(self, kind, payload):
        if kind == "reset":
            return self.reset(payload)
        if kind == "replay":
            return self.replay(payload)
        if kind == "finish":
            if self.episode is None:
                return {"error": "no episode"}
            if not self.episode.over:
                self.episode.finish(payload.get("reason", "agent_exit"), success=self.executor.success_now())
            return dict(self.episode.result, over=True, directory=self.episode.directory)
        if kind == "result":
            if self.episode is None:
                return {"error": "no episode"}
            if not self.episode.over:
                return {"over": False, "commands": self.episode.commands, "action_steps": self.episode.steps}
            return dict(self.episode.result, over=True, directory=self.episode.directory)
        return dispatch(self, lambda: self.episode, kind, payload)

    def loop(self):
        log(f"ready: agent {ARGS.host}:{ARGS.port}, admin 127.0.0.1:{ARGS.admin_port}, task {ARGS.task}")
        while APP.is_running():
            try:
                kind, payload, future = self.requests.get(timeout=0.2)
            except queue.Empty:
                continue
            if kind == "shutdown":
                future.set_result({"ok": True})
                break
            try:
                future.set_result(self.handle(kind, payload))
            except Exception as error:
                traceback.print_exc()
                if self.episode is not None and not self.episode.over:
                    try:
                        self.episode.finish("infra")
                    except Exception:
                        traceback.print_exc()
                future.set_result({"error": "internal server error", "exit_code": C.EXIT_EXEC_FAILED, "detail": type(error).__name__})


def main():
    server = Server()
    server.build()
    start_http(server, ARGS.host, ARGS.port, ARGS.admin_port)
    server.loop()
    try:
        server.env.close()
    finally:
        APP.close()


if __name__ == "__main__":
    main()
