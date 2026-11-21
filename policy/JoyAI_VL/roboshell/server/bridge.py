"""robo-server, bridge mode: the official evaluation client drives the simulator.

This module is the XPolicyLab policy. The official client calls update_obs()
with every observation and get_action() for the next action chunk; the agent
talks to the robo interface. A robo command becomes one
joint-target chunk, and the command's feedback is computed from the observation
the client sends back after the chunk ran. Nothing here touches the simulator:
the policy knows only the observations, the robot model and the camera model.
"""

import json
import os
import queue
import re
import subprocess
import threading
import time
import traceback

from roboshell.image_transport import encode_png
import numpy as np
import yaml

from roboshell import task_card
from roboshell.contract import v0 as C
from roboshell.server import geometry as geo
from roboshell.server.core import NoDepth, CAMERA_NAMES, Episode, log
from roboshell.server.http_api import dispatch, start_http
from roboshell.server.tools import load_tools, schema, select_profile

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ACTION_POLL_S = 20.0  # stay below the official RPC timeout without advancing physics
GATEWAY_ERRORS = re.compile(r"429|Too Many Requests|exceeded retry limit|at capacity|overloaded|rate limit|stream disconnected|50[234] |Bad Gateway")
RELAUNCH_WAIT_S = 60.0  # wait before starting the acting agent again after a gateway refusal
RELAUNCH_MAX = 30       # refusals tolerated per episode before the run is aborted as an infrastructure failure


class RobotSpec:
    def __init__(self, entity_origin_pose):
        self.entity_origin_pose = list(entity_origin_pose)


class BridgeExecutor:
    mode = "bridge"

    def __init__(self, cfg):
        self.cfg = cfg
        self.repo = cfg["robodojo_repo"]
        self.task = cfg["task_name"]
        with open(os.path.join(self.repo, "env_cfg", "robot", cfg.get("robot_config", "dual_x5") + ".yml")) as handle:
            robots = yaml.safe_load(handle)["robots"]
        self.specs = {}
        for index, robot in enumerate(robots):
            tag = "left" if index == 0 else "right"
            self.specs[tag] = RobotSpec(list(robot["default_root_pos"]) + list(robot["default_root_rot"]))
        with open(cfg.get("camera_model", os.path.join(ROOT, "agent", "camera_model.json"))) as handle:
            self.camera_model = json.load(handle)["cameras"]
        log("observation: RGB and camera matrices only, no depth")
        self._step_lim = self.read_step_lim()
        self.planner_obj = None
        self.obs = None
        self.pending = queue.Queue()
        self.executed = threading.Event()
        self.awaiting = False
        self.over = False
        self.reason = None
        self.verdict = None
        self.end_seen = threading.Event()
        self.delivered = 0
        self.dt_s = 0.004

    def read_step_lim(self):
        path = os.path.join(self.repo, "task", "RoboDojo", "tasks", f"{self.task}.py")
        with open(path) as handle:
            match = re.search(r"self\.step_lim\s*=\s*(\d+)", handle.read())
        if not match:
            raise ValueError(f"step_lim not found in {path}")
        return int(match.group(1))

    def build_planner(self):
        from env.planner_manager.curobo_planner import CuroboPlanner

        names = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"]
        root = os.path.join(self.repo, "Assets", "Robots", "x5")
        self.planner_obj = CuroboPlanner(
            robot_origin_pose=[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0], active_joints_name=names, all_joints=names,
            dt=self.dt_s, yml_path=os.path.join(root, "curobo.yml"), table_height=0.74 - self.specs["left"].entity_origin_pose[2],
        )

    # ---- Executor interface ----------------------------------------------
    def arm_tags(self):
        return list(self.specs)

    def dt(self):
        return self.dt_s

    def step_lim(self):
        return self._step_lim

    def steps(self):
        if self.verdict is not None and self.verdict.get("step") is not None:
            return int(self.verdict["step"])
        return self.delivered

    def planner(self, tag):
        return self.planner_obj

    def robot(self, tag):
        return self.specs[tag]

    def limits(self, tag):
        return None

    def joints(self, tag):
        return np.asarray(self.obs["state"][f"{tag}_arm_joint_state"], dtype=float)

    def ee(self, tag):
        return geo.pose_to_matrix(np.asarray(self.obs["state"][f"{tag}_ee_pose"], dtype=float))

    def gripper_command(self, tag):
        value = self.obs["state"].get(f"{tag}_ee_joint_state")
        return float(value[0]) if value is not None else 1.0

    def gripper_observed(self, tag):
        """The official observation's gripper value: the finger target after the control's clamp (see core.py)."""
        value = ((self.obs or {}).get("state") or {}).get(f"{tag}_ee_joint_state")
        return float(value[0]) if value is not None else None

    def end_reason(self):
        return self.reason or "episode_over"

    def run_chunk(self, chunk):
        """Hand the chunk to the client and wait until the client asks for the next one."""
        if self.over:
            return False
        actions = []
        for step in chunk:
            action = {}
            for tag, (joints, gripper) in step.items():
                action[f"{tag}_arm_joint_state"] = [float(v) for v in joints]
                action[f"{tag}_ee_joint_state"] = [float(gripper)]
            actions.append(action)
        self.executed.clear()
        if self.over:
            return False
        self.pending.put(actions)
        self.executed.wait()
        return not self.over

    def observe(self):
        png, cameras, depth = {}, {}, {}
        for source in CAMERA_NAMES:
            color = np.ascontiguousarray(np.asarray(self.obs["vision"][source]["color"])[:, :, :3]).astype(np.uint8)
            png[source] = encode_png(color)
            model = self.camera_model[source]
            if "extrinsics_world" in model:
                extrinsic = np.asarray(model["extrinsics_world"], dtype=float)
            else:
                tag = "left" if "left" in source else "right"
                extrinsic = self.ee(tag) @ np.asarray(model["ee_to_camera"], dtype=float)
            cameras[source] = {"intrinsics": np.asarray(model["intrinsics"], dtype=float), "extrinsics_world": extrinsic,
                               "size": [color.shape[1], color.shape[0]]}
        return {"png": png, "cameras": cameras, "depth": NoDepth()}

    def success_now(self):
        return None  # the policy cannot know; the official client records the outcome

    def progress_score(self, success):
        return None

    def private_note(self):
        return {}

    def final_state(self, episode):
        return {f"arm_{tag}": np.append(arm.joints(), arm.gripper()) for tag, arm in episode.arms.items()}

    def on_finish(self, episode, success):
        pass


class Model:
    """XPolicyLab policy: the official client on one side, the agent's robo interface on the other."""

    def __init__(self, cfg):
        self.cfg = cfg
        if os.path.exists(os.path.join(ROOT, "config.env")):
            for line in open(os.path.join(ROOT, "config.env")):
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, value = line.split("=", 1)
                    os.environ.setdefault(key.strip(), os.path.expandvars(value.strip().strip("'\"")))
        cfg.setdefault("robodojo_repo", os.environ.get("ROBODOJO_REPO"))
        self.run_root = cfg.get("run_root") or os.environ.get("ROBOSHELL_RUN_ROOT") or os.path.join(ROOT, "runs", "bridge", cfg["task_name"])
        os.makedirs(self.run_root, exist_ok=True)
        self.lane = str(cfg.get("lane") or os.environ.get("LANE") or "0")
        self.agent_started = 0
        self.budget = int(cfg.get("command_budget", 0))
        self.executor = BridgeExecutor(cfg)
        self.executor.build_planner()
        select_profile()
        log("profile: shared generic tools")
        self.registry = load_tools(cfg["task_name"])
        if self.registry:
            log(f"tools for {cfg['task_name']}: {sorted(self.registry)}")
        self.requests = queue.Queue()
        self.episode = None
        self.robot_tools = None
        self.episode_index = 0
        self.agent = None
        self.agent_dir = None
        self.relaunches = 0     # model-gateway refusals before the first command in this episode
        self.relaunch_at = 0    # when to start the acting agent again after such a refusal
        host = subprocess.run([os.path.join(ROOT, "agent", "api", "run_agent.sh"), "--gateway"], capture_output=True, text=True,
                              env=dict(os.environ, LANE=self.lane), check=True).stdout.strip()
        self.agent_port = int(cfg.get("agent_port") or os.environ.get("ROBO_PORT") or 28700)
        self.admin_port = int(cfg.get("admin_port") or self.agent_port + 1)
        start_http(self, host, self.agent_port, self.admin_port)
        threading.Thread(target=self.loop, daemon=True).start()
        log(f"bridge ready: agent {host}:{self.agent_port}, task {cfg['task_name']}, run root {self.run_root}")

    # ---- robo interface (worker thread) -------------------------------------
    def submit(self, kind, payload):
        from concurrent.futures import Future

        future = Future()
        self.requests.put((kind, payload, future))
        return future.result()

    def obs_cache(self):
        return self.episode.obs_cache if self.episode is not None else {}

    def tool_schema(self):
        return schema(self.registry)

    def loop(self):
        while True:
            kind, payload, future = self.requests.get()
            try:
                if kind == "result":
                    if self.episode is None:
                        future.set_result({"error": "no episode"})
                    elif not self.episode.over:
                        future.set_result({"over": False, "commands": self.episode.commands, "action_steps": self.episode.steps})
                    else:
                        future.set_result(dict(self.episode.result, over=True, directory=self.episode.directory))
                elif kind == "finish":
                    future.set_result({"error": "finish is not available in bridge mode: the official client ends the episode"})
                elif kind in ("reset", "replay", "shutdown"):
                    future.set_result({"error": f"{kind} is not available in bridge mode: the official client drives the episodes"})
                elif kind == "cmd" and payload.get("cmd") == "robot15" and self.robot_tools is not None:
                    future.set_result(self.robot_tools.call(payload))
                elif kind == "cmd" and self.robot_tools is not None and payload.get("cmd") == "robot15_status":
                    future.set_result(self.robot_tools.status())
                elif kind == "cmd" and self.robot_tools is not None and payload.get("cmd") == "robot15_turn":
                    future.set_result(self.robot_tools.end_turn())
                elif kind == "cmd" and payload.get("cmd") == "robot15_end_seen":
                    self.executor.end_seen.set()
                    future.set_result({"ok": True})
                elif kind == "cmd" and self.robot_tools is not None and payload.get("cmd") == "done":
                    with self.robot_tools.lock:
                        future.set_result(dispatch(self, lambda: self.episode, kind, payload))
                else:
                    future.set_result(dispatch(self, lambda: self.episode, kind, payload))
            except Exception as error:
                traceback.print_exc()
                future.set_result({"error": "internal server error", "exit_code": C.EXIT_EXEC_FAILED, "detail": type(error).__name__})

    # ---- XPolicyLab interface (client threads) -----------------------------
    def reset(self):
        self.end_episode("episode_over")
        self.executor.obs = None
        self.executor.over = False
        self.executor.reason = None
        self.executor.verdict = None
        self.executor.end_seen.clear()
        self.executor.awaiting = False
        self.executor.delivered = 0
        while not self.executor.pending.empty():
            self.executor.pending.get_nowait()
        self.episode = None
        log("reset")

    def update_obs(self, obs):
        self.executor.obs = obs

    def update_obs_batch(self, obs_list):
        self.update_obs(obs_list[0])

    def get_action(self):
        executor = self.executor
        if self.episode is None:
            self.start_episode()
        if self.relaunch_at and time.time() >= self.relaunch_at and not self.episode.over:
            self.relaunch_at = 0
            self.launch_agent()
        if executor.awaiting:
            executor.awaiting = False
            executor.executed.set()  # the previous chunk has been executed; the episode thread may continue
        if (not self.episode.over and self.agent is not None and self.agent.poll() is not None
                and executor.pending.empty()):
            # A stopped agent with an empty queue has no more actions.
            if self.episode.commands == 0 and self.gateway_refused() and self.relaunches < RELAUNCH_MAX:
                # The model gateway refused the first request (429, at capacity, ...). Nothing has moved yet: keep the
                # attempt for the record and start the agent again, as the evolution loop does; the client gets holds.
                self.relaunches += 1
                os.rename(self.agent_dir, f"{self.agent_dir}_gateway{self.relaunches}")
                self.agent = None
                self.relaunch_at = time.time() + RELAUNCH_WAIT_S
                log(f"model gateway refused the acting agent before its first command; relaunch {self.relaunches} in {RELAUNCH_WAIT_S:.0f} s")
            elif (self.episode.commands == 0 and self.agent_started and time.time() - self.agent_started < 120
                    and self.agent.returncode != 0):
                # The agent container never worked (no network left for Docker, image missing, credentials...). Letting
                # the run continue would record every remaining episode as a failed one. Stop the policy server instead.
                # An agent that completed its session (exit 0) without a command chose to do nothing: that is a result.
                tail = ""
                try:
                    with open(os.path.join(self.agent_dir, "..", "agent.out")) as handle:
                        tail = handle.read()[-300:].strip().replace("\n", " | ")
                except OSError:
                    pass
                log(f"FATAL: the acting agent exited within 120 s without a single command ({tail}). Aborting the evaluation: this is an infrastructure failure, not a result.")
                os._exit(70)
            else:
                log("agent exited without done")
                self.episode.finish("agent_exit")
        if self.episode.over:
            return []
        try:
            actions = executor.pending.get(timeout=ACTION_POLL_S)
        except queue.Empty:
            return [] if self.episode.over else [{"hexashell": "wait"}]
        executor.awaiting = True
        executor.delivered += len(actions)
        return actions

    def get_action_batch(self, env_idx_list=None):
        return [self.get_action()]

    def episode_over(self, verdict=None):
        """Close the Preview episode with the verdict supplied by RoboDojo."""
        if verdict is not None:
            executor = self.executor
            executor.verdict = verdict
            executor.over = True
            executor.executed.set()
            if executor.awaiting and self.agent is not None and self.agent.poll() is None:
                # Let the in-flight cell record its last tool response before stopping the agent.
                executor.end_seen.wait(timeout=5)
        self.end_episode("episode_over")
        if self.episode is not None and verdict is not None:
            self.episode.record_verdict(verdict)
        return {"ok": True}

    def on_trial_end(self, result=None):
        return self.episode_over(result)

    def start_episode(self):
        self.episode_index += 1
        name = f"{self.cfg['task_name']}_ep{self.episode_index:02d}_{time.strftime('%Y%m%dT%H%M%S')}"
        directory = os.path.join(self.run_root, name)
        os.makedirs(directory, exist_ok=True)
        info = (self.executor.obs.get("additional_info") or {}).get("hexashell") or {}
        task = self.cfg["task_name"]
        episode = Episode(self.executor, task, int(self.cfg.get("seed", 0)), info.get("layout"),
                          directory, self.budget, self.registry)
        episode.refresh_obs()
        from roboshell.server.robot15 import RobotTools

        try:
            self.robot_tools = RobotTools(episode)
        except Exception:
            episode.log_file.close()
            raise
        self.episode = episode
        first = self.robot_tools.host.snap
        episode.instruction = first.instruction
        episode.prompt = task_card.task_card(first)
        with open(os.path.join(directory, "instruction.txt"), "w", encoding="utf-8") as handle:
            handle.write(episode.instruction + "\n")
        with open(os.path.join(directory, "prompt.txt"), "w", encoding="utf-8") as handle:
            handle.write(episode.prompt)
        with open(os.path.join(directory, "context.json"), "w", encoding="utf-8") as handle:
            json.dump(task_card.info(), handle)
        self.agent_dir = os.path.join(directory, "agent")
        self.relaunches = 0
        self.relaunch_at = 0
        self.launch_agent()

    def launch_agent(self):
        episode, directory = self.episode, self.episode.directory
        for name in (f"roboshell-agent-{self.lane}", f"roboshell-egress-{self.lane}"):
            subprocess.run(["docker", "rm", "-f", name], capture_output=True)
        env = dict(os.environ, LANE=self.lane, ROBO_PORT=str(self.agent_port),
                   WALL=str(self.cfg.get("agent_wall_s", 5400)))
        command = [os.path.join(ROOT, "agent", "api", "run_agent.sh"), self.agent_dir, episode.prompt]
        with open(os.path.join(directory, "agent.out"), "a") as output:
            self.agent = subprocess.Popen(command, env=env, stdout=output, stderr=subprocess.STDOUT)
        self.agent_started = time.time()
        log(f"episode {os.path.basename(directory)}: agent started, instruction {episode.instruction!r}")

    def gateway_refused(self):
        """True if the acting agent's session ended on a model-gateway error (a fixed list of gateway error messages)."""
        try:
            with open(os.path.join(self.agent_dir, "model_events.jsonl"), errors="replace") as handle:
                for line in handle:
                    event = json.loads(line)
                    if event.get("type") == "error" and GATEWAY_ERRORS.search(event["error"]):
                        return True
        except (OSError, ValueError):
            pass
        return False

    def end_episode(self, reason):
        executor = self.executor
        executor.over = True
        executor.reason = reason
        executor.executed.set()
        if self.robot_tools is not None:
            self.robot_tools.close()
            self.robot_tools = None
        if self.episode is not None and not self.episode.over:
            self.episode.finish(reason)
        if self.agent is not None:
            if self.agent.poll() is None:
                self.agent.terminate()
                try:
                    self.agent.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    self.agent.kill()
            for name in (f"roboshell-agent-{self.lane}", f"roboshell-egress-{self.lane}"):
                subprocess.run(["docker", "rm", "-f", name], capture_output=True)
            self.agent = None
