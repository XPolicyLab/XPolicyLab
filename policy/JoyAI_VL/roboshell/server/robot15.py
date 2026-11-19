"""Connect 1.5 robot tools to Preview's episode and action transport."""

import base64
import itertools
import json
import os
import threading
import time

import cv2

from vendor.robot15.hexashell.errors import CallFailed, Stop
from vendor.robot15.hexashell.motion.driver import EpisodeOver
from vendor.robot15.hexashell.motion.ik import speed_named
from vendor.robot15.hexashell.robot.host import MOTIONS, RobotHost, describe_verdict
from vendor.robot15.hexashell.robot.results import decode, encode
from vendor.robot15.hexashell.state import parse_obs
from vendor.robot15.hexashell.vision.scene import Scene
from vendor.robot15.hexashell.vision.sam3_client import DEFAULT_URL, Sam3Client, Sam3Error


class RobotTools:
    def __init__(self, episode):
        self.episode = episode
        self.executor = episode.executor
        self.observation = self.executor.obs
        first = parse_obs(self.observation, 0)
        self.files = {}
        self.captions = {}
        self.picture_index = itertools.count()
        self.lock = threading.Lock()
        url = os.environ.get("HEXASHELL_SAM3_URL") or DEFAULT_URL
        sam = Sam3Client(url)
        try:
            if not sam.health().get("ok"):
                raise Sam3Error("the service is not ready")
            self.scene = Scene(sam, log=self.log)
        except Sam3Error as exc:
            sam.close()
            raise RuntimeError(f"SAM3 at {url}: {exc}") from exc
        except Exception:
            sam.close()
            raise
        try:
            self.host = RobotHost(self, first, scene=self.scene, attach=self.attach, log=self.log,
                                  speed=speed_named(os.environ.get("HEXASHELL_MOTION", "v2")))
        except Exception:
            self.scene.close()
            raise

    @property
    def over(self):
        return self.episode.over or self.executor.over

    @property
    def verdict(self):
        return self.executor.verdict or self.episode.result

    def close(self):
        with self.lock:
            self.scene.close()

    def act(self, action):
        episode = self.episode
        if self.over:
            raise EpisodeOver(self.verdict)
        joints = {}
        for tag, arm in episode.arms.items():
            arm.gripper_target = float(action[f"{tag}_ee_joint_state"][0])
            joints[tag] = [action[f"{tag}_arm_joint_state"]]
        if not episode.run(joints) or self.over:
            raise EpisodeOver(self.verdict)
        self.observation = self.executor.obs
        return self.host.snap.seq + 1, self.observation

    def attach(self, kind, image, caption):
        ok, png = cv2.imencode(".png", cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        if not ok:
            raise ValueError("could not encode the robot picture")
        name = f"robot_{next(self.picture_index):06d}.png"
        self.files[name] = base64.b64encode(png).decode("ascii")
        self.captions[name] = caption

    def log(self, event):
        self.episode.log_file.write(json.dumps({"robot15": event}, default=str) + "\n")

    def status(self):
        verdict = self.executor.verdict or {}
        return {"over": self.over, "step": verdict.get("step", self.host.snap.step),
                "step_limit": verdict.get("step_limit", self.host.snap.step_limit),
                "end": describe_verdict(verdict) if self.over else None}

    def end_turn(self):
        with self.lock:
            self.host.pictures_shown()
            return self.status()

    def call(self, payload):
        with self.lock:
            return self._call(payload)

    def _call(self, payload):
        if self.over:
            return {"error": "the episode is over", "kind": "EpisodeOver"}
        self.files = {}
        self.captions = {}
        name = payload.get("name")
        episode = self.episode
        actions = self.host.driver.actions
        first_step, started = episode.steps, time.time()
        error = None
        try:
            args = decode(payload["args"])
            kwargs = decode(payload["kwargs"])
            if not isinstance(name, str) or not isinstance(args, tuple) or not isinstance(kwargs, dict):
                raise ValueError("invalid robot call")
            if self.observation is not self.executor.obs:
                previous = self.host.snap
                self.observation = self.executor.obs
                self.host.snap = parse_obs(self.observation, previous.seq + 1, previous)
                self.host.driver.now = self.host.snap
                for tag, arm in episode.arms.items():
                    self.host.driver.command[tag] = arm.joint_target.copy()
                    self.host.driver.grip_goal[tag] = arm.gripper_target
            if name in MOTIONS and not (name == "goto" and kwargs.get("dry_run")):
                episode.commands += 1
                if episode.budget_left is not None:
                    episode.budget_left -= 1
            result = self.host.dispatch(name, args, kwargs)
            if not payload.get("structured"):
                self.host.pictures_shown()
            reply = {"result": encode(result)}
            if self.files:
                reply["files"] = self.files
                reply["captions"] = self.captions
            return reply
        except CallFailed as exc:
            error = exc.text
            return {"error": exc.text, "kind": exc.kind}
        except Stop as exc:
            error = str(exc)
            return {"error": str(exc), "kind": "EpisodeOver"}
        except (KeyError, TypeError, ValueError) as exc:
            error = str(exc)
            return {"error": str(exc), "kind": "ValueError"}
        finally:
            if not episode.over and episode.budget_left is not None and episode.budget_left <= 0:
                episode.finish("budget", success=self.executor.success_now())
            if self.host.driver.actions != actions:
                episode.last_cmd = f"robot.{name}"
                episode.refresh_obs()
            episode.log_file.write(json.dumps({
                "t": round(started, 3), "wall_s": round(time.time() - started, 3),
                "request": {"cmd": f"robot.{name}"}, "feedback": {"error": error} if error else {},
                "exit_code": 2 if error else 0, "steps": [first_step, episode.steps],
            }) + "\n")
            episode.log_file.flush()
