"""Standard Model API: generated Python yields one native action at a time."""

import copy
import multiprocessing
from pathlib import Path
import time

import numpy as np

from XPolicyLab.model_template import ModelTemplate
from orchestrator.files import append_jsonl, write_json
from integrations.robodojo_policy.worker import run_worker


class Model(ModelTemplate):
    def __init__(self, model_cfg):
        self.cfg = model_cfg
        self.run = Path(model_cfg["run_dir"])
        self.episode = -1
        self.worker = self.connection = None
        self.obs = None
        self.pending = self.finished = False
        self.step_limit = None
        self.reset_requested = False

    def reset(self):
        self.close()
        self.obs = None
        self.pending = self.finished = False
        self.step_limit = None
        self.reset_requested = False

    def prepare_case(self, case_meta=None):
        self.step_limit = int(case_meta["native_step_limit"])

    def update_obs(self, obs):
        # Images have already been decoded by the official policy server.
        self.obs = copy.deepcopy(obs)

    def _transition(self, *, success=False, terminated=False, truncated=False):
        self.connection.send({"kind": "transition", "observation": self.obs,
                              "success": success, "terminated": terminated, "truncated": truncated})
        self.pending = False

    def get_action(self):
        if self.obs is None or self.step_limit is None:
            raise RuntimeError("prepare_case and update_obs must precede get_action")
        if self.worker is None:
            self.episode += 1
            context = multiprocessing.get_context("spawn")
            self.connection, child = context.Pipe()
            target = run_worker
            if self.cfg.get("diagnostic"):
                from integrations.robodojo_policy.probe import run_probe
                target = run_probe
            self.worker = context.Process(target=target,
                args=(child, self.cfg, self.obs, self.step_limit, self.episode))
            self.worker.start()
            child.close()
        elif self.pending:
            self._transition()
        if self.finished:
            if self.cfg.get("phase") == "playground":
                return []
            # Voluntary early stop does not set evaluator success/end flags.
            # Hold measured poses until the official success check or timeout.
            state = self.obs["state"]
            action = {k: copy.deepcopy(v) for k, v in state.items()
                      if k.endswith("ee_pose") or k.endswith("ee_joint_state")}
            self._record(action, source="stopped_policy_hold")
            return [action]
        if self.reset_requested:
            return []
        # Empty chunk means still computing. The deploy loop polls without
        # advancing physics, so long coding turns do not hit RPC timeouts.
        if not self.connection.poll(0.5):
            if not self.worker.is_alive():
                raise RuntimeError("Agent worker died without a completion message")
            return []
        message = self.connection.recv()
        if message["kind"] == "error":
            raise RuntimeError(message["error"])
        if message["kind"] == "finished":
            self.finished = True
            return []
        if message["kind"] == "reset":
            if self.cfg.get("phase") != "playground":
                raise PermissionError("Native reset is forbidden in Test")
            self.reset_requested = True
            return []
        if message["kind"] != "action":
            raise RuntimeError("Unexpected worker message")
        self.pending = True
        self._record(message["action"], source="agent_code")
        return [message["action"]]

    def _record(self, action, *, source):
        append_jsonl(self.run / "actions.jsonl", {
            "episode": self.episode, "source": source, "timestamp": time.time(),
            "action": {key: np.asarray(value).tolist() for key, value in action.items()},
        })

    def get_control(self):
        return {"reset": self.reset_requested, "finished": self.finished}

    def complete_reset(self, observation):
        if self.cfg.get("phase") != "playground" or not self.reset_requested:
            raise PermissionError("No Playground reset is pending")
        self.update_obs(observation)
        self.connection.send({"kind": "reset", "observation": self.obs})
        self.reset_requested = False

    def on_trial_end(self, result=None):
        result = result or {}
        if self.pending:
            self._transition(success=bool(result.get("success")),
                             terminated=bool(result.get("terminated")),
                             truncated=bool(result.get("truncated")))
        if self.cfg.get("phase") == "playground":
            return
        if self.worker is not None:
            self.worker.join(timeout=60)
            if self.worker.is_alive():
                raise RuntimeError("Worker did not close after official termination")
            if not self.finished and self.connection.poll():
                message = self.connection.recv()
                if message.get("kind") == "error":
                    raise RuntimeError(message["error"])
        write_json(self.run / f"trial_{self.episode:04d}.json", result)

    def update_obs_batch(self, obs_list):
        if len(obs_list) != 1:
            raise ValueError("This adapter uses one independent policy server per environment")
        self.update_obs(obs_list[0])

    def get_action_batch(self, env_idx_list=None):
        if env_idx_list not in (None, [0]):
            raise ValueError("Only a single environment is supported")
        return [self.get_action()]

    def close(self):
        if self.worker is not None:
            if self.worker.is_alive():
                self.worker.terminate()
            self.worker.join(timeout=10)
            self.worker = None
        if self.connection is not None:
            self.connection.close()
            self.connection = None
