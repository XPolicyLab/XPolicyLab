"""EmbodiedRSI L2 Python policy, driven by the official XPolicyLab evaluator."""

import atexit
import copy
import multiprocessing
import signal
import time
from pathlib import Path

import numpy as np

from XPolicyLab.model_template import ModelTemplate
from XPolicyLab.utils.process_data import get_robot_action_dim_info

from XPolicyLab.policy.EmbodiedRSI.runtime.workspace import prepare_run, verify_tree
from XPolicyLab.policy.EmbodiedRSI.runtime.xpolicylab.bridge import action_keys
from XPolicyLab.policy.EmbodiedRSI.runtime.files import append_jsonl, write_json
from XPolicyLab.policy.EmbodiedRSI.runtime.worker import run_worker


class _EnvironmentPolicy:
    """One independent coding worker; no mutable episode state is shared."""
    def __init__(self, cfg, robot_dims, frozen_files, episode):
        self.cfg, self.robot_dims, self.frozen_files = cfg, robot_dims, frozen_files
        self.run = Path(cfg["run_dir"])
        self.episode = episode
        self.worker = self.connection = None
        self.obs = None
        self.pending = self.finished = False
        self.step_limit = None

    def reset(self):
        """Evaluator episode boundary; never exposed as an agent Test reset."""
        self.close()
        self.obs = None
        self.pending = self.finished = False
        self.step_limit = None

    def prepare_case(self, case_meta=None):
        meta = case_meta or {}
        if self.cfg["diagnostic"] != bool(meta.get("diagnostic_environment")):
            raise ValueError("Diagnostic policy is restricted to the official debug environment")
        self.step_limit = int(meta["native_step_limit"])
        if self.step_limit <= 0:
            raise ValueError("Evaluator must supply a positive action limit")

    def update_obs(self, obs):
        # The official WS server has already decoded every camera to RGB.
        self.obs = copy.deepcopy(obs)
        if "instruction" not in self.obs:
            self.obs["instruction"] = self.obs["instructions"]

    def _transition(self, *, success=False, terminated=False, truncated=False):
        self.connection.send({"kind": "transition", "observation": self.obs,
                              "success": success, "terminated": terminated, "truncated": truncated})
        self.pending = False

    def _action(self, action, source):
        action = {k: np.asarray(v, dtype=np.float64) for k, v in action.items()}
        ee = any(k.endswith("ee_pose") for k in action)
        arms, grippers = self.robot_dims["arm_dim"], self.robot_dims["ee_dim"]
        prefixes = [""] if len(arms) == 1 else ["left_", "right_"]
        if len(arms) not in (1, 2) or len(grippers) != len(arms):
            raise ValueError("Unsupported robot dimensions")
        expected = {}
        for prefix, arm, gripper in zip(prefixes, arms, grippers, strict=True):
            expected[prefix + ("ee_pose" if ee else "arm_joint_state")] = 7 if ee else arm
            expected[prefix + "ee_joint_state"] = gripper
        if action.keys() != expected.keys() or any(
            action[k].shape != (size,) or not np.isfinite(action[k]).all()
            for k, size in expected.items()
        ):
            raise ValueError("Action does not match the official robot dimensions")
        append_jsonl(self.run / "actions.jsonl", {
            "episode": self.episode, "source": source, "timestamp": time.time(),
            "action": {k: v.tolist() for k, v in action.items()},
        })
        return [action]

    def get_action(self):
        if self.obs is None or self.step_limit is None:
            raise RuntimeError("prepare_case and update_obs must precede get_action")
        if self.worker is None:
            verify_tree(self.run / "frozen_harness", self.frozen_files)
            context = multiprocessing.get_context("spawn")
            self.connection, child = context.Pipe()
            target = run_worker
            if self.cfg["diagnostic"]:
                from XPolicyLab.policy.EmbodiedRSI.tests.probe import run_probe
                target = run_probe
            self.worker = context.Process(target=target,
                args=(child, self.cfg, self.obs, self.step_limit, self.episode))
            self.worker.start()
            child.close()
        elif self.pending:
            self._transition()
        if self.finished:
            # Preserve the original voluntary-stop behavior. Success and the
            # action horizon are decided solely by the official environment.
            action = {k: copy.deepcopy(self.obs["state"][k])
                      for k in action_keys(self.obs["state"], ee=True)}
            return self._action(action, "stopped_policy_hold")
        if not self.connection.poll(0.5):
            if not self.worker.is_alive():
                raise RuntimeError("Agent worker died without completion")
            return []  # Still programming: do not advance simulator physics.
        message = self.connection.recv()
        if message["kind"] == "error":
            raise RuntimeError(message["error"])
        if message["kind"] == "finished":
            self.finished = True
            return []
        if message["kind"] != "action":
            raise RuntimeError("Unexpected worker message; Test never permits reset")
        self.pending = True
        return self._action(message["action"], "agent_code")

    def on_trial_end(self, result=None):
        result = result or {}
        if self.pending:
            self._transition(success=bool(result.get("success")),
                             terminated=bool(result.get("terminated")),
                             truncated=bool(result.get("truncated")))
        if self.worker is not None:
            self.worker.join(timeout=60)
            if self.worker.is_alive():
                raise RuntimeError("Worker did not close after official termination")
            if not self.finished and self.connection.poll():
                message = self.connection.recv()
                if message.get("kind") == "error":
                    raise RuntimeError(message["error"])
            episode = self.run / "episodes" / f"episode_{self.episode:04d}"
            verify_tree(episode / "workspace", self.frozen_files, components=("skills", "lessons"))
        write_json(self.run / f"trial_{self.episode:04d}.json", result)

    def close(self, *, terminate=True):
        if self.worker is not None:
            if terminate and self.worker.is_alive():
                self.worker.terminate()
            self.worker.join(timeout=45)
            if self.worker.is_alive():
                raise RuntimeError("Worker cleanup did not finish; inspect episode runtime/container record")
            self.worker = None
        if self.connection is not None:
            self.connection.close()
            self.connection = None


class Model(ModelTemplate):
    """Official batch interface with a separate agent and workspace per env_idx."""
    def __init__(self, model_cfg):
        super().__init__()
        self.contexts = {}
        self.ready = {}
        self.next_episode = 0
        self.robot_dims = get_robot_action_dim_info(model_cfg["env_cfg_type"])
        self.cfg, self.frozen_files = prepare_run(model_cfg)
        self.run = Path(self.cfg["run_dir"])
        atexit.register(self.close)
        if multiprocessing.current_process().name == "MainProcess":
            signal.signal(signal.SIGTERM, self._shutdown)

    @staticmethod
    def _shutdown(*_):
        raise KeyboardInterrupt

    def reset(self):
        self.close()
        self.contexts.clear()
        self.ready.clear()

    def prepare_case(self, case_meta=None):
        meta = case_meta or {}
        if self.contexts:
            raise RuntimeError("Reset the policy before preparing a new native batch")
        environments = meta.get("environments", [{"env_idx": 0}])
        indices = [entry["env_idx"] for entry in environments]
        if len(indices) != len(set(indices)) or any(type(i) is not int or i < 0 for i in indices):
            raise ValueError("Expected distinct nonnegative native environment indices")
        if len(indices) > 16:
            raise ValueError("This deployment permits at most 16 concurrent environments per GPU")
        if not self.cfg["eval_batch"] and indices != [0]:
            raise ValueError("Enable eval_batch for multiple environments")
        for entry in environments:
            index = entry["env_idx"]
            context = _EnvironmentPolicy(self.cfg, self.robot_dims, self.frozen_files, self.next_episode)
            context.prepare_case(meta)
            self.contexts[index] = context
            self.next_episode += 1
            append_jsonl(self.run / "episode-map.jsonl", {
                "episode": context.episode, "env_idx": index,
                "layout_id": entry.get("layout_id"), "task": self.cfg["task_name"],
                "seed": self.cfg["seed"],
            })  # Host-only identity; BridgeTask strips env_idx from agent observations.

    def update_obs(self, obs):
        self.contexts[0].update_obs(obs)

    def get_action(self):
        return self.contexts[0].get_action()

    def update_obs_batch(self, obs_list):
        indices = [obs["env_idx"] for obs in obs_list]
        if len(indices) != len(set(indices)):
            raise ValueError("Duplicate environment observations")
        for index, obs in zip(indices, obs_list, strict=True):
            self.contexts[index].update_obs(obs)

    def get_action_batch(self, env_idx_list=None):
        indices = list(self.contexts) if env_idx_list is None else list(env_idx_list)
        if not indices or len(indices) != len(set(indices)) or not set(indices) <= self.contexts.keys():
            raise ValueError("Expected distinct, prepared environment indices")
        if not self.ready.keys() <= set(indices):
            raise RuntimeError("Cannot remove an environment while its action is waiting")
        for index in indices:
            if index not in self.ready:
                action = self.contexts[index].get_action()
                if action:
                    self.ready[index] = action
        if len(self.ready) != len(indices):
            return [[] for _ in indices]
        # Barrier: never advance global simulator physics while another live
        # agent is still programming; cached actions must not be requested twice.
        result = [self.ready.pop(index) for index in indices]
        return result

    def on_trial_end(self, result=None):
        result = result or {}
        results = result.get("results", [{"env_idx": 0, **result}])
        for item in results:
            index = item["env_idx"]
            self.contexts[index].on_trial_end(item)
            self.ready.pop(index, None)

    def close(self):
        # Signal all workers before joining them. This also works during atexit,
        # when Python has already shut down thread-pool scheduling.
        for context in self.contexts.values():
            if context.worker is not None and context.worker.is_alive():
                context.worker.terminate()
        for context in self.contexts.values():
            context.close(terminate=False)
        self.contexts.clear()
        self.ready.clear()
