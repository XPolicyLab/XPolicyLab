"""Small policy boundary: frozen upstream agent, or explicitly labelled CPU IO probe."""

import copy
import json
import os
from pathlib import Path
import re
from types import SimpleNamespace

import numpy as np

from XPolicyLab.model_template import ModelTemplate
from XPolicyLab.utils.checkpoint_resolver import resolve_checkpoint_root
from XPolicyLab.utils.process_data import get_robot_action_dim_info
from .configuration import UPSTREAM_SHA, interface_probe_enabled, validate_scope, verified_upstream
from .frozen_inputs import validate_source_inputs, verify_pinned_inputs


class Model(ModelTemplate):
    def __init__(self, model_cfg):
        self.cfg = dict(model_cfg)
        validate_scope(self.cfg)
        self.probe = interface_probe_enabled()
        self.delegate = None
        self.obs = None
        self.run = None
        self.native_steps = None
        self.completed_trials = 0
        dimensions = get_robot_action_dim_info(self.cfg["env_cfg_type"])
        arm_count = len(dimensions["arm_dim"])
        if arm_count != len(dimensions["ee_dim"]) or arm_count not in (1, 2):
            raise ValueError("Unsupported robot action metadata")
        prefixes = [""] if arm_count == 1 else ["left_", "right_"]
        self.action_shapes = {}
        for index, prefix in enumerate(prefixes):
            self.action_shapes[prefix + "ee_pose"] = (7,)  # XYZ + scalar-first quaternion contract.
            self.action_shapes[prefix + "ee_joint_state"] = (dimensions["ee_dim"][index],)
        if self.probe:
            print("INTERFACE PROBE ONLY: no EmbodiedEvo model, physics, task score or leaderboard result", flush=True)
            return
        self._initialize_agent()

    def _initialize_agent(self):
        root = verified_upstream()
        policy_dir = Path(__file__).resolve().parent
        source_cfg = dict(self.cfg)
        if os.environ.get("EMBODIEDEVO_SOURCE_RUN"):
            source_cfg["checkpoint_path"] = os.environ["EMBODIEDEVO_SOURCE_RUN"]
        source = resolve_checkpoint_root(source_cfg, policy_dir / "checkpoints", policy_dir=policy_dir)
        run_text = os.environ.get("EMBODIEDEVO_RUN_DIR") or self.cfg.get("run_dir")
        if not run_text:
            raise ValueError("Set EMBODIEDEVO_RUN_DIR to a new timestamp-prefixed output directory")
        run = Path(run_text).expanduser().resolve()
        if not re.match(r"^\d{12}_", run.name):
            raise ValueError("Run directory basename must start YYYYMMDDHHMM_")
        if run == source or source in run.parents:
            raise ValueError("Run output must be outside the frozen source tree")
        if run.exists():
            raise FileExistsError("Run directory already exists; attempts must never be overwritten")
        required = [source / "protocol.json", source / "inputs/test.md",
                    source / "inputs/primitives/primitives.md",
                    root / "vendor/robodojo/.embodiedevo-provisioned.json"]
        for path in required:
            if not path.is_file():
                raise FileNotFoundError(f"Required frozen source/provision asset is missing: {path}")
        if not (source / "frozen_harness").is_dir():
            raise FileNotFoundError("The source has no frozen_harness directory")
        if os.environ.get("EMBODIEDEVO_AGENT_RECOVERY"):
            raise ValueError("Automatic agent retry must be disabled for this candidate evaluation")
        episodes = int(os.environ.get("EMBODIEDEVO_EPISODES", self.cfg.get("episodes") or 0))
        if episodes < 1:
            raise ValueError("Set EMBODIEDEVO_EPISODES to the exact requested native evaluator episode count")
        # The upstream helper verifies protocol/task/harness hashes, creates a new
        # run, preserves H, and records its native-action-limit wording changes.
        from integrations.robodojo_policy.run import pin
        from integrations.robodojo_policy.policy.EmbodiedEvo.model import Model as UpstreamModel
        from orchestrator.files import hashes
        state = json.loads((source / "protocol.json").read_text())
        validate_source_inputs(source, state, hashes)
        if state.get("definition", {}).get("config", {}).get("agent", {}).get("isolation") != "docker":
            raise ValueError("This pinned public worker requires Docker; native-runtime patches are not bundled")
        _, run, spec = pin(SimpleNamespace(
            source_run=source, run_dir=run, env_task=self.cfg["task_name"],
            seed=int(self.cfg["seed"]), episodes=episodes, gpu=str(self.cfg.get("gpu_id", "0"))
        ))
        pinned_input_hashes = verify_pinned_inputs(source, run, state, hashes)
        spec.update(verified_input_hashes=pinned_input_hashes, phase="test", diagnostic=False, policy_name="EmbodiedEvoRepro",
                    upstream_commit=UPSTREAM_SHA, scope="candidate native evaluation; not organizer verified")
        self.run, self.expected_harness = run, hashes(run / "frozen_harness")
        self.spec = spec
        (run / "candidate-adapter.json").write_text(json.dumps(spec, indent=2) + "\n")
        self.delegate = UpstreamModel(spec)

    def update_obs(self, obs):
        for camera in obs.get("vision", {}).values():
            color = camera.get("color")
            if not isinstance(color, np.ndarray) or color.ndim != 3 or color.shape[-1] != 3:
                raise TypeError("Expected an already-decoded HWC RGB array from the policy server")
        if self.delegate:
            self.delegate.update_obs(obs)
        else:
            self.obs = copy.deepcopy(obs)

    def update_obs_batch(self, obs_list):
        if len(obs_list) != 1:
            raise ValueError("Exactly one observation is supported")
        self.update_obs(obs_list[0])

    def get_action(self):
        if self.delegate:
            actions = self.delegate.get_action()
        else:
            if self.obs is None:
                raise RuntimeError("update_obs must precede get_action")
            actions = [{key: np.asarray(self.obs["state"][key], dtype=np.float32).copy()
                        for key in self.action_shapes}]
        if len(actions) > 1:
            raise ValueError("The upstream coding policy must yield at most one native action")
        for action in actions:
            if set(action) != set(self.action_shapes):
                raise ValueError("Action keys differ from the configured EE contract")
            for key, shape in self.action_shapes.items():
                value = np.asarray(action[key])
                if value.shape != shape or not np.isfinite(value).all():
                    raise ValueError(f"Invalid action shape or finite values for {key}")
        return actions

    def get_action_batch(self, env_idx_list=None):
        if env_idx_list is not None and list(env_idx_list) != [0]:
            raise ValueError("Only environment index 0 is supported")
        return [self.get_action()]

    def reset(self):
        # Evaluator resets policy state BETWEEN trials. There is no RPC that
        # resets the simulator from an agent's Test session.
        self.obs, self.native_steps = None, None
        if self.delegate:
            self._verify_harness()
            self.delegate.reset()

    def prepare_case(self, case_meta=None):
        if self.delegate:
            if self.completed_trials >= self.spec["episodes"]:
                raise RuntimeError("Native evaluator exceeded the declared episode count")
            self.delegate.prepare_case(case_meta)
        else:
            self.native_steps = (case_meta or {}).get("native_step_limit")

    def on_trial_end(self, result=None):
        if self.delegate:
            self.delegate.on_trial_end(result)
            self._verify_harness()
            self.completed_trials += 1

    def complete_reset(self, observation=None):
        raise PermissionError("Simulator reset is forbidden in Test")

    def _verify_harness(self):
        from orchestrator.files import hashes
        if hashes(self.run / "frozen_harness") != self.expected_harness:
            raise RuntimeError("Frozen Skills/Lessons changed during Test")

    def close(self):
        if self.delegate:
            self.delegate.close()
            self._verify_harness()
