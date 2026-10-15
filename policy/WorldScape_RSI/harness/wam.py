"""Checkpoint-backed World Action Models called as Skills by the inner Agent.

Reuse the adjacent PR #162 adapter's native explicit commit/discard interface.
No episode router or task-to-model lookup exists here.
"""
from __future__ import annotations

import importlib
import inspect
import os
from pathlib import Path

from . import contracts
from .release import ROOT, digest


def _verify_files(root, files, label):
    root = Path(root).resolve(strict=True)
    for relative, expected in files.items():
        path = (root / relative).resolve()
        if (Path(relative).is_absolute() or not path.is_relative_to(root)
                or not path.is_file() or digest(path) != expected):
            raise RuntimeError(label + " hash mismatch: " + relative)


def create_model(spec, task, deployment):
    from XPolicyLab.utils.checkpoint_resolver import resolve_checkpoint_root
    if spec.get("policy_name") != "WorldScape_Policy_2":
        raise ValueError("Only WSP World Action Models are permitted")
    cfg = dict(spec["config"])
    value = os.environ.get(spec["checkpoint_env"], "") or os.environ.get(spec.get("legacy_checkpoint_env", ""), "")
    if not value or not Path(value).is_absolute():
        raise ValueError("Set an absolute checkpoint directory in " + spec["checkpoint_env"])
    cfg["checkpoint_path"] = value
    policy_dir = ROOT.parent.parent / "WorldScape_Policy_2"
    root = resolve_checkpoint_root(cfg, policy_dir / "checkpoints", policy_dir=policy_dir)
    if root is None:
        raise FileNotFoundError("WAM checkpoint cannot be resolved")
    _verify_files(root, spec["checkpoint_files"], "WAM checkpoint")
    _verify_files(policy_dir, spec["source_files"], "Adjacent WSP adapter")
    upstream = os.environ.get(spec["worldscape_root_env"])
    if upstream:
        cfg["worldscape_root"] = str(Path(upstream).resolve(strict=True))
    device = os.environ.get(spec["device_env"], cfg.get("device", "cuda"))
    if device != "cpu" and device != "cuda" and not (device.startswith("cuda:") and device[5:].isdigit()):
        raise ValueError("Invalid WAM device binding")
    cfg.update(device=device, seed=int(deployment.get("seed", 0)), eval_batch=False)
    module = importlib.import_module("XPolicyLab.policy.WorldScape_Policy_2.model")
    if Path(inspect.getfile(module.Model)).resolve() != (policy_dir / "model.py").resolve():
        raise RuntimeError("WSP adapter imported from an unexpected checkout")
    model = module.Model(cfg)
    try:
        if model.get_runtime_info().get("requires_explicit_commit") is not True:
            raise RuntimeError("WAM Skill requires the native explicit-commit interface")
    except BaseException:
        model.close()
        raise
    return model


class WamSkills:
    def __init__(self, catalog, task, deployment, *, factory=None, dims=None):
        self.catalog = catalog
        self.task = task
        self.deployment = deployment
        self.factory = factory or create_model
        self.dims = dims
        self.models = {}
        self.latest = None
        self.continuous = None
        self.pending = None
        self.planned = self.executed = 0
        self.issued = False
        self.awaiting_observation = False

    def plan(self, skill_id):
        if skill_id not in self.catalog:
            raise ValueError("Unknown WAM Skill")
        if self.latest is None or self.pending is not None:
            raise RuntimeError("WAM requires a live observation and no unfinished prediction")
        if self.continuous != skill_id:
            self.native_selected()
            if skill_id not in self.models:
                self.models[skill_id] = self.factory(self.catalog[skill_id], self.task, self.deployment)
            self.models[skill_id].reset()
            self.models[skill_id].update_obs(self.latest)
            self.continuous = skill_id
        model = self.models[skill_id]
        # Set ownership before inference so failure cleanup cannot replay/forget it.
        self.pending = skill_id
        self.issued = False
        self.executed = 0
        actions = model.get_action()
        if not isinstance(actions, (list, tuple)) or not actions:
            raise ValueError("WAM returned an empty or malformed chunk")
        dims = self.dims or contracts.action_dimensions(self.deployment["env_cfg_type"])
        actions = [contracts.action(a, dims) for a in actions]
        self.planned = len(actions)
        return actions

    def before(self, action):
        if self.pending is not None:
            if self.awaiting_observation:
                raise RuntimeError("An issued WAM action still lacks execution evidence")
            self.issued = self.awaiting_observation = True

    def observe(self, observation, executed_action, interrupted=False):
        self.latest = observation
        if self.pending is None:
            return
        model = self.models[self.pending]
        if executed_action is None or not self.awaiting_observation:
            raise RuntimeError("WAM execution acknowledgement is out of order")
        self.awaiting_observation = False
        self.executed += 1
        if interrupted:
            model.discard_action({"execution_started": self.issued})
            self.pending = None
            self.planned = self.executed = 0
        elif self.executed == self.planned:
            # Commit before forwarding the final frame, matching the PR's lifecycle.
            model.commit_action({"executed_steps": {0: self.executed}, "finished_env_idx_list": []})
            self.pending = None
        model.update_obs(observation)

    def native_selected(self):
        if self.pending is not None:
            raise RuntimeError("Cannot change Skills during an unfinished WAM chunk")
        if self.continuous is not None:
            self.models[self.continuous].reset()
        self.continuous = None

    def close(self):
        models, self.models = self.models, {}
        pending, self.pending = self.pending, None
        errors = []
        for key, model in models.items():
            try:
                if key == pending:
                    model.discard_action({"execution_started": self.issued})
            except BaseException as error:
                errors.append(error)
            try:
                model.reset()
            except BaseException as error:
                errors.append(error)
            try:
                model.close()
            except BaseException as error:
                errors.append(error)
        self.latest = self.continuous = None
        self.awaiting_observation = self.issued = False
        self.planned = self.executed = 0
        if errors:
            raise errors[0]
