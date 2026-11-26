"""Bind ReStart's native noise adapter to the existing UniWAM serving interface."""

from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import sys

import numpy as np
import yaml

from XPolicyLab.policy.UniWAM import model as uniwam_xpl


METHOD_KEYS = frozenset({
    "embodpfn_expert_bundle",
    "embodpfn_fp_per_step",
    "embodpfn_head_feature_manifest",
    "embodpfn_head_parent_receipt",
    "embodpfn_head_query_trace",
    "embodpfn_inverse_reference",
    "embodpfn_pfn_checkpoint",
    "embodpfn_pfn_device",
    "embodpfn_pfn_devices",
    "embodpfn_pfn_python",
    "embodpfn_policy_receipt",
    "embodpfn_support",
})

ASSET_PATH_KEYS = METHOD_KEYS - {
    "embodpfn_fp_per_step", "embodpfn_pfn_device", "embodpfn_pfn_devices",
}


def _method_manifest(task: str) -> Path:
    explicit = os.environ.get("RESTART_METHOD_ARGS")
    if explicit:
        return Path(explicit).resolve(strict=True)
    directory = os.environ.get("RESTART_CONTEXT_DIR")
    if not directory:
        raise ValueError("Set RESTART_CONTEXT_DIR to the prepared context directory.")
    if not task or Path(task).name != task or task in {".", ".."}:
        raise ValueError("Expected a RoboTwin task name.")
    return (Path(directory) / f"{task}.json").resolve(strict=True)


def _read_method_arguments(path: Path, task: str) -> dict:
    with path.open() as stream:
        arguments = json.load(stream)
    if not isinstance(arguments, dict) or arguments.keys() - METHOD_KEYS:
        raise ValueError("The ReStart manifest accepts only the declared method asset paths and PFN devices.")
    for name in (
        "embodpfn_inverse_reference", "embodpfn_pfn_checkpoint", "embodpfn_pfn_device",
        "embodpfn_pfn_python", "embodpfn_policy_receipt", "embodpfn_support",
    ):
        if not arguments.get(name):
            raise ValueError(f"Declare {name} in the ReStart manifest.")

    for name in ASSET_PATH_KEYS & arguments.keys():
        arguments[name] = str((path.parent / Path(arguments[name]).expanduser()).resolve())
    support = Path(arguments["embodpfn_support"])
    if support.is_file():
        with np.load(support, allow_pickle=False) as archive:
            metadata = json.loads(archive["manifest"].item())
        origin = metadata["context_coverage"]
        coverage = origin["coverage"]
        context_task, rows = origin["task"], metadata["support_rows"]
    else:
        if not arguments.get("embodpfn_expert_bundle"):
            raise ValueError(f"Support bank not found: {support}. Supply the complete bank or an expert bundle.")
        with (Path(arguments["embodpfn_expert_bundle"]) / "MANIFEST.json").open() as stream:
            expert = json.load(stream)
        if expert["split"] != "clean50":
            raise ValueError("ReStart needs the official clean50 expert demonstrations.")
        context_task, rows, coverage = expert["task"], len(expert["rows"]), expert["context_coverage"]

    episodes = coverage["episodes"]
    if (context_task != task
            or coverage["version"] != "all-native-robotwin-replanning-windows.v1"
            or coverage.get("native_tail_contexts") is not True
            or len(episodes) != 50 or len({episode["episode"] for episode in episodes}) != 50
            or sum(len(episode["decision_frames"]) for episode in episodes) != rows
            or coverage["expected_rows"] != rows):
        raise ValueError("Use every original replan, including tail windows, from the task's 50 clean demonstrations.")
    return arguments


def _native_policy_directory() -> Path:
    import embodpfn

    directory = Path(embodpfn.__file__).resolve().parents[1] / ".deps/upstream/UniWAM/inference/robotwin"
    directory = directory.resolve(strict=True)
    path = str(directory)
    if path not in sys.path:
        sys.path.insert(0, path)
    spec = importlib.util.find_spec("uniwam")
    if spec is None or Path(spec.origin).resolve() != directory / "uniwam/__init__.py":
        raise ImportError("Install the pinned UniWAM source checkout declared by ReStart.")
    return directory / "uniwam"


class Model(uniwam_xpl.Model):
    def __init__(self, model_cfg: dict):
        with (uniwam_xpl.POLICY_DIR / "deploy.yml").open() as stream:
            upstream = yaml.safe_load(stream)
        super().__init__({**upstream, **model_cfg})

    def _load_policy(self):
        from embodpfn.official_recipe import robotwin_recipe
        from embodpfn.robotwin_policy import get_model

        native_recipe = robotwin_recipe(_native_policy_directory())["arguments"]
        for key in ("seed", "inference_mode", "num_inference_timesteps",
                    "history_action_noise_std", "future_video_denoise_fraction"):
            if self.model_cfg[key] != native_recipe[key]:
                raise ValueError(f"Use the original UniWAM {key}={native_recipe[key]}.")
        if self.model_cfg["bench_name"] != "RoboTwin":
            raise ValueError("This ReStart adapter supports RoboTwin.")
        task = self.model_cfg["task_name"]
        method = _read_method_arguments(_method_manifest(task), task)
        policy = get_model({
            **native_recipe, **method, "task_name": task,
            "ckpt_setting": str(self.checkpoint_model_dir),
            "wan_path": str(self.wan_path), "vlm_path": str(self.vlm_path),
            "embodpfn_noise_target": "action",
            "embodpfn_pfn_cache": "repr",
            "embodpfn_pfn_execution": "shared-batched",
        })
        # Make readiness mean that the complete context is already fitted.
        # Use the native plugin's own context preparation and attachment.
        policy._embodpfn_prepare_context()
        return policy
