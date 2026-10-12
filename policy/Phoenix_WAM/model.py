"""Training-faithful WAM bridge for the Phoenix_WAM RoboDojo evaluation."""

from __future__ import annotations

import sys
from functools import partial
from pathlib import Path
from typing import Any

import torch

from XPolicyLab.model_template import ModelTemplate

from ._adapter_base import StatefulWAMAdapter, as_bool, optional_float, optional_int
from .checkpoint_paths import resolve_policy_checkpoint
from .robot_profiles import SUPPORTED_ROBOTS

POLICY_DIR = Path(__file__).resolve().parent


def _local_path(value: Any) -> Path:
    path = Path(str(value)).expanduser()
    return path if path.is_absolute() else POLICY_DIR / path


def _activate_runtime(runtime_root: Path) -> None:
    source = str(runtime_root / "src")
    sys.path[:] = [source] + [entry for entry in sys.path if entry != source]
    for module_name in list(sys.modules):
        if module_name == "wam" or module_name.startswith("wam."):
            del sys.modules[module_name]


class Model(ModelTemplate):
    """XPolicyLab model adapter with one stateful WAM session per RoboDojo environment."""

    __init__ = StatefulWAMAdapter.__init__
    _adapt_obs = StatefulWAMAdapter._adapt_obs
    _ingest = StatefulWAMAdapter._ingest
    update_obs = StatefulWAMAdapter.update_obs
    _session_for = StatefulWAMAdapter._session_for
    update_obs_batch = StatefulWAMAdapter.update_obs_batch
    _dummy_actions = StatefulWAMAdapter._dummy_actions
    _predict = StatefulWAMAdapter._predict
    get_action = StatefulWAMAdapter.get_action
    get_action_batch = StatefulWAMAdapter.get_action_batch
    get_timing_rollout = StatefulWAMAdapter.get_timing_rollout
    reset = StatefulWAMAdapter.reset

    def _load_real_policy(self) -> None:
        mode = str(self.model_cfg.get("runtime_mode") or "joint96_real")
        if mode != "joint96_real":
            raise ValueError(f"Phoenix_WAM requires runtime_mode=joint96_real, got {mode}")
        self._load_joint96_policy()

    def _load_joint96_policy(self) -> None:
        from .joint96_assets import load_assets

        if self.env_cfg_type not in SUPPORTED_ROBOTS or self.action_type != "joint":
            raise ValueError("Phoenix_WAM supports arx_x5, piper, piper_x / joint only")
        if self.action_horizon != 32:
            raise ValueError("The real Joint96 checkpoint requires action_horizon=32")
        if str(self.model_cfg.get("mixed_precision") or "bf16") != "bf16":
            raise ValueError("Joint96 real evaluation requires mixed_precision=bf16")
        if str(self.model_cfg.get("rand_device") or "cpu") != "cpu":
            raise ValueError("Joint96 real evaluation requires rand_device=cpu")
        if float(self.model_cfg.get("action_hz") or 25.0) != 25.0:
            raise ValueError("The Joint96 evaluation recipe uses action_hz=25")
        if float(self.model_cfg.get("text_cfg_scale") or 1.0) != 1.0:
            raise ValueError("Joint96 real evaluation requires text_cfg_scale=1")
        if str(self.model_cfg.get("negative_prompt") or "") or as_bool(self.model_cfg.get("tiled")):
            raise ValueError("Joint96 real evaluation uses no negative prompt or tiling")
        if optional_float(self.model_cfg.get("sigma_shift")) is not None:
            raise ValueError("Joint96 uses the saved training noise schedules")
        if self.model_cfg.get("current_visual_mode", "repeat9_full_stvae") != "repeat9_full_stvae":
            raise ValueError("Joint96 real requires repeat9_full_stvae")
        print("[Joint96] Verifying source, metadata and policy checksum...", flush=True)
        manifest, assets = load_assets(resolve_policy_checkpoint(self.model_cfg), robot_type=self.env_cfg_type)
        requested_step = optional_int(self.model_cfg.get("expected_step"))
        if requested_step is not None and requested_step != manifest["step"]:
            raise ValueError("expected_step differs from pinned checkpoint")
        _activate_runtime(assets["runtime_root"])
        from .joint96_backend import RoboDojoRealJoint96Session, build_runtime

        device = str(self.model_cfg.get("device") or "cuda")
        device = "cuda:0" if device == "cuda" else device
        if not device.startswith("cuda") or not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for real Joint96 evaluation; use CPU preflight for metadata")
        self.runtime, session_cfg = build_runtime(
            assets["checkpoint"], assets["inference_config"], assets["dataset_stats"],
            expected_step=manifest["step"], assets=assets,
            output_dir=_local_path(self.model_cfg.get("audit_dir") or "logs/joint96"),
            device=device, denoise=int(self.model_cfg.get("num_inference_steps") or 10),
            replan=self.replan_steps, seed=int(self.model_cfg.get("seed") or 0),
        )
        self.session_factory = partial(RoboDojoRealJoint96Session, self.runtime, session_cfg)
        self.session = self.session_factory()
        print(f"[Phoenix_WAM] Joint96 real robot={self.env_cfg_type} slot={self.runtime.embodiment_id} step={manifest['step']} repeat9_full_stvae", flush=True)


__all__ = ["Model"]
