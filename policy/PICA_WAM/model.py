"""XPolicyLab adapter for PICA-WAM on the InternW0-Delta backbone (RoboTwin 2.0)."""

from __future__ import annotations

import json
import os
import sys
from functools import partial
from pathlib import Path
from typing import Any

import torch
from omegaconf import DictConfig, OmegaConf

from XPolicyLab.model_template import ModelTemplate
from XPolicyLab.utils.checkpoint_resolver import resolve_checkpoint_root

from ._adapter_base import StatefulWAMAdapter, as_bool, optional_float, optional_int

POLICY_DIR = Path(__file__).resolve().parent
CHECKPOINTS_DIR = POLICY_DIR / "checkpoints"
RUNTIME_ROOT = POLICY_DIR / "internw0_pica"
RELEASE_CHECKPOINT = "robotwin.pt"
TRAIN_TASK = "robotwin_pica"
SUPPORTED_BENCH = "RoboTwin"


def optional_str(value: Any) -> str | None:
    text = "" if value is None else str(value).strip()
    return None if text.lower() in {"", "none", "null"} else text


def _local_path(value: Any) -> Path:
    path = Path(str(value)).expanduser()
    return path if path.is_absolute() else POLICY_DIR / path


def _require_file(value: Any, name: str) -> Path:
    path = _local_path(value).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"{name} not found: {path}")
    return path


def _require_dir(value: Any, name: str) -> Path:
    path = _local_path(value).resolve()
    if not path.is_dir():
        raise FileNotFoundError(f"{name} not found: {path}")
    return path


def latest_run_weights(run_dir: Path) -> Path | None:
    """Latest training weight in <run_dir>/checkpoints/weights.

    Slots are reused round-robin, so the manifest (or file mtime) decides, not
    the slot index.
    """
    weights_dir = run_dir / "checkpoints" / "weights"
    manifest = weights_dir / "weights_manifest.json"
    if manifest.is_file():
        latest = (json.loads(manifest.read_text(encoding="utf-8")).get("latest") or {}).get("path")
        if latest and (weights_dir / latest).is_file():
            return weights_dir / latest
    slots = sorted(weights_dir.glob("slot_*.pt"), key=lambda path: path.stat().st_mtime)
    return slots[-1] if slots else None


def _activate_runtime(runtime_root: Path) -> None:
    source = str(runtime_root / "src")
    sys.path[:] = [source] + [entry for entry in sys.path if entry != source]
    for module_name in list(sys.modules):
        if module_name == "wam" or module_name.startswith("wam."):
            del sys.modules[module_name]


def compose_train_config(runtime_root: Path, task: str) -> DictConfig:
    """The training config of ``task``, composed from the vendored Hydra tree."""

    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra
    from wam.utils.config_resolvers import register_default_resolvers

    register_default_resolvers()
    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()
    with initialize_config_dir(version_base="1.3", config_dir=str(runtime_root / "configs")):
        return compose(config_name="train", overrides=[f"task={task}"])


class Model(ModelTemplate):
    """XPolicyLab model adapter with one stateful WAM session per environment."""

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

    def _resolve_checkpoint(self) -> tuple[Path, Path | None]:
        """Weights file, plus the training run directory when there is one."""

        explicit_keys = ("checkpoint_path", "ckpt_path", "model_path")
        try:
            checkpoint_path = resolve_checkpoint_root(
                self.model_cfg,
                CHECKPOINTS_DIR,
                policy_dir=POLICY_DIR,
                explicit_keys=explicit_keys,
                must_exist=True,
            )
        except FileNotFoundError:
            # No run directory: fall back to the release file, but never past a
            # checkpoint path that was given explicitly and does not exist.
            if any(optional_str(self.model_cfg.get(key)) for key in explicit_keys):
                raise
            checkpoint_path = CHECKPOINTS_DIR / RELEASE_CHECKPOINT
        run_dir = None
        if checkpoint_path.is_dir():
            run_dir = checkpoint_path
            checkpoint_path = latest_run_weights(run_dir) or (run_dir / RELEASE_CHECKPOINT)
        return _require_file(checkpoint_path, "checkpoint_path"), run_dir

    def _load_real_policy(self) -> None:
        bench_name = str(self.model_cfg.get("bench_name") or SUPPORTED_BENCH)
        if bench_name != SUPPORTED_BENCH:
            raise ValueError(f"PICA_WAM supports bench_name={SUPPORTED_BENCH}, got {bench_name!r}")
        checkpoint_path, run_dir = self._resolve_checkpoint()
        runtime_root = _require_dir(self.model_cfg.get("wam_root") or RUNTIME_ROOT, "wam_root")
        stats_path = _require_file(
            self.model_cfg.get("dataset_stats_path")
            or runtime_root / "assets" / "stats" / "robotwin.json",
            "dataset_stats_path",
        )
        base_model_dir = _require_dir(
            self.model_cfg.get("base_model_dir", "assets/Wan-AI/Wan2.2-TI2V-5B"),
            "base_model_dir",
        )
        vlm_model_path = _require_dir(
            self.model_cfg.get(
                "vlm_model_path", "assets/Alibaba-DAMO-Academy/RynnBrain1.1-2B"
            ),
            "vlm_model_path",
        )

        _activate_runtime(runtime_root)
        os.environ["DIFFSYNTH_SKIP_DOWNLOAD"] = "true"

        from .robotwin_session import (
            RobotWinPolicySession,
            RobotWinSessionConfig,
            _mixed_precision_to_model_dtype,
            _normalize_device,
            _resolve_training_video_contract,
            build_robotwin_shared_runtime,
        )

        module_path = Path(sys.modules["wam"].__file__).resolve()
        if not module_path.is_relative_to(runtime_root):
            raise RuntimeError(f"WAM source isolation failed: {module_path}")

        # The config a run was trained with travels with its run directory;
        # the release file was trained with the vendored robotwin_pica task.
        config_path = optional_str(self.model_cfg.get("train_config_path"))
        if config_path is None and run_dir is not None and (run_dir / "config.yaml").is_file():
            config_path = str(run_dir / "config.yaml")
        if config_path is not None:
            config_path = str(_require_file(config_path, "train_config_path"))
            train_cfg = OmegaConf.load(config_path)
        else:
            train_cfg = compose_train_config(runtime_root, TRAIN_TASK)
            config_path = f"{runtime_root / 'configs'}:task={TRAIN_TASK}"

        model_cfg = OmegaConf.create(OmegaConf.to_container(train_cfg.model, resolve=True))
        data_cfg = OmegaConf.create(OmegaConf.to_container(train_cfg.data.train, resolve=True))
        # The PICA alignment head and its EMA teacher only produce the training-time
        # alignment loss; inference never reads them, nor the compile settings.
        model_cfg.pica_alignment = None
        model_cfg.mot_compile_mode = "off"
        model_cfg.mot_compile_gradient_checkpointing = False
        model_cfg.model_id = str(base_model_dir)
        model_cfg.tokenizer_model_id = str(base_model_dir)
        model_cfg.redirect_common_files = False
        model_cfg.skip_dit_load_from_pretrain = True
        model_cfg.action_dit_pretrained_path = None
        model_cfg.understanding.vlm_model_path = str(vlm_model_path)
        model_cfg.load_text_encoder = True

        device = _normalize_device(str(self.model_cfg.get("device") or "cuda"))
        if device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for real WAM evaluation")
        dtype = _mixed_precision_to_model_dtype(
            str(self.model_cfg.get("mixed_precision") or "bf16")
        )
        self.runtime = build_robotwin_shared_runtime(
            model_cfg=model_cfg,
            processor_cfg=data_cfg.processor,
            checkpoint_path=str(checkpoint_path),
            dataset_stats_path=stats_path,
            device=device,
            model_dtype=dtype,
        )

        if [str(item.key) for item in data_cfg.processor.shape_meta.images] != [
            "cam_high",
            "cam_left_wrist",
            "cam_right_wrist",
        ]:
            raise ValueError("Evaluation camera order does not match training")
        video_layout, view_names = _resolve_training_video_contract(data_cfg)
        num_frames = int(data_cfg.num_frames)
        frequency_ratio = int(data_cfg.action_video_freq_ratio)
        session_cfg = RobotWinSessionConfig(
            action_horizon=self.action_horizon,
            action_hz=float(self.model_cfg.get("action_hz") or 50.0),
            replan_steps=self.replan_steps,
            num_inference_steps=int(self.model_cfg.get("num_inference_steps") or 10),
            sigma_shift=optional_float(self.model_cfg.get("sigma_shift")),
            seed=optional_int(self.model_cfg.get("seed")),
            rand_device=str(self.model_cfg.get("rand_device") or "cpu"),
            tiled=as_bool(self.model_cfg.get("tiled", False)),
            timing_enabled=as_bool(self.model_cfg.get("timing_enabled", False)),
            num_video_frames=(num_frames - 1) // frequency_ratio + 1,
            video_size=tuple(map(int, data_cfg.video_size)),
            video_layout=video_layout,
            video_view_names=view_names,
        )
        self.session_factory = partial(RobotWinPolicySession, self.runtime, session_cfg)
        self.session = self.session_factory()
        print(
            "[PICA_WAM] initialized "
            f"checkpoint={checkpoint_path} config={config_path} stats={stats_path} "
            f"horizon={self.action_horizon} replan={self.replan_steps} "
            f"denoise={session_cfg.num_inference_steps}",
            flush=True,
        )


__all__ = ["Model"]
