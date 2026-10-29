"""Load a complete policy and run a preprocessed observation contract.

Simulator-specific camera acquisition and physical action execution belong to
the benchmark adapter. This layer never guesses units, action representations,
history sampling, or an observation's camera order.
"""

from __future__ import annotations

import copy
import importlib
import inspect
import json
import math
from pathlib import Path
from typing import Any, Mapping

import torch
import yaml
from omegaconf import OmegaConf

from prismwam.config import load_config


def read_config(path: str | Path) -> dict[str, Any]:
    """Read a resolved YAML config; interpolations must resolve explicitly."""
    path = Path(path).expanduser().resolve()
    result = OmegaConf.to_container(load_config(path), resolve=True)
    if not isinstance(result, dict):
        raise TypeError("Policy configuration must be a mapping.")
    # Exported models resolve assets relative to their configuration, never the
    # caller's current working directory. Training configs retain their paths.
    if result.get("deployment", {}).get("portable_bundle", False):
        for key in (
            "rgb_vae_checkpoint_path",
            "depth_vae_checkpoint_path",
            "text_encoder_checkpoint_path",
            "tokenizer_path",
        ):
            value = result.get("model", {}).get(key)
            if value:
                result["model"][key] = str((path.parent / value).resolve())
        for key in ("statistics_path", "norm_stats_path", "text_cache_path"):
            value = result["deployment"].get(key)
            if value:
                result["deployment"][key] = str((path.parent / value).resolve())
    return result


def read_checkpoint(path: str | Path) -> dict[str, Any]:
    """Read tensor-only model weights without executing arbitrary pickle code."""
    path = Path(path).expanduser().resolve(strict=True)
    payload = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    if not isinstance(payload, dict) or not isinstance(payload.get("mot"), dict):
        raise ValueError("Expected a complete policy checkpoint containing 'mot'.")
    if not payload["mot"]:
        raise ValueError("The policy checkpoint contains no model weights.")
    return payload


def needs_text_encoder(config: Mapping[str, Any]) -> bool:
    """Models without a text cache encode instructions with their T5 encoder."""
    return "deployment" in config and not config["deployment"].get("text_cache_path")


def prepare_model_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """Disable training initializers without changing architecture or contracts."""
    result = copy.deepcopy(dict(config.get("model", config)))
    target = result.get("_target_", "")
    if not isinstance(target, str) or not target.startswith("prismwam."):
        raise ValueError(
            "Expected a PrismWAM factory target; export a release config first."
        )
    result["checkpoint_init_only"] = True
    if needs_text_encoder(config):
        # The T5 encoder and tokenizer come from the Wan2.2-TI2V-5B folder.
        result["load_text_encoder"] = True
        result["redirect_common_files"] = False
        result["tokenizer_model_id"] = result.get("model_id", "Wan-AI/Wan2.2-TI2V-5B")
    for key in ("depth_dit_checkpoint_path", "action_dit_pretrained_path"):
        if key in result:
            result[key] = None
    for key in (
        "skip_depth_dit_load_from_pretrain",
        "skip_rgb_dit_load_from_pretrain",
        "skip_action_dit_load_from_pretrain",
    ):
        if key in result:
            result[key] = True
    return result


def apply_evaluation_config(
    config: Mapping[str, Any], evaluation: Mapping[str, Any]
) -> dict[str, Any]:
    """Bind explicit inference settings without changing the training recipe."""
    result = copy.deepcopy(dict(config))
    if evaluation.get("conditioning_noise", False) is not False:
        raise ValueError("Deployment conditioning noise must be disabled.")
    if evaluation.get("rand_device", "cpu") != "cpu":
        raise ValueError("Release deployment samples reproducible noise on CPU.")
    deployment = result.setdefault("deployment", {})
    for key in ("action_horizon", "replan_steps", "num_inference_steps"):
        if key in evaluation:
            value = evaluation[key]
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"Invalid evaluation {key}: {value!r}")
            deployment[key] = value
    if "inference_shifts" in evaluation:
        shifts = evaluation["inference_shifts"]
        if not isinstance(shifts, Mapping) or set(shifts) != {"depth", "action"}:
            raise ValueError("inference_shifts requires exactly depth and action.")
        for value in shifts.values():
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value <= 0
            ):
                raise ValueError("Inference shifts must be finite positive numbers.")
        deployment["inference_shifts"] = dict(shifts)
    return result


INFERENCE_KEYS = (
    "benchmark",
    "action_horizon",
    "replan_steps",
    "num_inference_steps",
    "inference_shifts",
)


def read_evaluation_settings(
    config: Mapping[str, Any], evaluation: str | Path | Mapping[str, Any] | None = None
) -> dict[str, Any]:
    """Inference settings from an explicit file/mapping, else the config's `deployment`."""
    if evaluation is None:
        deployment = config.get("deployment") or {}
        settings = {key: deployment[key] for key in INFERENCE_KEYS if key in deployment}
        if not settings:
            raise ValueError(
                "No inference settings: pass an evaluation file or use a Stage-2 "
                "config, which records them under `deployment`."
            )
        return settings
    if isinstance(evaluation, (str, Path)):
        evaluation = yaml.safe_load(Path(evaluation).read_text())
    return dict(evaluation)


def audit_model_state(
    model: torch.nn.Module, payload: Mapping[str, Any]
) -> dict[str, int]:
    """Fail on a missing, extra, or incorrectly shaped trained policy tensor.

    Compact checkpoints store shared visual blocks once. Expansion here is a
    view of existing tensors; it does not allocate a second parameter backbone.
    """
    expected = model.mot.state_dict()
    observed = dict(payload["mot"])
    for key, value in list(observed.items()):
        if key.startswith("mixtures.rgb.blocks."):
            alternate = key.replace("mixtures.rgb.blocks.", "mixtures.video.blocks.", 1)
            if alternate in expected and alternate not in observed:
                observed[alternate] = value
    missing = sorted(set(expected) - set(observed))
    extra = sorted(set(observed) - set(expected))
    wrong = sorted(
        key
        for key in set(expected) & set(observed)
        if not isinstance(observed[key], torch.Tensor)
        or tuple(observed[key].shape) != tuple(expected[key].shape)
    )
    if missing or extra or wrong:
        raise ValueError(
            f"Checkpoint tensor mismatch: missing={missing[:12]}, "
            f"unexpected={extra[:12]}, wrong_shape={wrong[:12]}"
        )
    proprio = getattr(model, "proprio_encoder", None)
    if proprio is not None:
        actual = payload.get("proprio_encoder", {})
        desired = proprio.state_dict()
        if set(actual) != set(desired) or any(
            not isinstance(actual[key], torch.Tensor)
            or actual[key].shape != desired[key].shape
            for key in desired
        ):
            raise ValueError("Checkpoint proprio encoder does not match the policy.")
    rgb, depth = (
        getattr(model, "rgb_expert", None),
        getattr(model, "video_expert", None),
    )
    if rgb is not None and depth is not None and rgb.blocks is not depth.blocks:
        raise ValueError(
            "This release requires a shared RGB/Depth transformer backbone."
        )
    return {
        "checkpoint_tensor_keys": len(payload["mot"]),
        "expanded_tensor_keys": len(observed),
        "unique_model_parameters": sum(p.numel() for p in model.parameters()),
    }


class ActionStatistics:
    """Separate state/action z-score statistics with strict dimension checks."""

    def __init__(
        self,
        payload: Mapping[str, Any],
        dimension: int,
        *,
        require_separate_state: bool = False,
        state_dimension: int | None = None,
        expected_task: str | None = None,
    ):
        state_dimension = dimension if state_dimension is None else state_dimension
        if expected_task is not None:
            recorded_dimension = payload.get("action_dim")
            if recorded_dimension is not None and recorded_dimension != dimension:
                raise ValueError("Statistics action_dim does not match the robot task.")
            recorded_tasks = payload.get("tasks")
            if recorded_tasks is not None and (
                not isinstance(recorded_tasks, (list, tuple))
                or expected_task not in recorded_tasks
            ):
                raise ValueError(
                    "Statistics tasks do not include the selected robot task."
                )

        def vector(key: str, size: int) -> torch.Tensor:
            value = torch.as_tensor(payload[key], dtype=torch.float32)
            if value.shape != (size,) or not torch.isfinite(value).all():
                raise ValueError(f"Statistics '{key}' must be finite [{size}].")
            return value

        self.action_mean, self.action_std = (
            vector("mean", dimension),
            vector("std", dimension),
        )
        if "state_mean" in payload and "state_std" in payload:
            self.state_mean, self.state_std = (
                vector("state_mean", state_dimension),
                vector("state_std", state_dimension),
            )
        elif require_separate_state or state_dimension != dimension:
            raise ValueError(
                "This action representation requires independent State/Action statistics."
            )
        else:
            self.state_mean, self.state_std = self.action_mean, self.action_std
        if not (self.action_std > 0).all() or not (self.state_std > 0).all():
            raise ValueError(
                "Normalization standard deviations must be strictly positive."
            )
        self.schema = payload.get("action_schema_id")

    @classmethod
    def load(cls, path: str | Path, dimension: int, **kwargs) -> "ActionStatistics":
        path = Path(path).expanduser()
        payload = (
            json.loads(path.read_text())
            if path.suffix == ".json"
            else torch.load(path, map_location="cpu", weights_only=True)
        )
        return cls(payload, dimension, **kwargs)

    def normalize_state(self, state: torch.Tensor) -> torch.Tensor:
        return (state - self.state_mean.to(state)) / self.state_std.to(state)

    def denormalize_action(self, action: torch.Tensor) -> torch.Tensor:
        return action * self.action_std.to(action) + self.action_mean.to(action)


# Every recipe trains with this template unless its data config sets prompt_template.
PROMPT_TEMPLATE = (
    "A video recorded from a robot's point of view executing the following "
    "instruction: {task}"
)


class PrismWAMPolicy:
    """One inference stream, with caller-owned observation history.

    ``predict`` accepts normalized RGB/depth mosaics and either tokenized text
    context or a prompt if the model has its text encoder. State/action are in
    checkpoint units when statistics are supplied; otherwise both are normalized.
    """

    def __init__(
        self,
        model,
        config: Mapping[str, Any],
        *,
        audit=None,
        statistics: ActionStatistics | None = None,
    ):
        self.model = model.eval()
        self.config = copy.deepcopy(dict(config))
        self.audit = dict(audit or {})
        self.statistics = statistics
        data = self.config.get("data", {}).get("train", {})
        deployment = self.config.get("deployment", {})
        self.action_horizon = int(
            deployment.get("action_horizon", data.get("action_horizon", 32))
        )
        self.video_frames = int(
            deployment.get("num_video_frames", data.get("num_video_frames", 9))
        )
        self.inference_steps = int(deployment.get("num_inference_steps", 10))
        self.replan_steps = int(deployment.get("replan_steps", 24))
        if (
            not 1 <= self.replan_steps <= self.action_horizon
            or self.inference_steps < 1
        ):
            raise ValueError(
                "Invalid horizon, replanning interval, or denoising steps."
            )
        self.action_dim = int(model.action_expert.action_dim)
        self.proprio_dim = int(
            self.config.get("model", {}).get("proprio_dim", self.action_dim)
        )

    def format_prompt(self, instruction: str) -> str:
        """Wrap an instruction in the prompt template the model was trained with."""
        data = self.config.get("data", {}).get("train", {})
        return data.get("prompt_template", PROMPT_TEMPLATE).format(task=instruction)

    def reset(self) -> None:
        if hasattr(self.model, "reset_rgb_memory"):
            self.model.reset_rgb_memory()

    @torch.inference_mode()
    def predict(
        self, observation: Mapping[str, Any], *, seed: int = 42
    ) -> dict[str, Any]:
        allowed = {
            "input_image",
            "rgb_input_image",
            "rgb_history_images",
            "proprio",
            "context",
            "context_mask",
            "prompt",
        }
        unexpected = set(observation) - allowed
        if unexpected:
            raise ValueError(f"Unknown observation fields: {sorted(unexpected)}")
        if not {"input_image", "rgb_input_image"} <= set(observation):
            raise ValueError("Current depth and RGB images are required.")
        inputs = {}
        for key, value in observation.items():
            if key == "prompt":
                if value is not None and not isinstance(value, str):
                    raise TypeError("prompt must be text or null")
                inputs[key] = value
                continue
            if key == "proprio":
                # The native deployment normalizes CPU float32 robot state
                # before the model casts it for its proprio projection.
                dtype, device = torch.float32, "cpu"
            elif key.endswith("mask") or key.endswith("is_valid"):
                dtype, device = torch.bool, self.model.device
            else:
                dtype, device = self.model.torch_dtype, self.model.device
            tensor = torch.as_tensor(value, dtype=dtype, device=device)
            if tensor.dtype != torch.bool and not torch.isfinite(tensor).all():
                raise ValueError(f"Observation '{key}' contains non-finite values.")
            inputs[key] = tensor
        inputs.setdefault("prompt", None)
        if "proprio" in inputs:
            if inputs["proprio"].shape[-1] != self.proprio_dim:
                raise ValueError(
                    "Robot-state dimensionality differs from the checkpoint proprio encoder."
                )
            if inputs["proprio"].shape == (1, 1, self.proprio_dim):
                inputs["proprio"] = inputs["proprio"][:, 0]
            if inputs["proprio"].shape not in {
                (self.proprio_dim,),
                (1, self.proprio_dim),
            }:
                raise ValueError("Robot state must be [D], [1,D], or [1,1,D].")
            if self.statistics is not None:
                inputs["proprio"] = self.statistics.normalize_state(inputs["proprio"])
        output = self.model.infer_action(
            **inputs,
            action_horizon=self.action_horizon,
            num_video_frames=self.video_frames,
            num_inference_steps=self.inference_steps,
            seed=int(seed),
            rand_device="cpu",
        )
        action = output["action"].detach().float().cpu()
        if (
            action.shape != (self.action_horizon, self.action_dim)
            or not torch.isfinite(action).all()
        ):
            raise RuntimeError(
                f"Invalid predicted action chunk: shape={tuple(action.shape)}"
            )
        if self.statistics is not None:
            action = self.statistics.denormalize_action(action)
        return {
            "action": action,
            "replan_steps": self.replan_steps,
            "normalized": self.statistics is None,
        }


def load_policy(
    config: str | Path | Mapping[str, Any],
    checkpoint: str | Path,
    *,
    device: str = "cuda:0",
    dtype: torch.dtype = torch.bfloat16,
    statistics_path: str | Path | None = None,
    evaluation: str | Path | Mapping[str, Any] | None = None,
) -> PrismWAMPolicy:
    """Load exact model weights on CPU first, then transfer one shared model."""
    config = read_config(config) if isinstance(config, (str, Path)) else dict(config)
    if evaluation is not None:
        config = apply_evaluation_config(
            config, read_evaluation_settings(config, evaluation)
        )
    model_config = prepare_model_config(config)
    if needs_text_encoder(config):
        from prismwam.hub import download_text_encoder

        download_text_encoder(model_config["tokenizer_model_id"])
    if model_config.get("image_encoder_model_id"):
        from prismwam.hub import download_image_encoder

        download_image_encoder(model_config["image_encoder_model_id"])
    target = model_config.pop("_target_")
    module, name = target.rsplit(".", 1)
    factory = getattr(importlib.import_module(module), name)
    shifts = config.get("deployment", {}).get("inference_shifts")
    payload = read_checkpoint(checkpoint)
    # A factory that takes the checkpoint builds the model from it directly,
    # instead of initializing every weight first.
    builds_from_weights = "checkpoint_path" in inspect.signature(factory).parameters
    if builds_from_weights:
        model_config["checkpoint_path"] = str(checkpoint)
    model = factory(**model_config, model_dtype=dtype, device="cpu")
    audit = audit_model_state(model, payload)
    if not builds_from_weights:
        model.load_checkpoint(str(checkpoint))
    if shifts is not None:
        # Separate inference objects leave training schedules and signed metadata
        # untouched, including checkpoints trained with a different Action shift.
        for attr, branch in (
            ("infer_video_scheduler", "depth"),
            ("infer_action_scheduler", "action"),
        ):
            scheduler = copy.deepcopy(getattr(model, attr))
            scheduler.shift = float(shifts[branch])
            setattr(model, attr, scheduler)
        audit["inference_shifts"] = dict(shifts)
    del payload
    model.to(device).eval()
    model.device = torch.device(device)
    statistics = None
    if statistics_path is None:
        statistics_path = config.get("deployment", {}).get("statistics_path")
    action_dim = int(model.action_expert.action_dim)
    # Single-arm real-robot models keep 7-D statistics: the caller normalizes,
    # pads and un-normalizes, so the policy itself works in normalized units.
    native_dim = config.get("data", {}).get("train", {}).get("action_dim", action_dim)
    if statistics_path is not None and int(native_dim) == action_dim:
        statistics = ActionStatistics.load(
            statistics_path,
            action_dim,
            state_dimension=int(
                config.get("model", {}).get(
                    "proprio_dim", model.action_expert.action_dim
                )
            ),
        )
    return PrismWAMPolicy(model, config, audit=audit, statistics=statistics)
