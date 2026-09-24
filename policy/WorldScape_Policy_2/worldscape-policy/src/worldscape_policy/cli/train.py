"""Configuration-driven native WorldScape Policy training entrypoint."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
from torch import nn
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from tqdm.auto import tqdm

from worldscape_policy.cli.config_composition import load_composed_config
from worldscape_policy.cli.config_profiles import resolve_config_profiles
from worldscape_policy.training.freezing import (
    NativeFreezeConfig,
    native_freeze_policy,
)
from worldscape_policy.training.objective import CompositeObjective
from worldscape_policy.training.native_export import export_training_checkpoint
from worldscape_policy.training.prompt_schedule import PromptSchedule
from worldscape_policy.training.runtime import (
    NativeDistributedConfig,
    NativeTrainingRuntime,
)
from worldscape_policy.training.scheduler import (
    NativeLRSchedulerConfig,
    build_lr_scheduler,
)
from worldscape_policy.training.wandb_logger import WandbRunLogger
from worldscape_policy.training.checkpoint_resolve import (
    resolve_training_checkpoint_sources,
)
from worldscape_policy.training.trainer import NativeTrainer
from worldscape_policy.types import InteractionMode

_FORBIDDEN_TARGET_PARTS = (
    ".compat.",
    "WorldScapeWan22ActionHead",
)


def run_config(config: DictConfig) -> NativeTrainer:
    """Build and execute a native run; legacy action heads are rejected."""

    _reject_compat_targets(OmegaConf.to_container(config, resolve=True))
    for required in ("model", "optimizer", "noise_kernel", "data_loader", "training"):
        if required not in config:
            raise ValueError(f"native training config is missing {required!r}")

    distributed_options = (
        OmegaConf.to_container(config.distributed, resolve=True)
        if "distributed" in config
        else {}
    )
    if not isinstance(distributed_options, dict):
        raise TypeError("distributed config must be a mapping")
    distributed_options.pop("_target_", None)
    distributed_options.setdefault(
        "device",
        str(OmegaConf.select(config, "model.device", default="cpu")),
    )
    distributed_options.setdefault(
        "seed",
        int(OmegaConf.select(config, "data_loader.seed", default=torch.initial_seed())),
    )
    runtime = NativeTrainingRuntime(NativeDistributedConfig(**distributed_options))
    resolve_training_checkpoint_sources(config)
    _prepare_resume_data_loader_config(config)
    model = instantiate(config.model)
    prompt_schedule, projector_only_end = _build_prompt_schedule(config)
    if prompt_schedule is not None:
        configured_mode = getattr(model, "configured_mode", None)
        if (
            configured_mode is not None
            and InteractionMode.parse(configured_mode) is not InteractionMode.AUTO
        ):
            raise ValueError(
                "dual-prompt scheduling requires an Auto checkpoint containing both "
                "native conditioners"
            )

    freeze_report = _apply_freezing(config, model, prompt_schedule is not None)
    optimizer = _build_optimizer(config.optimizer, model)
    objective = (
        instantiate(config.objective)
        if "objective" in config
        else CompositeObjective()
    )
    _validate_planning_vlm_trainable(model, objective)
    noise_kernel = instantiate(config.noise_kernel)
    adapter = None
    if "batch_adapter" in config:
        adapter_options: dict[str, Any] = {}
        visual_memory = getattr(model, "visual_memory", None)
        codec = getattr(visual_memory, "codec", None)
        preprocessor = getattr(codec, "prepare_diffusion_video", None)
        encoder = getattr(codec, "encode_normalized", None)
        if not callable(preprocessor) or not callable(encoder):
            raise TypeError(
                "Wan training requires a visual codec with "
                "prepare_diffusion_video and encode_normalized"
            )
        adapter_options["diffusion_video_preprocessor"] = preprocessor
        adapter_options["video_latent_encoder"] = encoder
        adapter = instantiate(config.batch_adapter, **adapter_options)
    if "scheduler" in config:
        scheduler = instantiate(config.scheduler, optimizer=optimizer)
    else:
        scheduler = build_lr_scheduler(
            optimizer,
            total_steps=int(config.training.max_steps),
            config=NativeLRSchedulerConfig(
                schedule=str(config.training.get("lr_schedule", "linear")),
                warmup_ratio=float(config.training.get("warmup_ratio", 0.05)),
            ),
        )
    trainer_options = (
        OmegaConf.to_container(config.trainer, resolve=True)
        if "trainer" in config
        else {}
    )
    if not isinstance(trainer_options, dict):
        raise TypeError("trainer config must be a mapping")
    metadata = _checkpoint_config_metadata(config)
    training = config.training
    max_steps = int(training.max_steps)
    if max_steps <= 0:
        raise ValueError("training.max_steps must be positive")
    callbacks = instantiate(config.callbacks) if "callbacks" in config else None
    trainer = NativeTrainer(
        policy=model,
        optimizer=optimizer,
        objective=objective,
        noise_kernel=noise_kernel,
        scheduler=scheduler,
        batch_adapter=adapter,
        config_metadata=metadata,
        prompt_schedule=prompt_schedule,
        max_steps=max_steps,
        projector_only_end=projector_only_end,
        callbacks=callbacks,
        runtime=runtime,
        **trainer_options,
    )

    save_every = int(training.get("save_every", 0))
    save_at_end = bool(training.get("save_at_end", True))
    log_every = int(training.get("log_every", 1))
    if log_every <= 0:
        raise ValueError("training.log_every must be positive")
    if "save_final" in training:
        # Backward-compatible override for older recipes and CLI dot-lists.
        save_at_end = bool(training.save_final)
    checkpoint_dir = Path(str(training.get("checkpoint_dir", "checkpoints")))
    wandb_logger = (
        _build_wandb_logger(config, checkpoint_dir)
        if runtime.is_rank_zero
        else None
    )
    if runtime.is_rank_zero:
        _persist_freeze_report(freeze_report.as_dict(), checkpoint_dir)
    data_loader = instantiate(config.data_loader)
    trainer.attach_data_source(data_loader)
    resume_mode = str(training.get("resume_mode", "fast"))
    if "resume" in training and training.resume and resume_mode == "exact":
        _validate_resume_data_loader(data_loader)
    if "resume" in training and training.resume:
        trainer.load_checkpoint(
            training.resume,
            restore_data_state=resume_mode == "exact",
        )
        if resume_mode == "fast":
            print(
                json.dumps(
                    {
                        "resume_data_loader": {
                            "mode": "fast",
                            "checkpoint": str(training.resume),
                            "reproducible": False,
                            "reason": "worker prefetch/data position is intentionally reset",
                        }
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    iterator = iter(data_loader)
    if "resume" in training and training.resume:
        if resume_mode == "exact" and not _has_stateful_data_source(data_loader):
            iterator = _fast_forward_data(
                data_loader, iterator, trainer.data_batches_consumed
            )
            # Data iteration may use process RNGs. Restore exact RNG after replay.
            trainer.load_checkpoint(training.resume, notify_callbacks=False)
    run_steps_value = training.get("run_steps")
    stop_step = max_steps
    if run_steps_value is not None:
        run_steps = int(run_steps_value)
        if run_steps <= 0:
            raise ValueError("training.run_steps must be positive when configured")
        stop_step = min(max_steps, trainer.step + run_steps)
    if wandb_logger is not None:
        wandb_logger.start()
    trainer.train_start()
    last_periodic_step: int | None = None
    last_checkpoint_path: Path | None = None
    if "resume" in training and training.resume:
        resume_path = Path(str(training.resume))
        expected_name = f"checkpoint-{trainer.step}"
        if (
            resume_path.is_dir()
            and resume_path.name == expected_name
            and resume_path.parent.resolve(strict=False)
            == checkpoint_dir.resolve(strict=False)
        ):
            # A completed run can be relaunched without rewriting its checkpoint.
            last_periodic_step = trainer.step
            last_checkpoint_path = resume_path
    progress_bar = tqdm(
        total=stop_step,
        initial=trainer.step,
        desc="train",
        dynamic_ncols=True,
        disable=not runtime.is_rank_zero,
    )
    try:
        while trainer.step < stop_step:
            try:
                batch = next(iterator)
            except StopIteration:
                iterator = iter(data_loader)
                try:
                    batch = next(iterator)
                except StopIteration as exc:
                    raise RuntimeError("native data loader is empty") from exc
            metrics = trainer.train_step(batch)
            if runtime.is_rank_zero:
                logged_metrics = _console_metrics(metrics, trainer.last_step_timing)
                wandb_metrics = {
                    **logged_metrics,
                    "training_step_time": trainer.last_training_step_time,
                    "model_forward_time": trainer.last_model_forward_time,
                }
                progress_bar.update(trainer.step - progress_bar.n)
                assert wandb_logger is not None
                if metrics.get("optimizer_step", 0.0) == 1.0:
                    wandb_logger.log(wandb_metrics, step=trainer.step)
                    if trainer.step % log_every == 0:
                        progress_bar.write(_format_console_metrics(logged_metrics))
            if save_every > 0 and trainer.step % save_every == 0:
                last_checkpoint_path = trainer.save_checkpoint(
                    _checkpoint_path(
                        checkpoint_dir,
                        f"checkpoint-{trainer.step}",
                    )
                )
                if _export_every_checkpoint(config):
                    _export_native_bundle(
                        config,
                        last_checkpoint_path,
                        destination=last_checkpoint_path,
                        runtime=runtime,
                    )
                last_periodic_step = trainer.step
        terminal_checkpoint_created = False
        if (
            save_at_end
            and last_periodic_step != trainer.step
        ):
            last_checkpoint_path = trainer.save_checkpoint(
                _checkpoint_path(
                    checkpoint_dir,
                    f"checkpoint-{trainer.step}",
                )
            )
            terminal_checkpoint_created = True
        if save_at_end:
            if last_checkpoint_path is None:
                raise RuntimeError("terminal checkpoint was not created")
            if _native_export_enabled(config) and (
                terminal_checkpoint_created or not _export_every_checkpoint(config)
            ):
                _export_native_bundle(
                    config,
                    last_checkpoint_path,
                    destination=last_checkpoint_path,
                    runtime=runtime,
                )
    finally:
        progress_bar.close()
        if wandb_logger is not None:
            wandb_logger.finish()
        trainer.train_end()
        runtime.close()
    return trainer


def _checkpoint_config_metadata(config: DictConfig) -> dict[str, Any]:
    """Remove launch-time checkpoint and worker choices from recipe identity."""
    metadata = OmegaConf.to_container(config, resolve=True)
    if not isinstance(metadata, dict):
        raise TypeError("top-level training config must be a mapping")
    metadata.pop("pretrained_adapter_source_rows", None)
    metadata.pop("logging", None)
    metadata_training = metadata.get("training")
    if isinstance(metadata_training, dict):
        metadata_training.pop("resume", None)
        metadata_training.pop("resume_mode", None)
        metadata_training.pop("run_steps", None)
        metadata_training.pop("checkpoint_dir", None)
        metadata_training.pop("log_every", None)
    metadata_loader = metadata.get("data_loader")
    if isinstance(metadata_loader, dict):
        metadata_loader.pop("num_workers", None)
    metadata_trainer = metadata.get("trainer")
    if isinstance(metadata_trainer, dict):
        metadata_trainer.pop("log_prompt_text", None)
    metadata_model = metadata.get("model")
    if isinstance(metadata_model, dict):
        metadata_model.pop("checkpoint_dir", None)
        metadata_model.pop("initialization", None)
        metadata_model.pop("pretrained_action_adapter_index", None)
    return metadata


_HF_NO_DECAY_MODULES = (
    nn.LayerNorm,
    nn.GroupNorm,
    nn.InstanceNorm1d,
    nn.InstanceNorm2d,
    nn.InstanceNorm3d,
    nn.LocalResponseNorm,
    nn.BatchNorm1d,
    nn.BatchNorm2d,
    nn.BatchNorm3d,
    nn.SyncBatchNorm,
)


def _build_optimizer(optimizer_config: DictConfig, model: nn.Module) -> torch.optim.Optimizer:
    options = OmegaConf.to_container(optimizer_config, resolve=True)
    if not isinstance(options, dict):
        raise TypeError("optimizer config must be a mapping")
    hf_decay_groups = bool(options.pop("hf_trainer_decay_groups", False))
    if hf_decay_groups:
        weight_decay = float(options.pop("weight_decay", 0.0))
        parameters = _hf_trainer_decay_groups(model, weight_decay=weight_decay)
        has_parameters = any(group["params"] for group in parameters)
    else:
        parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
        has_parameters = bool(parameters)
    if not has_parameters:
        raise ValueError("native model has no trainable parameters")
    return instantiate(options, params=parameters, _convert_="all")


def _hf_trainer_decay_groups(
    model: nn.Module,
    *,
    weight_decay: float,
) -> list[dict[str, Any]]:
    decay: list[nn.Parameter] = []
    no_decay: list[nn.Parameter] = []
    seen: set[int] = set()
    for module_name, module in model.named_modules():
        for parameter_name, parameter in module.named_parameters(recurse=False):
            if not parameter.requires_grad:
                continue
            parameter_id = id(parameter)
            if parameter_id in seen:
                continue
            seen.add(parameter_id)
            full_name = (
                parameter_name
                if not module_name
                else f"{module_name}.{parameter_name}"
            )
            if isinstance(module, _HF_NO_DECAY_MODULES) or "bias" in full_name:
                no_decay.append(parameter)
            else:
                decay.append(parameter)
    return [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]


def _validate_planning_vlm_trainable(
    model: Any,
    objective: CompositeObjective,
) -> None:
    if objective.planning_ce_weight <= 0:
        return
    router = getattr(model, "condition_router", None)
    auto_conditioner = getattr(router, "auto", None)
    planning_vlm = getattr(auto_conditioner, "vlm", None)
    if planning_vlm is None:
        return
    vlm = getattr(planning_vlm, "vlm", None)
    if vlm is None:
        raise ValueError(
            "planning supervision requires an Auto planning VLM"
        )
    if not any(parameter.requires_grad for parameter in vlm.parameters()):
        raise ValueError(
            "planning supervision requires an unfrozen VLM; set freeze.vlm=false"
        )


def _apply_freezing(
    config: DictConfig,
    model: Any,
    dual_prompt: bool,
):
    if "freeze" not in config:
        raise ValueError("native training config is missing 'freeze'")
    freeze_node = config.freeze
    strict = bool(freeze_node.get("strict", True))
    if "config" not in freeze_node:
        raise ValueError("freeze config is missing 'config'")
    freeze_config = instantiate(freeze_node.config)
    if not isinstance(freeze_config, NativeFreezeConfig):
        raise TypeError("freeze.config must instantiate NativeFreezeConfig")
    configured_mode = getattr(model, "configured_mode", None)
    unused_paths: tuple[str, ...] = ()
    if not dual_prompt and configured_mode is not None:
        mode = InteractionMode.parse(configured_mode)
        unused_paths = (
            ("condition_router.interactive",)
            if mode is InteractionMode.AUTO
            else ("condition_router.auto",)
        )
    report = native_freeze_policy(freeze_config, strict=strict).apply(
        model,
        unused_module_paths=unused_paths,
    )
    unused_qformer = tuple(
        name for name in report.unused_trainable_names if ".qformer." in name
    )
    if unused_qformer:
        raise ValueError(
            "freeze policy leaves requested trainable QFormer parameters "
            "outside model forward: "
            + ", ".join(unused_qformer)
        )
    return report


def _persist_freeze_report(report: Mapping[str, object], checkpoint_dir: Path) -> None:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    destination = checkpoint_dir / "trainability-report.json"
    destination.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")


def _build_wandb_logger(
    config: DictConfig,
    checkpoint_dir: Path,
) -> WandbRunLogger:
    node = OmegaConf.select(config, "logging.wandb", default={})
    options = (
        OmegaConf.to_container(node, resolve=True)
        if OmegaConf.is_config(node)
        else dict(node)
        if isinstance(node, Mapping)
        else node
    )
    if not isinstance(options, dict):
        raise TypeError("logging.wandb must be a mapping")
    options.setdefault("enabled", False)
    options.setdefault("output_dir", str(checkpoint_dir))
    options.setdefault("project", "worldscape_policy")
    options.setdefault("name", checkpoint_dir.name)
    wandb_config = _checkpoint_config_metadata(config)
    wandb_config["runtime"] = {
        "data_loader_num_workers": int(
            OmegaConf.select(config, "data_loader.num_workers", default=0)
        ),
        "data_loader_prefetch_factor": int(
            OmegaConf.select(config, "data_loader.prefetch_factor", default=2)
        ),
        "data_loader_persistent_workers": bool(
            OmegaConf.select(
                config,
                "data_loader.persistent_workers",
                default=False,
            )
        ),
        "data_loader_pin_memory": bool(
            OmegaConf.select(config, "data_loader.pin_memory", default=False)
        ),
    }
    options["config"] = wandb_config
    return WandbRunLogger(**options)


def _console_metrics(
    metrics: Mapping[str, float],
    step_timing: Mapping[str, float] | None = None,
) -> dict[str, float]:
    """Keep routine console output focused on optimization signals."""

    keys = {
        "step",
        "lr",
        "loss",
        "grad_norm",
        "action_flow/weighted_loss",
        "video_flow/weighted_loss",
        "prompt_schedule/auto",
        "prompt_schedule/stage",
    }
    if metrics.get("semantic_forcing/skipped", 1.0) == 0.0:
        keys.update(
            {
                "semantic_forcing/weighted_loss",
            }
        )
    if metrics.get("planning_ce/skipped", 1.0) == 0.0:
        keys.update(
            {
                "planning_ce/loss",
                "planning_ce/weighted_loss",
                "planning_ce/valid_tokens",
            }
        )
    logged = {key: metrics[key] for key in sorted(keys) if key in metrics}
    if step_timing:
        logged.update(
            {
                key: step_timing[key]
                for key in sorted(step_timing)
                if key.startswith("step_time/")
            }
        )
    return logged


def _format_console_metrics(metrics: Mapping[str, float]) -> str:
    """Render stable JSON with four decimal places for optimization metrics."""

    fields: list[str] = []
    for key in sorted(metrics):
        value = metrics[key]
        if key in {"step", "prompt_schedule/stage"}:
            rendered = str(int(value))
        elif key == "lr":
            rendered = f"{value:.8g}"
        else:
            rendered = f"{value:.4f}"
        fields.append(f"{json.dumps(key)}: {rendered}")
    return "{" + ", ".join(fields) + "}"


def _checkpoint_path(checkpoint_dir: Path, name: str) -> Path:
    return checkpoint_dir / name


def _export_every_checkpoint(config: DictConfig) -> bool:
    node = config.get("native_export")
    return bool(
        _native_export_enabled(config)
        and node is not None
        and node.get("every_checkpoint", True)
    )


def _native_export_enabled(config: DictConfig) -> bool:
    node = config.get("native_export")
    return bool(node is not None and node.get("enabled", False))


def _export_native_bundle(
    config: DictConfig,
    source: Path,
    *,
    destination: Path,
    runtime: NativeTrainingRuntime,
) -> Path | None:
    node = config.get("native_export")
    if node is None or not bool(node.get("enabled", False)):
        return None
    if "transform_bundle" not in node:
        raise ValueError("enabled native_export requires transform_bundle")
    error: str | None = None
    exported: Path | None = None
    if runtime.is_rank_zero:
        try:
            transform_bundle = instantiate(node.transform_bundle)
            exported = export_training_checkpoint(
                source,
                destination,
                model_variant=str(node.get("model_variant", "wsp2-wan22-5b")),
                model_config=config.model.model_config,
                generation_config=config.model.generation_config,
                normalization=node.normalization,
                transform_bundle=transform_bundle,
                tokenizer_source=str(
                    node.get("tokenizer_source", config.model.tokenizer_path)
                ),
                provenance=node.provenance,
                git_commit=node.get("git_commit"),
                max_shard_size=str(
                    node.get(
                        "max_shard_size",
                        config.trainer.get("checkpoint_max_shard_size", "5GB"),
                    )
                ),
            )
        except Exception as exc:  # synchronize failure instead of hanging peers
            error = f"{type(exc).__name__}: {exc}"
    states = runtime.gather_rank_state({"native_export_error": error})
    rank_zero_error = states[0]["native_export_error"]
    if rank_zero_error is not None:
        raise RuntimeError(f"native checkpoint export failed: {rank_zero_error}")
    runtime.barrier()
    return exported


def _build_prompt_schedule(
    config: DictConfig,
) -> tuple[PromptSchedule | None, float]:
    if "prompt_schedule" not in config:
        return None, 0.0
    node = config.prompt_schedule
    enabled = bool(node.get("enabled", False))
    projector_only_end = float(node.get("projector_only_end", 0.0))
    if not enabled:
        if projector_only_end != 0:
            raise ValueError(
                "prompt_schedule.projector_only_end requires prompt scheduling"
            )
        return None, 0.0
    if "schedule" not in node:
        raise ValueError("enabled prompt_schedule is missing 'schedule'")
    schedule = instantiate(node.schedule)
    if not isinstance(schedule, PromptSchedule):
        raise TypeError("prompt_schedule.schedule must instantiate PromptSchedule")
    return schedule, projector_only_end


def _fast_forward_data(data_loader: Any, iterator: Any, batches: int) -> Any:
    for _ in range(batches):
        try:
            next(iterator)
        except StopIteration:
            iterator = iter(data_loader)
            try:
                next(iterator)
            except StopIteration as exc:
                raise RuntimeError("native data loader is empty") from exc
    return iterator


def _validate_resume_data_loader(data_loader: Any) -> None:
    num_workers = int(getattr(data_loader, "num_workers", 0))
    if num_workers != 0:
        raise ValueError(
            "native checkpoint resume requires data_loader.num_workers=0; "
            "worker RNG and prefetch state cannot be reconstructed exactly"
        )


def _prepare_resume_data_loader_config(config: DictConfig) -> None:
    """Select exact or fast data semantics before constructing the loader."""
    resume = OmegaConf.select(config, "training.resume")
    if not resume:
        return
    mode = str(OmegaConf.select(config, "training.resume_mode", default="fast"))
    if mode not in {"exact", "fast"}:
        raise ValueError("training.resume_mode must be 'exact' or 'fast'")
    if mode == "fast":
        return
    configured = int(OmegaConf.select(config, "data_loader.num_workers", default=0))
    if configured != 0:
        config.data_loader.num_workers = 0
        print(
            json.dumps(
                {
                    "resume_data_loader": {
                        "checkpoint": str(resume),
                        "num_workers": 0,
                        "overrode_num_workers": configured,
                        "reason": (
                            "exact resume cannot reconstruct worker RNG and "
                            "prefetch state"
                        ),
                    }
                },
                sort_keys=True,
            ),
            flush=True,
        )


def _has_stateful_data_source(data_loader: Any) -> bool:
    pending = [data_loader]
    visited: set[int] = set()
    while pending:
        source = pending.pop()
        if source is None or id(source) in visited:
            continue
        visited.add(id(source))
        if hasattr(source, "state_dict") and hasattr(source, "load_state_dict"):
            return source is not data_loader
        pending.extend(
            getattr(source, name, None)
            for name in ("batch_sampler", "sampler", "dataset")
        )
    return False


def _reject_compat_targets(value: object, path: str = "config") -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            child_path = f"{path}.{key}"
            if (
                key == "_target_"
                and isinstance(child, str)
                and any(part in child for part in _FORBIDDEN_TARGET_PARTS)
            ):
                raise ValueError(
                    f"{child_path} selects forbidden compatibility training target "
                    f"{child!r}"
                )
            _reject_compat_targets(child, child_path)
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _reject_compat_targets(child, f"{path}[{index}]")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        required=True,
        type=Path,
        help="Native YAML/JSON training configuration",
    )
    parser.add_argument(
        "overrides",
        nargs="*",
        help="Optional OmegaConf dot-list overrides applied after loading --config",
    )
    args = parser.parse_args(argv)
    config = load_composed_config(args.config)
    overrides = OmegaConf.from_dotlist(args.overrides) if args.overrides else None
    config = resolve_config_profiles(config, overrides=overrides)
    run_config(config)


if __name__ == "__main__":
    main()
