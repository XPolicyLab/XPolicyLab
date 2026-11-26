from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Any

import torch

from evals.common.artifacts import EvaluationArtifactWriter
from evals.common.backends import backend_components
from evals.common.evaluator import (
    EvaluationConfig,
    EvaluationRunner,
)
from evals.common.suite import TaskSuite
from worldscape_policy.native_builder import (
    build_wan22_policy_from_checkpoint,
    checkpoint_mode,
    checkpoint_supports_mode,
)
from worldscape_policy.types import InteractionMode
from worldscape_policy.rollout.session import PolicyRuntime


def main(argv: list[str] | None = None) -> int:
    log_level_name = os.environ.get("WSP_LOG_LEVEL", "INFO").upper()
    log_level = getattr(logging, log_level_name, logging.INFO)
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        force=True,
    )
    parser = argparse.ArgumentParser(
        prog="wsp-eval",
        description="Run a WorldScape task suite and write common artifacts",
    )
    parser.add_argument("--config", required=True, help="Evaluation YAML/JSON recipe")
    parser.add_argument(
        "--backend",
        choices=("hdf5", "libero", "agilex"),
        help="Override the recipe backend",
    )
    parser.add_argument("--checkpoint", help="Override checkpoint directory")
    parser.add_argument("--output-dir", help="Override artifact directory")
    parser.add_argument(
        "--live-hardware",
        action="store_true",
        help=(
            "Explicitly authorize physical AgileX commands; AgileX otherwise "
            "requires safe dry-run replay"
        ),
    )
    parser.add_argument(
        "overrides",
        nargs="*",
        help="Optional OmegaConf dot-list overrides applied after loading --config",
    )
    args = parser.parse_args(argv)

    config = _load_config(args.config, overrides=args.overrides)
    backend = args.backend or config.get("backend")
    checkpoint = args.checkpoint or config.get("checkpoint")
    output_dir = args.output_dir or config.get("output_dir")
    if backend not in {"hdf5", "libero", "agilex"}:
        parser.error("config backend must be hdf5, libero, or agilex")
    if not checkpoint:
        parser.error("checkpoint is required in the recipe or command line")
    if not output_dir:
        parser.error("output_dir is required in the recipe or command line")
    if args.live_hardware and backend != "agilex":
        parser.error("--live-hardware is valid only with backend=agilex")
    if backend == "agilex":
        from evals.agilex.evaluate import run_agilex_recipe

        try:
            summary = run_agilex_recipe(
                config,
                checkpoint=checkpoint,
                output_dir=output_dir,
                live_hardware=args.live_hardware,
            )
        except (TypeError, ValueError) as exc:
            parser.error(str(exc))
        print(
            f"Completed {summary['episodes']} episode(s); "
            f"success rate {summary['success_rate']:.3f}; artifacts: {output_dir}"
        )
        return 0

    # The builder below performs the single full checksum validation.
    checkpoint_primary_mode = checkpoint_mode(
        checkpoint, validate_artifacts=False
    )
    configured_mode = config.get("mode")
    mode = InteractionMode.parse(
        configured_mode
        if configured_mode is not None
        else checkpoint_primary_mode
    )
    if not checkpoint_supports_mode(checkpoint_primary_mode, mode):
        parser.error(
            f"checkpoint mode is {checkpoint_primary_mode.value!r}, "
            f"not requested {mode.value!r}"
        )
    device = str(config.get("device", "cuda"))
    visual_input_range = str(config.get("visual_input_range", "zero_one"))
    policy = build_wan22_policy_from_checkpoint(
        checkpoint,
        visual_input_range=visual_input_range,
        device=device,
        expected_mode=mode,
        validate_checkpoint_artifacts=bool(
            config.get("validate_checkpoint_artifacts", True)
        ),
    )
    environment, adapter = backend_components(str(backend), config)
    suite = TaskSuite.from_config(config)
    evaluation = EvaluationConfig(
        mode=mode,
        max_steps=int(config.get("max_steps", 100)),
        execution_timeout_s=(
            float(config["execution_timeout_s"])
            if config.get("execution_timeout_s") is not None
            else None
        ),
        device=device,
        control_frequency_hz=(
            float(config["control_frequency_hz"])
            if config.get("control_frequency_hz") is not None
            else None
        ),
    )
    seed = int(config.get("seed", 0))
    generator = torch.Generator(device=torch.device(device)).manual_seed(seed)
    result = EvaluationRunner(
        PolicyRuntime(policy),
        environment,
        adapter,
    ).run(suite, evaluation, generator=generator)
    effective_config = dict(config)
    effective_config.update(
        {
            "backend": backend,
            "checkpoint": str(checkpoint),
            "output_dir": str(output_dir),
            "mode": mode.value,
        }
    )
    summary = EvaluationArtifactWriter(
        output_dir,
        config_format=str(config.get("config_format", "yaml")),
        video_fps=int(config.get("video_fps", 10)),
    ).write(effective_config, result)
    print(
        f"Completed {summary['episodes']} episode(s); "
        f"success rate {summary['success_rate']:.3f}; artifacts: {output_dir}"
    )
    return 0


def _load_config(
    path: str | Path,
    *,
    overrides: list[str] | None = None,
) -> dict[str, Any]:
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"Evaluation config does not exist: {source}")
    try:
        from omegaconf import OmegaConf
    except ImportError as exc:
        raise ImportError("Evaluation recipes require the core hydra/omegaconf extra") from exc
    from worldscape_policy.cli.config_profiles import resolve_config_profiles

    if source.suffix.lower() == ".json":
        config = OmegaConf.create(json.loads(source.read_text()))
    else:
        config = OmegaConf.load(source)
    dotlist = OmegaConf.from_dotlist(overrides) if overrides else None
    resolved = resolve_config_profiles(config, overrides=dotlist)
    value = OmegaConf.to_container(resolved, resolve=True)
    if not isinstance(value, dict):
        raise TypeError("Evaluation config must contain a mapping")
    return value


if __name__ == "__main__":
    raise SystemExit(main())
