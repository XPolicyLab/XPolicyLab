# coding=utf-8
"""Lazy loading of a caller-supplied, compatible OpenPI checkpoint."""

from __future__ import annotations

import hashlib
import importlib.util
from typing import TYPE_CHECKING

from .openpi_configs import (
    ROBODOJO_PI05_CONFIG,
    get_inference_config,
    layout_errors,
    validate_norm_stats,
)

if TYPE_CHECKING:
    from ..policy import PolicyConfig


def asset_errors(config: PolicyConfig) -> list[str]:
    errors = layout_errors(config)
    if config.checkpoint is None:
        errors.append(
            "Provide --checkpoint or policies.<id>.checkpoint (the exact checkpoint directory)"
        )
    elif not (config.checkpoint / "params").is_dir():
        errors.append(
            f"Missing checkpoint params directory: {config.checkpoint / 'params'}"
        )
    if not config.train_config_name:
        errors.append("Provide --train-config or policies.<id>.train_config_name")
    if config.tokenizer_path is not None:
        if not config.tokenizer_path.is_file():
            errors.append(f"Missing tokenizer model: {config.tokenizer_path}")
        if config.train_config_name != ROBODOJO_PI05_CONFIG:
            errors.append("tokenizer_path is supported by the RoboDojo pi05 preset")
    elif config.train_config_name == ROBODOJO_PI05_CONFIG:
        errors.append("Provide --tokenizer-path or policies.<id>.tokenizer_path")
    if not config.normalization_id:
        errors.append("Provide --normalization-id or policies.<id>.normalization_id")
    elif config.checkpoint is not None:
        path = (
            config.checkpoint / "assets" / config.normalization_id / "norm_stats.json"
        )
        if not path.is_file():
            errors.append(f"Missing normalization assets: {path}")
    return errors


def require_assets(config: PolicyConfig) -> None:
    errors = asset_errors(config)
    if errors:
        raise ValueError("; ".join(errors))


def load_openpi(config: PolicyConfig):
    require_assets(config)
    try:
        from openpi.policies import policy_config
        from openpi.shared import normalize

        model_config = get_inference_config(config)
    except ImportError:
        raise RuntimeError(
            "Install third_party/openpi in the policy Python environment; see doc/python-environment.md"
        ) from None
    if not getattr(getattr(model_config, "model", None), "pi05", False):
        raise ValueError("The supplied OpenPI configuration does not select pi0.5")
    norm_stats = normalize.load(config.checkpoint / "assets" / config.normalization_id)
    validate_norm_stats(config, norm_stats)
    return policy_config.create_trained_policy(
        model_config, str(config.checkpoint), norm_stats=norm_stats
    )


def inspect_skill(config: PolicyConfig) -> dict:
    """Check local assets/dependency presence without loading weights or code."""
    errors = []
    required = ["numpy"]
    if config.name == "code-policy":
        if config.code_skills:
            from .code_bundle import code_identity

            try:
                code_identity(config)
            except (ValueError, TypeError, OSError) as error:
                errors.append(str(error))
        elif config.code_path is None:
            errors.append("Provide --code-path or policies.<id>.code_path")
        elif not config.code_path.is_file() or config.code_path.suffix != ".py":
            errors.append(f"Expected a Python program file: {config.code_path}")
        elif (
            config.code_sha256
            and hashlib.sha256(config.code_path.read_bytes()).hexdigest()
            != config.code_sha256
        ):
            errors.append("Code policy source SHA-256 mismatch")
    else:
        errors.extend(asset_errors(config))
        required.extend(["PIL", "openpi"])
    dependencies = {}
    for name in required:
        try:
            dependencies[name] = importlib.util.find_spec(name) is not None
        except (ValueError, ImportError):
            dependencies[name] = False
        if not dependencies[name]:
            errors.append(f"Missing dependency: {name}")
    return {
        "skill": config.name,
        "adapter": "implemented",
        "ready_for_load": not errors,
        "dependencies": dependencies,
        "errors": errors,
        "loaded": False,
        "scope": "Local asset/dependency presence only; model compatibility and execution are not verified",
    }
