# coding=utf-8
"""Inference presets built from public OpenPI configuration classes."""

from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..policy import PolicyConfig

ROBODOJO_PI05_CONFIG = "pi05_base_aloha_full_sim_arx-x5_seed_0"
ARX_DIMS = (6, 1, 6, 1)
ARX_CAMERAS = {"cam_high", "cam_left_wrist", "cam_right_wrist"}


def layout_errors(config: PolicyConfig) -> list[str]:
    if config.train_config_name != ROBODOJO_PI05_CONFIG:
        return []
    from ..policy import ARX_FIELDS

    errors = []
    for prefix in ("state", "action"):
        if (
            getattr(config, prefix + "_fields") != ARX_FIELDS
            or getattr(config, prefix + "_dims") != ARX_DIMS
        ):
            errors.append(
                f"RoboDojo pi05 requires the ARX X5 joint {prefix} layout (6, 1, 6, 1)"
            )
    if set(config.camera_map) != ARX_CAMERAS:
        errors.append(
            "RoboDojo pi05 requires cam_high, cam_left_wrist and cam_right_wrist model cameras"
        )
    return errors


def get_inference_config(config: PolicyConfig):
    from openpi.training import config as training_config

    if config.train_config_name != ROBODOJO_PI05_CONFIG:
        if config.tokenizer_path is not None:
            raise ValueError("tokenizer_path is supported by the RoboDojo pi05 preset")
        try:
            return training_config.get_config(config.train_config_name)
        except ValueError:
            raise ValueError(
                f"OpenPI configuration {config.train_config_name!r} is not registered; "
                "use a supported OrimaPI preset or an official OpenPI configuration"
            ) from None

    errors = layout_errors(config)
    if errors:
        raise ValueError("; ".join(errors))
    if config.checkpoint is None or not config.normalization_id:
        raise ValueError("RoboDojo pi05 requires checkpoint and normalization_id")
    if config.tokenizer_path is None or not config.tokenizer_path.is_file():
        raise ValueError("Provide a local tokenizer model with --tokenizer-path")

    from openpi.models.pi0_config import Pi0Config

    # Match the published RoboDojo ARX X5 recipe without changing OpenPI's registry.
    return training_config.TrainConfig(
        name=ROBODOJO_PI05_CONFIG,
        model=Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=50,
            max_token_len=200,
            discrete_state_input=True,
        ),
        data=training_config.SimpleDataConfig(
            assets=training_config.AssetsConfig(
                assets_dir=str(config.checkpoint / "assets"),
                asset_id=config.normalization_id,
            ),
            data_transforms=_robodojo_transforms,
            model_transforms=partial(
                _local_model_transforms, tokenizer_path=config.tokenizer_path
            ),
        ),
    )


def _robodojo_transforms(model_config):
    from openpi import transforms
    from openpi.policies.aloha_policy import AlohaInputs, AlohaOutputs

    mask = transforms.make_bool_mask(6, -1, 6, -1)
    return transforms.Group(
        inputs=[AlohaInputs(adapt_to_pi=False), transforms.DeltaActions(mask)],
        outputs=[transforms.AbsoluteActions(mask), AlohaOutputs(adapt_to_pi=False)],
    )


def _local_model_transforms(model_config, *, tokenizer_path):
    from openpi import transforms

    from .openpi_tokenizer import LocalPaligemmaTokenizer

    return transforms.Group(
        inputs=[
            transforms.InjectDefaultPrompt(None),
            transforms.ResizeImages(224, 224),
            transforms.TokenizePrompt(
                LocalPaligemmaTokenizer(model_config.max_token_len, tokenizer_path),
                discrete_state_input=model_config.discrete_state_input,
            ),
            transforms.PadStatesAndActions(model_config.action_dim),
        ]
    )


def validate_norm_stats(config: PolicyConfig, stats: dict) -> None:
    if config.train_config_name != ROBODOJO_PI05_CONFIG:
        return

    import numpy as np

    for name in ("state", "actions"):
        value = stats.get(name)
        for field in ("mean", "std", "q01", "q99"):
            array = np.asarray(getattr(value, field, None))
            if array.shape != (14,) or not np.isfinite(array).all():
                raise ValueError(
                    f"RoboDojo pi05 normalization requires finite {name}.{field} with shape (14,)"
                )
        if np.any(value.std < 0) or np.any(value.q99 < value.q01):
            raise ValueError(f"Invalid RoboDojo pi05 normalization ranges: {name}")
