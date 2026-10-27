from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf

logger = logging.getLogger(__name__)

WANDB_RUN_ID_FILE = ".wandb_run_id"


def _none_if_empty(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if text == "" or text.lower() == "null":
        return None
    return text


def _wandb_cfg_get(cfg: DictConfig | None, key: str, default: Any = None) -> Any:
    if cfg is None or "wandb" not in cfg or cfg.wandb is None:
        return default
    return cfg.wandb.get(key, default)


def resolve_train_wandb_run_id(
    output_dir: str | os.PathLike[str], cfg: DictConfig
) -> str | None:
    run_id = _none_if_empty(_wandb_cfg_get(cfg, "run_id"))
    if run_id is not None:
        return run_id

    run_id_file = Path(output_dir) / WANDB_RUN_ID_FILE
    if run_id_file.exists():
        return _none_if_empty(run_id_file.read_text(encoding="utf-8"))
    return None


def resolve_train_wandb_resume(
    output_dir: str | os.PathLike[str], cfg: DictConfig
) -> str:
    run_id = resolve_train_wandb_run_id(output_dir, cfg)
    configured_resume = _none_if_empty(_wandb_cfg_get(cfg, "resume"))
    if configured_resume is not None:
        return configured_resume
    return "allow" if run_id is not None else "never"


def save_wandb_run_id(
    output_dir: str | os.PathLike[str],
    cfg: DictConfig,
    run: Any,
) -> None:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    run_id = str(run.id)
    run_url = getattr(run, "url", None)
    (output_path / WANDB_RUN_ID_FILE).write_text(run_id, encoding="utf-8")

    OmegaConf.update(cfg, "wandb.run_id", run_id, merge=True, force_add=True)
    if run_url is not None:
        OmegaConf.update(cfg, "wandb.url", str(run_url), merge=True, force_add=True)

    config_path = output_path / "config.yaml"
    if config_path.exists():
        saved_cfg = OmegaConf.load(config_path)
        OmegaConf.update(saved_cfg, "wandb.run_id", run_id, merge=True, force_add=True)
        if run_url is not None:
            OmegaConf.update(
                saved_cfg, "wandb.url", str(run_url), merge=True, force_add=True
            )
        OmegaConf.save(saved_cfg, str(config_path))
