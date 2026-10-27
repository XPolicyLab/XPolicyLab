"""Single trainer for every retained ME_U0 experiment."""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, Optional

import torch
from accelerate.utils import set_seed
from torch.optim import Optimizer
from torch.utils.data import DataLoader, RandomSampler

from leap.core.config import instantiate
from leap.core.event import Event
from leap.data.world_unified.builder import build_world_unified_dataset
from leap.models.builder import (
    is_accelerate_deepspeed_checkpoint,
    load_deepspeed_model_checkpoint,
    load_pretrained_weights,
)
from leap.trainers.base_trainer import BaseTrainer


logger = logging.getLogger(__name__)

_ACTION_PROJECTION_KEYS = (
    "action_in",
    "action_out",
    "action_modality_embed",
    "state_in",
    "domain_embed",
)


def parse_config_int(value: Any, name: str, *, min_value: int = 0) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be an integer, got boolean")
    if isinstance(value, str):
        value = value.strip().replace(",", "").replace("_", "")
        try:
            value = int(value)
        except ValueError as exc:
            raise ValueError(f"{name} must be an integer") from exc
    elif isinstance(value, float) and value.is_integer():
        value = int(value)
    if not isinstance(value, int) or value < min_value:
        raise ValueError(f"{name} must be an integer >= {min_value}, got {value!r}")
    return value


def _set_epoch_recursive(value: Any, epoch: int) -> None:
    seen: set[int] = set()

    def visit(current: Any) -> None:
        if current is None or id(current) in seen:
            return
        seen.add(id(current))
        setter = getattr(current, "set_epoch", None)
        if callable(setter):
            setter(int(epoch))
        for attribute in (
            "dataset",
            "batch_sampler",
            "base_sampler",
            "sampler",
            "base_dataloader",
        ):
            visit(getattr(current, attribute, None))

    visit(value)


class MachEmbodiedUnifiedTrainer(BaseTrainer):
    """Map-style ME_U0 post-training with the canonical batch contract."""

    def __init__(self, cfg: Any, optimizer: Optional[Optimizer] = None) -> None:
        self.data_style = "map_style"
        self.batch_contract = "canonical_subtask"
        self._init_canonical(cfg, optimizer)

    def _init_canonical(
        self, cfg: Any, optimizer: Optional[Optimizer] = None
    ) -> None:
        if optimizer is not None:
            raise ValueError("ME_U0 canonical routes construct their own optimizer")
        self._deferred_pretrained: tuple[str, bool] | None = None
        training_seed = parse_config_int(
            cfg.training.seed, "training.seed", min_value=0
        )
        set_seed(training_seed, device_specific=False)
        BaseTrainer.__init__(self, cfg)
        if self._deferred_pretrained is not None:
            path, strict = self._deferred_pretrained
            self.strategy.print(
                f"[ME_U0] DeepSpeed model-only load: {path} (strict={strict})",
                flush=True,
            )
            load_deepspeed_model_checkpoint(self.model, path, strict=strict)
            self.strategy.wait_for_everyone()
        set_seed(
            training_seed + int(self.strategy.process_index),
            device_specific=False,
        )
        self.event_bus.subscribe(Event.EPOCH_START, self._keep_frozen_modules_eval)
        self._keep_frozen_modules_eval(self.state)


    def _build_model(self) -> torch.nn.Module:
        model_cfg = self.cfg.get("model", {})
        pretrained = model_cfg.get("pretrained_pth")
        model = instantiate(model_cfg).to(torch.bfloat16)
        if pretrained:
            strict = bool(model_cfg.get("pretrained_strict", True))
            if not strict:
                raise ValueError("ME_U0 raw checkpoint loading must be strict")
            if (
                int(os.environ.get("WORLD_SIZE", "1")) > 1
                and is_accelerate_deepspeed_checkpoint(str(pretrained))
            ):
                self._deferred_pretrained = (str(pretrained), True)
            else:
                self.strategy.print(
                    f"[ME_U0] strict-loading raw checkpoint: {pretrained}",
                    flush=True,
                )
                load_pretrained_weights(model, str(pretrained), strict=True)
        return model

    def build_optimizer(self, cfg: Dict[str, Any]) -> torch.optim.Optimizer:
        values = dict(cfg)
        values.pop("type", None)
        base_lr = float(values.pop("lr", 5e-5))
        action_lr = float(values.pop("action_lr", base_lr))
        vit_lr = float(values.pop("vit_lr", base_lr))
        betas = tuple(values.pop("betas", (0.9, 0.95)))
        weight_decay = float(values.pop("weight_decay", 0.0))
        backbone, vit, continuous = [], [], []
        for name, parameter in self.model.named_parameters():
            if not parameter.requires_grad:
                continue
            if "lance.vit_model." in name:
                target = vit
            elif any(key in name for key in _ACTION_PROJECTION_KEYS):
                target = continuous
            else:
                target = backbone
            target.append(parameter)
        groups = []
        if backbone:
            groups.append({"params": backbone, "lr": base_lr})
        if vit:
            groups.append({"params": vit, "lr": vit_lr})
        if continuous:
            groups.append({"params": continuous, "lr": action_lr})
        return torch.optim.AdamW(
            groups, betas=betas, weight_decay=weight_decay, **values
        )

    def _build_train_dataloader(self) -> DataLoader:
        """Build the map-style dataset and its global batch sampler."""

        data_cfg = self.cfg.data
        raw_dataset = instantiate(data_cfg.train)
        world_unified_cfg = data_cfg.get("world_unified", {}) or {}
        dataset = build_world_unified_dataset(
            raw_dataset,
            max_action_dim=int(world_unified_cfg.get("max_action_dim", 26)),
            max_state_dim=int(world_unified_cfg.get("max_state_dim", 26)),
        )
        collate_fn = instantiate(data_cfg.collate_fn)
        workers = int(data_cfg.get("num_workers", 4))
        base_sampler = RandomSampler(dataset)
        batch_sampler = instantiate(
            data_cfg.batch_sampler,
            base_sampler=base_sampler,
            sample_type_fn=dataset.get_source,
        )
        kwargs: Dict[str, Any] = {
            "dataset": dataset,
            "batch_sampler": batch_sampler,
            "num_workers": workers,
            "collate_fn": collate_fn,
            "pin_memory": torch.cuda.is_available(),
            "persistent_workers": workers > 0,
            "timeout": int(
                data_cfg.get("dataloader_timeout_seconds", 1800)
            ),
        }
        if workers > 0:
            kwargs["prefetch_factor"] = int(
                data_cfg.get("prefetch_factor", 2)
            )
        return DataLoader(**kwargs)

    def _keep_frozen_modules_eval(self, _state: Any) -> None:
        model = self.strategy.unwrap_model(self.model)
        lance = getattr(model, "lance", None)
        vit = getattr(lance, "vit_model", None) if lance is not None else None
        if vit is not None and bool(getattr(model.config, "freeze_vit", True)):
            vit.eval()
        vae_wrapper = getattr(model, "vae", None)
        # WanVideoVAE is a plain wrapper around ``WanVAE``, whose actual
        # nn.Module lives at ``model.vae.vae.model``. Keep a direct fallback
        # for compatible wrappers without claiming the wrong object was set to
        # eval mode.
        vae_container = getattr(vae_wrapper, "vae", vae_wrapper)
        vae_model = getattr(vae_container, "model", None)
        if vae_model is not None and hasattr(vae_model, "eval"):
            vae_model.eval()


    def _forward_step(self, batch: Any) -> Dict[str, torch.Tensor]:
        if not isinstance(batch, dict):
            raise TypeError("ME_U0 canonical batches require a dictionary")
        with torch.autocast(
            "cuda", dtype=torch.bfloat16, enabled=torch.cuda.is_available()
        ):
            output = self.model(batch)
        for key in ("loss", "loss_video", "loss_action"):
            if key not in output:
                raise KeyError(f"ME_U0 model output is missing {key}")
        return output


    def _train_epoch(self) -> None:
        _set_epoch_recursive(self.train_dataloader, self.state.epoch)
        BaseTrainer._train_epoch(self)
        return


__all__ = [
    "MachEmbodiedUnifiedTrainer",
    "parse_config_int",
]
