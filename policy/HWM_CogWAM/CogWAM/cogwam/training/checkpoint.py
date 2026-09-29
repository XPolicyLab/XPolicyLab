"""Checkpoint discovery, save and restore.

Two artifacts are written per save, and they serve different consumers:

* ``steps_<N>_pytorch_model.pt`` (or ``steps_<N>_model.safetensors``) is the
  portable, engine-independent weight file. Evaluation, serving and the next
  fine-tuning stage read this one.
* ``steps_<N>_state/`` is Accelerate's full training state -- optimizer moments,
  the LR schedule, the DeepSpeed shards. Only a resume of *this* run reads it,
  and only at the same world size.

``_SUCCESS`` inside the state directory is the commit marker. It is removed
before a rewrite and touched last, so a preemption mid-save can never leave a
partial optimizer shard looking resumable. ``trainer_extras.pt`` alongside it
carries the loop state Accelerate does not own -- the two dataloader epoch
counters. It is written by rank 0 and read by every rank, so only
rank-invariant values may go in it: seeding all ranks from one rank's RNG
snapshot would collapse their sampling streams onto each other.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

import torch
from accelerate.logging import get_logger
from omegaconf import OmegaConf

from cogwam.training.config_tracker import AccessTrackedConfig

logger = get_logger(__name__)

_FULL_STATE_PATTERN = re.compile(r"steps_(\d+)_state")


class CheckpointMixin:
    """Save/restore half of the trainer, kept separate from the training loop."""

    def _save_initial_configs(self):
        """Save full config and training script at the very start of training."""
        if not self.accelerator.is_main_process:
            return

        output_dir = Path(self.config.output_dir)

        # 1. Save config.full.yaml — the complete merged config (all parameters)
        if isinstance(self.config, AccessTrackedConfig):
            full_cfg = self.config.unwrap()
        else:
            full_cfg = self.config
        full_yaml_path = output_dir / "config.full.yaml"
        OmegaConf.save(full_cfg, full_yaml_path, resolve=True)
        logger.info(f"📝 Full config saved at {full_yaml_path}")

        # 2. Save config.yaml — accessed-only snapshot (will be updated at checkpoints)
        if isinstance(self.config, AccessTrackedConfig):
            self.config.save_accessed_config(output_dir / "config.yaml", use_original_values=False)
            logger.info(f"📊 Accessed config snapshot saved at {output_dir / 'config.yaml'}")

    def _init_checkpointing(self):
        """Initialize checkpoint directory and handle checkpoint loading."""
        self.checkpoint_dir = os.path.join(self.config.output_dir, "checkpoints")
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        pretrained_checkpoint = getattr(self.config.trainer, "pretrained_checkpoint", None)
        is_resume = bool(getattr(self.config.trainer, "is_resume", False))
        self.resume_from_checkpoint = pretrained_checkpoint

        if is_resume:
            explicit_state = getattr(self.config.trainer, "resume_state_path", None)
            if explicit_state:
                state_path, state_step = self._validate_full_state_checkpoint(explicit_state)
            else:
                state_path, state_step = self._get_latest_full_state_checkpoint(self.checkpoint_dir)
            if state_path:
                self._pending_full_state_path = state_path
                self.resume_from_checkpoint = state_path
                self.completed_steps = state_step
                logger.info(
                    "Preparing full optimizer/scheduler resume from %s at step %d",
                    state_path,
                    state_step,
                )
                return

            # Backward-compatible fallback for historical weight-only runs.
            # This cannot restore Adam moments; the scheduler is reconstructed
            # deterministically below and the limitation is made explicit.
            resume_from_checkpoint, self.completed_steps = self._get_latest_checkpoint(self.checkpoint_dir)
            if resume_from_checkpoint:
                self.resume_from_checkpoint = resume_from_checkpoint
                self.model = self.load_pretrained_backbones(self.model, self.resume_from_checkpoint, reload_modules=None)
                logger.info(
                    "Legacy weight-only resume from %s at step %d; optimizer moments are unavailable",
                    self.resume_from_checkpoint,
                    self.completed_steps,
                )
                return

            logger.warning(f"No valid checkpoint found in {self.checkpoint_dir}. Starting training from scratch.")
            self.completed_steps = 0

        if pretrained_checkpoint:
            reload_modules = getattr(self.config.trainer, "reload_modules", None)
            pretrained_strict = bool(getattr(self.config.trainer, "pretrained_strict", False))
            self.model = self.load_pretrained_backbones(
                self.model,
                pretrained_checkpoint,
                reload_modules=reload_modules,
                strict=pretrained_strict,
            )
            self.completed_steps = 0
            self.resume_from_checkpoint = pretrained_checkpoint
            logger.info(f"Loaded pretrained checkpoint: {pretrained_checkpoint}, steps: {self.completed_steps}")
        else:
            logger.info("No pretrained checkpoint provided. Starting training from scratch.")
            self.completed_steps = 0

    def _adjust_lr_scheduler_for_resume(self):
        """Adjust LR scheduler state after resuming from non-zero steps."""
        if self._pending_full_state_path is not None:
            return
        if self.completed_steps > 0:
            logger.info(f"Adjusting LR scheduler for resume from step {self.completed_steps}")
            for _ in range(self.completed_steps):
                self.lr_scheduler.step()
            logger.info(
                f"LR scheduler adjusted to step {self.completed_steps}, current LR: {self.lr_scheduler.get_last_lr()}"
            )

    @staticmethod
    def _full_state_metadata_path(state_path: str | Path) -> Path:
        return Path(state_path) / "trainer_state.json"

    @classmethod
    def _validate_full_state_checkpoint(cls, state_path: str | Path) -> tuple[str | None, int]:
        state_path = Path(state_path)
        success_path = state_path / "_SUCCESS"
        metadata_path = cls._full_state_metadata_path(state_path)
        if not state_path.is_dir() or not success_path.is_file() or not metadata_path.is_file():
            raise FileNotFoundError(
                f"Incomplete full training-state checkpoint: {state_path}; expected trainer_state.json and _SUCCESS"
            )
        with metadata_path.open("r", encoding="utf-8") as handle:
            metadata = json.load(handle)
        step = int(metadata["steps"])
        match = _FULL_STATE_PATTERN.fullmatch(state_path.name)
        if match is not None and int(match.group(1)) != step:
            raise ValueError(f"Full-state checkpoint step mismatch: directory={state_path.name}, metadata={step}")
        return str(state_path), step

    @classmethod
    def _get_latest_full_state_checkpoint(cls, checkpoint_dir: str | Path) -> tuple[str | None, int]:
        checkpoint_dir = Path(checkpoint_dir)
        if not checkpoint_dir.is_dir():
            return None, 0
        candidates: list[tuple[int, str]] = []
        for state_path in checkpoint_dir.iterdir():
            if not state_path.is_dir() or _FULL_STATE_PATTERN.fullmatch(state_path.name) is None:
                continue
            try:
                validated_path, step = cls._validate_full_state_checkpoint(state_path)
            except (FileNotFoundError, KeyError, TypeError, ValueError, json.JSONDecodeError):
                continue
            candidates.append((step, validated_path))
        if not candidates:
            return None, 0
        step, state_path = max(candidates, key=lambda item: item[0])
        return state_path, step

    def _restore_pending_full_training_state(self) -> None:
        state_path = self._pending_full_state_path
        if state_path is None:
            return
        metadata_path = self._full_state_metadata_path(state_path)
        with metadata_path.open("r", encoding="utf-8") as handle:
            metadata = json.load(handle)
        saved_world_size = int(metadata.get("world_size", self.accelerator.num_processes))
        if saved_world_size != int(self.accelerator.num_processes):
            raise RuntimeError(
                "DeepSpeed full-state resume requires the saved data-parallel world size: "
                f"saved={saved_world_size}, current={self.accelerator.num_processes}"
            )
        saved_accumulation = int(
            metadata.get(
                "gradient_accumulation_steps",
                self.accelerator.gradient_accumulation_steps,
            )
        )
        if saved_accumulation != int(self.accelerator.gradient_accumulation_steps):
            raise RuntimeError(
                "Full-state resume requires the saved gradient accumulation setting: "
                f"saved={saved_accumulation}, current={self.accelerator.gradient_accumulation_steps}"
            )
        self.accelerator.load_state(input_dir=state_path)
        extras_path = Path(state_path) / "trainer_extras.pt"
        if extras_path.is_file():
            extras = torch.load(extras_path, map_location="cpu", weights_only=False)
            self.vla_epoch_count = int(extras.get("vla_epoch_count", 0))
            self.semantic_epoch_count = int(extras.get("semantic_epoch_count", 0))
        logger.info(
            "Restored full training state from %s at step %d; current LR=%s",
            state_path,
            self.completed_steps,
            self.lr_scheduler.get_last_lr(),
        )

    def _save_checkpoint(self):
        """Save current training state."""
        checkpoint_path = os.path.join(self.checkpoint_dir, f"steps_{self.completed_steps}")
        if self.accelerator.is_main_process:
            save_format = getattr(self.config.trainer, "save_format", "pt")

            state_dict = self.accelerator.get_state_dict(self.model)
            if save_format == "safetensors":
                from safetensors.torch import save_file

                save_file(state_dict, checkpoint_path + "_model.safetensors")
            elif save_format == "pt":
                torch.save(state_dict, checkpoint_path + "_pytorch_model.pt")
            else:
                raise ValueError(f"Unsupported save_format `{save_format}`. Expected `pt` or `safetensors`.")

            summary_data = {"steps": self.completed_steps}
            with open(os.path.join(self.config.output_dir, "summary.jsonl"), "a") as f:
                f.write(json.dumps(summary_data) + "\n")
            self.accelerator.print(f"✅ Checkpoint saved at {checkpoint_path}")

            if isinstance(self.config, AccessTrackedConfig):
                logger.info("📊 Saving accessed configuration...")
                output_dir = Path(self.config.output_dir)
                self.config.save_accessed_config(output_dir / "config.yaml", use_original_values=False)
                logger.info("✅ Configuration files saved")

        # Do not let non-main ranks enter DeepSpeed's collective state save
        # while rank 0 is still materializing the portable weight checkpoint.
        self.accelerator.wait_for_everyone()
        if bool(getattr(self.config.trainer, "save_full_training_state", False)):
            full_state_path = checkpoint_path + "_state"
            # `_SUCCESS` is the commit marker.  Remove it before rewriting an
            # existing step so a preemption during save can never make a
            # partial optimizer shard look resumable.
            if self.accelerator.is_main_process:
                (Path(full_state_path) / "_SUCCESS").unlink(missing_ok=True)
            self.accelerator.wait_for_everyone()
            self.accelerator.save_state(output_dir=full_state_path)
            self.accelerator.wait_for_everyone()
            if self.accelerator.is_main_process:
                metadata = {
                    "steps": int(self.completed_steps),
                    "world_size": int(self.accelerator.num_processes),
                    "gradient_accumulation_steps": int(self.accelerator.gradient_accumulation_steps),
                }
                with self._full_state_metadata_path(full_state_path).open("w", encoding="utf-8") as handle:
                    json.dump(metadata, handle, ensure_ascii=True, indent=2)
                torch.save(
                    {
                        "vla_epoch_count": int(getattr(self, "vla_epoch_count", 0)),
                        "semantic_epoch_count": int(getattr(self, "semantic_epoch_count", 0)),
                    },
                    Path(full_state_path) / "trainer_extras.pt",
                )
                (Path(full_state_path) / "_SUCCESS").touch()
                logger.info("Saved resumable optimizer/scheduler state at %s", full_state_path)

        self.accelerator.wait_for_everyone()

    def _save_final_model(self):
        """Write the end-of-training weights outside the step-numbered series."""
        if not self.accelerator.is_main_process:
            return
        save_format = getattr(self.config.trainer, "save_format", "pt")
        final_checkpoint = os.path.join(self.config.output_dir, "final_model")
        os.makedirs(final_checkpoint, exist_ok=True)
        state_dict = self.accelerator.get_state_dict(self.model)
        if save_format == "safetensors":
            from safetensors.torch import save_file

            save_file(state_dict, os.path.join(final_checkpoint, "model.safetensors"))
        elif save_format == "pt":
            torch.save(state_dict, os.path.join(final_checkpoint, "pytorch_model.pt"))
        else:
            raise ValueError(f"Unsupported save_format `{save_format}`. Expected `pt` or `safetensors`.")
        logger.info(f"Training complete. Final model saved at {final_checkpoint}")


__all__ = ["CheckpointMixin"]
