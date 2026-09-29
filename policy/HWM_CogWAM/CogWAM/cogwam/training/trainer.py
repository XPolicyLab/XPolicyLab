"""The dual-stream CogWAM training loop.

Every optimizer step consumes two batches:

* the **physical** stream (12 frames/device) drives the action and world
  flow-matching objectives, and
* the **semantic** stream (6 frames/device, balanced 2 UPDATE / 2 hard KEEP /
  2 random KEEP) drives the ``<KEEP>``/``<UPDATE>`` next-token objective that
  keeps the shared VLM's language head alive.

The model fuses both into a single scalar returned as ``action_loss``; the
trainer never adds the two itself, because the semantic term is already scaled
by ``text_loss_weight`` inside the framework.
"""

from __future__ import annotations

import os
import time

import torch
import torch.distributed as dist
import wandb
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from tqdm import tqdm

from cogwam.training.checkpoint import CheckpointMixin
from cogwam.training.utils import (
    TrainerUtils,
    build_grad_norm_groups,
    group_grad_local_sqnorm,
)

logger = get_logger(__name__)


def collect_loss_metrics(output_dict: dict, total_loss: torch.Tensor) -> dict:
    """Expose weighted training objectives plus explicitly named raw losses."""

    action_raw = output_dict["mot_action_loss_raw"].detach()
    world_raw = output_dict["mot_world_loss_raw"].detach()
    text_raw = output_dict["mot_text_loss_raw"].detach()
    action_weighted = output_dict.get("mot_action_loss_weighted", action_raw).detach()
    world_weighted = output_dict.get("mot_world_loss_weighted", world_raw).detach()
    text_weighted = output_dict.get("mot_text_loss_weighted", text_raw).detach()
    metrics = {
        "train/task": "world_action_mot",
        "train/loss_total": float(total_loss.detach()),
        # Un-suffixed curves are the actual weighted objectives used by
        # backward. Explicit raw curves remain available for loss scale/debug
        # audits.
        "train/action_loss": float(action_weighted),
        "train/world_loss": float(world_weighted),
        "train/text_loss": float(text_weighted),
        "train/action_loss_raw": float(action_raw),
        "train/world_loss_raw": float(world_raw),
        "train/text_loss_raw": float(text_raw),
        "train/action_loss_weighted": float(action_weighted),
        "train/world_loss_weighted": float(world_weighted),
        "train/text_loss_weighted": float(text_weighted),
        "train/text_sample_count": float(output_dict["mot_text_sample_count"].detach()),
    }
    for source_key, metric_key in (
        ("mot_world_loss_weight", "train/world_loss_weight"),
        ("mot_text_loss_weight", "train/text_loss_weight"),
        # The two halves saturate for different reasons and must be read apart:
        # the decision term is one token of binary KEEP/UPDATE and is expected
        # to bottom out early, while the body term is full autoregressive
        # generation of "Memory Add: ... Current Subtask: ..." on the 2 UPDATE
        # rows of each 6-row semantic batch. Only the body term collapsing
        # means the auxiliary objective has stopped constraining the backbone.
        ("mot_text_decision_loss", "train/text_decision_loss"),
        ("mot_text_update_body_loss", "train/text_update_body_loss"),
        ("mot_text_decision_accuracy", "train/text_decision_accuracy"),
        ("mot_text_update_count", "train/text_update_count"),
        ("mot_text_scheduled_probability", "train/text_scheduled_probability"),
        ("mot_text_scheduled_count", "train/text_scheduled_count"),
        ("mot_text_scheduled_update_count", "train/text_scheduled_update_count"),
        ("mot_text_scheduled_fallback_count", "train/text_scheduled_fallback_count"),
    ):
        value = output_dict.get(source_key)
        if value is not None:
            metrics[metric_key] = float(value.detach())
    return metrics


class CogWAMTrainer(TrainerUtils, CheckpointMixin):
    """Native PyTorch + Accelerate + DeepSpeed loop, kept explicit and hackable."""

    def __init__(
        self,
        cfg,
        model,
        vla_train_dataloader,
        optimizer,
        lr_scheduler,
        accelerator,
        semantic_train_dataloader=None,
    ):
        self.config = cfg
        self.model = model
        self.vla_train_dataloader = vla_train_dataloader
        self.semantic_train_dataloader = semantic_train_dataloader
        self.semantic_batch_sampler = (
            semantic_train_dataloader.batch_sampler if semantic_train_dataloader is not None else None
        )
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.accelerator = accelerator

        self.completed_steps = 0
        self.vla_epoch_count = 0
        self.semantic_epoch_count = 0
        self._pending_full_state_path: str | None = None
        self.total_batch_size = self._calculate_total_batch_size()

        if bool(cfg.trainer.get("action_eval_enabled", False)):
            raise ValueError(
                "trainer.action_eval_enabled is not supported: RoboDojo evaluation carries "
                "persistent semantic memory across a rollout, so a stateless in-loop action-MSE "
                "probe measures the wrong quantity. Evaluate with cogwam.eval instead."
            )
        world_validation = cfg.trainer.get("world_validation", {})
        if hasattr(world_validation, "get") and bool(world_validation.get("enabled", False)):
            raise ValueError(
                "trainer.world_validation.enabled=true is not supported by this recipe; "
                "the released RoboDojo run has no held-out world-prediction split."
            )

        self._log_grad_norms = bool(cfg.trainer.get("log_grad_norms", False))
        # Cheap scalar norm for the main experiment panel.  Under DeepSpeed
        # this reads the engine's already-reduced norm and adds no collective;
        # per-head decomposition remains controlled separately above.
        self._log_global_grad_norm = bool(cfg.trainer.get("log_global_grad_norm", False))
        self._grad_groups = None
        self._using_deepspeed = "DEEPSPEED" in str(getattr(accelerator, "distributed_type", "")).upper()
        self._last_clip_norm = None
        self._grad_warned = False

        shared_vlm_diag_cfg = cfg.trainer.get("shared_vlm_gradient_diagnostics", {})
        self._shared_vlm_gradient_diagnostics_enabled = bool(
            hasattr(shared_vlm_diag_cfg, "get") and shared_vlm_diag_cfg.get("enabled", False)
        )
        self._shared_vlm_gradient_diagnostics_interval = int(
            shared_vlm_diag_cfg.get("interval", cfg.trainer.logging_frequency)
            if hasattr(shared_vlm_diag_cfg, "get")
            else cfg.trainer.logging_frequency
        )
        if self._shared_vlm_gradient_diagnostics_enabled:
            logging_frequency = int(cfg.trainer.logging_frequency)
            if self._shared_vlm_gradient_diagnostics_interval <= 0:
                raise ValueError("trainer.shared_vlm_gradient_diagnostics.interval must be positive")
            if self._shared_vlm_gradient_diagnostics_interval % logging_frequency != 0:
                raise ValueError(
                    "trainer.shared_vlm_gradient_diagnostics.interval must be "
                    "a multiple of trainer.logging_frequency so every diagnostic "
                    "is emitted to W&B"
                )
            if int(cfg.trainer.get("gradient_accumulation_steps", 1)) != 1:
                raise ValueError(
                    "shared-VLM interface gradient diagnostics currently require "
                    "trainer.gradient_accumulation_steps=1 so the measured query "
                    "gradient is exactly the optimizer step's full local batch"
                )

    # ------------------------------------------------------------------
    # setup
    # ------------------------------------------------------------------

    def prepare_training(self):
        rank = dist.get_rank() if dist.is_initialized() else 0
        seed = self.config.seed + rank if hasattr(self.config, "seed") else rank + 3047
        set_seed(seed)

        # Save config snapshots upfront so that even if a later setup step
        # (ckpt load / DeepSpeed init / dataloader build) crashes, the
        # produced run dir is still introspectable / from_pretrained-able.
        self._save_initial_configs()

        self._init_checkpointing()
        self._adjust_lr_scheduler_for_resume()

        freeze_modules = (
            self.config.trainer.freeze_modules
            if (self.config and hasattr(self.config.trainer, "freeze_modules"))
            else None
        )
        self.model = self.freeze_backbones(self.model, freeze_modules=freeze_modules)
        self.print_trainable_parameters(self.model)

        self._prepare_distributed_components()

        # The historical loop intentionally steps the raw Transformers
        # scheduler itself instead of wrapping it with Accelerator.  Register
        # that scheduler as a checkpointable object so save_state/load_state
        # also preserves its exact step and per-group LR without changing the
        # established stepping semantics.
        self.accelerator.register_for_checkpointing(self.lr_scheduler)
        if self.semantic_batch_sampler is not None:
            self.accelerator.register_for_checkpointing(self.semantic_batch_sampler)

        # Full optimizer/scheduler state can only be restored after Accelerate
        # has registered and wrapped every training object.
        self._restore_pending_full_training_state()

        self._validate_runtime_batch_contract()

        if self._log_grad_norms:
            try:
                base_model = self.accelerator.unwrap_model(self.model)
                self._grad_groups = build_grad_norm_groups(base_model)
                if self.accelerator.is_main_process:
                    logger.info(
                        "Grad-norm groups: %s (deepspeed=%s)",
                        {k: len(v) for k, v in self._grad_groups.items()},
                        self._using_deepspeed,
                    )
            except Exception as e:
                logger.warning("build_grad_norm_groups failed (%s); disabling per-head grad logging.", e)
                self._log_grad_norms = False
                self._grad_groups = None
        if self._shared_vlm_gradient_diagnostics_enabled and self.accelerator.is_main_process:
            logger.info(
                "Shared-VLM interface action/world gradient norms: interval=%d "
                "(weighted physical objectives; text loss excluded)",
                self._shared_vlm_gradient_diagnostics_interval,
            )

        self._init_wandb()

    def _prepare_distributed_components(self) -> None:
        """Initialize DeepSpeed from the physical loader, then shard the semantic one.

        Accelerate resolves DeepSpeed's ``train_micro_batch_size_per_gpu=auto``
        from every dataloader passed to one ``prepare`` call, and its default
        ``is_train_batch_min=True`` would pick the 6-sample semantic loader over
        the 12-sample physical one.  Prepare the engine with the physical loader
        alone; the semantic stream still needs distributed sharding, but must
        not participate in the engine's training-batch ABI.
        """

        prepared = self.setup_distributed_training(
            self.accelerator,
            self.model,
            self.optimizer,
            self.vla_train_dataloader,
        )
        self.model, self.optimizer, self.vla_train_dataloader = prepared
        if self.semantic_train_dataloader is not None:
            self.semantic_train_dataloader = self.accelerator.prepare_data_loader(self.semantic_train_dataloader)

    @staticmethod
    def _runtime_int(obj, name: str):
        value = getattr(obj, name, None)
        if value is None:
            return None
        value = value() if callable(value) else value
        return int(value)

    def _validate_runtime_batch_contract(self) -> None:
        """Fail immediately if Accelerate/DeepSpeed changed the YAML batch ABI."""

        expected_accumulation = int(self.config.trainer.gradient_accumulation_steps)
        active_accumulation = int(self.accelerator.gradient_accumulation_steps)
        if active_accumulation != expected_accumulation:
            raise RuntimeError(
                "Accelerate runtime accumulation differs from the training YAML: "
                f"yaml={expected_accumulation}, runtime={active_accumulation}."
            )

        expected_micro_batch = int(self.config.datasets.vla_data.per_device_batch_size)
        expected_global_batch = expected_micro_batch * int(self.accelerator.num_processes) * expected_accumulation
        declared_global_batch = self.config.trainer.get("expected_global_batch_size", None)
        if declared_global_batch is not None and expected_global_batch != int(declared_global_batch):
            raise RuntimeError(
                "Runtime topology violates trainer.expected_global_batch_size: "
                f"micro={expected_micro_batch} x world={self.accelerator.num_processes} x "
                f"accumulation={expected_accumulation} = {expected_global_batch}, "
                f"declared={int(declared_global_batch)}."
            )
        engine_values = {}
        if self._using_deepspeed:
            for name, expected in (
                ("gradient_accumulation_steps", expected_accumulation),
                ("train_micro_batch_size_per_gpu", expected_micro_batch),
                ("train_batch_size", expected_global_batch),
            ):
                active = self._runtime_int(self.model, name)
                if active is None:
                    raise RuntimeError(f"DeepSpeed engine does not expose {name}; cannot verify the batch contract.")
                engine_values[name] = active
                if active != expected:
                    raise RuntimeError(
                        f"DeepSpeed {name} differs from the training YAML/runtime topology: "
                        f"expected={expected}, active={active}."
                    )

        if self.accelerator.is_main_process:
            logger.info(
                "Verified runtime batch contract: micro=%d x world=%d x accumulation=%d = global=%d; deepspeed=%s",
                expected_micro_batch,
                self.accelerator.num_processes,
                expected_accumulation,
                expected_global_batch,
                engine_values or "disabled",
            )

    def _calculate_total_batch_size(self):
        """Calculate global batch size."""
        return (
            self.config.datasets.vla_data.per_device_batch_size
            * self.accelerator.num_processes
            * self.accelerator.gradient_accumulation_steps
        )

    def _init_wandb(self):
        """Initialize Weights & Biases."""
        if self.accelerator.is_main_process:
            wandb.init(
                name=self.config.run_id,
                dir=os.path.join(self.config.output_dir, "wandb"),
                project=self.config.wandb_project,
                group="vla-train",
            )

    # ------------------------------------------------------------------
    # data
    # ------------------------------------------------------------------

    def _create_data_iterators(self):
        """Create data iterators."""
        self.vla_iter = iter(self.vla_train_dataloader)
        if self.semantic_train_dataloader is not None:
            # The sampler epoch is checkpointed independently.  Seed the loop
            # counter from the restored value so the first exhaustion after a
            # resume advances e.g. 17 -> 18 instead of silently resetting to 1.
            self.semantic_epoch_count = int(getattr(self.semantic_batch_sampler, "epoch", self.semantic_epoch_count))
            self.semantic_iter = iter(self.semantic_train_dataloader)

    def _get_next_batch(self):
        """Get next physical batch (automatically handle data loop)."""
        try:
            batch_vla = next(self.vla_iter)
        except StopIteration:
            self.vla_iter, self.vla_epoch_count = TrainerUtils._reset_dataloader(
                self.vla_train_dataloader, self.vla_epoch_count
            )
            batch_vla = next(self.vla_iter)

        return batch_vla

    def _get_next_semantic_batch(self):
        if self.semantic_train_dataloader is None:
            return None
        try:
            return next(self.semantic_iter)
        except StopIteration:
            self.semantic_epoch_count += 1
            if self.semantic_batch_sampler is not None:
                self.semantic_batch_sampler.set_epoch(self.semantic_epoch_count)
            self.semantic_iter = iter(self.semantic_train_dataloader)
            return next(self.semantic_iter)

    # ------------------------------------------------------------------
    # gradient instrumentation
    # ------------------------------------------------------------------

    def _is_log_step(self, sync_gradients: bool) -> bool:
        """Return whether this synchronized optimizer step should be logged."""
        return bool(sync_gradients) and (self.completed_steps + 1) % self.config.trainer.logging_frequency == 0

    def _global_grad_norm(self):
        """Read the unscaled, globally reduced, pre-clipping gradient norm."""
        try:
            m = self.model
            if hasattr(m, "get_global_grad_norm"):
                v = m.get_global_grad_norm()
                if v is not None:
                    return float(v)
            eng = getattr(getattr(self.accelerator, "deepspeed_engine_wrapped", None), "engine", None)
            if eng is not None and hasattr(eng, "get_global_grad_norm"):
                v = eng.get_global_grad_norm()
                if v is not None:
                    return float(v)
        except Exception:
            pass
        return self._last_clip_norm

    def _compute_head_sqnorms(self):
        """Return the head group's local sharded squared gradient norm.

        This runs after backward and before step with no collective; one later
        scalar all-reduce recovers the exact global head norm.
        """
        if not self._grad_groups or "head" not in self._grad_groups:
            return None
        return group_grad_local_sqnorm(self._grad_groups["head"])

    def _finalize_grad_metrics(self, head_local_sq) -> dict:
        """Finalize the total / shared-backbone / head gradient norms.

        The head norm requires one scalar all-reduce; the disjoint shared norm
        is recovered with ``sqrt(total^2 - head^2)``.
        """
        m = {}
        total = self._global_grad_norm()
        if total is not None:
            m["train/grad_norm_total"] = total
        if head_local_sq is not None:
            if self._using_deepspeed and dist.is_available() and dist.is_initialized():
                dist.all_reduce(head_local_sq, op=dist.ReduceOp.SUM)
            head_sq = float(head_local_sq.item())
            m["train/grad_norm_head"] = head_sq**0.5
            if total is not None:
                m["train/grad_norm_shared"] = max(total * total - head_sq, 0.0) ** 0.5
        return m

    @staticmethod
    def _shared_vlm_interface_tensors(interface_tensors) -> tuple[torch.Tensor, ...]:
        if not isinstance(interface_tensors, (tuple, list)):
            raise TypeError("mot_shared_vlm_interface must be a tuple/list of tensors")
        tracked = tuple(tensor for tensor in interface_tensors if torch.is_tensor(tensor) and tensor.requires_grad)
        if not tracked:
            raise RuntimeError("shared-VLM interface gradient diagnostics found no differentiable planner-query tensors")
        return tracked

    @staticmethod
    def _shared_vlm_interface_metrics_from_gradients(
        tracked: tuple[torch.Tensor, ...],
        action_grads,
        world_grads,
    ) -> dict[str, float]:
        if len(action_grads) != len(tracked) or len(world_grads) != len(tracked):
            raise ValueError("shared-VLM gradient tuples must match interface tensors")
        squared_norms = torch.zeros(2, device=tracked[0].device, dtype=torch.float32)
        for action_grad, world_grad in zip(action_grads, world_grads, strict=True):
            if action_grad is None and world_grad is None:
                continue
            if action_grad is not None:
                squared_norms[0] += torch.sum(action_grad.detach().float().square())
            if world_grad is not None:
                squared_norms[1] += torch.sum(world_grad.detach().float().square())

        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(squared_norms, op=dist.ReduceOp.SUM)

        return {
            "train/shared_vlm_interface_grad_norm_action": float(squared_norms[0].clamp_min(0.0).sqrt()),
            "train/shared_vlm_interface_grad_norm_world": float(squared_norms[1].clamp_min(0.0).sqrt()),
        }

    @staticmethod
    def _prepare_shared_vlm_interface_gradient_probe(
        action_objective: torch.Tensor,
        world_objective: torch.Tensor,
        interface_tensors,
    ):
        """Prepare one auxiliary gradient before the real optimizer backward.

        The real backward already produces ``g_action + g_world`` at every
        retained planner-query tensor.  Computing only ``g_world`` here lets
        finalization recover ``g_action = g_total - g_world`` exactly, cutting
        the diagnostic from two auxiliary branch backwards to one.  The world
        expert is the smaller branch in this recipe.
        """

        if not torch.is_tensor(action_objective) or action_objective.ndim != 0:
            raise ValueError("mot_action_objective must be a scalar tensor")
        if not torch.is_tensor(world_objective) or world_objective.ndim != 0:
            raise ValueError("mot_world_objective must be a scalar tensor")
        tracked = CogWAMTrainer._shared_vlm_interface_tensors(interface_tensors)
        if any(tensor.is_leaf for tensor in tracked):
            raise RuntimeError(
                "optimized shared-VLM diagnostics require non-leaf planner-query "
                "interface tensors so retained gradients cannot alias parameters"
            )
        for tensor in tracked:
            tensor.retain_grad()
        world_grads = torch.autograd.grad(
            world_objective,
            tracked,
            retain_graph=True,
            allow_unused=True,
        )
        # ``retain_grad`` installs hooks that also observe ``autograd.grad``;
        # clear those auxiliary values so the subsequent real backward stores
        # only g_total rather than g_world + g_total.
        for tensor in tracked:
            tensor.grad = None
        return tracked, world_grads

    @staticmethod
    def _finalize_shared_vlm_interface_gradient_probe(probe) -> dict[str, float]:
        """Recover exact per-task interface gradients after the real backward."""

        tracked, world_grads = probe
        total_grads = tuple(tensor.grad for tensor in tracked)
        for tensor in tracked:
            tensor.grad = None

        action_grads = []
        for total_grad, world_grad in zip(total_grads, world_grads, strict=True):
            if total_grad is None:
                if world_grad is not None:
                    raise RuntimeError(
                        "real optimizer backward did not retain a shared-VLM "
                        "interface gradient required by the diagnostic"
                    )
                action_grads.append(None)
            elif world_grad is None:
                action_grads.append(total_grad)
            else:
                action_grads.append(total_grad - world_grad)

        return CogWAMTrainer._shared_vlm_interface_metrics_from_gradients(
            tracked,
            tuple(action_grads),
            world_grads,
        )

    def _should_measure_shared_vlm_interface_gradients(self, *, sync_gradients: bool) -> bool:
        return bool(
            sync_gradients
            and self._shared_vlm_gradient_diagnostics_enabled
            and (self.completed_steps + 1) % int(self._shared_vlm_gradient_diagnostics_interval) == 0
        )

    # ------------------------------------------------------------------
    # loop
    # ------------------------------------------------------------------

    def _log_metrics(self, metrics):
        """Record training metrics."""
        if self.completed_steps % self.config.trainer.logging_frequency == 0 and self.accelerator.is_main_process:
            last_lrs = self.lr_scheduler.get_last_lr()
            for i, group in enumerate(self.optimizer.param_groups):
                group_name = group.get("name", str(i))
                metrics[f"learning_rate/{group_name}"] = last_lrs[i] if i < len(last_lrs) else last_lrs[-1]
            metrics["epoch"] = round(self.completed_steps / len(self.vla_train_dataloader), 2)
            wandb.log(metrics, step=self.completed_steps)
            logger.info(f"Step {self.completed_steps}, Loss: {metrics})")

    def _log_training_config(self):
        """Record training config."""
        if self.accelerator.is_main_process:
            logger.info("***** Training Configuration *****")
            logger.info(f"  Total optimization steps = {self.config.trainer.max_train_steps}")
            logger.info(f"  Per device batch size = {self.config.datasets.vla_data.per_device_batch_size}")
            logger.info(f"  Gradient accumulation steps = {self.accelerator.gradient_accumulation_steps}")
            logger.info(f"  Total batch size = {self.total_batch_size}")

    def train(self):
        """Execute training loop."""
        self._log_training_config()
        self._create_data_iterators()
        progress_bar = tqdm(
            total=self.config.trainer.max_train_steps,
            initial=self.completed_steps,
            # Multi-node launchers merge stdout from every node, so
            # ``is_local_main_process`` produced one duplicate bar per machine.
            disable=not self.accelerator.is_main_process,
        )

        while self.completed_steps < self.config.trainer.max_train_steps:
            t_start_data = time.perf_counter()
            batch_vla = self._get_next_batch()
            batch_semantic = self._get_next_semantic_batch()
            t_end_data = time.perf_counter()

            t_start_model = time.perf_counter()
            step_metrics = self._train_step(batch_vla, batch_semantic=batch_semantic)
            t_end_model = time.perf_counter()
            data_elapsed = t_end_data - t_start_data
            model_elapsed = t_end_model - t_start_model
            did_optimizer_step = bool(self.accelerator.sync_gradients)

            if did_optimizer_step:
                progress_bar.update(1)
                self.completed_steps += 1

            # Show one progress record per optimizer step. With accumulation,
            # printing every micro-batch duplicates the same step number.
            if did_optimizer_step and self.accelerator.is_main_process:
                progress_bar.set_postfix(
                    {
                        "rank": self.accelerator.process_index,
                        "data_times": f"{data_elapsed:.3f}",
                        "model_times": f"{model_elapsed:.3f}",
                    }
                )

            # Log/save are optimizer-step events.  An accumulation micro-batch
            # keeps ``completed_steps`` unchanged; running these hooks there
            # duplicates checkpoint writes.
            if did_optimizer_step:
                step_metrics["timing/data"] = data_elapsed
                step_metrics["timing/model"] = model_elapsed
                self._log_metrics(step_metrics)

            if (
                did_optimizer_step
                and self.completed_steps % self.config.trainer.save_interval == 0
                and self.completed_steps > 0
            ):
                self._save_checkpoint()

            if self.completed_steps >= self.config.trainer.max_train_steps:
                break

        self._finalize_training()

    def _train_step(self, batch_vla, batch_semantic=None):
        """Execute single training step over the paired physical/semantic batch."""
        with self.accelerator.accumulate(self.model):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                output_dict = self.model.forward(
                    batch_vla,
                    semantic_examples=batch_semantic,
                    global_step=self.completed_steps,
                )
                # The framework already folds the weighted world and semantic
                # terms into this scalar; summing anything else here would
                # double-count them.
                total_loss = output_dict["action_loss"]

            sync = bool(self.accelerator.sync_gradients)
            will_log = self._is_log_step(sync)
            shared_vlm_gradient_metrics: dict[str, float] = {}
            shared_vlm_gradient_probe = None
            if self._should_measure_shared_vlm_interface_gradients(sync_gradients=sync):
                required = (
                    "mot_action_objective",
                    "mot_world_objective",
                    "mot_shared_vlm_interface",
                )
                missing = [key for key in required if key not in output_dict]
                if missing:
                    raise RuntimeError(
                        "Shared-VLM gradient diagnostics require World--Action "
                        f"MoT outputs {missing}; active output keys={sorted(output_dict)}"
                    )
                shared_vlm_gradient_probe = self._prepare_shared_vlm_interface_gradient_probe(
                    output_dict["mot_action_objective"],
                    output_dict["mot_world_objective"],
                    output_dict["mot_shared_vlm_interface"],
                )
            self.accelerator.backward(total_loss)
            if shared_vlm_gradient_probe is not None:
                shared_vlm_gradient_metrics = self._finalize_shared_vlm_interface_gradient_probe(
                    shared_vlm_gradient_probe,
                )

            self._last_clip_norm = None
            if sync and self.config.trainer.gradient_clipping is not None:
                clipped = self.accelerator.clip_grad_norm_(
                    self.model.parameters(), self.config.trainer.gradient_clipping
                )
                if clipped is not None:
                    try:
                        self._last_clip_norm = float(clipped)
                    except Exception:
                        self._last_clip_norm = None

            want_global_grad = will_log and self._log_global_grad_norm
            want_grad = will_log and self._log_grad_norms and self._grad_groups is not None

            head_local_sq = None
            if want_grad:
                try:
                    head_local_sq = self._compute_head_sqnorms()
                except Exception as e:
                    if not self._grad_warned and self.accelerator.is_main_process:
                        logger.warning("per-head grad-norm failed (%s); skipping per-head grad logs hereafter.", e)
                    self._grad_warned = True
                    head_local_sq = None

            self.optimizer.step()
            # Only step the LR scheduler when gradients are actually synced
            # (i.e., not mid-accumulation). Without this guard the scheduler
            # runs gradient_accumulation_steps times faster than intended,
            # causing warmup to end too early and cosine decay to bottom out
            # at min_lr well before max_train_steps is reached.
            if sync:
                self.lr_scheduler.step()

            grad_metrics = dict(shared_vlm_gradient_metrics)
            if want_global_grad:
                try:
                    total_grad_norm = self._global_grad_norm()
                    if total_grad_norm is not None:
                        grad_metrics["train/grad_norm"] = total_grad_norm
                except Exception as e:
                    if not self._grad_warned and self.accelerator.is_main_process:
                        logger.warning("global grad-norm logging failed (%s); skipping.", e)
                    self._grad_warned = True
            if want_grad:  # Optional detailed shared/head decomposition.
                try:
                    grad_metrics.update(self._finalize_grad_metrics(head_local_sq))
                except Exception as e:
                    if not self._grad_warned and self.accelerator.is_main_process:
                        logger.warning("grad-norm finalize failed (%s); skipping.", e)
                    self._grad_warned = True
            # Accelerate performs ``step``/``zero_grad`` only on the sync
            # micro-batch.  Clearing at that micro-batch's start discards the
            # gradients accumulated by all preceding micro-batches.  Clearing
            # here, after step and optional gradient logging, preserves the
            # configured effective batch size.
            self.optimizer.zero_grad()

        if not will_log:
            return {}
        metrics = collect_loss_metrics(output_dict, total_loss)
        metrics.update(grad_metrics)
        return metrics

    def _finalize_training(self):
        """Training end processing."""
        self._save_final_model()

        if self.accelerator.is_main_process:
            wandb.finish()

        self.accelerator.wait_for_everyone()


__all__ = ["CogWAMTrainer", "collect_loss_metrics"]
