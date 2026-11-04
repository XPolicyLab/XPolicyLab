"""Optimizer boundaries shared by ordinary Accelerate and DeepSpeed training."""
from __future__ import annotations

import math

from accelerate.utils import DistributedType


def configure_deepspeed_clipping(accelerator, max_grad_norm: float) -> None:
    """Set the actual engine threshold before Accelerator.prepare initializes it."""
    if not math.isfinite(max_grad_norm) or max_grad_norm < 0:
        raise ValueError("max_grad_norm must be finite and >= 0 (0 disables clipping).")
    plugin = accelerator.state.deepspeed_plugin
    if plugin is not None:
        plugin.deepspeed_config["gradient_clipping"] = float(max_grad_norm)


class TrainingStep:
    """Accumulate main/CF gradients, then update once at the batch boundary.

    The critic must occupy its own optimizer group. A hook on the *base*
    AdamW optimizer skips that group's update when the entire accumulation
    window has no CF backward. In ZeRO-1/2 the hook sees the FP32 partition,
    after ZeRO has materialized its gradient, rather than the model's bf16
    parameters. This preserves parameters, Adam moments and Adam step counts.
    """

    def __init__(self, accelerator, model, optimizer, *, max_grad_norm: float,
                 critic_group_index: int | None):
        self.accelerator = accelerator
        self.model = model
        self.optimizer = optimizer
        self.max_grad_norm = float(max_grad_norm)
        self.critic_has_grad = False
        self.engine = None
        self._critic_hook = None
        if accelerator.distributed_type == DistributedType.DEEPSPEED:
            self.engine = accelerator.deepspeed_engine_wrapped.engine
            actual = float(self.engine.gradient_clipping())
            if actual != self.max_grad_norm:
                raise RuntimeError(
                    f"DeepSpeed gradient_clipping={actual} differs from max_grad_norm={self.max_grad_norm}."
                )
            base_optimizer = self.engine.basic_optimizer
        else:
            base_optimizer = getattr(optimizer, "optimizer", optimizer)

        if critic_group_index is not None:
            # Parameter identities survive optimizer state loading and ZeRO's
            # temporary replacement of param_groups during its per-group step.
            critic_ids = {id(p) for p in base_optimizer.param_groups[critic_group_index]["params"]}

            def skip_unused_critic(opt, args, kwargs):
                if not self.critic_has_grad:
                    for group in opt.param_groups:
                        for parameter in group["params"]:
                            if id(parameter) in critic_ids:
                                parameter.grad = None

            self._critic_hook = base_optimizer.register_step_pre_hook(skip_unused_critic)

    def backward(self, loss, *, final_backward: bool = True, critic: bool = False) -> None:
        if self.engine is None:
            self.accelerator.backward(loss)
        else:
            # Accelerator.backward also calls engine.step at a sync boundary.
            # Use engine.backward directly, with explicit GAS scaling: our
            # training_loss() entry point bypasses engine.forward(), where
            # recent DeepSpeed versions install the automatic scaling hook.
            # Only the final backward marks the optimizer update boundary.
            self.engine.set_gradient_accumulation_boundary(
                self.accelerator.sync_gradients and final_backward
            )
            self.engine.backward(
                loss / self.engine.gradient_accumulation_steps(), scale_wrt_gas=False
            )
        self.critic_has_grad = self.critic_has_grad or critic

    def step(self):
        """Return the pre-clipping norm, or None inside an accumulation window."""
        if not self.accelerator.sync_gradients:
            return None
        if self.engine is not None:
            if not self.engine.is_gradient_accumulation_boundary():
                raise RuntimeError("Missing final backward before the DeepSpeed optimizer step.")
            self.engine.step()
            grad_norm = self.engine.get_global_grad_norm()
        else:
            grad_norm = self.accelerator.clip_grad_norm_(
                self.model.parameters(), self.max_grad_norm or math.inf
            )
            self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)
        self.critic_has_grad = False
        return grad_norm
