"""Compile the full joint MoT without changing observations or denoising math.

TorchDynamo/AOT capture uses the native cudagraphs backend. Native attention,
normalization, RoPE and residual operators keep their eager implementation and
BF16 rounding. RGB/DINO encoders, hooks, mask construction and all four Euler
updates remain outside the compiled region.
"""
from __future__ import annotations

from contextlib import contextmanager
import time
from typing import Any

import torch


class AccelerationController:
    """One model's native compiled joint CUDA graph, with explicit failures."""

    def __init__(self, model):
        if model.device.type != "cuda" or model.torch_dtype != torch.bfloat16:
            raise ValueError("CR-WM compilation requires the frozen CUDA BF16 runtime.")
        self.original = model.mot._forward_joint_inner
        self.enabled = True
        self.calls = 0
        self.compiled_graphs = 0
        self.compile_seconds = 0.0
        self.warmed_up = False
        self.error = None
        self.device_index = model.device.index if model.device.index is not None else torch.cuda.current_device()
        from torch._dynamo.utils import counters
        self.skip_counter_before = counters["inductor"]["cudagraph_skips"]
        # Record actual Dynamo/AOT compilation, then separately verify that
        # native CUDA graph capture occurred and was not silently skipped.
        from torch._dynamo.backends.registry import lookup_backend
        cudagraphs = lookup_backend("cudagraphs")

        def counted_backend(graph_module, example_inputs, **kwargs):
            started = time.perf_counter()
            try:
                artifact = cudagraphs(graph_module, example_inputs, **kwargs)
            except Exception as exc:
                self.error = f"{type(exc).__name__}: {exc}"
                raise
            self.compile_seconds += time.perf_counter() - started
            self.compiled_graphs += 1
            return artifact

        # Frozen parameters remain at their loaded addresses; graph replays
        # must stage fresh observation tensors rather than recopy model weights.
        for parameter in model.mot.parameters():
            torch._dynamo.mark_static_address(parameter, guard=False)
        self.compiled = torch.compile(self.original, backend=counted_backend,
                                      mode="default", fullgraph=False, dynamic=False)

    def __call__(self, *args, **kwargs):
        if not self.enabled:
            return self.original(*args, **kwargs)
        try:
            # An inherited environment may otherwise swallow compiler failures
            # and run an entire graph eagerly. Such failures must reach the
            # worker and its evaluation log.
            torch.compiler.cudagraph_mark_step_begin()
            with torch._dynamo.config.patch(suppress_errors=False):
                output = self.compiled(*args, **kwargs)
            if self.compiled_graphs == 0:
                raise RuntimeError("MoT produced no compiled graph; refusing an unverified eager fallback.")
            from torch._dynamo.utils import counters
            if counters["inductor"]["cudagraph_skips"] > self.skip_counter_before:
                raise RuntimeError("Native cudagraph backend skipped capture; refusing an eager fallback.")
            self.calls += 1
            # CUDA graph outputs own graph-pool storage. Copy both results
            # before the next denoise step can replay into the same buffers.
            return tuple(tensor.clone() for tensor in output)
        except Exception as exc:
            self.error = f"{type(exc).__name__}: {exc}"
            raise RuntimeError(f"CR-WM full-joint compilation failed: {self.error}") from exc

    @contextmanager
    def eager_for_benchmark(self):
        """Temporarily run the saved identical body for parity measurement."""
        previous = self.enabled
        self.enabled = False
        try:
            yield
        finally:
            self.enabled = previous

    def report(self) -> dict[str, Any]:
        from torch._inductor.cudagraph_trees import get_manager
        manager = get_manager(self.device_index, create_if_none_exists=False)
        roots = list(manager.get_roots()) if manager is not None else []
        captured = sum(getattr(node, "graph", None) is not None for node in roots)
        return dict(kind="torch_compile_native_joint_cuda_graph", backend="cudagraphs", scope="joint_mot", mode="default", fullgraph=False,
                    dynamic=False, enabled=self.enabled, warmed_up=self.warmed_up,
                    cudnn_sdp_enabled=torch.backends.cuda.cudnn_sdp_enabled(),
                    compiled_graphs=self.compiled_graphs, compiled_calls=self.calls,
                    captured_cuda_graph_roots=captured,
                    compile_seconds=self.compile_seconds, error=self.error)


def install_acceleration(model) -> AccelerationController:
    """Install once after checkpoint loading and prepare_for_inference()."""
    existing = getattr(model, "_crwm_acceleration", None)
    if existing is not None:
        if not isinstance(existing, AccelerationController):
            raise RuntimeError("Unknown existing CR-WM acceleration wrapper.")
        return existing
    controller = AccelerationController(model)
    model.mot._forward_joint_inner = controller
    model._crwm_acceleration = controller
    return controller


def warmup_policy(policy, observation: dict, instruction: str, *, calls: int = 2) -> dict:
    """Warm with the actual first observation, without issuing robot actions.

    predict() does not enqueue or execute commands. Restore policy accounting
    and RNG state so this setup work does not count as evaluated replans. The
    first instruction's correctly computed text context remains cached.
    """
    if calls < 1:
        raise ValueError("Warmup requires at least one full-joint inference call.")
    controller = getattr(policy.model, "_crwm_acceleration", None)
    if not isinstance(controller, AccelerationController) or not controller.enabled:
        raise RuntimeError("Install and enable CR-WM acceleration before warmup.")
    old_count, old_seconds = policy.replan_count, policy.inference_seconds
    cpu_rng = torch.get_rng_state()
    cuda_rng = torch.cuda.get_rng_state(policy.model.device)
    started = time.perf_counter()
    try:
        for _ in range(calls):
            policy.predict(observation, instruction)
        torch.cuda.synchronize(policy.model.device)
        if controller.report()["captured_cuda_graph_roots"] == 0:
            raise RuntimeError("Warmup did not capture a native joint CUDA graph.")
        controller.warmed_up = True
    finally:
        policy.replan_count, policy.inference_seconds = old_count, old_seconds
        torch.set_rng_state(cpu_rng)
        torch.cuda.set_rng_state(cuda_rng, policy.model.device)
    return dict(warmup_calls=calls, warmup_seconds=time.perf_counter() - started,
                acceleration=controller.report())
