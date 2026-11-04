from typing import Any, Optional, Sequence, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from flexwam.utils.logging_config import get_logger
logger = get_logger(__name__)

class _ActionDenoiseCudaGraph:
    """Replay one `_predict_action_noise_with_cache` step from static buffers."""

    def __init__(self, eager_fn, device: torch.device) -> None:
        self.eager_fn = eager_fn
        self.device = device
        self.graph: Optional[torch.cuda.CUDAGraph] = None
        self.failed = False
        self.static: dict[str, Any] = {}
        self.static_kv: Optional[list[dict[str, torch.Tensor]]] = None
        self.static_out: Optional[torch.Tensor] = None
        self.spec: Optional[tuple[Any, ...]] = None

    @staticmethod
    def _spec(kwargs: dict[str, Any]) -> tuple[Any, ...]:
        kv = kwargs["video_kv_cache"]
        router = kwargs.get("action_video_router_weights")
        log_scales = kwargs.get("action_video_router_log_scales")
        proprio = kwargs["proprio"]
        return (
            tuple(kwargs["latents_action"].shape),
            str(kwargs["latents_action"].dtype),
            tuple(kwargs["timestep_action"].shape),
            tuple(kwargs["context"].shape),
            tuple(kwargs["context_mask"].shape),
            tuple(kwargs["attention_mask"].shape),
            tuple(proprio.shape),
            None if router is None else tuple(router.shape),
            None if log_scales is None else (tuple(log_scales.shape), str(log_scales.dtype)),
            int(kwargs["video_seq_len"]),
            len(kv),
            tuple(kv[0]["k"].shape) if kv else None,
        )

    def _allocate(self, kwargs: dict[str, Any]) -> None:
        def _clone(tensor: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
            return None if tensor is None else tensor.clone()

        self.static = {
            "latents_action": kwargs["latents_action"].clone(),
            "timestep_action": kwargs["timestep_action"].clone(),
            "context": kwargs["context"].clone(),
            "context_mask": kwargs["context_mask"].clone(),
            "attention_mask": kwargs["attention_mask"].clone(),
            "proprio": kwargs["proprio"].clone(),
            "action_video_router_weights": _clone(kwargs.get("action_video_router_weights")),
            "action_video_router_log_scales": _clone(kwargs.get("action_video_router_log_scales")),
            "video_seq_len": int(kwargs["video_seq_len"]),
            "video_meta": kwargs.get("video_meta"),
        }
        self.static_kv = [{"k": layer["k"].clone(), "v": layer["v"].clone()} for layer in kwargs["video_kv_cache"]]

    def _copy_inputs(self, kwargs: dict[str, Any]) -> None:
        self.static["latents_action"].copy_(kwargs["latents_action"])
        self.static["timestep_action"].copy_(kwargs["timestep_action"])
        self.static["context"].copy_(kwargs["context"])
        self.static["context_mask"].copy_(kwargs["context_mask"])
        self.static["attention_mask"].copy_(kwargs["attention_mask"])
        self.static["proprio"].copy_(kwargs["proprio"])
        if self.static["action_video_router_weights"] is not None:
            self.static["action_video_router_weights"].copy_(kwargs["action_video_router_weights"])
        if self.static["action_video_router_log_scales"] is not None:
            self.static["action_video_router_log_scales"].copy_(kwargs["action_video_router_log_scales"])
        assert self.static_kv is not None
        for dst, src in zip(self.static_kv, kwargs["video_kv_cache"]):
            dst["k"].copy_(src["k"])
            dst["v"].copy_(src["v"])

    def _call_eager_static(self) -> torch.Tensor:
        assert self.static_kv is not None
        return self.eager_fn(
            latents_action=self.static["latents_action"],
            timestep_action=self.static["timestep_action"],
            context=self.static["context"],
            context_mask=self.static["context_mask"],
            video_kv_cache=self.static_kv,
            attention_mask=self.static["attention_mask"],
            video_seq_len=self.static["video_seq_len"],
            video_meta=self.static["video_meta"],
            action_video_router_weights=self.static["action_video_router_weights"],
            action_video_router_log_scales=self.static["action_video_router_log_scales"],
            action_attention_stats=None,
            action_attention_heatmaps=None,
            proprio=self.static["proprio"],
        )

    def __call__(self, **kwargs: Any) -> torch.Tensor:
        if self.failed:
            return self.eager_fn(**kwargs)
        if kwargs.get("action_attention_stats") is not None or kwargs.get("action_attention_heatmaps") is not None:
            return self.eager_fn(**kwargs)
        spec = self._spec(kwargs)
        if self.graph is not None and spec != self.spec:
            logger.warning("Action CUDA graph input shapes changed; falling back to eager.")
            self.failed = True
            self.graph = None
            return self.eager_fn(**kwargs)
        if self.graph is None:
            return self._capture(kwargs, spec)
        self._copy_inputs(kwargs)
        assert self.graph is not None
        self.graph.replay()
        assert self.static_out is not None
        return self.static_out.clone()

    def _capture(self, kwargs: dict[str, Any], spec: tuple[Any, ...]) -> torch.Tensor:
        self.spec = spec
        self._allocate(kwargs)
        self._copy_inputs(kwargs)
        try:
            torch.cuda.synchronize(self.device)
            warmup = torch.cuda.Stream(device=self.device)
            warmup.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(warmup):
                for _ in range(2):
                    self.static_out = self._call_eager_static()
            torch.cuda.current_stream().wait_stream(warmup)
            torch.cuda.synchronize(self.device)
            self._copy_inputs(kwargs)
            self.graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.graph):
                self.static_out = self._call_eager_static()
            logger.info("Captured CUDA graph for one action denoise step.")
            self._copy_inputs(kwargs)
            self.graph.replay()
            assert self.static_out is not None
            return self.static_out.clone()
        except Exception as exc:
            logger.warning("Failed to capture action denoise CUDA graph (%s); using eager.", exc)
            self.failed = True
            self.graph = None
            return self.eager_fn(**kwargs)
