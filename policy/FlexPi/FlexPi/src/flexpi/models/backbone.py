"""Wan2.2 MoT backbone for FlexPi.

Owned by this package, so ``FlexPiBackbone`` has no import dependency on
the ``FlexPi`` model that extends it. The 7 methods that
FlexPi fully overrides without ever calling back into
(``forward``, ``from_wan22_pretrained``, ``infer``, ``infer_action``,
``infer_joint``, ``save_checkpoint``, ``training_loss``) are omitted.

Everything here — the VAE encode/decode path, the MoT attention-mask builders,
the flow-matching denoise loop, ``prepare_for_inference`` and its
torch.compile / quantization / TensorRT / CUDA-graph wiring — is shared
machinery that FlexPi uses unchanged.
"""
import contextlib
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from flexpi.composite_layouts import LayoutSpec, get_layout
from flexpi.utils.logging_config import get_logger

from .action_dit import ActionDiT
from .helpers.dino import select_aux_frame_slots
from .mot import MoT
from .schedulers.scheduler_continuous import WanContinuousFlowMatchScheduler
from .wan_video_dit import sinusoidal_embedding_1d

logger = get_logger(__name__)



class FlexPiBackbone(torch.nn.Module):
    def __init__(
        self,
        video_expert,
        action_expert: ActionDiT,
        mot: MoT,
        vae,
        text_encoder=None,
        tokenizer=None,
        text_dim: Optional[int] = None,
        proprio_dim: Optional[int] = None,
        device: str = "cpu",
        torch_dtype: torch.dtype = torch.float32,
        video_train_shift: float = 5.0,
        video_infer_shift: float = 5.0,
        video_num_train_timesteps: int = 1000,
        action_train_shift: float = 5.0,
        action_infer_shift: float = 5.0,
        action_num_train_timesteps: int = 1000,
        loss_lambda_video: float = 1.0,
        loss_lambda_action: float = 1.0,
        composite_layout: Union[str, LayoutSpec, None] = "tshape_robotwin_384x320_uniform",
        composite_layout_slot_key_map: Optional[Mapping[str, str]] = None,
    ):
        super().__init__()
        # Resolve the composite layout once. Defaults to RoboTwin so existing
        # checkpoints / configs keep working unchanged. Subclasses use
        # ``self._layout`` and ``self._slot_key_map`` in build_inputs to drive
        # composite assembly, DINO/pointmap encoder dispatch, and val-vis.
        self._layout: LayoutSpec = get_layout(composite_layout)
        self._slot_key_map = self._layout.resolve_slot_key_map(
            composite_layout_slot_key_map,
        )

        self.video_expert = video_expert
        self.action_expert = action_expert
        self.mot = mot
        # Keep trainer compatibility: optimizer and freeze logic use `model.dit`.
        self.dit = self.mot

        self.vae = vae
        self.text_encoder = text_encoder
        self.tokenizer = tokenizer
        if text_dim is None:
            if self.text_encoder is None:
                raise ValueError("`text_dim` is required when `text_encoder` is not loaded.")
            text_dim = int(self.text_encoder.dim)
        self.text_dim = int(text_dim)
        self.proprio_dim = None if proprio_dim is None else int(proprio_dim)
        if self.proprio_dim is not None:
            self.proprio_encoder = nn.Linear(self.proprio_dim, self.text_dim).to(torch_dtype)
        else:
            self.proprio_encoder = None

        self.train_video_scheduler = WanContinuousFlowMatchScheduler(
            num_train_timesteps=video_num_train_timesteps,
            shift=video_train_shift,
        )
        self.infer_video_scheduler = WanContinuousFlowMatchScheduler(
            num_train_timesteps=video_num_train_timesteps,
            shift=video_infer_shift,
        )
        self.train_action_scheduler = WanContinuousFlowMatchScheduler(
            num_train_timesteps=action_num_train_timesteps,
            shift=action_train_shift,
        )
        self.infer_action_scheduler = WanContinuousFlowMatchScheduler(
            num_train_timesteps=action_num_train_timesteps,
            shift=action_infer_shift,
        )
        # Optional aliases for consistency with Wan22Core naming.
        self.train_scheduler = self.train_video_scheduler
        self.infer_scheduler = self.infer_video_scheduler

        self.device = torch.device(device)
        self.torch_dtype = torch_dtype
        self.loss_lambda_video = float(loss_lambda_video)
        self.loss_lambda_action = float(loss_lambda_action)
        self.offload_text_encoder = False
        # Set by set_action_dim_mask(); None keeps every sampling path unchanged.
        self._action_dim_mask: Optional[torch.Tensor] = None

        self.to(self.device)

    def set_action_dim_mask(self, action_dim_is_pad: Optional[torch.Tensor]) -> None:
        """Register which action-head channels this embodiment does not fill.

        ``[A]`` (or ``[B, A]``, first row taken) bool, True = padded. Training
        zeroes these channels on the noisy/target action via the per-batch
        ``action_dim_is_pad`` and masks them out of the loss; at deploy there is
        no batch, so the server sets the mask once. Without it the padded
        channels start as N(0,1) and leak into the real ones through the action
        DiT at every denoise step. Leaving it unset keeps sampling bit-identical.
        """
        if action_dim_is_pad is None:
            self._action_dim_mask = None
            return
        m = torch.as_tensor(action_dim_is_pad)
        if m.ndim == 2:
            m = m[0]
        if m.ndim != 1:
            raise ValueError(f"action_dim_is_pad must be [A] or [B, A]; got {tuple(m.shape)}")
        width = int(self.action_expert.action_dim)
        if int(m.shape[0]) != width:
            raise ValueError(
                f"action_dim_is_pad width {int(m.shape[0])} != action head width {width}"
            )
        self._action_dim_mask = m.to(device=self.device, dtype=torch.bool)

    def _zero_padded_action(self, latents_action: torch.Tensor) -> torch.Tensor:
        """Zero the padded channels of an action latent. No-op when unset.

        Applied to the initial noise and after every scheduler step, so a padded
        channel starts at exactly 0 and cannot drift.
        """
        m = self._action_dim_mask
        if m is None:
            return latents_action
        return latents_action * (~m).to(device=latents_action.device, dtype=latents_action.dtype)

    # Subclasses: override this set if you add new pretrained modules that
    # must stay frozen (e.g. a DINO encoder). The trainer freezes everything,
    # then unfreezes all children NOT in this set. A runtime assertion checks
    # that every nn.Module child is either in this set or has trainable params.
    FROZEN_MODULES: set[str] = {"vae", "text_encoder"}

    def _layout_kwargs(self) -> dict:
        """Standard layout kwargs to forward to encoder calls (DINO,
        pointmap). Routes ``self._layout`` and the resolved ``slot_key_map``
        so encoders use the model's chosen layout regardless of any legacy
        ``concat_mode="tshape_robotwin_384x320_uniform"`` arg supplied alongside.

        For DINO encode calls, prefer ``_dino_encode_kwargs`` instead — it also
        forwards the model's pooled ``self.dino_cam_patches`` so the encoder's
        output token count matches the model's RoPE freqs when
        ``dino_pool_factor>1``. ``PointmapEncoder.encode_composite`` uses slot
        HW rather than per-cam patches, so it calls this directly.
        """
        return {"layout": self._layout, "slot_key_map": self._slot_key_map}

    def _dino_encode_kwargs(self) -> dict:
        """Layout kwargs + ``cam_patches`` for ``DinoEncoder.encode_video`` /
        ``encode_frames``. Includes the model's ``self.dino_cam_patches`` when
        defined (subclasses with DINO) so pool-factor changes propagate to the
        encoder. Returns plain ``_layout_kwargs()`` for models without DINO.
        """
        out = self._layout_kwargs()
        cam_patches = getattr(self, "dino_cam_patches", None)
        if cam_patches is not None:
            out["cam_patches"] = cam_patches
        out["pool_mode"] = getattr(self, "dino_pool_mode", "avg")
        return out


    def to(self, *args, **kwargs):
        # Temporarily remove text_encoder from module tree so super().to()
        # doesn't move it — nn.Module.to() moves ALL registered submodules.
        te = None
        if self.offload_text_encoder and self.text_encoder is not None:
            te = self.text_encoder
            del self._modules["text_encoder"]
        super().to(*args, **kwargs)
        if te is not None:
            self._modules["text_encoder"] = te
        elif self.text_encoder is not None:
            self.text_encoder.to(*args, **kwargs)
        self.mot.to(*args, **kwargs)
        self.vae.to(*args, **kwargs)
        return self

    def prepare_for_inference(
        self,
        torch_compile: bool = False,
        torch_compile_mode: str = "reduce-overhead",
        quantization: str | None = None,
        torch_compile_scope: str = "step",
        attn_backend: str = "sdpa",
        torch_compile_backend: str = "inductor",
        compile_encoders: bool = False,
        trt_joint_engine_path: str | None = None,
        trt_joint_free_video_blocks: bool = False,
        trt_joint_prefill_split_engine_path: str | None = None,
        trt_joint_decode_split_engine_path: str | None = None,
        trt_prefill_engine_path: str | None = None,
        solver: str = "euler",
        glue_cache: bool = False,
        encoder_cuda_graph: bool = False,
        joint_loop_cuda_graph: bool = False,
    ) -> None:
        """Optimize model for inference with torch.compile and/or INT8 quantization.

        Call after ``load_checkpoint()`` and ``eval()``. Quantization is applied
        first, then torch.compile wraps the (possibly quantized) methods.

        Args:
            torch_compile: Compile MoT inference methods with ``torch.compile``.
            torch_compile_mode: Compilation mode — ``"default"``,
                ``"reduce-overhead"`` (CUDA graphs, best for fixed-shape loops),
                or ``"max-autotune"``.
            quantization: Weight quantization method. Supports ``"int8"``
                (INT8 weight-only), ``"fp8"`` (FP8 dynamic activation + FP8
                weight, per-tensor scaling — needs SM89+ hardware, e.g.
                Ada/Hopper/Blackwell), and ``"fp8_video"`` (fp8 on the video
                expert only — speeds up the M>=534 prefill/joint GEMMs while
                keeping the M=32 action expert bf16, where fp8 is a net
                loss), all via torchao. Requires ``pip install torchao``.
            torch_compile_scope: ``"step"`` (default) compiles one denoise step
                and re-enters Python between steps. ``"loop"`` compiles the
                entire denoise loop (all steps + scheduler updates unrolled)
                into a single graph — zero host round-trips per step, at the
                cost of a longer first-call compile. Ignored when
                ``torch_compile=False``; falls back to per-step dispatch when a
                path has no loop-scope variant (e.g. dynamic step-skip).
            attn_backend: ``"sdpa"`` (default) runs joint attention as SDPA
                with a dense bool mask (memory-efficient backend).
                ``"flex"`` serves the static per-regime joint mask as a
                FlexAttention BlockMask (~2x faster masked attention; joint
                denoise paths only — the action-only KV-cache path and HBridge
                outer layers stay on SDPA).
                ``"auto"`` picks the fastest SDPA kernel per site: the
                action-fast-path masks are all-True by construction, so they
                are dropped and SDPA dispatches flash (~4x over the masked
                memory-efficient kernel at deploy shapes); joint denoise
                paths keep their non-trivial dense mask but run with
                cuDNN-first backend priority (~2.6x at joint shapes on
                Blackwell). Same math as "sdpa" — only the kernel changes.
                Mask drops apply on the compiled fast paths
                (``torch_compile=True``).
            compile_encoders: Also compile the observation encoders (VAE
                first-frame encode, DINO encode, pointmap encode) — outside
                the MoT compile targets but ~20 ms/call at deploy shapes.
                Off by default; requires ``torch_compile=True``.
            trt_joint_engine_path: Serialized TensorRT engine for the flex
                JOINT denoise core (built offline by
                ``scripts/inference_opt/trt_onnx_joint_engine.py``). When set, the joint
                core routes through the engine while encoders, pre/post_dit
                and schedulers stay in torch; the joint step itself runs
                uncompiled (the engine call cannot live inside an inductor
                CUDA graph) and off-shape calls fall back to the torch path.
                Incompatible with ``attn_backend="flex"`` (the engine
                consumes the dense joint mask). The engine carries its own
                fp16 weights, so pair with ``quantization=null``.
            trt_joint_free_video_blocks: With the engine active, release the
                torch-side video-expert block weights (~8.5 GB — the engine
                duplicates them in fp16). Frees VRAM for co-resident
                consumers (SAPIEN/Vulkan sim rendering); the torch fallback
                for the joint core then RAISES instead of falling back, and
                action-regime prefill / checkpointing become unavailable.
            trt_prefill_engine_path: Serialized TensorRT engine for the
                action-only KV-cache prefill (the 30-layer video-expert pass
                over the anchor tokens — the biggest action-path phase after
                denoise). Built offline by
                ``scripts/inference_opt/trt_onnx_prefill_engine.py``; shadows
                ``mot.prefill_video_cache``, off-shape calls fall back to
                torch. Independent of ``trt_joint_engine_path`` (different
                regime).
            solver: Inference ODE solver — ``"euler"`` (default, unchanged) or
                ``"dpmpp_2m"`` (DPM-Solver++(2M), a 2nd-order multistep that
                holds quality at fewer denoise steps). Engages on the eager /
                per-step joint denoise loop (including the TRT-engine path);
                the loop-scope-compiled joint/action paths stay Euler (a
                per-step host sync would break their CUDA-graph capture). Off =
                byte-identical.
            glue_cache: Memoize per-call/per-step CONSTANT inference glue —
                DINO/pointmap RoPE freqs, the joint attention mask, and the
                HBridge self-masks — keyed on shapes + device + regime bits.
                All are pure functions of those keys, so hits are bit-identical
                to a rebuild; kills ~25 ms of host-side CPU (and the GPU idle
                it causes) per joint TRT-engine call. Inference-only (memos are
                bypassed under ``self.training``). Off (default) = rebuild
                every time, exactly the prior behavior.
            encoder_cuda_graph: Capture each of the three first-frame anchor
                encoders (VAE / DINO / pointmap) as a CUDA graph on its first
                deploy-signature call and replay it as a single launch — the
                encoders are host-LAUNCH-rate-limited (§9.6), so this removes
                their ~7-12 ms of per-call launch cost each. Bitwise replay
                self-check at capture; any capture failure or off-signature
                call (val full-video encodes, tiled) falls back to eager.
                Mutually exclusive with ``compile_encoders``. Off (default) =
                eager encoders, exactly the prior behavior.
            joint_loop_cuda_graph: Capture the entire 10-step joint denoise
                loop (``_joint_denoise_loop_body``: per-step pre_dit → joint
                core → post_dit → x0→v → Euler updates → frame-0 reclamps) as
                ONE CUDA graph and replay it per infer call — removing the
                per-step eager host-launch cost that the TRT-engine path pays
                because inductor cannot trace the opaque engine call (manual
                capture can). Bit-identical to the eager loop under Euler +
                step-skip-off (verified by a warmup-vs-replay self-check);
                engages only in that regime and falls back to eager otherwise.
                Off (default) = the per-step eager loop, exactly the prior
                behavior. MEASURED CAVEAT: with ``trt_joint_engine_path``
                active the capture fails — the TRT engine's ``execute_async_v3``
                enqueue is not CUDA-graph-capturable (bare capture error, not a
                hoistable op) — and silently falls back to eager, so it is a
                no-op on the deploy engine path (the host-overhead ceiling it
                targets is only ~3%). Retained for the non-engine compiled path
                / a future capturable engine.
        """
        if torch_compile_scope not in ("step", "loop"):
            raise ValueError(
                f"`torch_compile_scope` must be 'step' or 'loop', got {torch_compile_scope!r}"
            )
        if attn_backend not in ("sdpa", "flex", "auto"):
            raise ValueError(
                f"`attn_backend` must be 'sdpa', 'flex', or 'auto', got {attn_backend!r}"
            )
        if torch_compile_backend not in ("inductor", "tensorrt"):
            raise ValueError(
                f"`torch_compile_backend` must be 'inductor' or 'tensorrt', "
                f"got {torch_compile_backend!r}"
            )
        self._glue_cache_enabled = bool(glue_cache)
        self._glue_cache = {}
        if encoder_cuda_graph and compile_encoders:
            raise ValueError(
                "`encoder_cuda_graph` and `compile_encoders` are mutually "
                "exclusive — both wrap the same encoder methods."
            )
        self._joint_loop_cuda_graph_enabled = bool(joint_loop_cuda_graph)
        if joint_loop_cuda_graph and torch_compile and torch_compile_scope == "loop":
            raise ValueError(
                "`joint_loop_cuda_graph` and torch_compile_scope='loop' are "
                "mutually exclusive — both capture the whole denoise loop "
                "(manual CUDA graph vs inductor)."
            )
        if torch_compile_backend == "tensorrt":
            if quantization is not None:
                raise ValueError(
                    "torch_compile_backend='tensorrt' is incompatible with torchao "
                    "quantization (tensor subclasses don't lower to TRT). Use "
                    "quantization=null."
                )
            try:
                import torch_tensorrt  # noqa: F401 — registers the dynamo backend
            except ImportError:
                raise ImportError(
                    "torch_compile_backend='tensorrt' requires torch-tensorrt. "
                    "Install with: pip install torch-tensorrt"
                )
        if quantization is not None:
            quantization = str(quantization).strip().lower()
            if quantization in ("none", "null", ""):
                quantization = None
        if trt_joint_engine_path is not None:
            if attn_backend == "flex":
                raise ValueError(
                    "trt_joint_engine_path requires attn_backend 'sdpa' or "
                    "'auto' — the engine consumes the dense joint mask, but "
                    "'flex' serves it as a BlockMask (mask=None at the core)."
                )
            if not Path(trt_joint_engine_path).is_file():
                raise FileNotFoundError(
                    f"trt_joint_engine_path not found: {trt_joint_engine_path}"
                )

        if quantization is not None:
            try:
                from torchao.quantization import quantize_
            except ImportError:
                raise ImportError(
                    "Quantization requires torchao. Install with: pip install torchao"
                )
            if quantization == "int8":
                from torchao.quantization import int8_weight_only
                quantize_(self.mot, int8_weight_only())
                logger.info("Applied INT8 weight-only quantization to MoT backbone.")
            elif quantization in ("fp8", "fp8_video"):
                from torchao.quantization import (
                    Float8DynamicActivationFloat8WeightConfig,
                    PerTensor,
                )

                # FP8 kernels (torch._scaled_mm) need both GEMM dims % 16 == 0;
                # skip odd-shaped Linears (e.g. the 14-dim action head).
                def _fp8_filter(m: torch.nn.Module, fqn: str) -> bool:
                    return (
                        isinstance(m, torch.nn.Linear)
                        and m.in_features % 16 == 0
                        and m.out_features % 16 == 0
                    )

                # "fp8_video": quantize only the video expert. Its linears run
                # at M>=534 (prefill / joint streams) where fp8 wins ~35%; the
                # action expert runs at M=32 where dynamic-quant overhead makes
                # fp8 a net loss — leave it bf16.
                target = (
                    self.mot.mixtures["video"] if quantization == "fp8_video" else self.mot
                )
                quantize_(
                    target,
                    Float8DynamicActivationFloat8WeightConfig(granularity=PerTensor()),
                    filter_fn=_fp8_filter,
                )
                logger.info(
                    "Applied FP8 dynamic-activation + FP8-weight (per-tensor) "
                    "quantization to %s.",
                    "video expert only" if quantization == "fp8_video" else "MoT backbone",
                )
            else:
                raise ValueError(
                    f"Unsupported quantization method: {quantization!r}. "
                    "Supported: 'int8', 'fp8', 'fp8_video'."
                )

        # Lazy-compile: set flags here; the per-step compiled methods are wrapped
        # on first call inside `_predict_action_noise_with_cache` (and the joint
        # equivalents in subclasses). This avoids compiling the original
        # `mot.forward_action_with_video_cache` directly — that path has graph
        # breaks (dict cache, validation, dynamic mask slicing) that prevent
        # CUDA Graph capture.
        self._compile_inference = bool(torch_compile)
        self._compile_mode = torch_compile_mode
        self._compile_scope = torch_compile_scope
        self._infer_attn_backend = attn_backend
        self._compile_backend = torch_compile_backend
        if solver != "euler":
            n = 0
            for attr in (
                "infer_video_scheduler", "infer_action_scheduler",
                "infer_dino_scheduler", "infer_pointmap_scheduler",
            ):
                sch = getattr(self, attr, None)
                if sch is not None and hasattr(sch, "set_solver"):
                    sch.set_solver(solver)
                    n += 1
            if torch_compile and torch_compile_scope == "loop":
                logger.warning(
                    "solver=%r set but torch_compile_scope='loop' — the "
                    "loop-scope joint/action graphs stay Euler; the solver "
                    "only engages on the eager/per-step (incl. TRT-engine) "
                    "joint path.", solver,
                )
            logger.info("Inference solver set to %r on %d scheduler(s).", solver, n)
        if trt_joint_engine_path is not None:
            from .inference_opt.trt_joint import install_trt_joint_adapter
            self._trt_joint_runner = install_trt_joint_adapter(
                self, trt_joint_engine_path,
                free_video_blocks=trt_joint_free_video_blocks,
            )
        if trt_joint_decode_split_engine_path is not None:
            if trt_joint_engine_path is not None:
                raise ValueError(
                    "trt_joint_decode_split_engine_path and trt_joint_engine_path "
                    "both shadow mot._forward_joint_inner — pick one."
                )
            if trt_joint_prefill_split_engine_path is None:
                raise ValueError(
                    "trt_joint_decode_split_engine_path requires "
                    "trt_joint_prefill_split_engine_path."
                )
            for _p in (trt_joint_prefill_split_engine_path, trt_joint_decode_split_engine_path):
                if not Path(_p).is_file():
                    raise FileNotFoundError(f"split engine not found: {_p}")
            if attn_backend == "flex":
                raise ValueError(
                    "trt_joint split engines require attn_backend 'sdpa' or 'auto' "
                    "(engines consume the dense joint mask)."
                )
            from .inference_opt.trt_joint_split import install_trt_joint_split_adapter
            self._trt_joint_runner = install_trt_joint_split_adapter(
                self, trt_joint_prefill_split_engine_path,
                trt_joint_decode_split_engine_path,
                free_video_blocks=trt_joint_free_video_blocks,
            )
        if trt_prefill_engine_path is not None:
            if not Path(trt_prefill_engine_path).is_file():
                raise FileNotFoundError(
                    f"trt_prefill_engine_path not found: {trt_prefill_engine_path}"
                )
            from .inference_opt.trt_prefill import install_trt_prefill_adapter
            self._trt_prefill_runner = install_trt_prefill_adapter(
                self, trt_prefill_engine_path
            )
            logger.info(
                "TRT joint engine active (%s); joint denoise step will run "
                "uncompiled around the engine call.",
                trt_joint_engine_path,
            )
        if encoder_cuda_graph:
            from .inference_opt.encoder_graphs import install_encoder_cuda_graphs
            install_encoder_cuda_graphs(self)
        if joint_loop_cuda_graph and hasattr(self, "_joint_denoise_loop_body"):
            from .inference_opt.joint_loop_graph import install_joint_loop_cuda_graph
            install_joint_loop_cuda_graph(self)
        if torch_compile and attn_backend == "auto":
            # Salt the inductor cache dir: "auto" changes the SDPA kernel
            # choice (cuDNN-first priority) but that choice is NOT part of the
            # FX cache key, so artifacts compiled under the default priority
            # would be silently reused — either keeping the old slow kernel or
            # tripping stride asserts when the extern SDPA re-dispatches at
            # runtime. A per-backend cache dir keeps both worlds consistent.
            import getpass
            import os
            import tempfile
            base = os.environ.get("TORCHINDUCTOR_CACHE_DIR")
            if base is None:
                base = os.path.join(
                    tempfile.gettempdir(), f"torchinductor_{getpass.getuser()}"
                )
            if not base.endswith("-sdpa-auto"):
                os.environ["TORCHINDUCTOR_CACHE_DIR"] = base + "-sdpa-auto"
            try:  # env var is snapshotted behind an lru_cache; refresh it
                from torch._inductor.runtime.cache_dir_utils import cache_dir
                cache_dir.cache_clear()
            except (ImportError, AttributeError):
                pass
        if torch_compile and torch_compile_backend == "tensorrt":
            # TRT engines replace inductor's CUDA graphs; use torch-tensorrt's
            # own runtime cudagraphs to keep graph-launch semantics.
            import torch_tensorrt
            torch_tensorrt.runtime.set_cudagraphs_mode(True)
        if torch_compile:
            # Move RoPE caches to GPU once so per-call `.to(tokens.device)` in
            # action_dit / wan_video_dit pre_dit becomes a same-device no-op
            # and inductor can capture a CUDA Graph (a CPU tensor leaking into
            # the compiled region forces "skipping cudagraphs due to cpu
            # device"). RoPE caches are kept as plain attributes (not buffers)
            # to avoid nn.Module.to(bfloat16) silently downcasting their
            # complex128 dtype, so we move them explicitly here.
            if self.device.type == "cuda":
                if hasattr(self.action_expert, "freqs") and torch.is_tensor(self.action_expert.freqs):
                    self.action_expert.freqs = self.action_expert.freqs.to(self.device)
                if hasattr(self.video_expert, "freqs"):
                    vf = self.video_expert.freqs
                    if isinstance(vf, (tuple, list)) and all(torch.is_tensor(t) for t in vf):
                        self.video_expert.freqs = type(vf)(t.to(self.device) for t in vf)
                    elif torch.is_tensor(vf):
                        self.video_expert.freqs = vf.to(self.device)
            logger.info(
                "torch_compile enabled (mode=%r); compiled denoise step will be "
                "built lazily on first infer_action call.",
                torch_compile_mode,
            )
        if compile_encoders and torch_compile and self.device.type == "cuda":
            # Observation encoders run once per infer call outside the MoT
            # compile targets (~20 ms combined at deploy shapes). Instance
            # attributes shadow the bound methods, so the original class
            # methods stay untouched. Deploy shapes are fixed, so
            # reduce-overhead CUDA graphs apply cleanly.
            compiled = []
            if getattr(self, "dino_encoder", None) is not None:
                self.dino_encoder.encode_video = torch.compile(
                    self.dino_encoder.encode_video, mode=torch_compile_mode, fullgraph=False,
                )
                compiled.append("dino_encoder.encode_video")
            if hasattr(self, "_encode_input_image_latents_tensor"):
                self._encode_input_image_latents_tensor = torch.compile(
                    self._encode_input_image_latents_tensor, mode=torch_compile_mode, fullgraph=False,
                )
                compiled.append("_encode_input_image_latents_tensor")
            if hasattr(self, "_encode_first_frame_pointmap_raw"):
                # Deploy feeds raw uint16 depth; reduce-overhead CUDA graphs
                # fail on uint16 static-input copies ("foreach_tensor_copy not
                # implemented for 'UInt16'") — compile this one without graphs.
                pm_mode = (
                    "default" if torch_compile_mode == "reduce-overhead"
                    else torch_compile_mode
                )
                self._encode_first_frame_pointmap_raw = torch.compile(
                    self._encode_first_frame_pointmap_raw, mode=pm_mode, fullgraph=False,
                )
                compiled.append("_encode_first_frame_pointmap_raw")
            logger.info("compile_encoders: compiled %s", ", ".join(compiled) or "nothing")

    def _compile_for_inference(self, fn):
        """torch.compile ``fn`` with the configured inference backend.

        inductor (default): ``mode=self._compile_mode`` — reduce-overhead
        captures CUDA graphs. tensorrt: torch-tensorrt's dynamo backend lowers
        subgraphs to TRT engines (bf16/fp16/fp32; mode is inductor-only so it
        is not passed).
        """
        if getattr(self, "_compile_backend", "inductor") == "tensorrt":
            return torch.compile(
                fn,
                backend="tensorrt",
                options={
                    "enabled_precisions": {torch.float32, torch.float16, torch.bfloat16},
                    "min_block_size": 1,
                    "truncate_double": True,
                },
                fullgraph=False,
            )
        return torch.compile(
            fn,
            mode=getattr(self, "_compile_mode", "reduce-overhead"),
            fullgraph=False,
        )

    @staticmethod
    def _check_resize_height_width(height, width, num_frames):
        if height % 16 != 0:
            height = (height + 15) // 16 * 16
        if width % 16 != 0:
            width = (width + 15) // 16 * 16
        if num_frames % 4 != 1:
            num_frames = (num_frames + 3) // 4 * 4 + 1
        return height, width, num_frames

    @torch.no_grad()
    def encode_prompt(self, prompt: Union[str, Sequence[str]]):
        if self.text_encoder is None or self.tokenizer is None:
            raise ValueError(
                "Prompt encoding requires loaded text encoder/tokenizer. "
                "Set `load_text_encoder=true` or provide precomputed `context/context_mask`."
            )
        if self.offload_text_encoder:
            self.text_encoder.cpu()
            ids, mask = self.tokenizer(prompt, return_mask=True, add_special_tokens=True)
            ids = ids.to("cpu")
            mask = mask.to("cpu", dtype=torch.bool)
            prompt_emb = self.text_encoder(ids, mask)
        else:
            ids, mask = self.tokenizer(prompt, return_mask=True, add_special_tokens=True)
            ids = ids.to(self.device)
            mask = mask.to(self.device, dtype=torch.bool)
            prompt_emb = self.text_encoder(ids, mask)
        # FIXME: original implementation's zero padding is visible in cross-attn.
        seq_lens = mask.gt(0).sum(dim=1).long()
        for i, v in enumerate(seq_lens):
            prompt_emb[i, v:] = 0
        mask = torch.ones_like(mask)
        return prompt_emb.to(device=self.device), mask

    def _append_proprio_to_context(
        self,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        proprio: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.proprio_encoder is None or proprio is None:
            return context, context_mask
        if proprio.ndim != 2:
            raise ValueError(f"`proprio` must be 2D [B, D], got shape {tuple(proprio.shape)}")
        if self.proprio_dim is None or proprio.shape[1] != self.proprio_dim:
            raise ValueError(
                f"`proprio` last dim must be {self.proprio_dim}, got {proprio.shape[1]}"
            )
        proprio_token = self.proprio_encoder(
            proprio.to(device=self.device, dtype=context.dtype).unsqueeze(1)
        ).to(dtype=context.dtype) # [B, 1, D]
        proprio_mask = torch.ones((context_mask.shape[0], 1), dtype=torch.bool, device=context_mask.device)
        return (
            torch.cat([context, proprio_token], dim=1),
            torch.cat([context_mask, proprio_mask], dim=1),
        )

    @torch.no_grad()
    def _encode_video_latents(self, video_tensor, tiled=False, tile_size=(30, 52), tile_stride=(15, 26)):
        z = self.vae.encode(
            video_tensor,
            device=self.device,
            tiled=tiled,
            tile_size=tile_size,
            tile_stride=tile_stride,
        )
        return z

    @torch.no_grad()
    def _encode_input_image_latents_tensor(self, input_image: torch.Tensor, tiled=False, tile_size=(30, 52), tile_stride=(15, 26)):
        if input_image.ndim == 3:
            input_image = input_image.unsqueeze(0)
        if input_image.ndim != 4 or input_image.shape[0] != 1 or input_image.shape[1] != 3:
            raise ValueError(
                f"`input_image` must have shape [1,3,H,W] or [3,H,W], got {tuple(input_image.shape)}"
            )
        image = input_image.to(device=self.device)[0].unsqueeze(1)
        z = self.vae.encode([image], device=self.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)
        if isinstance(z, list):
            z = z[0].unsqueeze(0)
        return z

    def _decode_latents(self, latents, tiled=False, tile_size=(30, 52), tile_stride=(15, 26)):
        video_tensor = self.vae.decode(latents, device=self.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)
        video_tensor = video_tensor.squeeze(0).detach().float().clamp(-1, 1)
        video_tensor = ((video_tensor + 1.0) * 127.5).to(torch.uint8).cpu()
        frames = []
        for t in range(video_tensor.shape[1]):
            frame = video_tensor[:, t].permute(1, 2, 0).numpy()
            frames.append(Image.fromarray(frame))
        return frames

    def build_inputs(self, sample, tiled: bool = False):
        # Per-camera dataset path: assemble composite from per_cam on GPU and
        # write it back into `sample` so subclass build_inputs methods (which
        # call super() and then read sample['video'] again) see the composite.
        # Layout-aware: uses ``self._layout`` + ``self._slot_key_map`` so any
        # registered layout works without code changes.
        if "per_cam" in sample and "video" not in sample:
            from flexpi.per_cam_compose import compose_from_per_cam
            per_cam_gpu = {
                k: v.to(device=self.device, dtype=self.torch_dtype, non_blocking=True)
                for k, v in sample["per_cam"].items()
            }
            sample["video"] = compose_from_per_cam(
                per_cam_gpu, self._layout, slot_key_map=self._slot_key_map,
            )
        video = sample["video"]
        if "context" not in sample or "context_mask" not in sample:
            raise ValueError(
                "FlexPi training requires `sample['context']` and `sample['context_mask']`."
            )
        context = sample["context"]
        context_mask = sample["context_mask"]
        proprio = sample.get("proprio", None)
        if video.ndim != 5:
            raise ValueError(f"`sample['video']` must be 5D [B, 3, T, H, W], got shape {tuple(video.shape)}")
        if video.shape[1] != 3:
            raise ValueError(f"`sample['video']` channel dimension must be 3, got shape {tuple(video.shape)}")

        batch_size, _, num_frames, height, width = video.shape
        if height % 16 != 0 or width % 16 != 0:
            raise ValueError(
                f"Video spatial dims must be multiples of 16, got H={height}, W={width}"
            )
        if num_frames % 4 != 1:
            raise ValueError(f"Video T must satisfy T % 4 == 1, got T={num_frames}")
        if num_frames <= 1:
            raise ValueError(f"Video T must be > 1 for action-conditioned training, got T={num_frames}")

        if "action" not in sample:
            raise ValueError("`sample['action']` is required for FlexPi training.")

        action = sample["action"]
        if action.ndim != 3:
            raise ValueError(f"`sample['action']` must be 3D [B, T, a_dim], got shape {tuple(action.shape)}")
        action_horizon = int(action.shape[1])
        if action_horizon % (num_frames - 1) != 0:
            raise ValueError(
                f"`sample['action']` temporal dimension must be divisible by video transitions ({num_frames - 1}), got {action_horizon}"
            )

        action_is_pad = sample.get("action_is_pad", None)
        if action_is_pad is not None:
            if action_is_pad.ndim != 2:
                raise ValueError(
                    f"`sample['action_is_pad']` must be 2D [B, T], got shape {tuple(action_is_pad.shape)}"
                )
            if action_is_pad.shape[0] != batch_size or action_is_pad.shape[1] != action_horizon:
                raise ValueError(
                    "`sample['action_is_pad']` shape mismatch: "
                    f"got {tuple(action_is_pad.shape)} vs expected ({batch_size}, {action_horizon})"
                )

        action_dim = int(action.shape[-1])
        action_dim_is_pad = sample.get("action_dim_is_pad", None)
        if action_dim_is_pad is not None:
            if action_dim_is_pad.ndim != 2:
                raise ValueError(
                    f"`sample['action_dim_is_pad']` must be 2D [B, A], got shape {tuple(action_dim_is_pad.shape)}"
                )
            if action_dim_is_pad.shape[0] != batch_size or action_dim_is_pad.shape[1] != action_dim:
                raise ValueError(
                    "`sample['action_dim_is_pad']` shape mismatch: "
                    f"got {tuple(action_dim_is_pad.shape)} vs expected ({batch_size}, {action_dim})"
                )

        image_is_pad = sample.get("image_is_pad", None)
        if image_is_pad is not None:
            if image_is_pad.ndim != 2:
                raise ValueError(
                    f"`sample['image_is_pad']` must be 2D [B, T], got shape {tuple(image_is_pad.shape)}"
                )
            if image_is_pad.shape[0] != batch_size or image_is_pad.shape[1] != num_frames:
                raise ValueError(
                    "`sample['image_is_pad']` shape mismatch: "
                    f"got {tuple(image_is_pad.shape)} vs expected ({batch_size}, {num_frames})"
                )

        input_video = video.to(device=self.device, dtype=self.torch_dtype, non_blocking=True)
        input_latents = self._encode_video_latents(input_video, tiled=tiled)

        first_frame_latents = None
        fuse_flag = False
        if getattr(self.video_expert, "fuse_vae_embedding_in_latents", False):
            first_frame_latents = input_latents[:, :, 0:1]
            fuse_flag = True

        if context.ndim != 3 or context_mask.ndim != 2:
            raise ValueError(
                f"`context/context_mask` must be [B,L,D]/[B,L], got {tuple(context.shape)} and {tuple(context_mask.shape)}"
            )
        context = context.to(device=self.device, dtype=self.torch_dtype, non_blocking=True)
        context_mask = context_mask.to(device=self.device, dtype=torch.bool, non_blocking=True)
        if self.proprio_encoder is not None:
            if proprio is None:
                raise ValueError("`sample['proprio']` is required when `proprio_dim` is enabled.")
            if proprio.ndim != 3:
                raise ValueError(f"`sample['proprio']` must be 3D [B, T, d], got shape {tuple(proprio.shape)}")
            if proprio.shape[2] != self.proprio_dim:
                raise ValueError(
                    f"`sample['proprio']` last dim must be {self.proprio_dim}, got {proprio.shape[2]}"
                )
            proprio = proprio[:, 0, :] # [B, D]
            context, context_mask = self._append_proprio_to_context(
                context=context,
                context_mask=context_mask,
                proprio=proprio.to(device=self.device, dtype=self.torch_dtype),
            )
        action = action.to(device=self.device, dtype=self.torch_dtype, non_blocking=True)

        if action_is_pad is not None:
            action_is_pad = action_is_pad.to(device=self.device, dtype=torch.bool, non_blocking=True)
        if action_dim_is_pad is not None:
            action_dim_is_pad = action_dim_is_pad.to(device=self.device, dtype=torch.bool, non_blocking=True)
        if image_is_pad is not None:
            image_is_pad = image_is_pad.to(device=self.device, dtype=torch.bool, non_blocking=True)

        return {
            "context": context,
            "context_mask": context_mask,
            "input_latents": input_latents,
            "first_frame_latents": first_frame_latents,
            "fuse_vae_embedding_in_latents": fuse_flag,
            "action": action,
            "action_is_pad": action_is_pad,
            "action_dim_is_pad": action_dim_is_pad,
            "image_is_pad": image_is_pad,
        }

    @torch.no_grad()
    def _build_mot_attention_mask(
        self,
        video_seq_len: int,
        action_seq_len: int,
        video_tokens_per_frame: int,
        device: torch.device,
    ) -> torch.Tensor:
        total_seq_len = video_seq_len + action_seq_len
        mask = torch.zeros((total_seq_len, total_seq_len), dtype=torch.bool, device=device)

        # video -> video
        mask[:video_seq_len, :video_seq_len] = self.video_expert.build_video_to_video_mask(
            video_seq_len=video_seq_len,
            video_tokens_per_frame=video_tokens_per_frame,
            device=device,
        )
        # action -> action
        mask[video_seq_len:, video_seq_len:] = True
        # action -> first-frame video only
        first_frame_tokens = min(video_tokens_per_frame, video_seq_len)
        mask[video_seq_len:, :first_frame_tokens] = True
        return mask

    @torch.no_grad()
    def _build_hbridge_self_masks(
        self,
        video_seq_len: int,
        action_seq_len: int,
        video_tokens_per_frame: int,
        device: torch.device,
    ) -> tuple[Optional[list[int]], Optional[list[torch.Tensor]]]:
        """Per-sub-stream self-masks for HBridge outer layers.

        For the base FlexPi (V + A), the sub-streams are ``[V, A]`` and each
        gets its own internal self-mask. Returns ``(None, None)`` when HBridge
        is disabled so MoT skips the per-sub-stream attention path entirely.
        """
        if not self.mot.hbridge_enabled:
            return None, None
        video_self_mask = self.video_expert.build_video_to_video_mask(
            video_seq_len=video_seq_len,
            video_tokens_per_frame=video_tokens_per_frame,
            device=device,
        )
        action_self_mask = torch.ones((action_seq_len, action_seq_len), dtype=torch.bool, device=device)
        return [video_seq_len, action_seq_len], [video_self_mask, action_self_mask]

    @torch.no_grad()
    def _build_action_only_attention_mask(
        self,
        action_seq_len: int,
        device: torch.device,
    ) -> Optional[torch.Tensor]:
        """Action-only attention mask for HBridge outer layers in fast inference.

        When HBridge is enabled, in outer layers action queries skip the cached
        video K/V entirely and self-attend instead. Returns ``None`` when HBridge
        is disabled so the cached path falls through to standard joint attention.
        """
        if not self.mot.hbridge_enabled:
            return None
        return torch.ones((action_seq_len, action_seq_len), dtype=torch.bool, device=device)

    @staticmethod
    def _aux_per_frame_is_pad(
        image_is_pad: torch.Tensor,
        aux_temporal_stride: int,
        keep_far: bool = False,
    ) -> torch.Tensor:
        """Per-RGB-frame is_pad mask for the DINO stream.

        Mirrors ``DinoEncoder.encode_video`` temporal indexing exactly (slot
        selection routes through ``select_aux_frame_slots``; ``keep_far``
        mirrors the encoder's ``stride_keep_far``):

            num_latent   = (T_video - 1) // 4 + 1
            vae_indices  = [min(4*i, T_video-1) for i in range(num_latent)]
            aux_indices  = [vae_indices[i] for i in
                            select_aux_frame_slots(num_latent,
                                                   aux_temporal_stride,
                                                   keep_far)]

        These ``aux_indices`` are positions INTO the strided video tensor
        (equivalently INTO ``image_is_pad``, since the dataset emits
        ``image_is_pad`` at the same strided granularity — length 9 by
        default). The leading entry is the first-frame anchor; callers
        that drop it from the loss should slice ``[:, 1:]`` on the result.

        Args:
            image_is_pad: ``[B, T_video]`` bool.
            aux_temporal_stride: ``dino_temporal_stride``.

        Returns:
            ``[B, F_aux]`` bool matching the encoder's output frame count
            for the given stride.
        """
        if image_is_pad.ndim != 2:
            raise ValueError(
                f"image_is_pad must be [B, T_video]; got shape {tuple(image_is_pad.shape)}"
            )
        T = image_is_pad.shape[1]
        num_latent = (T - 1) // 4 + 1
        if num_latent < 1:
            raise ValueError(f"T_video={T} produces no latent frames.")
        vae_indices = [min(4 * i, T - 1) for i in range(num_latent)]
        aux_indices = [
            vae_indices[i]
            for i in select_aux_frame_slots(num_latent, aux_temporal_stride, keep_far)
        ]
        return image_is_pad[:, aux_indices]

    @staticmethod
    def _masked_loss_reduction(
        loss_per_frame: torch.Tensor,
        is_pad_per_frame: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Mean over the frame dim, masking out padded frames.

        Equivalent (up to fp32 reordering) to ``loss_per_frame.mean(dim=1)``
        when ``is_pad_per_frame`` is None or all-False.

        Args:
            loss_per_frame: ``[B, F]`` float.
            is_pad_per_frame: ``[B, F]`` bool or None.

        Returns:
            ``[B]`` masked-mean per sample. Samples whose frames are ALL
            padded yield 0 (numerator zero; denominator clamped to 1 to
            avoid div-by-zero — gradient is still 0 since ``valid`` is
            all-zero).
        """
        if is_pad_per_frame is None:
            return loss_per_frame.mean(dim=1)
        if is_pad_per_frame.shape != loss_per_frame.shape:
            raise ValueError(
                "Loss/mask shape mismatch: "
                f"loss={tuple(loss_per_frame.shape)}, mask={tuple(is_pad_per_frame.shape)}"
            )
        valid = (~is_pad_per_frame).to(
            device=loss_per_frame.device, dtype=loss_per_frame.dtype,
        )
        return (loss_per_frame * valid).sum(dim=1) / valid.sum(dim=1).clamp(min=1.0)

    @staticmethod
    def _aggregate_image_is_pad_to_video_latents(
        image_is_pad: torch.Tensor,
        temporal_factor: int,
        include_initial: bool,
    ) -> torch.Tensor:
        """Aggregate per-image-frame `is_pad` into per-latent-frame `is_pad`.

        A latent step is marked padded only when **all** ``temporal_factor`` raw
        image frames it covers are padded — conservative but principled (matches
        the existing video-loss treatment).

        Args:
            image_is_pad: ``[B, T_image]`` bool. ``T_image - 1`` must be
                divisible by ``temporal_factor``. ``T_image`` equals the
                dataset's video-subsample count (9 with the default
                ``num_frames=33``, ``action_video_freq_ratio=4``).
            temporal_factor: VAE temporal compression factor (4 for WAN 2.2).
            include_initial: if True, the returned mask includes the leading
                latent (which always covers raw frame 0). Set False when the
                loss already drops the leading latent.

        Returns:
            ``[B, T_lat]`` bool, where ``T_lat = (T_image - 1) // factor``
            (without init) or ``1 + that`` (with init).
        """
        if temporal_factor <= 0:
            raise ValueError(f"temporal_factor must be positive, got {temporal_factor}.")
        if image_is_pad.shape[1] < 1:
            raise ValueError("`image_is_pad` must contain at least one frame.")
        if (image_is_pad.shape[1] - 1) % temporal_factor != 0:
            raise ValueError(
                "Cannot align `image_is_pad` with video latent steps: "
                f"num_frames={image_is_pad.shape[1]}, temporal_factor={temporal_factor}."
            )
        tail_is_pad = image_is_pad[:, 1:]
        latent_tail_is_pad = tail_is_pad.view(
            image_is_pad.shape[0], -1, temporal_factor,
        ).all(dim=2)
        if include_initial:
            return torch.cat([image_is_pad[:, :1], latent_tail_is_pad], dim=1)
        return latent_tail_is_pad

    def _compute_video_loss_per_sample(
        self,
        pred_video: torch.Tensor,
        target_video: torch.Tensor,
        image_is_pad: Optional[torch.Tensor],
        include_initial_video_step: bool,
    ) -> torch.Tensor:
        video_loss_token = F.mse_loss(pred_video.float(), target_video.float(), reduction="none").mean(dim=(1, 3, 4))
        if image_is_pad is None:
            return video_loss_token.mean(dim=1)

        video_is_pad = self._aggregate_image_is_pad_to_video_latents(
            image_is_pad,
            temporal_factor=int(self.vae.temporal_downsample_factor),
            include_initial=include_initial_video_step,
        )

        if video_is_pad.shape[1] != video_loss_token.shape[1]:
            raise ValueError(
                "Video-loss mask shape mismatch: "
                f"mask steps={video_is_pad.shape[1]}, loss steps={video_loss_token.shape[1]}."
            )

        valid = (~video_is_pad).to(device=video_loss_token.device, dtype=video_loss_token.dtype)
        valid_sum = valid.sum(dim=1).clamp(min=1.0)
        return (video_loss_token * valid).sum(dim=1) / valid_sum


    @torch.no_grad()
    def _predict_joint_noise(
        self,
        latents_video: torch.Tensor,
        latents_action: torch.Tensor,
        timestep_video: torch.Tensor,
        timestep_action: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        fuse_vae_embedding_in_latents: bool,
        gt_action: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if getattr(self, "_compile_inference", False) and self.device.type == "cuda":
            if not getattr(self, "_joint_step_is_compiled", False):
                self._joint_step_compiled = torch.compile(
                    self._predict_joint_noise_impl,
                    mode=getattr(self, "_compile_mode", "reduce-overhead"),
                    fullgraph=False,
                )
                self._joint_step_is_compiled = True
            return self._joint_step_compiled(
                latents_video=latents_video,
                latents_action=latents_action,
                timestep_video=timestep_video,
                timestep_action=timestep_action,
                context=context,
                context_mask=context_mask,
                fuse_vae_embedding_in_latents=fuse_vae_embedding_in_latents,
                gt_action=gt_action,
            )
        return self._predict_joint_noise_impl(
            latents_video=latents_video,
            latents_action=latents_action,
            timestep_video=timestep_video,
            timestep_action=timestep_action,
            context=context,
            context_mask=context_mask,
            fuse_vae_embedding_in_latents=fuse_vae_embedding_in_latents,
            gt_action=gt_action,
        )

    def _predict_joint_noise_impl(
        self,
        latents_video: torch.Tensor,
        latents_action: torch.Tensor,
        timestep_video: torch.Tensor,
        timestep_action: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        fuse_vae_embedding_in_latents: bool,
        gt_action: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        video_pre = self.video_expert.pre_dit(
            x=latents_video,
            timestep=timestep_video,
            context=context,
            context_mask=context_mask,
            action=gt_action,
            fuse_vae_embedding_in_latents=fuse_vae_embedding_in_latents,
        )
        action_pre = self.action_expert.pre_dit(
            action_tokens=latents_action,
            timestep=timestep_action,
            context=context,
            context_mask=context_mask,
        )

        attention_mask = self._build_mot_attention_mask(
            video_seq_len=video_pre["tokens"].shape[1],
            action_seq_len=action_pre["tokens"].shape[1],
            video_tokens_per_frame=int(video_pre["meta"]["tokens_per_frame"]),
            device=video_pre["tokens"].device,
        )
        sub_stream_lens, sub_stream_self_masks = self._build_hbridge_self_masks(
            video_seq_len=video_pre["tokens"].shape[1],
            action_seq_len=action_pre["tokens"].shape[1],
            video_tokens_per_frame=int(video_pre["meta"]["tokens_per_frame"]),
            device=video_pre["tokens"].device,
        )

        # Use the flat-arg MoT inner (compile-friendly: no dict iteration,
        # no validation, no gradient checkpointing). Equivalent to
        # `self.mot(embeds_all={"video": ..., "action": ...}, ...)` for valid
        # inputs; this method is `@torch.no_grad()` inference-only.
        video_tokens_out, action_tokens_out = self.mot._forward_joint_inner(
            video_tokens=video_pre["tokens"],
            action_tokens=action_pre["tokens"],
            video_freqs=video_pre["freqs"],
            action_freqs=action_pre["freqs"],
            video_t_mod=video_pre["t_mod"],
            action_t_mod=action_pre["t_mod"],
            video_context_payload={"context": video_pre["context"], "mask": video_pre["context_mask"]},
            action_context_payload={"context": action_pre["context"], "mask": action_pre["context_mask"]},
            attention_mask=attention_mask,
            sub_stream_lens=sub_stream_lens,
            sub_stream_self_masks=sub_stream_self_masks,
        )

        pred_video = self.video_expert.post_dit(video_tokens_out, video_pre)
        pred_action = self.action_expert.post_dit(action_tokens_out, action_pre)
        return pred_video, pred_action

    @torch.no_grad()
    def _predict_action_noise(
        self,
        first_frame_latents: torch.Tensor,
        latents_action: torch.Tensor,
        timestep_action: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        fuse_vae_embedding_in_latents: bool,
    ) -> torch.Tensor:
        timestep_video = torch.zeros_like(timestep_action, dtype=first_frame_latents.dtype, device=self.device)
        video_pre = self.video_expert.pre_dit(
            x=first_frame_latents,
            timestep=timestep_video,
            context=context,
            context_mask=context_mask,
            action=None,
            fuse_vae_embedding_in_latents=fuse_vae_embedding_in_latents,
        )
        action_pre = self.action_expert.pre_dit(
            action_tokens=latents_action,
            timestep=timestep_action,
            context=context,
            context_mask=context_mask,
        )

        attention_mask = self._build_mot_attention_mask(
            video_seq_len=video_pre["tokens"].shape[1],
            action_seq_len=action_pre["tokens"].shape[1],
            video_tokens_per_frame=int(video_pre["meta"]["tokens_per_frame"]),
            device=video_pre["tokens"].device,
        )
        sub_stream_lens, sub_stream_self_masks = self._build_hbridge_self_masks(
            video_seq_len=video_pre["tokens"].shape[1],
            action_seq_len=action_pre["tokens"].shape[1],
            video_tokens_per_frame=int(video_pre["meta"]["tokens_per_frame"]),
            device=video_pre["tokens"].device,
        )
        tokens_out = self.mot(
            embeds_all={
                "video": video_pre["tokens"],
                "action": action_pre["tokens"],
            },
            attention_mask=attention_mask,
            freqs_all={
                "video": video_pre["freqs"],
                "action": action_pre["freqs"],
            },
            context_all={
                "video": {
                    "context": video_pre["context"],
                    "mask": video_pre["context_mask"],
                },
                "action": {
                    "context": action_pre["context"],
                    "mask": action_pre["context_mask"],
                },
            },
            t_mod_all={
                "video": video_pre["t_mod"],
                "action": action_pre["t_mod"],
            },
            sub_stream_lens=sub_stream_lens,
            sub_stream_self_masks=sub_stream_self_masks,
        )
        pred_action = self.action_expert.post_dit(tokens_out["action"], action_pre)
        return pred_action

    def _denoise_step_compiled(
        self,
        latents_action: torch.Tensor,
        timestep_action: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        video_cache_k: list[torch.Tensor],
        video_cache_v: list[torch.Tensor],
        action_attention_mask: torch.Tensor,
        action_freqs: torch.Tensor,
        action_only_attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Single denoise step: pre_dit + MoT inner + post_dit, fused for compile.

        Inlines `ActionDiT.pre_dit` and `ActionDiT.post_dit` (mathematically
        identical) so the entire per-step computation is one graph with no
        Python-level glue. Designed to be wrapped with
        ``torch.compile(mode="reduce-overhead")`` for CUDA Graph capture.

        Caller must hoist out of the compiled region:
        - flat ``video_cache_k``/``video_cache_v`` lists (was list[dict])
        - sliced ``action_attention_mask`` (was the joint mask)
        - precomputed ``action_freqs``
        - optional ``action_only_attention_mask`` for HBridge outer layers
        """
        ae = self.action_expert
        t = ae.time_embedding(sinusoidal_embedding_1d(ae.freq_dim, timestep_action))
        t_mod = ae.time_projection(t).unflatten(1, (6, ae.hidden_dim))
        tokens = ae.action_encoder(latents_action)
        seq_len = tokens.shape[1]
        context_emb = ae.text_embedding(context)
        # None = all-visible context (attn_backend="auto" drops trivial masks).
        context_attn_mask = (
            context_mask.unsqueeze(1).expand(-1, seq_len, -1)
            if context_mask is not None else None
        )

        tokens = self.mot._forward_action_with_video_cache_inner(
            action_tokens=tokens,
            action_freqs=action_freqs,
            action_t_mod=t_mod,
            action_context_payload={
                "context": context_emb,
                "mask": context_attn_mask,
            },
            video_cache_k=video_cache_k,
            video_cache_v=video_cache_v,
            action_attention_mask=action_attention_mask,
            action_only_attention_mask=action_only_attention_mask,
        )
        return ae.head(tokens)

    @torch.no_grad()
    def _predict_action_noise_with_cache(
        self,
        latents_action: torch.Tensor,
        timestep_action: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        video_kv_cache: list[dict[str, torch.Tensor]],
        attention_mask: torch.Tensor,
        video_seq_len: int,
    ) -> torch.Tensor:
        if getattr(self, "_compile_inference", False) and self.device.type == "cuda":
            # Lazy-compile on first call.
            if not getattr(self, "_denoise_step_is_compiled", False):
                self._denoise_step_compiled = self._compile_for_inference(
                    self._denoise_step_compiled,
                )
                self._denoise_step_is_compiled = True
            # Hoist dynamic slicing + dict-flatten OUT of the compiled region.
            # attn_backend="auto" callers pass attention_mask=None (the fast-path
            # mask is all-True by construction) — keep the slice None-safe.
            action_seq_len = latents_action.shape[1]
            total_seq_len = video_seq_len + action_seq_len
            action_attention_mask = (
                attention_mask[video_seq_len:total_seq_len, :total_seq_len]
                if attention_mask is not None else None
            )
            action_freqs = self.action_expert.freqs[:action_seq_len].view(action_seq_len, 1, -1).to(latents_action.device)
            cache_k = [c["k"] for c in video_kv_cache]
            cache_v = [c["v"] for c in video_kv_cache]
            # NOTE: never mask-drop action_only_attention_mask — its presence
            # is the HBridge outer-layer routing flag (`hbridge_active`), not
            # just a visibility mask.
            action_only_attention_mask = self._build_action_only_attention_mask(
                action_seq_len=action_seq_len,
                device=latents_action.device,
            )
            return self._denoise_step_compiled(
                latents_action=latents_action,
                timestep_action=timestep_action,
                context=context,
                context_mask=context_mask,
                video_cache_k=cache_k,
                video_cache_v=cache_v,
                action_attention_mask=action_attention_mask,
                action_freqs=action_freqs,
                action_only_attention_mask=action_only_attention_mask,
            )
        # ORIGINAL PATH — verbatim, untouched. Delegated to a grad-safe helper
        # so callers that need autograd can reuse the same forward without
        # inheriting this method's ``@torch.no_grad()`` decorator.
        return self._predict_action_grad_safe(
            latents_action=latents_action,
            timestep_action=timestep_action,
            context=context,
            context_mask=context_mask,
            video_kv_cache=video_kv_cache,
            attention_mask=attention_mask,
            video_seq_len=video_seq_len,
        )

    def _use_loop_compile(self) -> bool:
        """True when torch_compile_scope="loop" should take the whole-loop path."""
        return (
            getattr(self, "_compile_inference", False)
            and getattr(self, "_compile_scope", "step") == "loop"
            and self.device.type == "cuda"
        )

    def _drop_trivial_mask(self, mask: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        """attn_backend="auto": an all-True mask is equivalent to no mask, and
        None lets SDPA dispatch flash/cuDNN instead of the masked
        memory-efficient kernel. Costs one small device sync — call once per
        infer call (hoisted), never per step/layer."""
        if (
            getattr(self, "_infer_attn_backend", "sdpa") == "auto"
            and mask is not None
            and bool(mask.all())
        ):
            return None
        return mask

    def _sdpa_priority_ctx(self):
        """attn_backend="auto": cuDNN-first SDPA priority for the joint denoise
        paths — cuDNN ingests the dense bool joint mask nearly for free
        (~2.6x over the memory-efficient fallback at joint shapes on sm_120).
        Entered around the compiled calls: kernel selection happens inside,
        at trace/capture time. Null context for other backends.

        torch < 2.9 gate: inductor's cuDNN-SDPA meta kernel mispredicts the
        output layout there (runtime stride assert, measured on 2.7.1), so
        the priority is skipped — "auto" still gets the action-fast-path
        mask drops, the joint paths just stay on the default backend."""
        if getattr(self, "_infer_attn_backend", "sdpa") == "auto":
            torch_minor = tuple(int(x) for x in torch.__version__.split(".")[:2])
            if torch_minor < (2, 9):
                if not getattr(self, "_warned_sdpa_priority_gate", False):
                    logger.warning(
                        "attn_backend='auto': skipping cuDNN-first joint SDPA "
                        "priority on torch %s (<2.9 inductor cudnn-sdpa stride "
                        "bug); action-fast-path mask drops remain active.",
                        torch.__version__,
                    )
                    self._warned_sdpa_priority_gate = True
                return contextlib.nullcontext()
            from torch.nn.attention import SDPBackend, sdpa_kernel

            return sdpa_kernel(
                [
                    SDPBackend.CUDNN_ATTENTION,
                    SDPBackend.FLASH_ATTENTION,
                    SDPBackend.EFFICIENT_ATTENTION,
                ],
                set_priority=True,
            )
        return contextlib.nullcontext()

    def _action_denoise_loop_body(
        self,
        latents_action: torch.Tensor,
        infer_timesteps: torch.Tensor,
        infer_deltas: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        video_cache_k: list[torch.Tensor],
        video_cache_v: list[torch.Tensor],
        action_attention_mask: torch.Tensor,
        action_freqs: torch.Tensor,
        action_only_attention_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """torch_compile_scope="loop" target: all denoise steps + Euler updates
        unrolled into one graph, so reduce-overhead captures a single CUDA
        Graph per call — zero Python round-trips between steps. Math is
        identical to N calls of ``_denoise_step_compiled`` + ``scheduler.step``.
        ``FlexPiBackbone._denoise_step_compiled`` (class attr) is always the raw body —
        the Tier-1 lazy wrap only shadows the *instance* attribute.
        """
        for i in range(infer_timesteps.shape[0]):
            timestep_action = infer_timesteps[i].reshape(1).to(dtype=latents_action.dtype)
            pred_action = FlexPiBackbone._denoise_step_compiled(
                self,
                latents_action=latents_action,
                timestep_action=timestep_action,
                context=context,
                context_mask=context_mask,
                video_cache_k=video_cache_k,
                video_cache_v=video_cache_v,
                action_attention_mask=action_attention_mask,
                action_freqs=action_freqs,
                action_only_attention_mask=action_only_attention_mask,
            )
            latents_action = self.infer_action_scheduler.step(
                pred_action, infer_deltas[i], latents_action,
            )
            latents_action = self._zero_padded_action(latents_action)
        return latents_action

    def _predict_action_grad_safe(
        self,
        latents_action: torch.Tensor,
        timestep_action: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        video_kv_cache: list[dict[str, torch.Tensor]],
        attention_mask: torch.Tensor,
        video_seq_len: int,
    ) -> torch.Tensor:
        """Action-velocity prediction without ``@torch.no_grad()``.

        Identical math to ``_predict_action_noise_with_cache``'s ORIGINAL PATH
        but skips the compiled / CUDA-graph branch (which is incompatible with
        autograd).
        """
        action_pre = self.action_expert.pre_dit(
            action_tokens=latents_action,
            timestep=timestep_action,
            context=context,
            context_mask=context_mask,
        )
        action_only_attention_mask = self._build_action_only_attention_mask(
            action_seq_len=action_pre["tokens"].shape[1],
            device=latents_action.device,
        )
        action_tokens = self.mot.forward_action_with_video_cache(
            action_tokens=action_pre["tokens"],
            action_freqs=action_pre["freqs"],
            action_t_mod=action_pre["t_mod"],
            action_context_payload={
                "context": action_pre["context"],
                "mask": action_pre["context_mask"],
            },
            video_kv_cache=video_kv_cache,
            attention_mask=attention_mask,
            video_seq_len=video_seq_len,
            action_only_attention_mask=action_only_attention_mask,
        )
        return self.action_expert.post_dit(action_tokens, action_pre)



    @staticmethod
    def _filter_shape_mismatches(pretrained_state, model_state, label="", strict_shape: bool = True):
        """Filter shape-mismatched keys between checkpoint and model.

        strict_shape=True (default): any mismatch raises ValueError with full list.
        strict_shape=False: legacy behavior — log warning and silently drop.

        Returns (filtered_state_dict, list_of_skipped_key_descriptions).
        """
        filtered = {}
        skipped = []
        for key, value in pretrained_state.items():
            if key in model_state and value.shape != model_state[key].shape:
                skipped.append(
                    f"{key}: ckpt {list(value.shape)} vs model {list(model_state[key].shape)}"
                )
                continue
            filtered[key] = value
        if skipped:
            msg = (
                f"{len(skipped)} shape-mismatched key(s) in {label or 'state_dict'}:\n  "
                + "\n  ".join(skipped)
            )
            if strict_shape:
                raise ValueError(
                    "Checkpoint shape mismatch with strict_shape=True. "
                    "Pass strict_shape=False explicitly to opt into silent drop.\n" + msg
                )
            logger.warning(msg)
        return filtered, skipped

    def load_checkpoint(self, path, optimizer=None, strict_shape: bool = True):
        payload = torch.load(path, map_location="cpu")
        if "mot" in payload:
            filtered, _ = self._filter_shape_mismatches(
                payload["mot"], self.mot.state_dict(), label="mot",
                strict_shape=strict_shape,
            )
            missing, unexpected = self.mot.load_state_dict(filtered, strict=False)
            if missing:
                logger.info("MoT keys kept at random init: %d  (first 10: %s)",
                            len(missing), missing[:10])
        elif "dit" in payload:
            logger.warning("Loading legacy `dit` checkpoint into video expert only.")
            self.video_expert.load_state_dict(payload["dit"], strict=False)
        else:
            raise ValueError(f"Checkpoint missing both `mot` and `dit` keys: {path}")

        if self.proprio_encoder is not None:
            if "proprio_encoder" in payload:
                filtered, _ = self._filter_shape_mismatches(
                    payload["proprio_encoder"],
                    self.proprio_encoder.state_dict(),
                    label="proprio_encoder",
                    strict_shape=strict_shape,
                )
                self.proprio_encoder.load_state_dict(filtered, strict=False)
            else:
                logger.warning("Checkpoint has no `proprio_encoder` weights; keeping current `proprio_encoder` params.")
        elif "proprio_encoder" in payload:
            logger.warning("Checkpoint contains `proprio_encoder` weights but current model has `proprio_dim=None`; ignoring.")

        if optimizer is not None and "optimizer" in payload:
            optimizer.load_state_dict(payload["optimizer"])
        return payload
