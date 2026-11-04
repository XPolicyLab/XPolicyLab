from typing import Any, Optional, Sequence, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from flexwam.utils.logging_config import get_logger
logger = get_logger(__name__)

from flexwam.inference.timing import _InferenceTimer, _synchronize_for_timing
import time
from flexwam.inference.cuda_graph import _ActionDenoiseCudaGraph
from flexwam.models.architectures.mot import resolve_action_video_query_start

class InferenceMixin:
    @torch.no_grad()
    def _predict_joint_noise(
        self,
        latents_video: torch.Tensor,
        latents_action: torch.Tensor,
        timestep_video: torch.Tensor,
        timestep_action: torch.Tensor,
        video_context: torch.Tensor,
        video_context_mask: torch.Tensor,
        action_context: torch.Tensor,
        action_context_mask: torch.Tensor,
        gt_action: Optional[torch.Tensor] = None,
        proprio: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        video_pre = self._video_expert_pre_dit(
            proprio=proprio,
            x=latents_video,
            timestep=timestep_video,
            context=video_context,
            context_mask=video_context_mask,
            action=self._video_gt_action_condition(gt_action),
            fuse_vae_embedding_in_latents=True,
        )
        action_pre = self._action_expert_pre_dit(
            action_tokens=latents_action,
            timestep=timestep_action,
            action_context=action_context,
            action_context_mask=action_context_mask,
            proprio=proprio,
        )

        attention_mask = self._build_mot_attention_mask(
            video_seq_len=video_pre["tokens"].shape[1],
            action_seq_len=action_pre["tokens"].shape[1],
            video_tokens_per_frame=int(video_pre["meta"]["tokens_per_frame"]),
            device=video_pre["tokens"].device,
            action_prefix_len=self._action_prefix_len(action_pre),
            num_views=int(video_pre["meta"].get("num_views", 1)),
            video_meta=video_pre["meta"],
        )
        tokens_out, _, _ = self._run_mot_video_action(
            video_pre=video_pre,
            action_pre=action_pre,
            proprio=proprio,
            attention_mask=attention_mask,
            include_critic=False,
        )

        pred_video = self.video_expert.post_dit(tokens_out["video"], video_pre)
        pred_action = self.action_expert.post_dit(tokens_out["action"], action_pre)
        return pred_video, pred_action

    @torch.no_grad()
    def _predict_action_noise_with_cache_eager(
        self,
        latents_action: torch.Tensor,
        timestep_action: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        video_kv_cache: list[dict[str, torch.Tensor]],
        attention_mask: torch.Tensor,
        video_seq_len: int,
        video_meta: Optional[dict[str, Any]] = None,
        action_video_router_weights: Optional[torch.Tensor] = None,
        action_attention_stats: Optional[list[dict[str, float]]] = None,
        action_attention_heatmaps: Optional[dict[str, Any]] = None,
        proprio: Optional[torch.Tensor] = None,
        action_video_router_log_scales: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        action_pre = self._action_expert_pre_dit(
            action_tokens=latents_action,
            timestep=timestep_action,
            action_context=context,
            action_context_mask=context_mask,
            proprio=proprio,
        )
        router_output = {"scales": action_video_router_weights}
        if action_video_router_log_scales is not None:
            router_output["log_scales"] = action_video_router_log_scales
        # Use the same configured attention path as training and CF probes.
        action_tokens = self._forward_action_with_video_cache(
            action_pre=action_pre,
            video_kv_cache=video_kv_cache,
            attention_mask=attention_mask,
            video_seq_len=video_seq_len,
            video_meta=video_meta,
            router_output=router_output,
            action_attention_stats=action_attention_stats,
            action_attention_heatmaps=action_attention_heatmaps,
        )
        return self.action_expert.post_dit(action_tokens, action_pre)

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
        video_meta: Optional[dict[str, Any]] = None,
        action_video_router_weights: Optional[torch.Tensor] = None,
        action_attention_stats: Optional[list[dict[str, float]]] = None,
        action_attention_heatmaps: Optional[dict[str, Any]] = None,
        proprio: Optional[torch.Tensor] = None,
        action_video_router_log_scales: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        kwargs = dict(
            latents_action=latents_action,
            timestep_action=timestep_action,
            context=context,
            context_mask=context_mask,
            video_kv_cache=video_kv_cache,
            attention_mask=attention_mask,
            video_seq_len=video_seq_len,
            video_meta=video_meta,
            action_video_router_weights=action_video_router_weights,
            action_video_router_log_scales=action_video_router_log_scales,
            action_attention_stats=action_attention_stats,
            action_attention_heatmaps=action_attention_heatmaps,
            proprio=proprio,
        )
        # GPU `.item()` is illegal during CUDA graph capture. Resolve the static
        # action-to-video query start on the host before entering the graph.
        if proprio is not None:
            state_tokens = self._action_state_tokens(proprio)
            action_prefix_len = 0 if state_tokens is None else int(state_tokens.shape[1])
        else:
            action_prefix_len = 0
        action_seq_len = int(latents_action.shape[1]) + action_prefix_len
        resolve_action_video_query_start(
            attention_mask,
            int(video_seq_len),
            action_seq_len,
            action_prefix_len,
        )
        ensure_freqs = getattr(self.action_expert, "ensure_freqs_device", None)
        if ensure_freqs is not None:
            ensure_freqs(latents_action.device)
        if (
            self.use_cuda_graph
            and torch.cuda.is_available()
            and self.device.type == "cuda"
            and action_attention_stats is None
            and action_attention_heatmaps is None
        ):
            if self._action_denoise_cuda_graph is None:
                self._action_denoise_cuda_graph = _ActionDenoiseCudaGraph(
                    self._predict_action_noise_with_cache_eager,
                    device=self.device,
                )
            return self._action_denoise_cuda_graph(**kwargs)
        return self._predict_action_noise_with_cache_eager(**kwargs)

    @torch.no_grad()
    def infer_joint(
        self,
        prompt: Optional[str],
        input_image: torch.Tensor,
        num_video_frames: int,
        action_horizon: int,
        action: Optional[torch.Tensor] = None,
        proprio: Optional[torch.Tensor] = None,
        context: Optional[torch.Tensor] = None,
        context_mask: Optional[torch.Tensor] = None,
        negative_prompt: Optional[str] = None,
        text_cfg_scale: float = 1.0,
        num_inference_steps: int = 20,
        sigma_shift: Optional[float] = None,
        seed: Optional[int] = None,
        rand_device: str = "cpu",
        tiled: bool = False,
    ) -> dict[str, Any]:
        if proprio is None:
            raise ValueError("FlexWAM inference requires `proprio` state input.")
        self.eval()
        if input_image.ndim == 3:
            input_image = input_image.unsqueeze(0)
        input_image = self._canonicalize_input_image_tensor(input_image)
        if input_image.ndim == 4:
            _, _, height, width = input_image.shape
        else:
            _, _, _, height, width = input_image.shape
        self._validate_video_spatial_size(height, width)
        if num_video_frames % 4 != 1:
            raise ValueError(
                f"`num_video_frames` must satisfy T % 4 == 1, got {num_video_frames}"
            )
        if action is not None:
            if action.ndim == 2:
                action = action.unsqueeze(0)
            if action.ndim != 3 or action.shape[0] != 1 or action.shape[1] != action_horizon:
                # NOTE: This enforces action condition to have the same shape as action horizon to predict, which may be unnecessary
                raise ValueError(
                    f"`action` must have shape [1, T, a_dim] or [T, a_dim], got {tuple(action.shape)} with action_horizon={action_horizon}"
                )
            action = action.to(device=self.device, dtype=self.torch_dtype)
        if proprio.ndim == 1:
            proprio = proprio.unsqueeze(0)
        elif proprio.ndim == 2 and proprio.shape[0] == 1:
            pass
        else:
            raise ValueError(f"`proprio` must be [D] or [1,D], got shape {tuple(proprio.shape)}")
        if proprio.shape[1] != self.proprio_dim:
            raise ValueError(f"`proprio` last dim must be {self.proprio_dim}, got {proprio.shape[1]}")
        proprio = proprio.to(device=self.device, dtype=self.torch_dtype)

        latent_t = (num_video_frames - 1) // self.vae.temporal_downsample_factor + 1
        latent_h = height // self.vae.upsampling_factor
        latent_w = width // self.vae.upsampling_factor

        video_generator = None if seed is None else torch.Generator(device=rand_device).manual_seed(seed)
        action_generator = None if seed is None else torch.Generator(device=rand_device).manual_seed(seed)
        if input_image.ndim == 5:
            latent_shape = (1, input_image.shape[1], self.vae.model.z_dim, latent_t, latent_h, latent_w)
        else:
            latent_shape = (1, self.vae.model.z_dim, latent_t, latent_h, latent_w)
        latents_video = torch.randn(
            latent_shape,
            generator=video_generator,
            device=rand_device,
            dtype=torch.float32,
        ).to(device=self.device, dtype=self.torch_dtype)
        latents_action = torch.randn(
            (1, action_horizon, self.action_expert.action_dim),
            generator=action_generator,
            device=rand_device,
            dtype=torch.float32,
        ).to(device=self.device, dtype=self.torch_dtype)

        input_image = input_image.to(device=self.device, dtype=self.torch_dtype)
        first_frame_latents = self._encode_input_image_latents_tensor(input_image=input_image, tiled=tiled)
        self._copy_first_frame_(latents_video, first_frame_latents.clone())

        use_prompt = prompt is not None
        use_context = context is not None or context_mask is not None
        if use_prompt and use_context:
            raise ValueError("`prompt` and `context/context_mask` are mutually exclusive.")
        if not use_prompt and not use_context:
            raise ValueError("Either `prompt` or both `context/context_mask` must be provided.")

        if use_prompt:
            context, context_mask = self.encode_prompt(prompt)
        else:
            if context is None or context_mask is None:
                raise ValueError("`context` and `context_mask` must be both provided together.")
            if context.ndim == 2:
                context = context.unsqueeze(0)
            if context_mask.ndim == 1:
                context_mask = context_mask.unsqueeze(0)
            if context.ndim != 3 or context_mask.ndim != 2:
                raise ValueError(
                    f"`context/context_mask` must be [B,L,D]/[B,L], got {tuple(context.shape)} and {tuple(context_mask.shape)}"
                )
            context = context.to(device=self.device, dtype=self.torch_dtype, non_blocking=True)
            context_mask = context_mask.to(device=self.device, dtype=torch.bool, non_blocking=True)
        context_payload = self._build_video_action_contexts(
            context=context,
            context_mask=context_mask,
            proprio=proprio,
        )
        video_context = context_payload["video_context"]
        video_context_mask = context_payload["video_context_mask"]
        action_context = context_payload["action_context"]
        action_context_mask = context_payload["action_context_mask"]

        infer_timesteps_video, infer_deltas_video = self.infer_video_scheduler.build_inference_schedule(
            num_inference_steps=num_inference_steps,
            device=self.device,
            dtype=latents_video.dtype,
            shift_override=sigma_shift,
        )
        infer_timesteps_action, infer_deltas_action = self.infer_action_scheduler.build_inference_schedule(
            num_inference_steps=num_inference_steps,
            device=self.device,
            dtype=latents_action.dtype,
            shift_override=sigma_shift,
        )
        denoise_step_ms: list[float] = []
        denoise_t0 = time.perf_counter()
        _synchronize_for_timing(self.device)
        for step_t_video, step_delta_video, step_t_action, step_delta_action in zip(
            infer_timesteps_video,
            infer_deltas_video,
            infer_timesteps_action,
            infer_deltas_action,
        ):
            step_t0 = time.perf_counter()
            _synchronize_for_timing(self.device)
            timestep_video = step_t_video.unsqueeze(0).to(dtype=latents_video.dtype, device=self.device)
            timestep_action = step_t_action.unsqueeze(0).to(dtype=latents_action.dtype, device=self.device)

            pred_video_posi, pred_action_posi = self._predict_joint_noise(
                latents_video=latents_video,
                latents_action=latents_action,
                timestep_video=timestep_video,
                timestep_action=timestep_action,
                video_context=video_context,
                video_context_mask=video_context_mask,
                action_context=action_context,
                action_context_mask=action_context_mask,
                gt_action=action,
                proprio=proprio,
            )
            pred_video = pred_video_posi
            pred_action = pred_action_posi

            latents_video = self.infer_video_scheduler.step(pred_video, step_delta_video, latents_video)
            latents_action = self.infer_action_scheduler.step(pred_action, step_delta_action, latents_action)
            self._copy_first_frame_(latents_video, first_frame_latents.clone())
            _synchronize_for_timing(self.device)
            denoise_step_ms.append((time.perf_counter() - step_t0) * 1000.0)
        _synchronize_for_timing(self.device)
        denoise_ms = (time.perf_counter() - denoise_t0) * 1000.0

        action_out = latents_action[0].detach().to(device="cpu", dtype=torch.float32)
        return {
            "video": self._decode_latents(latents_video, tiled=tiled),
            "action": action_out,
            "infer_timing_ms": {
                "denoise_ms": float(denoise_ms),
                "denoise_step_ms": [float(ms) for ms in denoise_step_ms],
            },
        }

    @torch.no_grad()
    def infer_action(
        self,
        prompt: Optional[str],
        input_image: torch.Tensor,
        action_horizon: int,
        proprio: Optional[torch.Tensor] = None,
        context: Optional[torch.Tensor] = None,
        context_mask: Optional[torch.Tensor] = None,
        negative_prompt: Optional[str] = None,
        text_cfg_scale: float = 1.0,
        num_inference_steps: int = 20,
        sigma_shift: Optional[float] = None,
        seed: Optional[int] = None,
        rand_device: str = "cpu",
        tiled: bool = False,
        return_action_attention_stats: bool = False,
    ) -> dict[str, Any]:
        if proprio is None:
            raise ValueError("FlexWAM inference requires `proprio` state input.")
        self.eval()
        timer = _InferenceTimer(self.device)
        if str(getattr(self.video_expert, "video_attention_mask_mode", "")) != "first_frame_causal":
            raise ValueError(
                "`infer_action` requires `video_attention_mask_mode='first_frame_causal'`."
            )

        if input_image.ndim == 3:
            input_image = input_image.unsqueeze(0)
        input_image = self._canonicalize_input_image_tensor(input_image)
        if input_image.ndim == 4:
            _, _, height, width = input_image.shape
        else:
            _, _, _, height, width = input_image.shape
        self._validate_video_spatial_size(height, width)
        if proprio.ndim == 1:
            proprio = proprio.unsqueeze(0)
        elif proprio.ndim == 2 and proprio.shape[0] == 1:
            pass
        else:
            raise ValueError(f"`proprio` must be [D] or [1,D], got shape {tuple(proprio.shape)}")
        if proprio.shape[1] != self.proprio_dim:
            raise ValueError(f"`proprio` last dim must be {self.proprio_dim}, got {proprio.shape[1]}")
        proprio = proprio.to(device=self.device, dtype=self.torch_dtype)

        generator = None if seed is None else torch.Generator(device=rand_device).manual_seed(seed)
        latents_action = torch.randn(
            (1, action_horizon, self.action_expert.action_dim),
            generator=generator,
            device=rand_device,
            dtype=torch.float32,
        ).to(device=self.device, dtype=self.torch_dtype)

        encode_t0 = timer.mark()
        input_image = input_image.to(device=self.device, dtype=self.torch_dtype)
        first_frame_latents = self._encode_input_image_latents_tensor(input_image=input_image, tiled=tiled)
        encode_ms = timer.elapsed_ms(encode_t0)

        use_prompt = prompt is not None
        use_context = context is not None or context_mask is not None
        if use_prompt and use_context:
            raise ValueError("`prompt` and `context/context_mask` are mutually exclusive.")
        if not use_prompt and not use_context:
            raise ValueError("Either `prompt` or both `context/context_mask` must be provided.")

        text_t0 = timer.mark()
        if use_prompt:
            context, context_mask = self.encode_prompt(prompt)
        else:
            if context is None or context_mask is None:
                raise ValueError("`context` and `context_mask` must be both provided together.")
            if context.ndim == 2:
                context = context.unsqueeze(0)
            if context_mask.ndim == 1:
                context_mask = context_mask.unsqueeze(0)
            if context.ndim != 3 or context_mask.ndim != 2:
                raise ValueError(
                    f"`context/context_mask` must be [B,L,D]/[B,L], got {tuple(context.shape)} and {tuple(context_mask.shape)}"
                )
            context = context.to(device=self.device, dtype=self.torch_dtype, non_blocking=True)
            context_mask = context_mask.to(device=self.device, dtype=torch.bool, non_blocking=True)
        text_ms = timer.elapsed_ms(text_t0)
        context_payload = self._build_video_action_contexts(
            context=context,
            context_mask=context_mask,
            proprio=proprio,
        )
        video_context = context_payload["video_context"]
        video_context_mask = context_payload["video_context_mask"]
        action_context = context_payload["action_context"]
        action_context_mask = context_payload["action_context_mask"]
        state_tokens = self._action_state_tokens(proprio)
        state_prefix_len = 0 if state_tokens is None else int(state_tokens.shape[1])
        # Text is either WAN cross-attention context or LingBot's video prefix.
        action_prefix_len = state_prefix_len

        prefill_t0 = timer.mark()
        timestep_video = torch.zeros(
            (first_frame_latents.shape[0],),
            dtype=first_frame_latents.dtype,
            device=self.device,
        )
        video_pre = self._video_expert_pre_dit(
            proprio=proprio,
            x=first_frame_latents,
            timestep=timestep_video,
            context=video_context,
            context_mask=video_context_mask,
            action=None,
            fuse_vae_embedding_in_latents=True,
        )
        video_seq_len = int(video_pre["tokens"].shape[1])
        attention_mask = self._build_mot_attention_mask(
            video_seq_len=video_seq_len,
            action_seq_len=int(latents_action.shape[1] + action_prefix_len),
            video_tokens_per_frame=int(video_pre["meta"]["tokens_per_frame"]),
            device=video_pre["tokens"].device,
            action_prefix_len=action_prefix_len,
            num_views=int(video_pre["meta"].get("num_views", 1)),
            video_meta=video_pre["meta"],
        )
        layer_index = int(self.action_router_feature_layer_index)
        video_out = self._forward_video_branch(
            video_pre=video_pre,
            attention_mask=attention_mask,
            proprio=proprio,
            snapshot_layer_indices=(layer_index,),
        )
        first_frame_features = self._extract_first_frame_view_tokens(
            video_out["layer_tokens"][layer_index],
            video_pre["meta"],
        )
        first_frame_features = first_frame_features.detach()
        router_output = self._build_action_router_output(
            proprio=proprio,
            video_pre=video_pre,
            first_frame_features=first_frame_features,
        )
        video_kv_cache = video_out["kv_cache"]
        prefill_ms = timer.elapsed_ms(prefill_t0)

        infer_timesteps_action, infer_deltas_action = self.infer_action_scheduler.build_inference_schedule(
            num_inference_steps=num_inference_steps,
            device=self.device,
            dtype=latents_action.dtype,
            shift_override=sigma_shift,
        )
        action_attention_records: Optional[list[dict[str, float]]] = [] if return_action_attention_stats else None
        action_attention_heatmaps: Optional[dict[str, Any]] = (
            {
                "primary": [],
                "late": [],
                "all": [],
                "primary_layers": tuple(range(12, 19)),
                "late_layers": tuple(range(18, 25)),
                "num_steps": len(infer_timesteps_action),
            }
            if return_action_attention_stats
            else None
        )
        denoise_step_ms: list[float] = []
        denoise_t0 = timer.mark()
        for step_idx, (step_t_action, step_delta_action) in enumerate(zip(infer_timesteps_action, infer_deltas_action)):
            step_t0 = timer.mark()
            timestep_action = step_t_action.unsqueeze(0).to(dtype=latents_action.dtype, device=self.device)
            if action_attention_heatmaps is not None:
                action_attention_heatmaps["step_index"] = int(step_idx)

            pred_action_posi = self._predict_action_noise_with_cache(
                latents_action=latents_action,
                timestep_action=timestep_action,
                context=action_context,
                context_mask=action_context_mask,
                video_kv_cache=video_kv_cache,
                attention_mask=attention_mask,
                video_seq_len=video_seq_len,
                video_meta=video_pre["meta"],
                proprio=proprio,
                **self._action_router_mot_kwargs(router_output),
                action_attention_stats=action_attention_records,
                action_attention_heatmaps=action_attention_heatmaps,
            )
            pred_action = pred_action_posi

            latents_action = self.infer_action_scheduler.step(pred_action, step_delta_action, latents_action)
            denoise_step_ms.append(timer.elapsed_ms(step_t0))
        denoise_ms = timer.elapsed_ms(denoise_t0)

        out = {
            "action": latents_action[0].detach().to(device="cpu", dtype=torch.float32),
            "infer_timing_ms": {
                "encode_ms": float(encode_ms),
                "text_ms": float(text_ms),
                "prefill_ms": float(prefill_ms),
                "denoise_ms": float(denoise_ms),
                "denoise_step_ms": [float(ms) for ms in denoise_step_ms],
                "model_internal_ms": float(timer.total_ms()),
            },
        }
        out["router_weights"] = self._router_weights_to_cpu(router_output)
        if action_attention_records is not None:
            out["action_attention_stats"] = self._summarize_action_attention_stats(
                action_attention_records,
                num_views=int(video_pre["meta"].get("num_views", 1)),
                view_names=self._camera_view_names(int(video_pre["meta"].get("num_views", 1))),
            )
        if action_attention_heatmaps is not None:
            out["action_attention_heatmaps"] = self._summarize_action_attention_heatmaps(
                action_attention_heatmaps,
                num_views=int(video_pre["meta"].get("num_views", 1)),
                view_names=self._camera_view_names(int(video_pre["meta"].get("num_views", 1))),
            )
        return out

    @torch.no_grad()
    def infer(
        self,
        prompt: Optional[str],
        input_image: torch.Tensor,
        num_frames: int,
        action: Optional[torch.Tensor] = None,
        action_horizon: Optional[int] = None,
        proprio: Optional[torch.Tensor] = None,
        context: Optional[torch.Tensor] = None,
        context_mask: Optional[torch.Tensor] = None,
        negative_prompt: Optional[str] = None,
        text_cfg_scale: float = 5.0,
        action_cfg_scale: float = 1.0,
        num_inference_steps: int = 20,
        sigma_shift: Optional[float] = None,
        seed: Optional[int] = None,
        rand_device: str = "cpu",
        tiled: bool = False,
    ):
        return self.infer_joint(
            prompt=prompt,
            input_image=input_image,
            num_video_frames=num_frames,
            action_horizon=action_horizon,
            action=action,
            proprio=proprio,
            context=context,
            context_mask=context_mask,
            negative_prompt=negative_prompt,
            text_cfg_scale=text_cfg_scale,
            num_inference_steps=num_inference_steps,
            sigma_shift=sigma_shift,
            seed=seed,
            rand_device=rand_device,
            tiled=tiled,
        )
