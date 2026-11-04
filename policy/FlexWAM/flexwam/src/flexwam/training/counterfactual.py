from typing import Any, Optional, Sequence, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from flexwam.utils.logging_config import get_logger
logger = get_logger(__name__)


class CounterfactualMixin:
    def set_train_progress(
        self,
        step: int,
        max_steps: Optional[int] = None,
        *,
        epoch: Optional[int] = None,
        stage: Optional[int] = None,
        stage3_epoch: Optional[int] = None,
        stage3_epochs: Optional[int] = None,
        stage4_epoch: Optional[int] = None,
        stage4_epochs: Optional[int] = None,
        stage4_progress: Optional[float] = None,
    ) -> None:
        del max_steps, epoch, stage3_epoch, stage3_epochs
        self._progress_step = int(step)
        self.training_stage = None if stage is None else int(stage)
        self._stage4_epoch = 0 if stage4_epoch is None else int(stage4_epoch)
        self._stage4_epochs = 1 if stage4_epochs is None else max(int(stage4_epochs), 1)
        self._stage4_progress = None if stage4_progress is None else float(stage4_progress)
        router_on = bool(getattr(self, "camera_router_enabled", True))
        # `teacher=cf` removes the critic. Stage 4 then matches the router to the
        # counterfactual target while action loss still trains the same router.
        critic_on = bool(getattr(self, "camera_critic_enabled", True)) and not self._router_cf_direct()
        if self.training_stage is None:
            self.action_camera_router_train_enabled = router_on
            self.action_camera_critic_train_enabled = critic_on
        else:
            self.action_camera_router_train_enabled = router_on and self.training_stage >= 2
            self.action_camera_critic_train_enabled = critic_on and self.training_stage in (3, 4)
        router = getattr(self, "action_camera_router", None)
        if router is not None:
            router.requires_grad_(self.action_camera_router_train_enabled)
            router.train(self.action_camera_router_train_enabled)
        critic = getattr(self, "action_camera_critic", None)
        if critic is not None:
            critic.requires_grad_(self.action_camera_critic_train_enabled)
            critic.train(self.action_camera_critic_train_enabled)

    def _router_cf_direct(self) -> bool:
        return str(getattr(self, "router_cf_teacher", "critic")) == "cf"

    def _should_run_pixel_critic_cf(self) -> bool:
        # Align with trainer logging: `set_train_progress` receives completed
        # `global_step`, then this call produces the next completed step.
        completed_step = int(getattr(self, "_progress_step", 0)) + 1
        return completed_step > 0 and completed_step % int(self.router_cf_interval) == 0

    def _pixel_critic_stage4_progress(self) -> float:
        if self.training_stage is None:
            return 1.0
        explicit = getattr(self, "_stage4_progress", None)
        if explicit is not None:
            return min(max(float(explicit), 0.0), 1.0)
        epochs = max(int(getattr(self, "_stage4_epochs", 1)), 1)
        epoch = max(int(getattr(self, "_stage4_epoch", 0)), 0)
        if epochs <= 1:
            return 1.0
        return min(max(epoch / float(epochs - 1), 0.0), 1.0)

    def _pixel_critic_kl_hparams(self) -> tuple[float, float]:
        # KL is the critic's guidance of the router, and it is active only in stage 4.
        # Stage 5 drops critic guidance, KL, and counterfactual probes.
        if self.training_stage is not None and int(self.training_stage) != 4:
            return float(self.router_cf_kl_temperature_start), 0.0
        progress = self._pixel_critic_stage4_progress()
        temperature = self.router_cf_kl_temperature_start + (
            self.router_cf_kl_temperature_end - self.router_cf_kl_temperature_start
        ) * progress
        kl_lambda = self.router_cf_kl_lambda_start + (
            self.router_cf_kl_lambda_end - self.router_cf_kl_lambda_start
        ) * progress
        return float(temperature), float(kl_lambda)

    def _mask_camera_pixels(
        self,
        video: torch.Tensor,
        view_idx: int,
        drop_ratio: float,
    ) -> torch.Tensor:
        """Black out `drop_ratio` of spatial pixels on one camera, all frames."""
        if video.ndim != 6:
            raise ValueError(
                f"Pixel-critic CF expects 6D video [B,V,3,T,H,W], got {tuple(video.shape)}."
            )
        batch_size, num_views, _, _, height, width = video.shape
        if view_idx < 0 or view_idx >= num_views:
            raise ValueError(f"Invalid camera index {view_idx} for {num_views} views.")
        masked = video.clone()
        n_pix = int(height * width)
        n_drop = int(round(float(drop_ratio) * n_pix))
        if n_drop <= 0:
            return masked
        fill = video.new_tensor(self.router_cf_pixel_fill)
        if n_drop >= n_pix:
            masked[:, view_idx] = fill
            return masked
        # Exact spatial ratio: randomly choose n_drop pixels per sample, shared across frames.
        scores = torch.rand((batch_size, n_pix), device=video.device, dtype=torch.float32)
        drop_idx = torch.topk(scores, k=n_drop, dim=-1, largest=True).indices
        drop = torch.zeros((batch_size, n_pix), device=video.device, dtype=torch.bool)
        drop.scatter_(1, drop_idx, True)
        drop = drop.view(batch_size, 1, 1, height, width)
        view = masked[:, view_idx]
        masked[:, view_idx] = view.masked_fill(drop, fill)
        return masked

    @staticmethod
    def _zscore_per_sample(delta: torch.Tensor, eps: float) -> torch.Tensor:
        """Standardize the V camera deltas of each sample independently."""
        mean = delta.mean(dim=-1, keepdim=True)
        var = delta.var(dim=-1, keepdim=True, unbiased=False)
        return (delta - mean) / (var.sqrt() + float(eps))

    def _rerun_action_loss_from_video(
        self,
        *,
        video: torch.Tensor,
        noise_video: torch.Tensor,
        timestep_video: torch.Tensor,
        video_context: torch.Tensor,
        video_context_mask: torch.Tensor,
        action: torch.Tensor,
        action_pre: dict[str, Any],
        target_action: torch.Tensor,
        action_is_pad: Optional[torch.Tensor],
        action_weight: torch.Tensor,
        proprio: torch.Tensor,
        attention_mask: torch.Tensor,
        tiled: bool,
        force_uniform_router: bool = False,
        state_keep_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        input_latents = self._encode_video_latents(video, tiled=tiled)
        latents = self.train_video_scheduler.add_noise(input_latents, noise_video, timestep_video)
        self._copy_first_frame_(latents, self._latent_first_frame(input_latents))
        video_pre = self._video_expert_pre_dit(
            proprio=proprio,
            state_keep_mask=state_keep_mask,
            x=latents,
            timestep=timestep_video,
            context=video_context,
            context_mask=video_context_mask,
            action=self._video_gt_action_condition(action),
            fuse_vae_embedding_in_latents=True,
        )
        tokens_out, _, _ = self._run_mot_video_action(
            video_pre=video_pre,
            action_pre=action_pre,
            proprio=proprio,
            attention_mask=attention_mask,
            force_uniform_router=force_uniform_router,
            include_critic=False,
            state_keep_mask=state_keep_mask,
        )
        pred_action = self.action_expert.post_dit(tokens_out["action"], action_pre)
        drop_action_loss_per_sample = self._compute_action_loss_per_sample(
            pred_action=pred_action,
            target_action=target_action,
            action_is_pad=action_is_pad,
        )
        return drop_action_loss_per_sample * action_weight

    def _cf_video_pre_from_ctx(self, ctx: dict[str, Any]) -> dict[str, Any]:
        input_latents = self._encode_video_latents(ctx["input_video"], tiled=ctx["tiled"])
        latents = self.train_video_scheduler.add_noise(
            input_latents, ctx["noise_video"], ctx["timestep_video"]
        )
        self._copy_first_frame_(latents, self._latent_first_frame(input_latents))
        return self._video_expert_pre_dit(
            proprio=ctx["proprio"],
            state_keep_mask=ctx.get("state_keep_mask"),
            x=latents,
            timestep=ctx["timestep_video"],
            context=ctx["video_context"],
            context_mask=ctx["video_context_mask"],
            action=self._video_gt_action_condition(ctx["action"]),
            fuse_vae_embedding_in_latents=True,
        )

    def _action_loss_with_cached_video(
        self,
        *,
        video_pre: dict[str, Any],
        video_out: dict[str, Any],
        ctx: dict[str, Any],
        router_output: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        action_tokens = self._forward_action_with_video_cache(
            action_pre=ctx["action_pre"],
            video_kv_cache=video_out["kv_cache"],
            attention_mask=ctx["attention_mask"],
            video_seq_len=int(video_pre["tokens"].shape[1]),
            video_meta=video_pre["meta"],
            router_output=router_output,
        )
        pred_action = self.action_expert.post_dit(action_tokens, ctx["action_pre"])
        action_loss_per_sample = self._compute_action_loss_per_sample(
            pred_action=pred_action,
            target_action=ctx["target_action"],
            action_is_pad=ctx["action_is_pad"],
        )
        return action_loss_per_sample * ctx["action_weight"]

    def _router_zero_cf_deltas(self, ctx: dict[str, Any]) -> torch.Tensor:
        """Action-loss deltas from zeroing each camera's mixer scale.

        Baseline is frozen 1:1:1. Each camera is then set to 0 in turn
        (1:1:0 / 1:0:1 / 0:1:1). Video encode + VideoDiT run once; only
        ActionDiT is rerun. Counterfactual forwards are no-grad.
        """
        self._resample_cf_action_if_needed(ctx)
        num_views = int(ctx["num_views"])
        with torch.no_grad():
            video_pre = self._cf_video_pre_from_ctx(ctx)
            video_out = self._forward_video_branch(
                video_pre=video_pre,
                attention_mask=ctx["attention_mask"],
                proprio=ctx.get("proprio"),
            )
            base_router = self._uniform_action_router_output(
                batch_size=int(video_pre["tokens"].shape[0]),
                num_views=num_views,
                device=video_pre["tokens"].device,
                dtype=video_pre["tokens"].dtype,
            )
            weighted_full_loss = self._action_loss_with_cached_video(
                video_pre=video_pre,
                video_out=video_out,
                ctx=ctx,
                router_output=base_router,
            )
            deltas = []
            for view_idx in range(num_views):
                dropped_router = self._router_output_zero_view(base_router, view_idx)
                weighted_drop_loss = self._action_loss_with_cached_video(
                    video_pre=video_pre,
                    video_out=video_out,
                    ctx=ctx,
                    router_output=dropped_router,
                )
                deltas.append(weighted_drop_loss - weighted_full_loss)
        return torch.stack(deltas, dim=1)

    def _compute_pixel_critic_router_cf_loss(
        self,
        *,
        video_pre: dict[str, Any],
        action_pre: dict[str, Any],
        target_action: torch.Tensor,
        action_is_pad: Optional[torch.Tensor],
        action_weight: torch.Tensor,
        full_action_loss_per_sample: torch.Tensor,
        proprio: torch.Tensor,
        attention_mask: torch.Tensor,
        input_video: torch.Tensor,
        noise_video: torch.Tensor,
        timestep_video: torch.Tensor,
        video_context: torch.Tensor,
        video_context_mask: torch.Tensor,
        action: torch.Tensor,
        tiled: bool,
        router_features: Optional[torch.Tensor] = None,
        critic_features: Optional[torch.Tensor] = None,
        state_keep_mask: Optional[torch.Tensor] = None,
    ) -> tuple[Optional[torch.Tensor], dict[str, float]]:
        video_meta = video_pre["meta"]
        num_views = int(video_meta.get("num_views", 1))
        if num_views <= 1:
            return None, {}
        if self._router_cf_direct():
            if router_features is None:
                raise ValueError("Direct router CF requires router first-frame features.")
            return self._queue_direct_router_cf(
                video_meta=video_meta,
                router_features=router_features,
                action_pre=action_pre,
                target_action=target_action,
                action_is_pad=action_is_pad,
                action_weight=action_weight,
                full_action_loss_per_sample=full_action_loss_per_sample,
                proprio=proprio,
                attention_mask=attention_mask,
                input_video=input_video,
                noise_video=noise_video,
                timestep_video=timestep_video,
                video_context=video_context,
                video_context_mask=video_context_mask,
                action=action,
                tiled=tiled,
                state_keep_mask=state_keep_mask,
                num_views=num_views,
            )
        if critic_features is None:
            critic_features = router_features
        if critic_features is None:
            raise ValueError("Pixel-critic CF requires critic first-frame features.")

        critic_logits = self._call_camera_head(
            self.action_camera_critic,
            first_frame_features=critic_features.detach(),
            proprio=proprio,
            state_keep_mask=state_keep_mask,
            video_meta=video_meta,
            return_logits=True,
        )
        dummy = critic_logits.float().sum() * 0.0
        self.router_cf_train_step += 1
        self._deferred_pixel_cf_ctx = None
        if not self._should_run_pixel_critic_cf():
            return dummy, {"router_cf_ran": 0.0}

        # Re-encode after the main backward so DiT activations are already freed.
        self._deferred_pixel_cf_ctx = {
            "video_meta": video_meta,
            "router_features": None if router_features is None else router_features.detach(),
            "critic_features": critic_features.detach(),
            "action_pre": self._detach_cf_payload(action_pre),
            "target_action": target_action.detach(),
            "action_is_pad": None if action_is_pad is None else action_is_pad.detach(),
            "action_weight": action_weight.detach(),
            "full_action_loss_per_sample": full_action_loss_per_sample.detach(),
            "proprio": proprio.detach(),
            "state_keep_mask": None if state_keep_mask is None else state_keep_mask.detach(),
            "attention_mask": attention_mask.detach(),
            "input_video": input_video.detach(),
            "noise_video": noise_video.detach(),
            "timestep_video": timestep_video.detach(),
            "video_context": video_context.detach(),
            "video_context_mask": video_context_mask.detach(),
            "action": action.detach(),
            "tiled": bool(tiled),
            "num_views": int(num_views),
        }
        return dummy, {}

    def _queue_direct_router_cf(
        self,
        *,
        video_meta: dict[str, Any],
        router_features: torch.Tensor,
        action_pre: dict[str, Any],
        target_action: torch.Tensor,
        action_is_pad: Optional[torch.Tensor],
        action_weight: torch.Tensor,
        full_action_loss_per_sample: torch.Tensor,
        proprio: torch.Tensor,
        attention_mask: torch.Tensor,
        input_video: torch.Tensor,
        noise_video: torch.Tensor,
        timestep_video: torch.Tensor,
        video_context: torch.Tensor,
        video_context_mask: torch.Tensor,
        action: torch.Tensor,
        tiled: bool,
        state_keep_mask: Optional[torch.Tensor],
        num_views: int,
    ) -> tuple[Optional[torch.Tensor], dict[str, float]]:
        """Defer the counterfactual target so it can supervise the router after the main backward."""
        self.router_cf_train_step += 1
        self._deferred_pixel_cf_ctx = None
        if not self._should_run_pixel_critic_cf():
            return None, {"router_cf_ran": 0.0}
        self._deferred_pixel_cf_ctx = {
            "video_meta": video_meta,
            "router_features": router_features.detach(),
            "critic_features": None,
            "action_pre": self._detach_cf_payload(action_pre),
            "target_action": target_action.detach(),
            "action_is_pad": None if action_is_pad is None else action_is_pad.detach(),
            "action_weight": action_weight.detach(),
            "full_action_loss_per_sample": full_action_loss_per_sample.detach(),
            "proprio": proprio.detach(),
            "state_keep_mask": None if state_keep_mask is None else state_keep_mask.detach(),
            "attention_mask": attention_mask.detach(),
            "input_video": input_video.detach(),
            "noise_video": noise_video.detach(),
            "timestep_video": timestep_video.detach(),
            "video_context": video_context.detach(),
            "video_context_mask": video_context_mask.detach(),
            "action": action.detach(),
            "tiled": bool(tiled),
            "num_views": int(num_views),
            "direct_router": True,
        }
        return None, {}

    def _kl_router_to_logits(
        self,
        ctx: dict[str, Any],
        teacher_logits: torch.Tensor,
        temperature: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        teacher = torch.softmax(teacher_logits.detach().float() / float(temperature), dim=-1)
        features = ctx.get("router_features")
        if features is None:
            features = ctx.get("critic_features")
        if features is None:
            raise ValueError("Router KL requires first-frame features.")
        student_weights, _ = self._call_camera_head(
            self.action_camera_router,
            first_frame_features=features,
            proprio=ctx["proprio"],
            state_keep_mask=ctx.get("state_keep_mask"),
            video_meta=ctx["video_meta"],
            return_logits=False,
        )
        student = student_weights.float().clamp(min=1e-8)
        kl = (teacher * (teacher.clamp(min=1e-8).log() - student.log())).sum(dim=-1).mean()
        return kl, teacher

    @staticmethod
    def _detach_cf_payload(value: Any) -> Any:
        if torch.is_tensor(value):
            return value.detach()
        if isinstance(value, dict):
            return {key: CounterfactualMixin._detach_cf_payload(item) for key, item in value.items()}
        if isinstance(value, list):
            return [CounterfactualMixin._detach_cf_payload(item) for item in value]
        if isinstance(value, tuple):
            return tuple(CounterfactualMixin._detach_cf_payload(item) for item in value)
        return value

    def _cf_action_context(
        self,
        action: torch.Tensor,
        action_pre: dict[str, Any],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if "input_context" in action_pre:
            # WAN's `context` is already projected and its mask is query-expanded.
            return action_pre["input_context"], action_pre["input_context_mask"]
        context = action_pre.get("context")
        mask = action_pre.get("context_mask")
        if context is not None:
            return context, mask
        batch = int(action.shape[0])
        return (
            action.new_empty((batch, 0, self.context_input_dim)),
            torch.empty((batch, 0), device=action.device, dtype=torch.bool),
        )

    @torch.no_grad()
    def _resample_cf_action_if_needed(self, ctx: dict[str, Any]) -> bool:
        """Rebuild ActionDiT inputs for the CF intervention if a sigma range is set.

        Main-step video/action t is left unchanged. Only the deferred 4-forward
        critic GT uses this resampled action t.
        """
        sigma_min = getattr(self, "router_cf_action_sigma_min", None)
        sigma_max = getattr(self, "router_cf_action_sigma_max", None)
        if sigma_min is None:
            return False
        action = ctx["action"]
        batch = int(action.shape[0])
        sigma = torch.empty((batch,), device=action.device, dtype=torch.float32).uniform_(
            float(sigma_min),
            float(sigma_max),
        )
        timestep = (sigma * float(self.train_action_scheduler.num_train_timesteps)).to(
            dtype=action.dtype
        )
        noise_action = torch.randn_like(action)
        noisy_action = self.train_action_scheduler.add_noise(action, noise_action, timestep)
        target_action = self.train_action_scheduler.training_target(action, noise_action, timestep)
        action_weight = self.train_action_scheduler.training_weight(timestep).to(
            device=action.device, dtype=action.dtype
        )
        # σ=1 makes training_weight ~ 0; keep per-sample ranking defined.
        action_weight = torch.where(
            action_weight.abs() < 1e-8,
            torch.ones_like(action_weight),
            action_weight,
        )
        action_context, action_context_mask = self._cf_action_context(action, ctx["action_pre"])
        action_pre = self._action_expert_pre_dit(
            action_tokens=noisy_action,
            timestep=timestep,
            action_context=action_context,
            action_context_mask=action_context_mask,
            proprio=ctx["proprio"],
        )
        ctx["action_pre"] = self._detach_cf_payload(action_pre)
        ctx["target_action"] = target_action.detach()
        ctx["action_weight"] = action_weight.detach()
        ctx["action_sigma"] = sigma.detach()
        return True

    def _pixel_critic_cf_deltas(self, ctx: dict[str, Any], drop_ratio: float) -> torch.Tensor:
        """Action-loss deltas from dropping each camera. Counterfactual forwards are no-grad.

        `router_cf.intervention=router_zero`: freeze mixer at 1:1:1, then set each camera
        scale to 0. Otherwise pixel-drop each camera. `router_cf.uniform_probe`:
        also rerun the unmasked clip with a frozen 1:1:1 mixer, and use that
        mixer on every pixel drop.
        """
        if str(getattr(self, "router_cf_intervention", "pixel")) == "router_zero":
            return self._router_zero_cf_deltas(ctx)
        resampled = self._resample_cf_action_if_needed(ctx)
        num_views = int(ctx["num_views"])
        force_uniform = bool(getattr(self, "router_cf_uniform_probe", False))
        rerun_kwargs = {
            "noise_video": ctx["noise_video"],
            "timestep_video": ctx["timestep_video"],
            "video_context": ctx["video_context"],
            "video_context_mask": ctx["video_context_mask"],
            "action": ctx["action"],
            "action_pre": ctx["action_pre"],
            "target_action": ctx["target_action"],
            "action_is_pad": ctx["action_is_pad"],
            "action_weight": ctx["action_weight"],
            "proprio": ctx["proprio"],
            "state_keep_mask": ctx.get("state_keep_mask"),
            "attention_mask": ctx["attention_mask"],
            "tiled": ctx["tiled"],
            "force_uniform_router": force_uniform,
        }
        with torch.no_grad():
            # Resampled action t cannot reuse the main-step full loss.
            if force_uniform or resampled:
                weighted_full_loss = self._rerun_action_loss_from_video(
                    video=ctx["input_video"],
                    **rerun_kwargs,
                )
            else:
                weighted_full_loss = ctx["full_action_loss_per_sample"] * ctx["action_weight"]
            deltas = []
            for view_idx in range(num_views):
                masked_video = self._mask_camera_pixels(ctx["input_video"], view_idx, drop_ratio)
                weighted_drop_loss = self._rerun_action_loss_from_video(
                    video=masked_video,
                    **rerun_kwargs,
                )
                deltas.append(weighted_drop_loss - weighted_full_loss)
        return torch.stack(deltas, dim=1)

    def run_deferred_pixel_critic_cf(self) -> tuple[Optional[torch.Tensor], dict[str, float]]:
        ctx = self._deferred_pixel_cf_ctx
        self._deferred_pixel_cf_ctx = None
        if ctx is None:
            return None, {}

        intervention = str(getattr(self, "router_cf_intervention", "pixel"))
        if intervention == "router_zero":
            drop_ratio = 0.0
        else:
            drop_ratio = float(
                torch.empty((), device="cpu").uniform_(
                    self.router_cf_pixel_drop_min,
                    self.router_cf_pixel_drop_max,
                ).item()
            )
        delta = self._pixel_critic_cf_deltas(ctx, drop_ratio)
        num_views = int(ctx["num_views"])
        z_target = self._zscore_per_sample(delta, self.router_cf_zscore_eps).detach()
        if ctx.get("direct_router"):
            return self._direct_router_cf_loss(ctx, delta, z_target, drop_ratio, intervention, num_views)
        critic_logits = self._call_camera_head(
            self.action_camera_critic,
            first_frame_features=ctx["critic_features"],
            proprio=ctx["proprio"],
            state_keep_mask=ctx.get("state_keep_mask"),
            video_meta=ctx["video_meta"],
            return_logits=True,
        )
        critic_loss = F.mse_loss(critic_logits.float(), z_target.float())
        z_mse_zero = float((z_target.float() ** 2).mean().item())
        delta_std = float(delta.detach().float().std(dim=-1, unbiased=False).mean().item())

        temperature, kl_lambda = self._pixel_critic_kl_hparams()
        metrics: dict[str, float] = {
            "loss_router_cf_critic": float(critic_loss.detach().item()),
            "router_cf_z_mse_zero": z_mse_zero,
            "router_cf_delta_std": delta_std,
            "router_cf_kl_temperature": float(temperature),
            "router_cf_kl_lambda": float(kl_lambda),
            "router_cf_ran": 1.0,
            "router_cf_uniform_probe": 1.0 if bool(getattr(self, "router_cf_uniform_probe", False)) else 0.0,
            "router_cf_intervention_router_zero": 1.0 if intervention == "router_zero" else 0.0,
        }
        if intervention != "router_zero":
            metrics["router_cf_pixel_drop"] = drop_ratio
        if getattr(self, "router_cf_action_sigma_min", None) is not None:
            metrics["router_cf_action_sigma_min"] = float(self.router_cf_action_sigma_min)
            metrics["router_cf_action_sigma_max"] = float(self.router_cf_action_sigma_max)
        action_sigma = ctx.get("action_sigma")
        if action_sigma is not None:
            metrics["router_cf_action_sigma"] = float(action_sigma.float().mean().item())
        view_names = self._camera_view_names(num_views)
        mean_delta = delta.detach().float().mean(dim=0)
        mean_z = z_target.float().mean(dim=0)
        mean_critic = critic_logits.detach().float().mean(dim=0)
        for view_idx in range(num_views):
            name = view_names[view_idx] if view_idx < len(view_names) else f"view{view_idx}"
            metrics[f"router_cf_delta_{name}"] = float(mean_delta[view_idx].item())
            metrics[f"router_cf_z_{name}"] = float(mean_z[view_idx].item())
            metrics[f"critic_{name}"] = float(mean_critic[view_idx].item())

        loss = critic_loss
        apply_kl = kl_lambda > 0.0 and (
            self.training_stage is None or int(self.training_stage) == 4
        )
        if apply_kl:
            teacher = torch.softmax(critic_logits.detach().float() / float(temperature), dim=-1)
            student_weights, _ = self._call_camera_head(
                self.action_camera_router,
                first_frame_features=ctx.get("router_features", ctx["critic_features"]),
                proprio=ctx["proprio"],
                state_keep_mask=ctx.get("state_keep_mask"),
                video_meta=ctx["video_meta"],
                return_logits=False,
            )
            student = student_weights.float().clamp(min=1e-8)
            kl = (teacher * (teacher.clamp(min=1e-8).log() - student.log())).sum(dim=-1).mean()
            loss = loss + float(kl_lambda) * kl
            metrics["loss_router_cf_kl"] = float(kl.detach().item())
            mean_teacher = teacher.detach().mean(dim=0)
            for view_idx in range(num_views):
                name = view_names[view_idx] if view_idx < len(view_names) else f"view{view_idx}"
                metrics[f"router_cf_teacher_{name}"] = float(mean_teacher[view_idx].item())

        metrics["loss_router_cf"] = float(loss.detach().item())
        return loss, metrics

    def _direct_router_cf_loss(
        self,
        ctx: dict[str, Any],
        delta: torch.Tensor,
        z_target: torch.Tensor,
        drop_ratio: float,
        intervention: str,
        num_views: int,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Match the router to softmax(z-scored action-loss deltas). No critic is involved."""
        temperature, kl_lambda = self._pixel_critic_kl_hparams()
        kl, teacher = self._kl_router_to_logits(ctx, z_target, temperature)
        loss = kl * float(kl_lambda)
        z_mse_zero = float((z_target.float() ** 2).mean().item())
        delta_std = float(delta.detach().float().std(dim=-1, unbiased=False).mean().item())
        metrics: dict[str, float] = {
            "loss_router_cf_kl": float(kl.detach().item()),
            "router_cf_z_mse_zero": z_mse_zero,
            "router_cf_delta_std": delta_std,
            "router_cf_kl_temperature": float(temperature),
            "router_cf_kl_lambda": float(kl_lambda),
            "router_cf_ran": 1.0,
            "router_cf_direct_teacher": 1.0,
            "router_cf_uniform_probe": 1.0 if bool(getattr(self, "router_cf_uniform_probe", False)) else 0.0,
            "router_cf_intervention_router_zero": 1.0 if intervention == "router_zero" else 0.0,
        }
        if intervention != "router_zero":
            metrics["router_cf_pixel_drop"] = drop_ratio
        if getattr(self, "router_cf_action_sigma_min", None) is not None:
            metrics["router_cf_action_sigma_min"] = float(self.router_cf_action_sigma_min)
            metrics["router_cf_action_sigma_max"] = float(self.router_cf_action_sigma_max)
        action_sigma = ctx.get("action_sigma")
        if action_sigma is not None:
            metrics["router_cf_action_sigma"] = float(action_sigma.float().mean().item())
        view_names = self._camera_view_names(num_views)
        mean_delta = delta.detach().float().mean(dim=0)
        mean_z = z_target.float().mean(dim=0)
        mean_teacher = teacher.detach().mean(dim=0)
        for view_idx in range(num_views):
            name = view_names[view_idx] if view_idx < len(view_names) else f"view{view_idx}"
            metrics[f"router_cf_delta_{name}"] = float(mean_delta[view_idx].item())
            metrics[f"router_cf_z_{name}"] = float(mean_z[view_idx].item())
            metrics[f"router_cf_teacher_{name}"] = float(mean_teacher[view_idx].item())
        metrics["loss_router_cf"] = float(loss.detach().item())
        return loss, metrics

    def _compute_router_cf_loss(
        self,
        *,
        video_pre: dict[str, Any],
        action_pre: dict[str, Any],
        target_action: torch.Tensor,
        action_is_pad: Optional[torch.Tensor],
        action_weight: torch.Tensor,
        full_action_loss_per_sample: torch.Tensor,
        proprio: Optional[torch.Tensor],
        attention_mask: torch.Tensor,
        input_video: Optional[torch.Tensor] = None,
        noise_video: Optional[torch.Tensor] = None,
        timestep_video: Optional[torch.Tensor] = None,
        video_context: Optional[torch.Tensor] = None,
        video_context_mask: Optional[torch.Tensor] = None,
        action: Optional[torch.Tensor] = None,
        tiled: bool = False,
        router_features: Optional[torch.Tensor] = None,
        critic_features: Optional[torch.Tensor] = None,
        state_keep_mask: Optional[torch.Tensor] = None,
    ) -> tuple[Optional[torch.Tensor], dict[str, float]]:
        video_meta = video_pre["meta"]
        num_views = int(video_meta.get("num_views", 1))
        if num_views <= 1:
            return None, {}

        direct = self._router_cf_direct()
        stage = None if self.training_stage is None else int(self.training_stage)
        if direct:
            # Stage 5 keeps the router on action loss only, same as the critic path.
            if stage is not None and stage != 4:
                return None, {}
            if not bool(getattr(self, "action_camera_router_train_enabled", False)):
                return None, {}
        else:
            if stage is not None and stage not in (3, 4):
                return None, {}
            if not bool(getattr(self, "action_camera_critic_train_enabled", False)):
                return None, {}
        if (
            input_video is None
            or noise_video is None
            or timestep_video is None
            or video_context is None
            or video_context_mask is None
            or action is None
        ):
            raise ValueError("Pixel-critic CF is missing video/noise tensors from `training_loss`.")
        return self._compute_pixel_critic_router_cf_loss(
            video_pre=video_pre,
            action_pre=action_pre,
            target_action=target_action,
            action_is_pad=action_is_pad,
            action_weight=action_weight,
            full_action_loss_per_sample=full_action_loss_per_sample,
            proprio=proprio,
            attention_mask=attention_mask,
            input_video=input_video,
            noise_video=noise_video,
            timestep_video=timestep_video,
            video_context=video_context,
            video_context_mask=video_context_mask,
            action=action,
            tiled=tiled,
            router_features=router_features,
            critic_features=critic_features,
            state_keep_mask=state_keep_mask,
        )
