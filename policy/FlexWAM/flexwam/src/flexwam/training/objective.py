from typing import Any, Optional, Sequence, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from flexwam.utils.logging_config import get_logger
logger = get_logger(__name__)


class TrainingObjectiveMixin:
    @staticmethod
    def _compute_action_loss_per_sample(
        pred_action: torch.Tensor,
        target_action: torch.Tensor,
        action_is_pad: Optional[torch.Tensor],
    ) -> torch.Tensor:
        action_loss_token = F.mse_loss(pred_action.float(), target_action.float(), reduction="none").mean(dim=2)
        if action_is_pad is None:
            return action_loss_token.mean(dim=1)
        valid = (~action_is_pad).to(device=action_loss_token.device, dtype=action_loss_token.dtype)
        valid_sum = valid.sum(dim=1).clamp(min=1.0)
        return (action_loss_token * valid).sum(dim=1) / valid_sum

    def build_inputs(self, sample, tiled: bool = False):
        video = sample["video"]
        if "context" not in sample or "context_mask" not in sample:
            raise ValueError(
                "FlexWAM training requires `sample['context']` and `sample['context_mask']`."
            )
        context = sample["context"]
        context_mask = sample["context_mask"]
        proprio = sample.get("proprio", None)
        if proprio is None:
            raise ValueError("FlexWAM training requires `sample['proprio']`.")
        expected_num_views = int(getattr(self.video_expert, "num_views", 1))
        video = self._canonicalize_video_tensor(video, expected_num_views=expected_num_views)

        if video.ndim == 5:
            batch_size, _, num_frames, height, width = video.shape
        else:
            batch_size, num_views, _, num_frames, height, width = video.shape
            if num_views != expected_num_views:
                raise ValueError(
                    f"`sample['video']` has {num_views} views but video expert expects {expected_num_views}."
                )
        self._validate_video_spatial_size(height, width)
        if num_frames % 4 != 1:
            raise ValueError(f"Video T must satisfy T % 4 == 1, got T={num_frames}")
        if num_frames <= 1:
            raise ValueError(f"Video T must be > 1 for action-conditioned training, got T={num_frames}")

        if "action" not in sample:
            raise ValueError("`sample['action']` is required for FlexWAM training.")

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
        video_is_pad = sample.get("video_is_pad", None)
        if video_is_pad is not None and video_is_pad.ndim != 2:
            raise ValueError(
                f"`sample['video_is_pad']` must be 2D [B, T_latent_loss], got shape {tuple(video_is_pad.shape)}"
            )
        
        input_video = video.to(device=self.device, dtype=self.torch_dtype, non_blocking=True)
        input_latents = self._encode_video_latents(input_video, tiled=tiled)

        first_frame_latents = self._latent_first_frame(input_latents)

        if video_is_pad is not None:
            expected_video_loss_steps = self._latent_num_frames(input_latents) - 1
            if video_is_pad.shape[0] != batch_size or video_is_pad.shape[1] != expected_video_loss_steps:
                raise ValueError(
                    "`sample['video_is_pad']` shape mismatch: "
                    f"got {tuple(video_is_pad.shape)} vs expected ({batch_size}, {expected_video_loss_steps})"
                )

        if context.ndim != 3 or context_mask.ndim != 2:
            raise ValueError(
                f"`context/context_mask` must be [B,L,D]/[B,L], got {tuple(context.shape)} and {tuple(context_mask.shape)}"
            )
        context = context.to(device=self.device, dtype=self.torch_dtype, non_blocking=True)
        context_mask = context_mask.to(device=self.device, dtype=torch.bool, non_blocking=True)
        if proprio.ndim != 3:
            raise ValueError(f"`sample['proprio']` must be 3D [B, T, d], got shape {tuple(proprio.shape)}")
        if proprio.shape[2] != self.proprio_dim:
            raise ValueError(
                f"`sample['proprio']` last dim must be {self.proprio_dim}, got {proprio.shape[2]}"
            )
        proprio = proprio[:, 0, :] # [B, D]
        proprio_for_router = proprio.to(device=self.device, dtype=self.torch_dtype)
        context_payload = self._build_video_action_contexts(
            context=context,
            context_mask=context_mask,
            proprio=proprio_for_router,
        )
        action = action.to(device=self.device, dtype=self.torch_dtype, non_blocking=True)

        if action_is_pad is not None:
            action_is_pad = action_is_pad.to(device=self.device, dtype=torch.bool, non_blocking=True)
        if image_is_pad is not None:
            image_is_pad = image_is_pad.to(device=self.device, dtype=torch.bool, non_blocking=True)
        if video_is_pad is not None:
            video_is_pad = video_is_pad.to(device=self.device, dtype=torch.bool, non_blocking=True)

        return {
            "context": context_payload["action_context"],
            "context_mask": context_payload["action_context_mask"],
            "video_context": context_payload["video_context"],
            "video_context_mask": context_payload["video_context_mask"],
            "action_context": context_payload["action_context"],
            "action_context_mask": context_payload["action_context_mask"],
            "input_video": input_video,
            "input_latents": input_latents,
            "first_frame_latents": first_frame_latents,
            "action": action,
            "proprio": proprio_for_router,
            "action_is_pad": action_is_pad,
            "image_is_pad": image_is_pad,
            "video_is_pad": video_is_pad,
        }

    def _compute_video_loss_per_sample(
        self,
        pred_video: torch.Tensor,
        target_video: torch.Tensor,
        image_is_pad: Optional[torch.Tensor],
        video_is_pad: Optional[torch.Tensor],
        return_view_losses: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, Optional[torch.Tensor]]:
        if pred_video.ndim == 5:
            video_loss_token = F.mse_loss(pred_video.float(), target_video.float(), reduction="none").mean(dim=(1, 3, 4))
        elif pred_video.ndim == 6:
            video_loss_token = F.mse_loss(pred_video.float(), target_video.float(), reduction="none").mean(dim=(2, 4, 5))
        else:
            raise ValueError(f"Expected 5D/6D video prediction, got shape {tuple(pred_video.shape)}")

        def _finish(per_token_loss: torch.Tensor, mask: Optional[torch.Tensor]):
            if mask is not None:
                if mask.shape[0] != per_token_loss.shape[0] or mask.shape[1] != per_token_loss.shape[-1]:
                    raise ValueError(
                        "Video-loss mask shape mismatch: "
                        f"mask shape={tuple(mask.shape)}, loss shape={tuple(per_token_loss.shape)}."
                    )
                valid = (~mask).to(device=per_token_loss.device, dtype=per_token_loss.dtype)
                valid_sum = valid.sum(dim=1).clamp(min=1.0)
                if per_token_loss.ndim == 2:
                    per_sample = (per_token_loss * valid).sum(dim=1) / valid_sum
                    if return_view_losses:
                        return per_sample, None
                    return per_sample
                per_view_sample = (per_token_loss * valid.unsqueeze(1)).sum(dim=2) / valid_sum.unsqueeze(1)
            else:
                if per_token_loss.ndim == 2:
                    per_sample = per_token_loss.mean(dim=1)
                    if return_view_losses:
                        return per_sample, None
                    return per_sample
                per_view_sample = per_token_loss.mean(dim=2)

            per_sample = per_view_sample.mean(dim=1)
            if return_view_losses:
                return per_sample, per_view_sample
            return per_sample

        if video_is_pad is not None:
            return _finish(video_loss_token, video_is_pad)

        if image_is_pad is None:
            return _finish(video_loss_token, None)

        temporal_factor = int(self.vae.temporal_downsample_factor)
        if temporal_factor <= 0:
            raise ValueError(f"`vae.temporal_downsample_factor` must be positive, got {temporal_factor}.")
        if image_is_pad.shape[1] < 1:
            raise ValueError("`image_is_pad` must contain at least one frame.")
        if (image_is_pad.shape[1] - 1) % temporal_factor != 0:
            raise ValueError(
                "Cannot align `image_is_pad` with video latent steps: "
                f"num_frames={image_is_pad.shape[1]}, temporal_downsample_factor={temporal_factor}."
            )

        tail_is_pad = image_is_pad[:, 1:]
        video_is_pad = tail_is_pad.view(image_is_pad.shape[0], -1, temporal_factor).all(dim=2)

        if video_is_pad.shape[1] != video_loss_token.shape[-1]:
            raise ValueError(
                "Video-loss mask shape mismatch: "
                f"mask steps={video_is_pad.shape[1]}, loss steps={video_loss_token.shape[-1]}."
            )

        return _finish(video_loss_token, video_is_pad)

    def training_loss(self, sample, tiled: bool = False):
        inputs = self.build_inputs(sample, tiled=tiled)
        input_latents = inputs["input_latents"]
        batch_size = input_latents.shape[0]
        video_context = inputs["video_context"]
        video_context_mask = inputs["video_context_mask"]
        action_context = inputs["action_context"]
        action_context_mask = inputs["action_context_mask"]
        action = inputs["action"]
        proprio = inputs["proprio"]
        state_keep_mask = self._sample_state_keep_mask(proprio)
        action_is_pad = inputs["action_is_pad"]
        image_is_pad = inputs["image_is_pad"]
        video_is_pad = inputs["video_is_pad"]

        noise_video = torch.randn_like(input_latents)
        timestep_video = self.train_video_scheduler.sample_training_t(
            batch_size=batch_size,
            device=self.device,
            dtype=input_latents.dtype,
        )
        latents = self.train_video_scheduler.add_noise(input_latents, noise_video, timestep_video)
        target_video = self.train_video_scheduler.training_target(input_latents, noise_video, timestep_video)

        self._copy_first_frame_(latents, inputs["first_frame_latents"])

        noise_action = torch.randn_like(action)
        timestep_action = self.train_action_scheduler.sample_training_t(
            batch_size=batch_size,
            device=self.device,
            dtype=action.dtype,
        )
        noisy_action = self.train_action_scheduler.add_noise(action, noise_action, timestep_action)
        target_action = self.train_action_scheduler.training_target(action, noise_action, timestep_action)

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
        action_pre = self._action_expert_pre_dit(
            action_tokens=noisy_action,
            timestep=timestep_action,
            action_context=action_context,
            action_context_mask=action_context_mask,
            proprio=proprio,
        )

        video_tokens = video_pre["tokens"]
        action_tokens = action_pre["tokens"]

        attention_mask = self._build_mot_attention_mask(
            video_seq_len=video_tokens.shape[1],
            action_seq_len=action_tokens.shape[1],
            video_tokens_per_frame=int(video_pre["meta"]["tokens_per_frame"]),
            device=video_tokens.device,
            action_prefix_len=self._action_prefix_len(action_pre),
            num_views=int(video_pre["meta"].get("num_views", 1)),
            video_meta=video_pre["meta"],
        )
        tokens_out, router_output, head_features = self._run_mot_video_action(
            video_pre=video_pre,
            action_pre=action_pre,
            proprio=proprio,
            attention_mask=attention_mask,
            state_keep_mask=state_keep_mask,
        )

        pred_video = self.video_expert.post_dit(tokens_out["video"], video_pre)

        pred_action = self.action_expert.post_dit(tokens_out["action"], action_pre)

        # The clean conditioning frame is excluded from the video loss.
        pred_video = self._latent_tail(pred_video)
        target_video = self._latent_tail(target_video)

        loss_video_per_sample, loss_video_per_view_sample = self._compute_video_loss_per_sample(
            pred_video=pred_video,
            target_video=target_video,
            image_is_pad=image_is_pad,
            video_is_pad=video_is_pad,
            return_view_losses=True,
        )
        video_weight = self.train_video_scheduler.training_weight(timestep_video).to(
            loss_video_per_sample.device, dtype=loss_video_per_sample.dtype
        )
        loss_video = (loss_video_per_sample * video_weight).mean()

        action_loss_per_sample = self._compute_action_loss_per_sample(
            pred_action=pred_action,
            target_action=target_action,
            action_is_pad=action_is_pad,
        )

        action_weight = self.train_action_scheduler.training_weight(timestep_action).to(
            action_loss_per_sample.device, dtype=action_loss_per_sample.dtype
        )
        loss_action = (action_loss_per_sample * action_weight).mean()

        router_cf_loss, router_cf_metrics = self._compute_router_cf_loss(
            video_pre=video_pre,
            action_pre=action_pre,
            target_action=target_action,
            action_is_pad=action_is_pad,
            action_weight=action_weight,
            full_action_loss_per_sample=action_loss_per_sample,
            proprio=proprio,
            attention_mask=attention_mask,
            state_keep_mask=state_keep_mask,
            input_video=inputs["input_video"],
            noise_video=noise_video,
            timestep_video=timestep_video,
            video_context=video_context,
            video_context_mask=video_context_mask,
            action=action,
            tiled=tiled,
            router_features=head_features["router"],
            critic_features=head_features.get("critic"),
        )

        loss_total = self.loss_lambda_video * loss_video + self.loss_lambda_action * loss_action
        if router_cf_loss is not None:
            loss_total = loss_total + float(self.router_cf_lambda) * router_cf_loss
        loss_dict = {
            "loss_video": self.loss_lambda_video * float(loss_video.detach().item()),
            "loss_action": self.loss_lambda_action * float(loss_action.detach().item()),
        }
        if router_cf_metrics:
            for key, value in router_cf_metrics.items():
                if key == "loss_router_cf":
                    loss_dict[key] = float(self.router_cf_lambda) * value
                else:
                    loss_dict[key] = value
        if loss_video_per_view_sample is not None:
            video_weight_per_view = video_weight.view(-1, 1)
            loss_video_per_view = (loss_video_per_view_sample * video_weight_per_view).mean(dim=0)
            view_names = self._camera_view_names(len(loss_video_per_view))
            for view_idx, view_loss in enumerate(loss_video_per_view):
                view_name = view_names[view_idx] if view_idx < len(view_names) else f"view{view_idx}"
                loss_dict[f"loss_video_{view_name}"] = self.loss_lambda_video * float(view_loss.detach().item())
        mean_weights = router_output["weights"].detach().float().mean(dim=0)
        view_names = self._camera_view_names(len(mean_weights))
        for view_idx, router_weight in enumerate(mean_weights):
            view_name = view_names[view_idx] if view_idx < len(view_names) else f"view{view_idx}"
            loss_dict[f"router_{view_name}"] = float(router_weight.item())
        return loss_total, loss_dict
