import time
from typing import Any, Optional, Sequence, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from ...utils.logging_config import get_logger

from .action_dit import ActionDiT
from .future_3d_expert import Future3DExpert
from .helpers.loader import load_wan22_ti2v_5b_components
from .mot import MoT
from .schedulers.scheduler_continuous import WanContinuousFlowMatchScheduler
from lerobot.policies.WSA_Base.da3_teacher import DA3BackboneTeacher
from lerobot.utils.constants import SAMPLE_ACTION_LOSS_MASK

logger = get_logger(__name__)


class WSALarge(torch.nn.Module):
    """MoT world model with video, future-3D, and action experts."""

    def __init__(
        self,
        video_expert,
        action_expert: ActionDiT,
        mot: MoT,
        vae,
        future_3d_expert: Future3DExpert,
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
        mask_action_dim_padding_loss: bool = False,
        loss_lambda_3d: float = 0.0,
        da3_model_path_or_name: str = "depth-anything/DA3-LARGE-1.1",
        da3_code_root: str | None = None,
        da3_teacher_process_res: int = 504,
        da3_teacher_layers: tuple[int, ...] | None = None,
        da3_layer_weights: tuple[float, ...] = (1.0, 1.2, 1.4, 1.6),
        log_da3_teacher_timing: bool = False,
        future_3d_view_attention_layout: str = "auto",
    ):
        super().__init__()
        if future_3d_expert is None:
            raise ValueError("WSA_Large requires a Future3DExpert.")
        if future_3d_view_attention_layout not in {"auto", "horizontal", "vertical", "robotwin", "full"}:
            raise ValueError(
                "future_3d_view_attention_layout must be one of: "
                "auto, horizontal, vertical, robotwin, full"
            )
        self.video_expert = video_expert
        self.action_expert = action_expert
        self.future_3d_expert = future_3d_expert
        self.future_3d_view_attention_layout = future_3d_view_attention_layout
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
        self.mask_action_dim_padding_loss = bool(mask_action_dim_padding_loss)
        self.loss_lambda_3d = float(loss_lambda_3d)
        self.da3_teacher_layers = None if da3_teacher_layers is None else tuple(int(idx) for idx in da3_teacher_layers)
        self.da3_layer_weights = tuple(float(weight) for weight in da3_layer_weights)
        self.log_da3_teacher_timing = bool(log_da3_teacher_timing)
        if self.da3_teacher_layers is not None and len(self.da3_teacher_layers) != len(self.da3_layer_weights):
            raise ValueError("da3_teacher_layers and da3_layer_weights must have the same length.")
        if self.future_3d_expert is not None and self.loss_lambda_3d > 0:
            teacher_dtype = torch.bfloat16 if torch_dtype == torch.bfloat16 else torch.float32
            self.da3_teacher = DA3BackboneTeacher(
                model_path_or_name=da3_model_path_or_name,
                code_root=da3_code_root,
                process_res=da3_teacher_process_res,
                dtype=teacher_dtype,
                teacher_layers=self.da3_teacher_layers,
            )
            if self.da3_teacher_layers is None:
                self.da3_teacher_layers = tuple(int(idx) for idx in self.da3_teacher.teacher_layers)
            if int(self.da3_teacher.feature_dim) != int(self.future_3d_expert.da3_query_dim):
                raise ValueError(
                    f"DA3 teacher feature dim ({self.da3_teacher.feature_dim}) does not match "
                    f"future_3d_expert.da3_query_dim ({self.future_3d_expert.da3_query_dim})."
                )
        else:
            self.da3_teacher = None

        self.to(self.device)

    @classmethod
    def from_wan22_pretrained(
        cls,
        device: str = "cuda",
        torch_dtype: torch.dtype = torch.bfloat16,
        model_id: str = "Wan-AI/Wan2.2-TI2V-5B",
        tokenizer_model_id: str = "Wan-AI/Wan2.1-T2V-1.3B",
        tokenizer_max_len: int = 512,
        load_text_encoder: bool = True,
        proprio_dim: Optional[int] = None,
        redirect_common_files: bool = True,
        video_dit_config: dict[str, Any] | None = None,
        action_dit_config: dict[str, Any] | None = None,
        future_3d_config: dict[str, Any] | None = None,
        action_dit_pretrained_path: str | None = None,
        future_3d_pretrained_path: str | None = None,
        skip_dit_load_from_pretrain: bool = False,
        mot_checkpoint_mixed_attn: bool = True,
        video_train_shift: float = 5.0,
        video_infer_shift: float = 5.0,
        video_num_train_timesteps: int = 1000,
        action_train_shift: float = 5.0,
        action_infer_shift: float = 5.0,
        action_num_train_timesteps: int = 1000,
        loss_lambda_video: float = 1.0,
        loss_lambda_action: float = 1.0,
        mask_action_dim_padding_loss: bool = False,
        loss_lambda_3d: float = 0.0,
        da3_model_path_or_name: str = "depth-anything/DA3-LARGE-1.1",
        da3_code_root: str | None = None,
        da3_teacher_process_res: int = 504,
        da3_teacher_layers: tuple[int, ...] | None = None,
        da3_layer_weights: tuple[float, ...] = (1.0, 1.2, 1.4, 1.6),
        log_da3_teacher_timing: bool = False,
        future_3d_view_attention_layout: str = "auto",
    ):
        if video_dit_config is None:
            raise ValueError("`video_dit_config` is required for WSALarge.from_wan22_pretrained().")
        if "text_dim" not in video_dit_config:
            raise ValueError("`video_dit_config['text_dim']` is required for WSA_Large.")

        components = load_wan22_ti2v_5b_components(
            device=device,
            torch_dtype=torch_dtype,
            model_id=model_id,
            tokenizer_model_id=tokenizer_model_id,
            tokenizer_max_len=tokenizer_max_len,
            redirect_common_files=redirect_common_files,
            dit_config=video_dit_config,
            skip_dit_load_from_pretrain=skip_dit_load_from_pretrain,
            load_text_encoder=load_text_encoder,
        )

        video_expert = components.dit
        action_expert = ActionDiT.from_pretrained(
            action_dit_config=action_dit_config,
            action_dit_pretrained_path=action_dit_pretrained_path,
            skip_dit_load_from_pretrain=skip_dit_load_from_pretrain,
            device=device,
            torch_dtype=torch_dtype,
        )
        if int(action_expert.num_heads) != int(video_expert.num_heads):
            raise ValueError("ActionDiT `num_heads` must match video expert for MoT mixed attention.")
        if int(action_expert.attn_head_dim) != int(video_expert.attn_head_dim):
            raise ValueError("ActionDiT `attn_head_dim` must match video expert for MoT mixed attention.")
        if int(len(action_expert.blocks)) != int(len(video_expert.blocks)):
            raise ValueError("ActionDiT `num_layers` must match video expert.")

        mixtures = {"video": video_expert}
        if future_3d_config is None:
            raise ValueError("`future_3d_config` is required for WSA_Large.")
        future_3d_expert = Future3DExpert.from_pretrained(
            future_3d_config=future_3d_config,
            future_3d_pretrained_path=future_3d_pretrained_path,
            skip_dit_load_from_pretrain=skip_dit_load_from_pretrain,
            device=device,
            torch_dtype=torch_dtype,
        )
        if int(future_3d_expert.num_heads) != int(video_expert.num_heads):
            raise ValueError("Future3DExpert `num_heads` must match video expert for MoT mixed attention.")
        if int(future_3d_expert.attn_head_dim) != int(video_expert.attn_head_dim):
            raise ValueError("Future3DExpert `attn_head_dim` must match video expert for MoT mixed attention.")
        if int(len(future_3d_expert.blocks)) != int(len(video_expert.blocks)):
            raise ValueError("Future3DExpert `num_layers` must match video expert.")
        mixtures["future_3d"] = future_3d_expert
        mixtures["action"] = action_expert

        mot = MoT(
            mixtures=mixtures,
            mot_checkpoint_mixed_attn=mot_checkpoint_mixed_attn,
        )

        model = cls(
            video_expert=video_expert,
            action_expert=action_expert,
            mot=mot,
            vae=components.vae,
            future_3d_expert=future_3d_expert,
            text_encoder=components.text_encoder,
            tokenizer=components.tokenizer,
            text_dim=int(video_dit_config["text_dim"]),
            proprio_dim=proprio_dim,
            device=device,
            torch_dtype=torch_dtype,
            video_train_shift=video_train_shift,
            video_infer_shift=video_infer_shift,
            video_num_train_timesteps=video_num_train_timesteps,
            action_train_shift=action_train_shift,
            action_infer_shift=action_infer_shift,
            action_num_train_timesteps=action_num_train_timesteps,
            loss_lambda_video=loss_lambda_video,
            loss_lambda_action=loss_lambda_action,
            mask_action_dim_padding_loss=mask_action_dim_padding_loss,
            loss_lambda_3d=loss_lambda_3d,
            da3_model_path_or_name=da3_model_path_or_name,
            da3_code_root=da3_code_root,
            da3_teacher_process_res=da3_teacher_process_res,
            da3_teacher_layers=da3_teacher_layers,
            da3_layer_weights=da3_layer_weights,
            log_da3_teacher_timing=log_da3_teacher_timing,
            future_3d_view_attention_layout=future_3d_view_attention_layout,
        )
        model.model_paths = {
            "video_dit": components.dit_path,
            "vae": components.vae_path,
            "text_encoder": components.text_encoder_path,
            "tokenizer": components.tokenizer_path,
            "action_dit_backbone": (
                "SKIPPED_PRETRAIN" if skip_dit_load_from_pretrain else action_dit_pretrained_path
            ),
            "future_3d_expert": "enabled",
        }
        return model

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        self.mot.to(*args, **kwargs)
        if self.text_encoder is not None:
            self.text_encoder.to(*args, **kwargs)
        self.vae.to(*args, **kwargs)
        return self

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
        ids, mask = self.tokenizer(prompt, return_mask=True, add_special_tokens=True)
        ids = ids.to(self.device)
        mask = mask.to(self.device, dtype=torch.bool)
        prompt_emb = self.text_encoder(ids, mask)
        seq_lens = mask.gt(0).sum(dim=1).long()
        for i, v in enumerate(seq_lens):
            prompt_emb[i, v:] = 0
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
        video = sample["video"]
        if "context" not in sample or "context_mask" not in sample:
            raise ValueError(
                "WSA_Large training requires `sample['context']` and `sample['context_mask']`."
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
            raise ValueError("`sample['action']` is required for WSA_Large training.")

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

        action_dim_is_pad = sample.get("action_dim_is_pad", None)
        if action_dim_is_pad is not None:
            if action_dim_is_pad.ndim != 2:
                raise ValueError(
                    f"`sample['action_dim_is_pad']` must be 2D [B, D], got shape {tuple(action_dim_is_pad.shape)}"
                )
            if action_dim_is_pad.shape[0] != batch_size or action_dim_is_pad.shape[1] != action.shape[2]:
                raise ValueError(
                    "`sample['action_dim_is_pad']` shape mismatch: "
                    f"got {tuple(action_dim_is_pad.shape)} vs expected ({batch_size}, {action.shape[2]})"
                )

        sample_action_loss_mask = sample.get(SAMPLE_ACTION_LOSS_MASK, None)
        if sample_action_loss_mask is not None:
            if sample_action_loss_mask.ndim > 2:
                raise ValueError(
                    f"`sample['{SAMPLE_ACTION_LOSS_MASK}']` must be scalar, [B], or [B,1], "
                    f"got shape {tuple(sample_action_loss_mask.shape)}"
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

        future_3d_images = sample.get("future_3d_images", None)
        future_3d_img_masks = sample.get("future_3d_img_masks", None)
        if future_3d_images is not None:
            if future_3d_images.ndim != 5:
                raise ValueError(
                    f"`sample['future_3d_images']` must be 5D [B,V,3,H,W], got shape {tuple(future_3d_images.shape)}"
                )
            if future_3d_images.shape[0] != batch_size or future_3d_images.shape[2] != 3:
                raise ValueError(
                    "`sample['future_3d_images']` shape mismatch: "
                    f"got {tuple(future_3d_images.shape)}, expected batch={batch_size} and 3 channels."
                )
            future_3d_images = future_3d_images.to(device=self.device, dtype=self.torch_dtype, non_blocking=True)
        if future_3d_img_masks is not None:
            if future_3d_img_masks.ndim != 2:
                raise ValueError(
                    f"`sample['future_3d_img_masks']` must be 2D [B,V], got shape {tuple(future_3d_img_masks.shape)}"
                )
            if future_3d_img_masks.shape[0] != batch_size:
                raise ValueError(
                    f"`sample['future_3d_img_masks']` batch mismatch: got {tuple(future_3d_img_masks.shape)}."
                )
            future_3d_img_masks = future_3d_img_masks.to(device=self.device, dtype=torch.bool, non_blocking=True)

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
        if sample_action_loss_mask is not None:
            sample_action_loss_mask = sample_action_loss_mask.to(
                device=self.device, dtype=torch.float32, non_blocking=True
            )
            if sample_action_loss_mask.ndim == 0:
                sample_action_loss_mask = sample_action_loss_mask.expand(batch_size)
            if sample_action_loss_mask.ndim == 2:
                if sample_action_loss_mask.shape[1] != 1:
                    raise ValueError(
                        f"`sample['{SAMPLE_ACTION_LOSS_MASK}']` must be scalar, [B], or [B,1], "
                        f"got shape {tuple(sample_action_loss_mask.shape)}"
                    )
                sample_action_loss_mask = sample_action_loss_mask.squeeze(-1)
            if sample_action_loss_mask.ndim != 1 or sample_action_loss_mask.shape[0] != batch_size:
                raise ValueError(
                    f"`sample['{SAMPLE_ACTION_LOSS_MASK}']` batch mismatch: "
                    f"got {tuple(sample_action_loss_mask.shape)}, expected ({batch_size},)."
                )
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
            "sample_action_loss_mask": sample_action_loss_mask,
            "image_is_pad": image_is_pad,
            "future_3d_images": future_3d_images,
            "future_3d_img_masks": future_3d_img_masks,
        }

    @torch.no_grad()
    def _resolve_future_3d_view_attention_layout(self) -> str:
        layout = self.future_3d_view_attention_layout
        if layout != "auto":
            return layout

        num_views = int(self.future_3d_expert.da3_num_views)
        if num_views <= 1:
            return "full"
        if num_views == 3:
            return "robotwin"
        return "horizontal"

    @torch.no_grad()
    def _first_frame_view_token_indices(
        self,
        video_grid_size: tuple[int, int, int] | Sequence[int] | None,
        video_tokens_per_frame: int,
        device: torch.device,
    ) -> list[torch.Tensor] | None:
        num_views = int(self.future_3d_expert.da3_num_views)
        layout = self._resolve_future_3d_view_attention_layout()
        if layout == "full" or num_views <= 1:
            return None
        if video_grid_size is None:
            raise ValueError("Future-3D view-aware attention requires video_pre['meta']['grid_size'].")
        if len(video_grid_size) != 3:
            raise ValueError(f"`video_grid_size` must be (F,H,W), got {video_grid_size}")

        _, grid_h, grid_w = (int(value) for value in video_grid_size)
        if grid_h * grid_w != int(video_tokens_per_frame):
            raise ValueError(
                "Video grid does not match tokens_per_frame: "
                f"H*W={grid_h * grid_w}, tokens_per_frame={video_tokens_per_frame}."
            )

        token_grid = torch.arange(video_tokens_per_frame, device=device).view(grid_h, grid_w)
        if layout == "horizontal":
            if grid_w % num_views != 0:
                raise ValueError(
                    "horizontal Future-3D view attention requires grid width divisible by da3_num_views: "
                    f"W={grid_w}, views={num_views}."
                )
            view_w = grid_w // num_views
            return [
                token_grid[:, view_idx * view_w : (view_idx + 1) * view_w].reshape(-1)
                for view_idx in range(num_views)
            ]

        if layout == "vertical":
            if grid_h % num_views != 0:
                raise ValueError(
                    "vertical Future-3D view attention requires grid height divisible by da3_num_views: "
                    f"H={grid_h}, views={num_views}."
                )
            view_h = grid_h // num_views
            return [
                token_grid[view_idx * view_h : (view_idx + 1) * view_h, :].reshape(-1)
                for view_idx in range(num_views)
            ]

        if layout == "robotwin":
            if num_views != 3:
                raise ValueError("robotwin Future-3D view attention requires da3_num_views=3.")
            if grid_h % 3 != 0 or grid_w % 2 != 0:
                raise ValueError(
                    "robotwin Future-3D view attention requires grid_h divisible by 3 and grid_w divisible by 2: "
                    f"H={grid_h}, W={grid_w}."
                )
            top_h = (2 * grid_h) // 3
            mid_w = grid_w // 2
            return [
                token_grid[:top_h, :].reshape(-1),
                token_grid[top_h:, :mid_w].reshape(-1),
                token_grid[top_h:, mid_w:].reshape(-1),
            ]

        raise ValueError(f"Unsupported future_3d_view_attention_layout={layout!r}")

    @torch.no_grad()
    def _view_video_token_indices(
        self,
        video_grid_size: tuple[int, int, int] | Sequence[int] | None,
        video_tokens_per_frame: int,
        video_seq_len: int,
        device: torch.device,
    ) -> list[torch.Tensor] | None:
        frame_view_indices = self._first_frame_view_token_indices(
            video_grid_size=video_grid_size,
            video_tokens_per_frame=video_tokens_per_frame,
            device=device,
        )
        if frame_view_indices is None:
            return None
        if video_seq_len % video_tokens_per_frame != 0:
            raise ValueError(
                "View-aware video token span must be frame-aligned: "
                f"video_seq_len={video_seq_len}, tokens_per_frame={video_tokens_per_frame}."
            )

        num_frames = video_seq_len // video_tokens_per_frame
        frame_offsets = torch.arange(num_frames, device=device) * int(video_tokens_per_frame)
        view_token_indices = []
        for indices in frame_view_indices:
            view_token_indices.append((indices.unsqueeze(0) + frame_offsets[:, None]).reshape(-1))
        return view_token_indices

    def _future_3d_video_read_seq_len(self, video_seq_len: int, current_2d_end: int) -> int:
        del video_seq_len
        return current_2d_end

    @torch.no_grad()
    def _fill_future_3d_attention(
        self,
        mask: torch.Tensor,
        future_3d_start: int,
        future_3d_end: int,
        video_seq_len: int,
        current_2d_end: int,
        action_start: int,
        video_tokens_per_frame: int,
        video_grid_size: tuple[int, int, int] | Sequence[int] | None,
    ) -> None:
        video_read_seq_len = self._future_3d_video_read_seq_len(
            video_seq_len=video_seq_len,
            current_2d_end=current_2d_end,
        )
        future_3d_seq_len = future_3d_end - future_3d_start
        view_token_indices = self._view_video_token_indices(
            video_grid_size=video_grid_size,
            video_tokens_per_frame=video_tokens_per_frame,
            video_seq_len=video_read_seq_len,
            device=mask.device,
        )
        if view_token_indices is None:
            mask[future_3d_start:future_3d_end, :video_read_seq_len] = True
        else:
            num_views = len(view_token_indices)
            if future_3d_seq_len % num_views != 0:
                raise ValueError(
                    "Future-3D query token count must be divisible by the number of view groups: "
                    f"query_tokens={future_3d_seq_len}, views={num_views}."
                )
            query_tokens_per_view = future_3d_seq_len // num_views
            for view_idx, video_indices in enumerate(view_token_indices):
                query_start = future_3d_start + view_idx * query_tokens_per_view
                query_end = query_start + query_tokens_per_view
                mask[query_start:query_end, video_indices] = True

        mask[future_3d_start:future_3d_end, future_3d_start:future_3d_end] = True
        mask[future_3d_start:future_3d_end, action_start:] = True

    @torch.no_grad()
    def _build_mot_attention_mask(
        self,
        video_seq_len: int,
        action_seq_len: int,
        video_tokens_per_frame: int,
        device: torch.device,
        future_3d_seq_len: int = 0,
        video_grid_size: tuple[int, int, int] | Sequence[int] | None = None,
    ) -> torch.Tensor:
        total_seq_len = video_seq_len + future_3d_seq_len + action_seq_len
        mask = torch.zeros((total_seq_len, total_seq_len), dtype=torch.bool, device=device)

        # video -> video
        mask[:video_seq_len, :video_seq_len] = self.video_expert.build_video_to_video_mask(
            video_seq_len=video_seq_len,
            video_tokens_per_frame=video_tokens_per_frame,
            device=device,
        )
        future_3d_start = video_seq_len
        future_3d_end = future_3d_start + future_3d_seq_len
        action_start = future_3d_end
        first_frame_tokens = min(video_tokens_per_frame, video_seq_len)
        if video_seq_len > first_frame_tokens:
            mask[:first_frame_tokens, first_frame_tokens:video_seq_len] = False
            mask[first_frame_tokens:video_seq_len, :video_seq_len] = True

        if future_3d_seq_len > 0:
            # Main FastWAM-like mask:
            # current 2D -> current 2D
            # subgoal 2D -> current/subgoal 2D + subgoal 3D
            # subgoal 3D -> current 2D + subgoal 3D + action
            # action -> current 2D + subgoal 3D + action
            self._fill_future_3d_attention(
                mask=mask,
                future_3d_start=future_3d_start,
                future_3d_end=future_3d_end,
                video_seq_len=video_seq_len,
                current_2d_end=first_frame_tokens,
                action_start=action_start,
                video_tokens_per_frame=video_tokens_per_frame,
                video_grid_size=video_grid_size,
            )
            if video_seq_len > first_frame_tokens:
                mask[first_frame_tokens:video_seq_len, future_3d_start:future_3d_end] = True

        # action -> action
        mask[action_start:, action_start:] = True
        # action -> first-frame video only
        mask[action_start:, :first_frame_tokens] = True
        # action -> future-3D bridge tokens
        if future_3d_seq_len > 0:
            mask[action_start:, future_3d_start:future_3d_end] = True
        return mask

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
        latent_tail_is_pad = tail_is_pad.view(image_is_pad.shape[0], -1, temporal_factor).all(dim=2)
        if include_initial_video_step:
            video_is_pad = torch.cat([image_is_pad[:, :1], latent_tail_is_pad], dim=1)
        else:
            video_is_pad = latent_tail_is_pad

        if video_is_pad.shape[1] != video_loss_token.shape[1]:
            raise ValueError(
                "Video-loss mask shape mismatch: "
                f"mask steps={video_is_pad.shape[1]}, loss steps={video_loss_token.shape[1]}."
            )

        valid = (~video_is_pad).to(device=video_loss_token.device, dtype=video_loss_token.dtype)
        valid_sum = valid.sum(dim=1).clamp(min=1.0)
        return (video_loss_token * valid).sum(dim=1) / valid_sum

    @staticmethod
    def _prepare_da3_teacher_inputs(
        future_images: torch.Tensor,
        img_masks: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        valid_view_masks = img_masks.to(dtype=torch.bool)
        valid_view_counts = valid_view_masks.sum(dim=1)
        incomplete_samples = (valid_view_counts > 0) & (valid_view_counts < future_images.shape[1])
        if not incomplete_samples.any():
            return future_images, valid_view_masks

        num_views = future_images.shape[1]
        batch_indices = torch.arange(future_images.shape[0], device=future_images.device)
        primary_view_indices = valid_view_masks.to(dtype=torch.int64).argmax(dim=1)
        primary_images = future_images[batch_indices, primary_view_indices]

        teacher_images = future_images.clone()
        invalid_view_masks = ~valid_view_masks
        if invalid_view_masks.any():
            teacher_images[invalid_view_masks] = primary_images.unsqueeze(1).expand(-1, num_views, -1, -1, -1)[
                invalid_view_masks
            ]
        return teacher_images, valid_view_masks

    def _get_3d_token_mask(self, img_masks: torch.Tensor, target_len: int) -> torch.Tensor:
        if self.future_3d_expert is None:
            raise RuntimeError("Future3DExpert is not enabled.")
        token_mask = img_masks.unsqueeze(-1).expand(
            -1,
            -1,
            self.future_3d_expert.da3_tokens_per_view,
        ).reshape(img_masks.shape[0], -1)
        if token_mask.shape[1] == target_len:
            return token_mask
        token_mask = token_mask[:, None, :].to(dtype=torch.float32)
        token_mask = F.interpolate(token_mask, size=target_len, mode="nearest")
        return token_mask[:, 0, :].to(dtype=torch.bool)

    def _compute_future_3d_loss(
        self,
        future_3d_layer_tokens: tuple[torch.Tensor, ...] | None,
        future_images: torch.Tensor | None,
        img_masks: torch.Tensor | None,
        collect_logs: bool = True,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if self.future_3d_expert is None or self.da3_teacher is None or future_3d_layer_tokens is None:
            reference = future_images if future_images is not None else next(self.parameters())
            return reference.new_zeros((), dtype=torch.float32), {}
        if future_images is None:
            raise ValueError(
                "Future-3D loss is enabled but the batch has no `future_3d_images`. "
                "Set dataset.return_future_3d_images=true."
            )
        if img_masks is None:
            img_masks = torch.ones(
                future_images.shape[:2],
                dtype=torch.bool,
                device=future_images.device,
            )

        teacher_images, teacher_img_masks = self._prepare_da3_teacher_inputs(future_images, img_masks)
        loss_logs: dict[str, torch.Tensor] = {}
        teacher_forward_start = time.perf_counter() if collect_logs and self.log_da3_teacher_timing else None
        with torch.no_grad():
            teacher_layers = self.da3_teacher(teacher_images)
        if teacher_forward_start is not None:
            loss_logs["time_3d_teacher_forward_s"] = torch.tensor(
                time.perf_counter() - teacher_forward_start,
                device=teacher_images.device,
                dtype=torch.float32,
            )

        if len(teacher_layers) != len(future_3d_layer_tokens):
            raise ValueError(
                f"Expected {len(future_3d_layer_tokens)} DA3 teacher layers, got {len(teacher_layers)}"
            )

        predicted_queries = self.future_3d_expert.project_query_layers(future_3d_layer_tokens)
        token_mask = self._get_3d_token_mask(teacher_img_masks, teacher_layers[0].shape[1])

        total_loss = teacher_images.new_zeros((), dtype=torch.float32)
        for pred, target, weight, teacher_layer_idx, query_layer_idx in zip(
            predicted_queries,
            teacher_layers,
            self.da3_layer_weights,
            self.da3_teacher_layers or (),
            self.future_3d_expert.query_layer_indices,
            strict=True,
        ):
            target = target.to(device=pred.device, dtype=pred.dtype)
            if pred.shape[1] != target.shape[1]:
                raise ValueError(
                    f"Projected future-3D query length ({pred.shape[1]}) does not match DA3 target length "
                    f"({target.shape[1]})."
                )
            if pred.shape[-1] != target.shape[-1]:
                raise ValueError(
                    f"Projected future-3D query dim ({pred.shape[-1]}) does not match DA3 target dim "
                    f"({target.shape[-1]})."
                )

            pred_valid = pred[token_mask]
            target_valid = target[token_mask]
            if pred_valid.numel() == 0:
                continue

            pred_norm = F.normalize(pred_valid.float(), p=2, dim=-1)
            target_norm = F.normalize(target_valid.detach().float(), p=2, dim=-1)
            cos_loss = (1.0 - (pred_norm * target_norm).sum(dim=-1)).mean()

            pred_ln = F.layer_norm(pred_valid.float(), normalized_shape=(pred_valid.shape[-1],))
            target_ln = F.layer_norm(target_valid.detach().float(), normalized_shape=(target_valid.shape[-1],))
            mse_loss = F.mse_loss(pred_ln, target_ln)

            layer_loss = (cos_loss + mse_loss) * float(weight)
            total_loss = total_loss + layer_loss
            loss_logs[f"loss_3d_q{query_layer_idx}_t{teacher_layer_idx}"] = layer_loss.detach()

        if predicted_queries:
            total_loss = total_loss / len(predicted_queries)
        return total_loss, loss_logs

    def _future_3d_query_noise_sigma(
        self,
        batch_size: int,
        timestep: torch.Tensor | None,
        scheduler: WanContinuousFlowMatchScheduler | None,
        dtype: torch.dtype,
    ) -> torch.Tensor | None:
        if self.future_3d_expert is None:
            return None
        if getattr(self.future_3d_expert, "query_mode", "query_token") == "query_token":
            return None
        min_sigma = float(getattr(self.future_3d_expert, "query_noise_min_sigma", 0.0))
        max_sigma = float(getattr(self.future_3d_expert, "query_noise_max_sigma", 1.0))
        sigma_source = getattr(self.future_3d_expert, "query_sigma_source", "constant")
        if sigma_source == "constant":
            return torch.full((batch_size,), max_sigma, device=self.device, dtype=dtype)
        if timestep is None or scheduler is None:
            return None
        sigma = timestep.to(device=self.device, dtype=torch.float32) / float(scheduler.num_train_timesteps)
        return sigma.clamp(min_sigma, max_sigma).to(dtype=dtype)

    def _future_3d_pre_dit(
        self,
        batch_size: int,
        dtype: torch.dtype,
        timestep: torch.Tensor | None = None,
        scheduler: WanContinuousFlowMatchScheduler | None = None,
    ) -> tuple[dict[str, Any], torch.Tensor | None]:
        if self.future_3d_expert is None:
            raise RuntimeError("Future3DExpert is not enabled.")
        query_noise_sigma = self._future_3d_query_noise_sigma(
            batch_size=batch_size,
            timestep=timestep,
            scheduler=scheduler,
            dtype=dtype,
        )
        if getattr(self.future_3d_expert, "query_mode", "query_token") == "query_token":
            timestep = None
        elif query_noise_sigma is not None and scheduler is not None:
            timestep = (
                query_noise_sigma.to(device=self.device, dtype=torch.float32)
                * float(scheduler.num_train_timesteps)
            ).to(dtype=dtype)
        pre = self.future_3d_expert.pre_dit(
            batch_size=batch_size,
            device=self.device,
            dtype=dtype,
            timestep=timestep,
            query_noise_sigma=query_noise_sigma,
        )
        return pre, query_noise_sigma

    def training_loss(
        self,
        sample,
        tiled: bool = False,
        collect_loss_dict: bool = True,
        collect_detailed_loss_dict: bool = True,
        loss_dict_as_tensors: bool = False,
    ):
        inputs = self.build_inputs(sample, tiled=tiled)
        input_latents = inputs["input_latents"]
        batch_size = input_latents.shape[0]
        context = inputs["context"]
        context_mask = inputs["context_mask"]
        action = inputs["action"]
        sample_action_loss_mask = inputs.get("sample_action_loss_mask", None)
        action_dim_is_pad = inputs.get("action_dim_is_pad", None)
        image_is_pad = inputs["image_is_pad"]
        future_3d_images = inputs["future_3d_images"]
        future_3d_img_masks = inputs["future_3d_img_masks"]

        noise_video = torch.randn_like(input_latents)
        timestep_video = self.train_video_scheduler.sample_training_t(
            batch_size=batch_size,
            device=self.device,
            dtype=input_latents.dtype,
        )
        latents = self.train_video_scheduler.add_noise(input_latents, noise_video, timestep_video)
        target_video = self.train_video_scheduler.training_target(input_latents, noise_video, timestep_video)

        if inputs["first_frame_latents"] is not None:
            latents[:, :, 0:1] = inputs["first_frame_latents"]

        noise_action = torch.randn_like(action)
        timestep_action = self.train_action_scheduler.sample_training_t(
            batch_size=batch_size,
            device=self.device,
            dtype=action.dtype,
        )
        noisy_action = self.train_action_scheduler.add_noise(action, noise_action, timestep_action)
        target_action = self.train_action_scheduler.training_target(action, noise_action, timestep_action)

        video_pre = self.video_expert.pre_dit(
            x=latents,
            timestep=timestep_video,
            context=context,
            context_mask=context_mask,
            action=action,
            fuse_vae_embedding_in_latents=inputs["fuse_vae_embedding_in_latents"],
        )

        action_pre = self.action_expert.pre_dit(
            action_tokens=noisy_action,
            timestep=timestep_action,
            context=context,
            context_mask=context_mask,
        )
        future_3d_pre = None
        future_3d_query_sigma = None
        if self.future_3d_expert is not None:
            future_3d_pre, future_3d_query_sigma = self._future_3d_pre_dit(
                batch_size=batch_size,
                dtype=self.torch_dtype,
                timestep=timestep_video,
                scheduler=self.train_video_scheduler,
            )

        video_tokens = video_pre["tokens"]
        action_tokens = action_pre["tokens"]
        future_3d_tokens = None if future_3d_pre is None else future_3d_pre["tokens"]
        future_3d_seq_len = 0 if future_3d_tokens is None else int(future_3d_tokens.shape[1])

        attention_mask = self._build_mot_attention_mask(
            video_seq_len=video_tokens.shape[1],
            action_seq_len=action_tokens.shape[1],
            video_tokens_per_frame=int(video_pre["meta"]["tokens_per_frame"]),
            device=video_tokens.device,
            future_3d_seq_len=future_3d_seq_len,
            video_grid_size=video_pre["meta"]["grid_size"],
        )
        embeds_all = {
            "video": video_tokens,
            "action": action_tokens,
        }
        freqs_all = {
            "video": video_pre["freqs"],
            "action": action_pre["freqs"],
        }
        context_all = {
            "video": {
                "context": video_pre["context"],
                "mask": video_pre["context_mask"],
            },
            "action": {
                "context": action_pre["context"],
                "mask": action_pre["context_mask"],
            },
        }
        t_mod_all = {
            "video": video_pre["t_mod"],
            "action": action_pre["t_mod"],
        }
        collect_expert_layers = None
        if future_3d_pre is not None:
            embeds_all = {
                "video": video_tokens,
                "future_3d": future_3d_tokens,
                "action": action_tokens,
            }
            freqs_all["future_3d"] = future_3d_pre["freqs"]
            context_all["future_3d"] = None
            t_mod_all["future_3d"] = future_3d_pre["t_mod"]
            if self.da3_teacher is not None:
                collect_expert_layers = {"future_3d": self.future_3d_expert.query_layer_indices}

        mot_out = self.mot(
            embeds_all=embeds_all,
            attention_mask=attention_mask,
            freqs_all=freqs_all,
            context_all=context_all,
            t_mod_all=t_mod_all,
            collect_expert_layers=collect_expert_layers,
        )
        future_3d_layer_tokens = None
        if collect_expert_layers is not None:
            tokens_out, collected = mot_out
            future_3d_layer_tokens = collected["future_3d"]
        else:
            tokens_out = mot_out

        pred_video = self.video_expert.post_dit(tokens_out["video"], video_pre)

        pred_action = self.action_expert.post_dit(tokens_out["action"], action_pre)

        include_initial_video_step = inputs["first_frame_latents"] is None
        if inputs["first_frame_latents"] is not None:
            pred_video = pred_video[:, :, 1:]
            target_video = target_video[:, :, 1:]

        loss_video_per_sample = self._compute_video_loss_per_sample(
            pred_video=pred_video,
            target_video=target_video,
            image_is_pad=image_is_pad,
            include_initial_video_step=include_initial_video_step,
        )
        video_weight = self.train_video_scheduler.training_weight(timestep_video).to(
            loss_video_per_sample.device, dtype=loss_video_per_sample.dtype
        )
        loss_video = (loss_video_per_sample * video_weight).mean()

        action_loss_raw = F.mse_loss(
            pred_action.float(),
            target_action.float(),
            reduction="none",
        )
        action_dim_valid_mask: torch.Tensor | None = None
        if self.mask_action_dim_padding_loss and action_dim_is_pad is not None:
            action_dim_valid_mask = (~action_dim_is_pad).to(
                device=action_loss_raw.device,
                dtype=action_loss_raw.dtype,
            )
            valid_dim_count = action_dim_valid_mask.sum(dim=1).clamp(min=1.0)
            action_loss_token = (
                action_loss_raw * action_dim_valid_mask[:, None, :]
            ).sum(dim=2) / valid_dim_count[:, None]
        else:
            action_loss_token = action_loss_raw.mean(dim=2)  # [B, T]
        action_loss_per_sample = action_loss_token.mean(dim=1)

        action_weight = self.train_action_scheduler.training_weight(timestep_action).to(
            action_loss_per_sample.device, dtype=action_loss_per_sample.dtype
        )
        weighted_action_loss = action_loss_per_sample * action_weight
        if sample_action_loss_mask is not None:
            sample_valid = sample_action_loss_mask.to(
                device=weighted_action_loss.device,
                dtype=weighted_action_loss.dtype,
            ) > 0.5
            if sample_valid.any():
                loss_action = weighted_action_loss[sample_valid].mean()
            else:
                loss_action = weighted_action_loss.new_zeros(())
        else:
            loss_action = weighted_action_loss.mean()
        loss_3d, loss_3d_logs = self._compute_future_3d_loss(
            future_3d_layer_tokens=future_3d_layer_tokens,
            future_images=future_3d_images,
            img_masks=future_3d_img_masks,
            collect_logs=collect_loss_dict and collect_detailed_loss_dict,
        )

        loss_video_w = self.loss_lambda_video * loss_video
        loss_action_w = self.loss_lambda_action * loss_action
        loss_3d_w = self.loss_lambda_3d * loss_3d
        loss_total = loss_video_w + loss_action_w + loss_3d_w
        if not collect_loss_dict:
            return loss_total, {}

        if loss_dict_as_tensors:
            loss_dict = {
                "loss_video": loss_video.detach(),
                "loss_action": loss_action.detach(),
                "loss_3d": loss_3d.detach(),
                "loss_video_w": loss_video_w.detach(),
                "loss_action_w": loss_action_w.detach(),
                "loss_3d_w": loss_3d_w.detach(),
            }
        else:
            loss_dict = {
                "loss_video": float(loss_video.detach().item()),
                "loss_action": float(loss_action.detach().item()),
                "loss_3d": float(loss_3d.detach().item()),
                "loss_video_w": float(loss_video_w.detach().item()),
                "loss_action_w": float(loss_action_w.detach().item()),
                "loss_3d_w": float(loss_3d_w.detach().item()),
            }
        if not collect_detailed_loss_dict:
            return loss_total, loss_dict

        dim_weight = torch.ones_like(action_loss_token, dtype=action_loss_raw.dtype)
        action_weight_for_dim = action_weight.to(device=dim_weight.device, dtype=dim_weight.dtype)
        if action_weight_for_dim.ndim == 0:
            action_weight_for_dim = action_weight_for_dim.expand(action_loss_raw.shape[0])
        dim_weight = dim_weight * action_weight_for_dim[:, None]
        if sample_action_loss_mask is not None:
            dim_weight = dim_weight * (
                sample_action_loss_mask.to(device=dim_weight.device, dtype=dim_weight.dtype) > 0.5
            )[:, None].to(dtype=dim_weight.dtype)
        if action_dim_valid_mask is not None:
            dim_weight_3d = dim_weight[:, :, None] * action_dim_valid_mask[:, None, :]
            dim_denom = dim_weight_3d.sum(dim=(0, 1)).clamp(min=1.0)
            loss_action_by_dim = (action_loss_raw * dim_weight_3d).sum(dim=(0, 1)) / dim_denom
        else:
            dim_denom = dim_weight.sum(dim=(0, 1)).clamp(min=1.0)
            loss_action_by_dim = (action_loss_raw * dim_weight[:, :, None]).sum(dim=(0, 1)) / dim_denom
        for dim_idx, dim_loss in enumerate(loss_action_by_dim.detach().float().cpu().tolist()):
            loss_dict[f"loss_action_dim{dim_idx}"] = float(dim_loss)
        if future_3d_query_sigma is not None:
            loss_dict["future_3d_query_sigma"] = float(future_3d_query_sigma.detach().float().mean().item())
        for key, value in loss_3d_logs.items():
            loss_dict[key] = float(value.detach().item())
        return loss_total, loss_dict

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
        future_3d_pre = None
        if self.future_3d_expert is not None:
            future_3d_pre, _ = self._future_3d_pre_dit(
                batch_size=latents_action.shape[0],
                dtype=self.torch_dtype,
                timestep=timestep_video,
                scheduler=self.infer_video_scheduler,
            )
        future_3d_seq_len = 0 if future_3d_pre is None else int(future_3d_pre["tokens"].shape[1])

        attention_mask = self._build_mot_attention_mask(
            video_seq_len=video_pre["tokens"].shape[1],
            action_seq_len=action_pre["tokens"].shape[1],
            video_tokens_per_frame=int(video_pre["meta"]["tokens_per_frame"]),
            device=video_pre["tokens"].device,
            future_3d_seq_len=future_3d_seq_len,
            video_grid_size=video_pre["meta"]["grid_size"],
        )
        embeds_all = {
            "video": video_pre["tokens"],
            "action": action_pre["tokens"],
        }
        freqs_all = {
            "video": video_pre["freqs"],
            "action": action_pre["freqs"],
        }
        context_all = {
            "video": {
                "context": video_pre["context"],
                "mask": video_pre["context_mask"],
            },
            "action": {
                "context": action_pre["context"],
                "mask": action_pre["context_mask"],
            },
        }
        t_mod_all = {
            "video": video_pre["t_mod"],
            "action": action_pre["t_mod"],
        }
        if future_3d_pre is not None:
            embeds_all = {
                "video": video_pre["tokens"],
                "future_3d": future_3d_pre["tokens"],
                "action": action_pre["tokens"],
            }
            freqs_all["future_3d"] = future_3d_pre["freqs"]
            context_all["future_3d"] = None
            t_mod_all["future_3d"] = future_3d_pre["t_mod"]

        tokens_out = self.mot(
            embeds_all=embeds_all,
            attention_mask=attention_mask,
            freqs_all=freqs_all,
            context_all=context_all,
            t_mod_all=t_mod_all,
        )

        pred_video = self.video_expert.post_dit(tokens_out["video"], video_pre)
        pred_action = self.action_expert.post_dit(tokens_out["action"], action_pre)
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
        future_3d_pre = None
        if self.future_3d_expert is not None:
            future_3d_pre, _ = self._future_3d_pre_dit(
                batch_size=latents_action.shape[0],
                dtype=self.torch_dtype,
                timestep=timestep_action,
                scheduler=self.infer_action_scheduler,
            )
        future_3d_seq_len = 0 if future_3d_pre is None else int(future_3d_pre["tokens"].shape[1])

        attention_mask = self._build_mot_attention_mask(
            video_seq_len=video_pre["tokens"].shape[1],
            action_seq_len=action_pre["tokens"].shape[1],
            video_tokens_per_frame=int(video_pre["meta"]["tokens_per_frame"]),
            device=video_pre["tokens"].device,
            future_3d_seq_len=future_3d_seq_len,
            video_grid_size=video_pre["meta"]["grid_size"],
        )
        embeds_all = {
            "video": video_pre["tokens"],
            "action": action_pre["tokens"],
        }
        freqs_all = {
            "video": video_pre["freqs"],
            "action": action_pre["freqs"],
        }
        context_all = {
            "video": {
                "context": video_pre["context"],
                "mask": video_pre["context_mask"],
            },
            "action": {
                "context": action_pre["context"],
                "mask": action_pre["context_mask"],
            },
        }
        t_mod_all = {
            "video": video_pre["t_mod"],
            "action": action_pre["t_mod"],
        }
        if future_3d_pre is not None:
            embeds_all = {
                "video": video_pre["tokens"],
                "future_3d": future_3d_pre["tokens"],
                "action": action_pre["tokens"],
            }
            freqs_all["future_3d"] = future_3d_pre["freqs"]
            context_all["future_3d"] = None
            t_mod_all["future_3d"] = future_3d_pre["t_mod"]
        tokens_out = self.mot(
            embeds_all=embeds_all,
            attention_mask=attention_mask,
            freqs_all=freqs_all,
            context_all=context_all,
            t_mod_all=t_mod_all,
        )
        pred_action = self.action_expert.post_dit(tokens_out["action"], action_pre)
        return pred_action

    @torch.no_grad()
    def infer_joint(
        self,
        prompt: Optional[str],
        input_image: torch.Tensor,
        num_video_frames: int,
        action_horizon: int,
        action: Optional[torch.Tensor] = None, # NOTE: this is gt action for conditioning videos, not for action expert
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
        test_action_with_infer_action: bool = True,
    ) -> dict[str, Any]:
        self.eval()
        if test_action_with_infer_action:
            if seed is None:
                raise ValueError("`test_action_with_infer_action=True` requires non-null `seed`.")
            action_only_out = self.infer_action(
                prompt=prompt,
                input_image=input_image.clone(),
                action_horizon=action_horizon,
                context=context.clone() if context is not None else None,
                context_mask=context_mask.clone() if context_mask is not None else None,
                num_inference_steps=num_inference_steps,
                sigma_shift=sigma_shift,
                seed=seed,
                rand_device=rand_device,
                tiled=tiled,
                proprio=proprio.clone() if proprio is not None else None,
            )["action"]
        
        if input_image.ndim == 3:
            input_image = input_image.unsqueeze(0)
        if input_image.ndim != 4 or input_image.shape[0] != 1 or input_image.shape[1] != 3:
            raise ValueError(
                f"`input_image` must have shape [1,3,H,W] or [3,H,W], got {tuple(input_image.shape)}"
            )
        _, _, height, width = input_image.shape
        checked_h, checked_w, checked_t = self._check_resize_height_width(height, width, num_video_frames)
        if (checked_h, checked_w) != (height, width):
            raise ValueError(
                f"`input_image` must be resized before infer, expected multiples of 16 but got HxW=({height},{width})"
            )
        if checked_t != num_video_frames:
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
        if proprio is not None:
            if self.proprio_dim is None:
                raise ValueError("`proprio` was provided but `proprio_dim=None` so `proprio_encoder` is disabled.")
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
        latents_video = torch.randn(
            (1, self.vae.model.z_dim, latent_t, latent_h, latent_w),
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
        latents_video[:, :, 0:1] = first_frame_latents.clone()
        fuse_flag = bool(getattr(self.video_expert, "fuse_vae_embedding_in_latents", False))

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
        if proprio is not None:
            context, context_mask = self._append_proprio_to_context(
                context=context,
                context_mask=context_mask,
                proprio=proprio,
            )

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
        for step_t_video, step_delta_video, step_t_action, step_delta_action in zip(
            infer_timesteps_video,
            infer_deltas_video,
            infer_timesteps_action,
            infer_deltas_action,
        ):
            timestep_video = step_t_video.unsqueeze(0).to(dtype=latents_video.dtype, device=self.device)
            timestep_action = step_t_action.unsqueeze(0).to(dtype=latents_action.dtype, device=self.device)

            pred_video_posi, pred_action_posi = self._predict_joint_noise(
                latents_video=latents_video,
                latents_action=latents_action,
                timestep_video=timestep_video,
                timestep_action=timestep_action,
                context=context,
                context_mask=context_mask,
                fuse_vae_embedding_in_latents=fuse_flag,
                gt_action=action,
            )
            pred_video = pred_video_posi
            pred_action = pred_action_posi

            latents_video = self.infer_video_scheduler.step(pred_video, step_delta_video, latents_video)
            latents_action = self.infer_action_scheduler.step(pred_action, step_delta_action, latents_action)
            latents_video[:, :, 0:1] = first_frame_latents.clone()

        action_out = latents_action[0].detach().to(device="cpu", dtype=torch.float32)
        if test_action_with_infer_action:
            if not torch.allclose(action_out, action_only_out, atol=1e-2, rtol=1e-2):
                max_abs_diff = (action_out - action_only_out).abs().max().item()
                logger.warning(
                    f"Action from infer_joint and infer_action differ with max abs diff {max_abs_diff:.6f}. "
                )

        return {
            "video": self._decode_latents(latents_video, tiled=tiled),
            "action": action_out,
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
    ) -> dict[str, Any]:
        self.eval()

        if input_image.ndim == 3:
            input_image = input_image.unsqueeze(0)
        if input_image.ndim != 4 or input_image.shape[0] != 1 or input_image.shape[1] != 3:
            raise ValueError(
                f"`input_image` must have shape [1,3,H,W] or [3,H,W], got {tuple(input_image.shape)}"
            )
        _, _, height, width = input_image.shape
        if height % 16 != 0 or width % 16 != 0:
            raise ValueError(
                f"`input_image` must be resized before infer, expected multiples of 16 but got HxW=({height},{width})"
            )
        if proprio is not None:
            if self.proprio_dim is None:
                raise ValueError("`proprio` was provided but `proprio_dim=None` so `proprio_encoder` is disabled.")
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

        input_image = input_image.to(device=self.device, dtype=self.torch_dtype)
        first_frame_latents = self._encode_input_image_latents_tensor(input_image=input_image, tiled=tiled)
        fuse_flag = bool(getattr(self.video_expert, "fuse_vae_embedding_in_latents", False))

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
        if proprio is not None:
            context, context_mask = self._append_proprio_to_context(
                context=context,
                context_mask=context_mask,
                proprio=proprio,
            )

        infer_timesteps_action, infer_deltas_action = self.infer_action_scheduler.build_inference_schedule(
            num_inference_steps=num_inference_steps,
            device=self.device,
            dtype=latents_action.dtype,
            shift_override=sigma_shift,
        )
        for step_t_action, step_delta_action in zip(infer_timesteps_action, infer_deltas_action):
            timestep_action = step_t_action.unsqueeze(0).to(dtype=latents_action.dtype, device=self.device)

            pred_action_posi = self._predict_action_noise(
                first_frame_latents=first_frame_latents,
                latents_action=latents_action,
                timestep_action=timestep_action,
                context=context,
                context_mask=context_mask,
                fuse_vae_embedding_in_latents=fuse_flag,
            )
            pred_action = pred_action_posi

            latents_action = self.infer_action_scheduler.step(pred_action, step_delta_action, latents_action)

        return {
            "action": latents_action[0].detach().to(device="cpu", dtype=torch.float32),
        }

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

    def save_checkpoint(self, path, optimizer=None, step=None):
        payload = {
            "mot": self.mot.state_dict(),
            "step": step,
            "torch_dtype": str(self.torch_dtype),
        }
        if self.proprio_encoder is not None:
            payload["proprio_encoder"] = self.proprio_encoder.state_dict()
        if optimizer is not None:
            payload["optimizer"] = optimizer.state_dict()
        torch.save(payload, path)

    def load_checkpoint(self, path, optimizer=None):
        payload = torch.load(path, map_location="cpu")
        if "mot" in payload:
            self.mot.load_state_dict(payload["mot"], strict=False)
        elif "dit" in payload:
            logger.warning("Loading legacy `dit` checkpoint into video expert only.")
            self.video_expert.load_state_dict(payload["dit"], strict=False)
        else:
            raise ValueError(f"Checkpoint missing both `mot` and `dit` keys: {path}")
        if self.proprio_encoder is not None:
            if "proprio_encoder" in payload:
                self.proprio_encoder.load_state_dict(payload["proprio_encoder"], strict=True)
            else:
                logger.warning("Checkpoint has no `proprio_encoder` weights; keeping current `proprio_encoder` params.")
        elif "proprio_encoder" in payload:
            logger.warning("Checkpoint contains `proprio_encoder` weights but current model has `proprio_dim=None`; ignoring.")

        if optimizer is not None and "optimizer" in payload:
            optimizer.load_state_dict(payload["optimizer"])
        return payload

    def forward(self, *args, **kwargs):
        return self.training_loss(*args, **kwargs)
