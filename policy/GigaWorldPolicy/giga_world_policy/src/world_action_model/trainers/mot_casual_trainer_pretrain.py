import os

import torch
from diffusers.models import AutoencoderKLWan
from diffusers.video_processor import VideoProcessor

from ..models.transformer_wa_mot import MoTWorldActionTransformer
from ..trainer import DictConfig, ModuleDict
from .wa_casual_trainer_pretrain import CasualWATrainerPretrain
from .wa_trainer import get_model_path, process_transformer


class MoTCasualWATrainerPretrain(CasualWATrainerPretrain):
    """Casual GWP pretraining with a FastWAM-style MoT transformer."""

    def _validate_mot_checkpoint(self, checkpoint):
        if checkpoint is None:
            return
        if isinstance(checkpoint, (list, tuple)):
            for ckpt in checkpoint:
                self._validate_mot_checkpoint(ckpt)
            return
        state_dict = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]
        elif "model_state_dict" in state_dict:
            state_dict = state_dict["model_state_dict"]
        keys = tuple(state_dict.keys())
        if not any(k.startswith("transformer.mot.") or k.startswith("mot.") or ".mot." in k for k in keys):
            raise ValueError(
                "MoTCasualWATrainerPretrain only supports MoT checkpoints. "
                f"Checkpoint does not contain MoT keys: {checkpoint}"
            )

    def get_models(self, model_config: DictConfig):
        pretrained = get_model_path(model_config.pretrained)
        self.flow_shift = model_config.flow_shift
        self.action_flow_shift = float(model_config.get("action_flow_shift", self.flow_shift))
        self.expand_timesteps = model_config.get("expand_timesteps", False)
        self.action_loss_weight = float(model_config.get("action_loss_weight", 1.0))
        self.visual_loss_weight = float(model_config.get("visual_loss_weight", 1.0))
        self.use_gt_action_for_video = model_config.get("use_gt_action_for_video", False)
        self.action_repeats = model_config.get("action_repeats", 1)
        self.state_repeats = model_config.get("state_repeats", 1)
        self.action_dim = int(model_config.get("action_dim", 14))
        self.state_dim = int(model_config.get("state_dim", self.action_dim))
        self.view_interval = int(model_config.get("view_interval", 50))
        self.view_dir = model_config.view_dir

        model = {}
        vae_pretrained = model_config.get("vae_pretrained", os.path.join(pretrained, "vae"))
        vae_dtype = model.get("vae_dtype", self.dtype)
        vae = AutoencoderKLWan.from_pretrained(vae_pretrained)
        vae.requires_grad_(False)
        vae.to(self.device, dtype=vae_dtype)
        self.vae = vae
        self.vae_scale_factor_temporal = self.vae.config.scale_factor_temporal
        self.vae_scale_factor_spatial = self.vae.config.scale_factor_spatial
        self.latents_mean = torch.tensor(self.vae.config.latents_mean).view(1, self.vae.config.z_dim, 1, 1, 1).to(
            self.device, dtype=vae_dtype
        )
        self.latents_std = 1.0 / torch.tensor(self.vae.config.latents_std).view(1, self.vae.config.z_dim, 1, 1, 1).to(
            self.device, dtype=vae_dtype
        )
        self.video_processor = VideoProcessor(vae_scale_factor=self.vae_scale_factor_spatial)

        transformer_pretrained = model_config.get("transformer_pretrained", os.path.join(pretrained, "transformer"))
        transformer = MoTWorldActionTransformer.from_pretrained_video(
            transformer_pretrained=transformer_pretrained,
            torch_dtype=self.dtype,
            action_dim=self.action_dim,
            state_dim=self.state_dim,
            action_expert=model_config.get("action_expert", {}),
            mot_checkpoint_mixed_attn=bool(model_config.get("mot_checkpoint_mixed_attn", True)),
            video_attention_mask_mode=model_config.get("video_attention_mask_mode", "gwp_casual"),
            unpretrain=model_config.get("unpretrain", False),
        )
        process_transformer(transformer.video_expert, model_config.get("transformer", {}))
        transformer.to(self.device, dtype=self.dtype)
        model.update(transformer=transformer)

        checkpoint = model_config.get("checkpoint", None)
        strict = model_config.get("strict", True)
        self._validate_mot_checkpoint(checkpoint)
        self.load_checkpoint(checkpoint, list(model.values()), strict=strict)
        model = ModuleDict(model)
        model.train()

        if model_config.get("freeze_backbone", False):
            frozen_count, trainable_count = 0, 0
            for name, param in transformer.named_parameters():
                if name.startswith("mot.mixtures.video."):
                    param.requires_grad = False
                    frozen_count += 1
                else:
                    param.requires_grad = True
                    trainable_count += 1
            if self.process_index == 0:
                print(f"Freeze video backbone: {frozen_count} params frozen, {trainable_count} params trainable")

        if model_config.get("freeze_action", False):
            frozen_count, trainable_count = 0, 0
            for name, param in transformer.named_parameters():
                if name.startswith("mot.mixtures.action."):
                    param.requires_grad = False
                    frozen_count += 1
                else:
                    trainable_count += 1
            if self.process_index == 0:
                print(f"Freeze action expert: {frozen_count} params frozen, {trainable_count} params trainable")

        return model
