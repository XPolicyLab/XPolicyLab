"""PrismWAM*: the RoboTwin model with Wan2.1 image-to-video experts.

The RGB and depth experts share one Wan2.1 I2V block stack and attend to CLIP
features of the current RGB frame. The RGB expert observes five RGB frames
(stride 4) as two latents; its input positions are pure noise at t=1000, so its
pass does not depend on the denoising step and is cached once per action chunk.
Depth observes the current depth latent and predicts three latents (nine
frames). The action expert reads depth through the RGB -> Depth -> Action chain.

Only inference is released: the model is built directly from its weights.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import torch

from ..wan22.action_dit import ActionDiT
from ..wan22.helpers.io import ModelConfig, load_state_dict
from ..wan22.helpers.loader import (
    _load_registered_model,
    _load_wan_video_vae,
    _resolve_configs,
)
from ..wan22.mot import MoT
from ..wan22.prismwam import PrismWAM
from ..wan22.rgb_depth_action import build_rgb_depth_action_attention_mask
from ..wan22.wan_video_dit import precompute_freqs_cis, precompute_freqs_cis_3d
from ..wan22.wan_video_text_encoder import HuggingfaceTokenizer
from ..wan22.wan_video_vae import WanVideoVAE
from .clip import CLIPImageEncoder
from .dit import I2VExpert

CHECKPOINT_FORMAT = "prismwam_star_v1"
IMAGE_ENCODER_MODEL_ID = "Wan-AI/Wan2.1-I2V-14B-480P"
IMAGE_ENCODER_FILE = "models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth"
ACTION_DIT_CONFIG = dict(
    hidden_dim=1024,
    action_dim=14,
    ffn_dim=4096,
    text_dim=4096,
    freq_dim=256,
    eps=1e-6,
    num_heads=40,
    attn_head_dim=128,
    num_layers=40,
)
RGB_TIMESTEP = 1000.0


class PrismWAMStar(PrismWAM):
    """RGB history, current depth and robot state in; a 32-step joint chunk out."""

    rgb_history_num_frames = 5
    rgb_history_frame_stride = 4
    num_video_frames = 9
    action_horizon = 32

    def __init__(self, *, rgb_expert, rgb_vae, clip_encoder, **kwargs):
        super().__init__(**kwargs)
        if list(self.mot.expert_order) != ["rgb", "video", "action"]:
            raise ValueError("Expected the MoT experts rgb, video and action.")
        if self.mot.mixtures["rgb"] is not rgb_expert:
            raise ValueError("`rgb_expert` must be the MoT `rgb` expert.")
        if rgb_expert.blocks is not self.video_expert.blocks:
            raise ValueError("RGB and depth must share one transformer block stack.")
        self.rgb_expert = rgb_expert
        # The RGB VAE runs in float32 and CLIP in float16, whatever the model dtype.
        self.rgb_vae = rgb_vae.to(device=self.device).eval().requires_grad_(False)
        self.clip_encoder = (
            clip_encoder.to(device=self.device).eval().requires_grad_(False)
        )

    @torch.no_grad()
    def _encode_rgb_history(self, history: torch.Tensor) -> torch.Tensor:
        dtype = next(self.rgb_vae.parameters()).dtype
        with torch.autocast(device_type=self.device.type, enabled=False):
            latents = self.rgb_vae.encode(
                history.to(self.device, dtype=dtype), device=self.device, tiled=False
            )
        if isinstance(latents, list):
            latents = torch.stack(latents)
        return latents.to(self.device, dtype=self.torch_dtype)

    @torch.no_grad()
    def _encode_clip(self, history: torch.Tensor) -> torch.Tensor:
        """CLIP features of the current (last) RGB frame."""
        dtype = next(self.clip_encoder.parameters()).dtype
        with torch.autocast(device_type=self.device.type, enabled=False):
            features = self.clip_encoder(
                [history[0, :, -1:].to(self.device, dtype=dtype)]
            )
        return features.to(self.device, dtype=self.torch_dtype)

    @torch.no_grad()
    def _prefill_rgb(self, *, noise, condition, clip_fea, context, context_mask):
        pre = self.rgb_expert.pre_dit(
            x=noise,
            timestep=torch.full(
                (noise.shape[0],), RGB_TIMESTEP, device=self.device, dtype=torch.float32
            ),
            context=context,
            context_mask=context_mask,
            clip_fea=clip_fea,
            condition_latents=condition,
        )
        seq_len = int(pre["tokens"].shape[1])
        cache = self.mot.prefill_prefix_expert_cache(
            expert_name="rgb",
            tokens=pre["tokens"],
            freqs=pre["freqs"],
            t_mod=pre["t_mod"],
            context_payload=self._text_context_payload(
                pre["context"], pre["context_mask"]
            ),
            attention_mask=torch.ones(
                (seq_len, seq_len), dtype=torch.bool, device=self.device
            ),
        )
        return pre, cache

    @torch.no_grad()
    def _predict_depth_action_noise(
        self,
        *,
        rgb_pre,
        rgb_cache,
        depth,
        depth_condition,
        clip_fea,
        action,
        timestep_depth,
        timestep_action,
        context,
        context_mask,
    ):
        depth_pre = self.video_expert.pre_dit(
            x=depth,
            timestep=timestep_depth,
            context=context,
            context_mask=context_mask,
            clip_fea=clip_fea,
            condition_latents=depth_condition,
        )
        action_pre = self.action_expert.pre_dit(
            action_tokens=action,
            timestep=timestep_action,
            context=context,
            context_mask=context_mask,
        )
        rgb_len = int(rgb_pre["tokens"].shape[1])
        depth_len = int(depth_pre["tokens"].shape[1])
        attention_mask = build_rgb_depth_action_attention_mask(
            rgb_self_mask=torch.ones(
                (rgb_len, rgb_len), dtype=torch.bool, device=self.device
            ),
            depth_self_mask=torch.ones(
                (depth_len, depth_len), dtype=torch.bool, device=self.device
            ),
            action_seq_len=int(action_pre["tokens"].shape[1]),
            rgb_condition_seq_len=rgb_len,
        )
        outputs = self.mot.forward_with_prefix_expert_cache(
            prefix_expert_name="rgb",
            prefix_kv_cache=rgb_cache,
            embeds_all={"video": depth_pre["tokens"], "action": action_pre["tokens"]},
            attention_mask=attention_mask,
            freqs_all={"video": depth_pre["freqs"], "action": action_pre["freqs"]},
            context_all={
                name: self._text_context_payload(pre["context"], pre["context_mask"])
                for name, pre in (("video", depth_pre), ("action", action_pre))
            },
            t_mod_all={"video": depth_pre["t_mod"], "action": action_pre["t_mod"]},
        )
        return (
            self.video_expert.post_dit(outputs["video"], depth_pre),
            self.action_expert.post_dit(outputs["action"], action_pre),
        )

    @torch.no_grad()
    def infer_action(
        self,
        prompt: Optional[str],
        input_image: torch.Tensor,
        rgb_input_image: torch.Tensor,
        action_horizon: int,
        num_video_frames: int,
        rgb_history_images: Optional[torch.Tensor] = None,
        proprio: Optional[torch.Tensor] = None,
        context: Optional[torch.Tensor] = None,
        context_mask: Optional[torch.Tensor] = None,
        num_inference_steps: int = 10,
        sigma_shift: Optional[float] = None,
        seed: Optional[int] = None,
        rand_device: str = "cpu",
        tiled: bool = False,
    ) -> dict[str, Any]:
        self.eval()
        if (int(action_horizon), int(num_video_frames)) != (
            self.action_horizon,
            self.num_video_frames,
        ):
            raise ValueError(
                f"PrismWAM* predicts {self.action_horizon} actions and {self.num_video_frames} depth frames."
            )
        if str(rand_device) != "cpu" or tiled:
            raise ValueError("Inference samples its noise on CPU and does not tile.")
        frames = self.rgb_history_num_frames
        if rgb_history_images is None or tuple(rgb_history_images.shape[:3]) != (
            1,
            3,
            frames,
        ):
            raise ValueError(f"`rgb_history_images` must be [1,3,{frames},H,W].")
        if rgb_input_image.ndim == 3:
            rgb_input_image = rgb_input_image.unsqueeze(0)
        if not torch.equal(
            rgb_input_image.to(rgb_history_images), rgb_history_images[:, :, -1]
        ):
            raise ValueError("The RGB history must end with the current RGB frame.")
        if input_image.ndim == 3:
            input_image = input_image.unsqueeze(0)
        if tuple(input_image.shape[:2]) != (1, 1):
            raise ValueError("`input_image` must be one depth image [1,1,H,W].")
        if proprio is None or proprio.numel() != self.proprio_dim:
            raise ValueError(
                f"`proprio` must hold the {self.proprio_dim}-D robot state."
            )

        condition = self._encode_rgb_history(rgb_history_images)
        clip_fea = self._encode_clip(rgb_history_images)
        depth_condition = self._encode_input_image_latents_tensor(
            input_image.to(self.device, dtype=self.torch_dtype)
        ).to(dtype=self.torch_dtype)
        context, context_mask = self._resolve_inference_language_context(
            prompt=prompt, context=context, context_mask=context_mask
        )
        context, context_mask = self._append_proprio_to_context(
            context=context,
            context_mask=context_mask,
            proprio=proprio.reshape(1, -1).to(self.device, dtype=self.torch_dtype),
        )

        def noise(shape, offset):
            generator = (
                None
                if seed is None
                else torch.Generator("cpu").manual_seed(int(seed) + offset)
            )
            return torch.randn(
                shape, generator=generator, device="cpu", dtype=torch.float32
            ).to(self.device, dtype=self.torch_dtype)

        rgb_pre, rgb_cache = self._prefill_rgb(
            noise=noise(condition.shape, 0),
            condition=condition,
            clip_fea=clip_fea,
            context=context,
            context_mask=context_mask,
        )
        depth_latent_frames = (self.num_video_frames - 1) // 4 + 1
        depth = noise((1, 16, depth_latent_frames, *depth_condition.shape[-2:]), 1)
        action = noise((1, self.action_horizon, self.action_expert.action_dim), 2)
        depth_times, depth_deltas = self.infer_video_scheduler.build_inference_schedule(
            num_inference_steps=num_inference_steps,
            device=self.device,
            dtype=torch.float32,
            shift_override=sigma_shift,
        )
        action_times, action_deltas = (
            self.infer_action_scheduler.build_inference_schedule(
                num_inference_steps=num_inference_steps,
                device=self.device,
                dtype=torch.float32,
                shift_override=sigma_shift,
            )
        )
        for index in range(int(num_inference_steps)):
            pred_depth, pred_action = self._predict_depth_action_noise(
                rgb_pre=rgb_pre,
                rgb_cache=rgb_cache,
                depth=depth,
                depth_condition=depth_condition,
                clip_fea=clip_fea,
                action=action,
                timestep_depth=depth_times[index : index + 1],
                timestep_action=action_times[index : index + 1],
                context=context,
                context_mask=context_mask,
            )
            depth = self.infer_video_scheduler.step(
                pred_depth, depth_deltas[index], depth
            )
            action = self.infer_action_scheduler.step(
                pred_action, action_deltas[index], action
            )
        if not torch.isfinite(action).all():
            raise FloatingPointError("Non-finite action prediction.")
        return {"action": action[0].detach().to(device="cpu", dtype=torch.float32)}


def read_checkpoint(path: str | Path) -> dict[str, Any]:
    payload = torch.load(
        Path(path).expanduser(), map_location="cpu", weights_only=True, mmap=True
    )
    if not isinstance(payload, dict) or payload.get("format") != CHECKPOINT_FORMAT:
        raise ValueError(f"Not a PrismWAM* checkpoint: {path}")
    return payload


def expand_shared_blocks(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Checkpoints store the shared RGB/depth blocks once, under ``rgb``."""
    if any(key.startswith("mixtures.video.blocks.") for key in state):
        raise ValueError("The shared blocks must be stored once, under rgb.")
    expanded = dict(state)
    for key, value in state.items():
        if key.startswith("mixtures.rgb.blocks."):
            expanded["mixtures.video." + key[len("mixtures.rgb.") :]] = value
    return expanded


def _load_rgb_vae(path: str | Path) -> WanVideoVAE:
    state = load_state_dict(str(path), torch_dtype=torch.float32, device="cpu")
    if not all(key.startswith("model.") for key in state):
        state = {"model." + key: value for key, value in state.items()}
    vae = WanVideoVAE()
    vae.load_state_dict(state, strict=True)
    return vae.eval().requires_grad_(False)


def _load_clip(model_id: str) -> CLIPImageEncoder:
    config = ModelConfig(model_id=model_id, origin_file_pattern=IMAGE_ENCODER_FILE)
    config.download_if_necessary()
    state = load_state_dict(config.path, torch_dtype=torch.float16, device="cpu")
    state = {
        key: value
        for key, value in state.items()
        if key.startswith("visual.") or key == "log_scale"
    }
    with torch.device("meta"):
        clip = CLIPImageEncoder()
    clip.load_state_dict(state, strict=True, assign=True)
    return clip.eval().requires_grad_(False)


def create_prismwam_star(
    *,
    checkpoint_path: str,
    rgb_vae_checkpoint_path: str,
    depth_vae_checkpoint_path: str,
    image_encoder_model_id: str = IMAGE_ENCODER_MODEL_ID,
    model_id: str = "Wan-AI/Wan2.2-TI2V-5B",
    tokenizer_model_id: str = "Wan-AI/Wan2.2-TI2V-5B",
    tokenizer_max_len: int = 128,
    load_text_encoder: bool = True,
    redirect_common_files: bool = False,
    proprio_dim: int = 14,
    checkpoint_init_only: bool = True,
    model_dtype: torch.dtype = torch.bfloat16,
    device: str = "cpu",
) -> PrismWAMStar:
    """Build PrismWAM* from its weights.

    ``model_id`` and ``tokenizer_model_id`` name where the T5 text encoder and
    its tokenizer are read from, under ``PRISMWAM_WEIGHTS_ROOT`` (default
    ``./weights``); the CLIP image encoder comes from ``image_encoder_model_id``.
    """
    if not checkpoint_init_only:
        raise ValueError("PrismWAM* is built from its released weights only.")
    payload = read_checkpoint(checkpoint_path)
    with torch.device("meta"):
        rgb = I2VExpert(num_condition_frames=2)
        depth = I2VExpert(num_condition_frames=1)
        action = ActionDiT(**ACTION_DIT_CONFIG)
    depth.blocks = rgb.blocks
    mot = MoT(
        {"rgb": rgb, "video": depth, "action": action}, mot_checkpoint_mixed_attn=False
    )
    mot.load_state_dict(expand_shared_blocks(payload["mot"]), strict=True, assign=True)
    if any(p.is_meta for p in mot.parameters()):
        raise RuntimeError("Some PrismWAM* weights were not loaded.")
    mot.to(dtype=model_dtype)
    # Rotary tables are not stored; the meta construction left them empty.
    rgb.freqs = precompute_freqs_cis_3d(rgb.attn_head_dim)
    depth.freqs = precompute_freqs_cis_3d(depth.attn_head_dim)
    action.freqs = precompute_freqs_cis(action.attn_head_dim, end=1024)

    text_encoder = tokenizer = None
    if load_text_encoder:
        _, text_config, _, tokenizer_config = _resolve_configs(
            model_id=model_id,
            tokenizer_model_id=tokenizer_model_id,
            redirect_common_files=redirect_common_files,
        )
        text_config.download_if_necessary()
        tokenizer_config.download_if_necessary()
        text_encoder = _load_registered_model(
            text_config.path,
            "wan_video_text_encoder",
            torch_dtype=model_dtype,
            device="cpu",
        ).eval()
        tokenizer = HuggingfaceTokenizer(
            name=tokenizer_config.path,
            seq_len=int(tokenizer_max_len),
            clean="whitespace",
        )
    depth_vae = _load_wan_video_vae(
        depth_vae_checkpoint_path,
        torch_dtype=model_dtype,
        device="cpu",
        vae_variant="depth",
        depth_vae_require_latent_stats=True,
    ).eval()
    model = PrismWAMStar(
        rgb_expert=rgb,
        rgb_vae=_load_rgb_vae(rgb_vae_checkpoint_path),
        clip_encoder=_load_clip(image_encoder_model_id),
        video_expert=depth,
        action_expert=action,
        mot=mot,
        vae=depth_vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        text_dim=ACTION_DIT_CONFIG["text_dim"],
        proprio_dim=proprio_dim,
        device=device,
        torch_dtype=model_dtype,
    )
    model.proprio_encoder.load_state_dict(payload["proprio_encoder"], strict=True)
    return model
