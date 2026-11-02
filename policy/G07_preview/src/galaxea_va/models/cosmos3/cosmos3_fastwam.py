"""``Cosmos3FastWAM`` — Cosmos3-Nano wired into the FastWAM video+action policy.

Mirrors the trainer/eval contract of ``galaxea_va.models.wan22.fastwam.FastWAM``
(uncond base), but the MoT backbone is the vendored Cosmos3 reasoner(text/causal)
+ generator(video+action/full) with a fastwam-style asymmetric attention mask.
"""

from __future__ import annotations


import os

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from galaxea_va.utils.logging_config import get_logger

from ..wan22.schedulers.scheduler_continuous import WanContinuousFlowMatchScheduler
from .loader import COSMOS3_NANO_ROOT, load_cosmos3_transformer
from .vae_loader import load_wan22_vae
from .vision_encoder import load_cosmos3_vision_encoder, preprocess_image
from .transformer import Cosmos3Backbone

logger = get_logger(__name__)

_MODALITY_MARGIN = 15000  # unified_3d_mrope_temporal_modality_margin

# Cosmos3-Nano vision special-token ids (top-level config.json).
_IMAGE_TOKEN_ID = 151655
_VISION_START_TOKEN_ID = 151652
_VISION_END_TOKEN_ID = 151653


def _compile_inference(function):
    """Compile Cosmos3 inference cores with the Python wrapper and no CUDA Graph."""
    return torch.compile(
        function,
        fullgraph=True,
        options={"triton.cudagraphs": False, "cpp_wrapper": False},
    )


class Cosmos3FastWAM(nn.Module):
    supports_fdm = True
    evaluation_action_modes = ("first_frame",)
    def __init__(
        self,
        *,
        backbone: Cosmos3Backbone,
        vae,
        tokenizer,
        proprio_dim: int,
        action_dim: int,
        action_horizon: int,
        tokenizer_max_len: int = 128,
        reasoner_input: str = "text",
        vision_encoder=None,
        action_path: str = "cogen",
        video_mask_mode: str = "first_frame_causal",
        enable_fps_modulation: bool = False,
        fps: float | None = None,
        base_fps: float = 24.0,
        action_lr_mult: float = 5.0,
        trainable_scope: str = "generator",
        video_train_shift: float = 5.0,
        video_infer_shift: float = 5.0,
        action_train_shift: float = 5.0,
        action_infer_shift: float = 5.0,
        num_train_timesteps: int = 1000,
        loss_lambda_video: float = 1.0,
        loss_lambda_action: float = 10.0,
        gradient_checkpointing: bool = False,
        video_fdm_prob: float = 0.0,
        compile_training_denoise: bool = False,
        device: str = "cpu",
        torch_dtype: torch.dtype = torch.bfloat16,
    ):
        super().__init__()
        self.backbone = backbone
        self.vae = vae
        self.tokenizer = tokenizer
        self.proprio_dim = int(proprio_dim)
        self.action_dim = int(action_dim)
        self.action_horizon = int(action_horizon)
        if self.action_horizon <= 0:
            raise ValueError(f"`action_horizon` must be positive, got {self.action_horizon}.")
        self.tokenizer_max_len = int(tokenizer_max_len)
        self.reasoner_input = str(reasoner_input)
        if self.reasoner_input not in ("text", "first_frame_text"):
            raise ValueError(f"reasoner_input must be 'text' or 'first_frame_text', got {reasoner_input!r}")
        self.vision_encoder = vision_encoder
        if self.reasoner_input == "first_frame_text":
            if self.vision_encoder is None:
                raise ValueError("reasoner_input='first_frame_text' requires a `vision_encoder`.")
            self._vis_cfg = self.vision_encoder.cfg
            self.vision_merge_size = int(self.vision_encoder.cfg.spatial_merge_size)
        self.action_path = str(action_path)
        if self.action_path not in ("cogen", "split"):
            raise ValueError(f"action_path must be 'cogen' or 'split', got {action_path!r}")
        self.video_mask_mode = video_mask_mode
        self.enable_fps_modulation = bool(enable_fps_modulation)
        self.fps = fps
        self.base_fps = float(base_fps)
        self.action_lr_mult = float(action_lr_mult)
        self.trainable_scope = str(trainable_scope)
        if self.trainable_scope != "generator":
            raise ValueError(
                f"trainable_scope must be 'generator', got {trainable_scope!r}"
            )
        self.loss_lambda_video = float(loss_lambda_video)
        self.loss_lambda_action = float(loss_lambda_action)
        self.gradient_checkpointing = bool(gradient_checkpointing)
        self.video_fdm_prob = float(video_fdm_prob)
        if not 0 <= self.video_fdm_prob <= 1:
            raise ValueError("`video_fdm_prob` must be in [0, 1].")
        if self.video_fdm_prob and not self.supports_fdm:
            raise ValueError("Joint variants do not support FDM.")
        self.compile_training_denoise = bool(compile_training_denoise)

        self.train_video_scheduler = WanContinuousFlowMatchScheduler(num_train_timesteps, video_train_shift)
        self.infer_video_scheduler = WanContinuousFlowMatchScheduler(num_train_timesteps, video_infer_shift)
        self.train_action_scheduler = WanContinuousFlowMatchScheduler(num_train_timesteps, action_train_shift)
        self.infer_action_scheduler = WanContinuousFlowMatchScheduler(num_train_timesteps, action_infer_shift)

        self.device = torch.device(device)
        self.torch_dtype = torch_dtype

        if self.video_fdm_prob:
            if self.video_mask_mode == "bidirectional":
                raise ValueError("FDM requires an isolated input frame; use first_frame_causal or per_frame_causal.")
            self.backbone.fdm_action_encoder = torch.nn.Linear(
                self.action_dim, self.backbone.cfg.hidden_size, device=self.device, dtype=torch_dtype,
            )
        self.to(self.device)
        self.setup_train_freeze()

    # ------------------------------------------------------------------ build
    @classmethod
    def from_pretrained(
        cls,
        *,
        proprio_dim: int,
        action_dim: int,
        action_horizon: int,
        tokenizer_max_len: int = 128,
        reasoner_input: str = "text",
        video_mask_mode: str = "first_frame_causal",
        enable_fps_modulation: bool = False,
        fps: float | None = None,
        action_lr_mult: float = 5.0,
        trainable_scope: str = "generator",
        action_path: str = "cogen",
        action_hidden: int = 1024,
        action_ffn: int = 4096,
        video_scheduler,
        action_scheduler,
        loss,
        gradient_checkpointing: bool = False,
        video_fdm_prob: float = 0.0,
        compile_training_denoise: bool = False,
        device: str = "cuda",
        torch_dtype: torch.dtype = torch.bfloat16,
    ) -> "Cosmos3FastWAM":
        from transformers import AutoTokenizer

        video_train_shift = float(video_scheduler["train_shift"])
        video_infer_shift = float(video_scheduler["infer_shift"])
        action_train_shift = float(action_scheduler["train_shift"])
        action_infer_shift = float(action_scheduler["infer_shift"])
        num_train_timesteps = int(video_scheduler["num_train_timesteps"])
        if num_train_timesteps != int(action_scheduler["num_train_timesteps"]):
            raise ValueError(
                "Cosmos3 video/action schedulers must use the same "
                "`num_train_timesteps`."
            )
        loss_lambda_video = float(loss["lambda_video"])
        loss_lambda_action = float(loss["lambda_action"])

        checkpoint_root = COSMOS3_NANO_ROOT
        backbone = load_cosmos3_transformer(
            checkpoint_root, action_dim=action_dim, proprio_dim=proprio_dim,
            action_path=action_path,
            action_hidden=action_hidden, action_ffn=action_ffn,
            dtype=torch_dtype, device=device,
        )
        vae = load_wan22_vae(torch_dtype=torch_dtype, device=device)
        tok_dir = os.path.join(checkpoint_root, "text_tokenizer")
        tokenizer = AutoTokenizer.from_pretrained(tok_dir)
        vision_encoder = None
        if reasoner_input == "first_frame_text":
            vision_encoder = load_cosmos3_vision_encoder(
                checkpoint_root, dtype=torch_dtype, device=device,
            )
        return cls(
            backbone=backbone,
            vae=vae,
            tokenizer=tokenizer,
            proprio_dim=proprio_dim,
            action_dim=action_dim,
            action_horizon=action_horizon,
            tokenizer_max_len=tokenizer_max_len,
            reasoner_input=reasoner_input,
            vision_encoder=vision_encoder,
            action_path=action_path,
            video_mask_mode=video_mask_mode,
            enable_fps_modulation=enable_fps_modulation,
            fps=fps,
            action_lr_mult=action_lr_mult,
            trainable_scope=trainable_scope,
            video_train_shift=video_train_shift,
            video_infer_shift=video_infer_shift,
            action_train_shift=action_train_shift,
            action_infer_shift=action_infer_shift,
            num_train_timesteps=num_train_timesteps,
            loss_lambda_video=loss_lambda_video,
            loss_lambda_action=loss_lambda_action,
            gradient_checkpointing=gradient_checkpointing,
            video_fdm_prob=video_fdm_prob,
            compile_training_denoise=compile_training_denoise,
            device=device, torch_dtype=torch_dtype,
        )

    # ------------------------------------------------------------- freeze/opt
    def setup_train_freeze(self):
        """Freeze UND/VAE and train the complete ``generator`` scope."""
        self.requires_grad_(False)
        self.vae.requires_grad_(False)
        parameter_groups = self.backbone.generator_parameter_groups()
        trainable_parameters = (
            parameter_groups["generator"] + parameter_groups["action"]
        )
        for parameter in trainable_parameters:
            parameter.requires_grad_(True)
        self.set_train_mode_for_training()

    def set_train_mode_for_training(self):
        """Enter training mode while keeping frozen encoders in eval mode."""
        self.train()
        self.vae.eval()
        if self.vision_encoder is not None:
            self.vision_encoder.eval()

    # ------------------------------------------------------------------- vae

    # --------------------------------------------------------------- text
    def _tokenize_prompt(self, prompts) -> tuple[torch.Tensor, torch.Tensor]:
        if isinstance(prompts, str):
            prompts = [prompts]
        enc = self.tokenizer(
            list(prompts), padding="max_length", truncation=True,
            max_length=self.tokenizer_max_len, return_tensors="pt",
        )
        input_ids = enc["input_ids"].to(self.device)
        text_pad_mask = ~enc["attention_mask"].to(self.device).bool()  # True where PAD
        return input_ids, text_pad_mask

    # --------------------------------------------------- masks + position ids

    # TODO: Expand the training/inference compile boundary to include the tensor
    # part of UND image encoding and attention-mask construction, for both base
    # and IDM paths. Split Python image preprocessing/tokenization from
    # _encode_und_image first so only graph-friendly tensor work is compiled.
    # ---------------------------------------------------------- UND image input
    def _encode_und_image(self, images):
        """Encode ``[B, 3, H, W]`` images into UND vision tokens and their source grid."""
        imgs01 = ((images.detach().to(self.device, torch.float32).clamp(-1, 1) + 1.0) * 0.5)
        pvs = []
        grid = None
        for b in range(imgs01.shape[0]):
            pv, g = preprocess_image(imgs01[b], self._vis_cfg)
            pvs.append(pv)
            grid = g
        pixel_values = torch.cat(pvs, dim=0).to(self.device, self.torch_dtype)
        grid_thw = torch.cat([grid] * imgs01.shape[0], dim=0).to(self.device)
        merged, _deep = self.vision_encoder(pixel_values, grid_thw)  # [B*n_img, D]
        gh, gw = int(grid[0, 1]), int(grid[0, 2])
        m = self.vision_merge_size
        num_image_tokens = (gh // m) * (gw // m)
        vision_embeds = merged.reshape(
            imgs01.shape[0],
            num_image_tokens,
            merged.shape[-1],
        )
        return vision_embeds, (1, gh, gw)

    def _build_und_inputs(self, prompts, n_img: int = 0):
        """Build text-only or first-frame ``[vision_start | image | vision_end | text]`` inputs.

        ``image_token_mask`` is ``None`` for text-only inputs.
        """
        text_ids, text_pad_mask = self._tokenize_prompt(prompts)
        if n_img == 0:
            return text_ids, text_pad_mask, None

        batch_size = text_ids.shape[0]
        vs = torch.full(
            (batch_size, 1),
            _VISION_START_TOKEN_ID,
            dtype=torch.long,
            device=self.device,
        )
        ve = torch.full(
            (batch_size, 1),
            _VISION_END_TOKEN_ID,
            dtype=torch.long,
            device=self.device,
        )
        img = torch.full(
            (batch_size, n_img),
            _IMAGE_TOKEN_ID,
            dtype=torch.long,
            device=self.device,
        )
        input_ids = torch.cat([vs, img, ve, text_ids], dim=1)  # [B, 2+n_img+L]
        img_block_pad = torch.zeros(
            (batch_size, n_img + 2),
            dtype=torch.bool,
            device=self.device,
        )
        und_pad_mask = torch.cat([img_block_pad, text_pad_mask], dim=1)
        image_token_mask = torch.zeros_like(input_ids, dtype=torch.bool)
        image_token_mask[:, 1 : 1 + n_img] = True
        return input_ids, und_pad_mask, image_token_mask


    # ---------------------------------------------------------------- loss

    # ---------------------------------------------------------- inference

    # ---------------------------------------------------------------- eval

    # ------------------------------------------------------------ checkpoint

    def load_checkpoint(self, path):
        """Load trainable Cosmos3 weights.

        If an action/proprio I/O projection has a different checkpoint shape,
        keep that entire projection freshly initialized. All other shape
        mismatches remain load errors.

        Args:
            path: Checkpoint path produced by :meth:`save_checkpoint`.

        Returns:
            Loaded checkpoint payload.
        """
        payload = torch.load(path, map_location=self.device)
        if "backbone_trainable" in payload:
            backbone_state = payload["backbone_trainable"]
            current_state = self.backbone.state_dict()
            action_io_prefixes = ("proprio_encoder.", "action_in.", "action_out.", "fdm_action_encoder.")
            action_io_shape_mismatches = {
                key: (tuple(value.shape), tuple(current_state[key].shape))
                for key, value in backbone_state.items()
                if key in current_state and key.startswith(action_io_prefixes)
                and value.shape != current_state[key].shape
            }
            fresh_init_prefixes = tuple(prefix for prefix in action_io_prefixes if any(
                key.startswith(prefix) for key in action_io_shape_mismatches
            ))
            if fresh_init_prefixes:
                backbone_state = {
                    key: value
                    for key, value in backbone_state.items()
                    if not key.startswith(fresh_init_prefixes)
                }
                logger.warning(
                    "Checkpoint action/proprio I/O shapes do not match the current Cosmos3 "
                    "model; keeping %s freshly initialized. Shape mismatches: %s",
                    [prefix.removesuffix(".") for prefix in fresh_init_prefixes],
                    action_io_shape_mismatches,
                )

            missing, unexpected = self.backbone.load_state_dict(backbone_state, strict=False)
            missing_fdm = [key for key in missing if key.startswith("fdm_action_encoder.")
                           and not key.startswith(fresh_init_prefixes)]
            if missing_fdm:
                logger.warning("Checkpoint has no FDM action projection; keeping fresh parameters: %s", missing_fdm)
            if unexpected:
                logger.warning("load_checkpoint unexpected keys: %s", sorted(unexpected)[:8])
        else:
            raise ValueError(f"Checkpoint missing `backbone_trainable`: {path}")
        return payload

    # keep .to() device bookkeeping
    def to(self, *args, **kwargs):
        out = super().to(*args, **kwargs)
        for a in args:
            if isinstance(a, (str, torch.device)):
                self.device = torch.device(a)
        if "device" in kwargs and kwargs["device"] is not None:
            self.device = torch.device(kwargs["device"])
        return out
