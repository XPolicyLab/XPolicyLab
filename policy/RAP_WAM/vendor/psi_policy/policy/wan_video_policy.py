from __future__ import annotations

import contextlib
import inspect
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

from psi_policy.model.common.lora import convert_linear_layer_to_lora
from psi_policy.model.common.normalizer import LinearNormalizer
from psi_policy.model.wan.runtime import WanModel
from psi_policy.policy.base_policy import BaseImagePolicy


def _parse_torch_dtype(dtype_name: str) -> torch.dtype:
    mapping = {
        "float32": torch.float32,
        "fp32": torch.float32,
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
    }
    key = str(dtype_name).lower()
    if key not in mapping:
        raise ValueError(f"Unsupported dtype: {dtype_name}")
    return mapping[key]


class WanVideoPolicy(BaseImagePolicy):
    """Wan TI2V-5B 未来视频生成策略。"""

    allows_pretrained_init = True

    def __init__(
        self,
        *,
        obs_encoder,
        wan_checkpoint_dir: str,
        transformer_subfolder: str | None = None,
        num_train_timesteps: int = 1000,
        train_shift: float = 5.0,
        sampling_shift: float = 5.0,
        sigma_min: float = 0.003 / 1.002,
        sigma_max: float = 1.0,
        num_inference_steps: int = 40,
        loss_type: str = "mse",
        charbonnier_eps: float = 1.0e-3,
        prompt_embedding_path: str | None = None,
        use_text_encoder: bool = False,
        fixed_prompt: str = "",
        wan_dtype: str = "bfloat16",
        use_action_condition: bool = False,
        use_future_only_loss: bool = True,
        **kwargs,
    ) -> None:
        super().__init__()
        self.obs_encoder = obs_encoder
        self.wan_checkpoint_dir = str(Path(wan_checkpoint_dir).expanduser().resolve())
        self.num_train_timesteps = int(num_train_timesteps)
        self.train_shift = float(train_shift)
        self.sampling_shift = float(sampling_shift)
        self.sigma_min = float(sigma_min)
        self.sigma_max = float(sigma_max)
        self.num_inference_steps = int(num_inference_steps)
        self.loss_type = str(loss_type)
        self.charbonnier_eps = float(charbonnier_eps)
        self.prompt_embedding_path = (
            str(Path(prompt_embedding_path).expanduser().resolve())
            if prompt_embedding_path is not None
            else None
        )
        self.use_text_encoder = bool(use_text_encoder)
        self.fixed_prompt = str(fixed_prompt)
        self.use_action_condition = bool(use_action_condition)
        self.use_future_only_loss = bool(use_future_only_loss)
        self.kwargs = dict(kwargs)

        if self.use_text_encoder:
            raise NotImplementedError("WanVideoPolicy currently supports only fixed prompt embeddings.")
        if self.use_action_condition:
            raise NotImplementedError("WanVideoPolicy currently supports only use_action_condition=False.")

        checkpoint_root = Path(self.wan_checkpoint_dir)
        wan_dtype_value = _parse_torch_dtype(wan_dtype)
        if transformer_subfolder is None:
            if (checkpoint_root / "transformer").exists():
                transformer_subfolder = "transformer"
            elif (checkpoint_root / "model").exists():
                transformer_subfolder = "model"

        if transformer_subfolder is None:
            self.model = WanModel.from_pretrained(
                self.wan_checkpoint_dir,
                low_cpu_mem_usage=False,
                torch_dtype=wan_dtype_value,
            )
        else:
            self.model = WanModel.from_pretrained(
                self.wan_checkpoint_dir,
                subfolder=transformer_subfolder,
                low_cpu_mem_usage=False,
                torch_dtype=wan_dtype_value,
            )

        self.text_dim = int(self.model.text_dim)
        self.text_len = int(self.model.text_len)
        self.patch_size = tuple(self.model.patch_size)
        self.target_horizon = int(self.obs_encoder.target_horizon)
        self.total_video_frames = int(self.obs_encoder.total_video_frames)
        self.normalizer = LinearNormalizer()
        self._cached_prompt_context_cpu: torch.Tensor | None = None

        train_sigmas = self._build_sigmas(self.num_train_timesteps, shift=self.train_shift)
        self.register_buffer("train_sigmas", train_sigmas, persistent=False)
        self.register_buffer("train_timesteps", train_sigmas * self.num_train_timesteps, persistent=False)
        self._did_apply_dit_mode = False

    def _build_sigmas(self, num_steps: int, shift: float) -> torch.Tensor:
        base = torch.linspace(self.sigma_max, self.sigma_min, int(num_steps), dtype=torch.float32)
        return shift * base / (1.0 + (shift - 1.0) * base)

    def _load_prompt_context_from_file(self) -> torch.Tensor:
        if self.prompt_embedding_path is None:
            raise RuntimeError("prompt_embedding_path is not set.")
        prompt_path = Path(self.prompt_embedding_path)
        if not prompt_path.exists():
            raise FileNotFoundError(f"Prompt embedding file not found: {prompt_path}")

        payload = torch.load(prompt_path, map_location="cpu")
        prompt_context = payload.get("prompt_embedding", payload.get("embedding")) if isinstance(payload, dict) else payload
        if not isinstance(prompt_context, torch.Tensor):
            raise TypeError(f"Prompt embedding must be a tensor, got {type(prompt_context)}")
        if prompt_context.ndim != 2:
            raise ValueError(f"Prompt embedding must have shape [L, C], got {tuple(prompt_context.shape)}")
        if int(prompt_context.shape[1]) != self.text_dim:
            raise ValueError(
                f"Prompt embedding hidden dim mismatch: expected {self.text_dim}, got {prompt_context.shape[1]}"
            )
        return prompt_context[: self.text_len].detach().cpu().float()

    def _get_prompt_context(self) -> torch.Tensor | None:
        if self._cached_prompt_context_cpu is not None:
            return self._cached_prompt_context_cpu
        if self.prompt_embedding_path is None:
            return None
        self._cached_prompt_context_cpu = self._load_prompt_context_from_file()
        return self._cached_prompt_context_cpu

    def _build_context_list(self, batch_size: int) -> List[torch.Tensor]:
        prompt_context = self._get_prompt_context()
        model_dtype = self.model.text_embedding[0].weight.dtype
        contexts: List[torch.Tensor] = []
        for _ in range(batch_size):
            if prompt_context is None:
                merged = torch.zeros((0, self.text_dim), device=self.device, dtype=model_dtype)
            else:
                merged = prompt_context.to(device=self.device, dtype=model_dtype)
            if merged.shape[0] > self.text_len:
                merged = merged[: self.text_len]
            contexts.append(merged)
        return contexts

    def _latent_seq_len(self, latent_video: torch.Tensor) -> int:
        patch_t, patch_h, patch_w = self.patch_size
        _, _, t, h, w = latent_video.shape
        return (t // patch_t) * (h // patch_h) * (w // patch_w)

    def _format_timesteps_for_wan(
        self,
        timesteps: torch.Tensor,
        *,
        seq_len: int,
        device: torch.device,
    ) -> torch.Tensor:
        timesteps = timesteps.to(device=device, dtype=torch.float32)
        if timesteps.ndim == 0:
            timesteps = timesteps.unsqueeze(0)
        if timesteps.ndim == 1:
            timesteps = timesteps.unsqueeze(1)
        elif timesteps.ndim != 2:
            raise ValueError(f"Expected timesteps with shape [B] or [B,L], got {tuple(timesteps.shape)}")

        if timesteps.shape[1] not in (1, seq_len):
            raise ValueError(
                f"Expected timestep width 1 or seq_len={seq_len}, got {tuple(timesteps.shape)}"
            )
        return timesteps

    def _format_masked_timesteps_for_wan(
        self,
        timesteps: torch.Tensor,
        *,
        latent_condition_mask: torch.Tensor,
        seq_len: int,
        device: torch.device,
    ) -> torch.Tensor:
        timesteps = self._format_timesteps_for_wan(timesteps, seq_len=seq_len, device=device)
        if timesteps.shape[1] == 1:
            timesteps = timesteps.expand(-1, seq_len)

        patch_t, patch_h, patch_w = self.patch_size
        token_mask = latent_condition_mask[:, 0, ::patch_t, ::patch_h, ::patch_w].contiguous()
        token_mask = token_mask.view(token_mask.shape[0], -1)
        if token_mask.shape[1] != seq_len:
            raise ValueError(
                f"Masked timestep token count mismatch: expected seq_len={seq_len}, got {token_mask.shape[1]}"
            )
        return timesteps * token_mask.to(device=device, dtype=torch.float32)

    def _run_model(
        self,
        noisy_latents: torch.Tensor,
        timesteps: torch.Tensor,
        context_list: List[torch.Tensor],
        condition_latents: torch.Tensor,
        latent_condition_mask: torch.Tensor,
    ) -> torch.Tensor:
        model_device = next(self.model.parameters()).device
        model_dtype = next(self.model.parameters()).dtype
        noisy_latents = noisy_latents.to(device=model_device, dtype=model_dtype)
        condition_latents = condition_latents.to(device=model_device, dtype=model_dtype)
        latent_condition_mask = latent_condition_mask.to(device=model_device, dtype=model_dtype)
        context_list = [context.to(device=model_device, dtype=model_dtype) for context in context_list]

        model_inputs = latent_condition_mask * noisy_latents + (1.0 - latent_condition_mask) * condition_latents
        seq_len = self._latent_seq_len(model_inputs)
        timesteps_for_model = self._format_masked_timesteps_for_wan(
            timesteps,
            latent_condition_mask=latent_condition_mask,
            seq_len=seq_len,
            device=model_device,
        )

        autocast_context = contextlib.nullcontext()
        if model_device.type == "cuda" and model_dtype in (torch.float16, torch.bfloat16):
            autocast_context = torch.autocast(device_type="cuda", dtype=model_dtype)

        with autocast_context:
            predictions = self.model(
                x=[sample for sample in model_inputs],
                t=timesteps_for_model,
                context=context_list,
                seq_len=seq_len,
            )
        return torch.stack([prediction.float() for prediction in predictions], dim=0)

    def _compute_loss_tensor(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.loss_type == "l1":
            return F.l1_loss(pred, target)
        if self.loss_type == "smooth_l1":
            return F.smooth_l1_loss(pred, target)
        if self.loss_type == "charbonnier":
            diff = pred - target
            return torch.sqrt(diff * diff + self.charbonnier_eps * self.charbonnier_eps).mean()
        return F.mse_loss(pred, target)

    def set_normalizer(self, normalizer: LinearNormalizer) -> None:
        self.normalizer.load_state_dict(normalizer.state_dict())
        self.normalizer.to(device=self.device)

    def _apply_dit_train_mode(self, dit_mode: str, dit_lora_cfg: dict) -> None:
        if self._did_apply_dit_mode:
            return

        mode = str(dit_mode).strip().lower()
        if mode == "freeze":
            for param in self.model.parameters():
                param.requires_grad = False
        elif mode == "lora":
            for param in self.model.parameters():
                param.requires_grad = False
            convert_linear_layer_to_lora(
                self.model,
                include_module_names=list(dit_lora_cfg.get("include_module_names", [])),
                exclude_module_names=list(dit_lora_cfg.get("exclude_module_names", [])),
                lora_dim=int(dit_lora_cfg.get("lora_dim", 4)),
                lora_scaling=float(dit_lora_cfg.get("lora_scaling", 4.0)),
            )
            for name, param in self.model.named_parameters():
                if "lora_right_weight" in name or "lora_left_weight" in name:
                    param.requires_grad = True
        elif mode != "full_train":
            raise ValueError(f"Unsupported dit_mode: {dit_mode}")
        self._did_apply_dit_mode = True

    def get_optimizer(
        self,
        lr: float = 5.0e-5,
        weight_decay: float = 1.0e-4,
        betas: Tuple[float, float] = (0.9, 0.95),
        start_ckpt_path: str | None = None,
        start_normalizer_path: str | None = None,
        is_ckpt_dit_lora: bool = False,
        is_ckpt_obs_encoder_lora: bool = False,
        obs_encoder_mode: str = "freeze",
        obs_encoder_lora_cfg: dict | None = None,
        dit_mode: str = "full_train",
        dit_lora_cfg: dict | None = None,
    ):
        del start_ckpt_path
        del start_normalizer_path
        del is_ckpt_dit_lora
        del is_ckpt_obs_encoder_lora
        del obs_encoder_mode
        del obs_encoder_lora_cfg

        self._apply_dit_train_mode(dit_mode, dit_lora_cfg or {})
        param_dict = {name: p for name, p in self.named_parameters() if p.requires_grad}
        if not param_dict:
            raise RuntimeError(
                "No trainable parameters found after applying dit_mode. "
                "Please check optimizer.dit_mode and optimizer.dit_lora_cfg."
            )
        decay_params = [p for _, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for _, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {"params": decay_params, "weight_decay": float(weight_decay)},
            {"params": nodecay_params, "weight_decay": 0.0},
        ]

        fused_available = "fused" in inspect.signature(torch.optim.AdamW).parameters
        if fused_available:
            return torch.optim.AdamW(optim_groups, lr=float(lr), betas=betas, fused=True)
        return torch.optim.AdamW(optim_groups, lr=float(lr), betas=betas)

    def compute_loss(self, batch: dict, training: bool = True) -> torch.Tensor:
        del training
        if "obs" not in batch:
            raise KeyError("batch missing obs")

        # Wan obs encoder expects image tensors in [0, 1]. Do not apply the
        # dataset image normalizer here, otherwise images would be shifted twice.
        batch_inputs = self.obs_encoder.prepare_batch(batch["obs"], training=True)
        clean_latents = batch_inputs["clean_latents"]
        condition_latents = batch_inputs["condition_latents"]
        latent_condition_mask = batch_inputs["latent_condition_mask"]
        num_condition_latent_steps = int(batch_inputs["num_condition_latent_steps"])

        batch_size = clean_latents.shape[0]
        timestep_idx = torch.randint(
            0,
            self.train_timesteps.shape[0],
            (batch_size,),
            device=self.device,
        )
        timesteps = self.train_timesteps[timestep_idx]
        sigmas = self.train_sigmas[timestep_idx].view(batch_size, 1, 1, 1, 1).to(clean_latents.dtype)

        noise = torch.randn_like(clean_latents)
        noisy_latents = clean_latents * (1.0 - sigmas) + noise * sigmas
        target_velocity = noise - clean_latents

        pred_velocity = self._run_model(
            noisy_latents=noisy_latents,
            timesteps=timesteps,
            context_list=self._build_context_list(batch_size=batch_size),
            condition_latents=condition_latents,
            latent_condition_mask=latent_condition_mask,
        )

        if self.use_future_only_loss:
            pred_velocity = pred_velocity[:, :, num_condition_latent_steps:, :, :]
            target_velocity = target_velocity[:, :, num_condition_latent_steps:, :, :]
        return self._compute_loss_tensor(pred_velocity, target_velocity)

    def forward(self, batch: dict, training: bool = True) -> torch.Tensor:
        return self.compute_loss(batch=batch, training=training)

    @torch.no_grad()
    def predict_visual(
        self,
        obs_dict: Dict[str, torch.Tensor],
        output_path: Optional[str] = None,
        show_progress: bool = False,
        progress_desc: Optional[str] = None,
    ) -> Dict[str, torch.Tensor]:
        del output_path
        condition_inputs = self.obs_encoder.prepare_condition_only(obs_dict)
        condition_latents = condition_inputs["condition_latents"]
        latent_condition_mask = condition_inputs["latent_condition_mask"]
        batch_size, latent_channels, latent_t, latent_h, latent_w = condition_inputs["latent_shape"]

        latent = torch.randn(
            batch_size,
            latent_channels,
            latent_t,
            latent_h,
            latent_w,
            device=self.device,
            dtype=self.dtype,
        )
        condition_latents = condition_latents.to(device=self.device, dtype=latent.dtype)
        latent_condition_mask = latent_condition_mask.to(device=self.device, dtype=latent.dtype)
        latent = latent_condition_mask * latent + (1.0 - latent_condition_mask) * condition_latents

        sampling_sigmas = self._build_sigmas(self.num_inference_steps, shift=self.sampling_shift).to(
            device=self.device,
            dtype=torch.float32,
        )
        sampling_timesteps = sampling_sigmas * self.num_train_timesteps
        context_list = self._build_context_list(batch_size=batch_size)

        step_iterator = range(self.num_inference_steps)
        if show_progress:
            from tqdm.auto import tqdm

            step_iterator = tqdm(
                step_iterator,
                total=self.num_inference_steps,
                desc=progress_desc or "Wan sampling",
                leave=False,
                dynamic_ncols=True,
            )

        for step_idx in step_iterator:
            timestep = sampling_timesteps[step_idx].expand(batch_size)
            pred_velocity = self._run_model(
                noisy_latents=latent,
                timesteps=timestep,
                context_list=context_list,
                condition_latents=condition_latents,
                latent_condition_mask=latent_condition_mask,
            ).to(latent.dtype)
            sigma = sampling_sigmas[step_idx]
            sigma_next = (
                sampling_sigmas[step_idx + 1]
                if step_idx + 1 < self.num_inference_steps
                else torch.tensor(0.0, device=self.device, dtype=sampling_sigmas.dtype)
            )
            latent = latent + pred_velocity * (sigma_next - sigma)
            latent = latent_condition_mask * latent + (1.0 - latent_condition_mask) * condition_latents

        return {"visual_pred": latent.float()}

    @torch.no_grad()
    def decode_visual_tokens(self, visual_tokens: torch.Tensor) -> torch.Tensor:
        if visual_tokens.ndim == 5:
            decoded = self.obs_encoder.decode_full_latents(visual_tokens)
            return decoded[:, 1 : 1 + self.target_horizon]

        if visual_tokens.ndim == 6:
            batch_size, num_samples = visual_tokens.shape[:2]
            decoded_samples = []
            for sample_idx in range(num_samples):
                decoded = self.obs_encoder.decode_full_latents(visual_tokens[:, sample_idx])
                decoded_samples.append(decoded[:, 1 : 1 + self.target_horizon].unsqueeze(1))
            return torch.cat(decoded_samples, dim=1)

        raise ValueError(f"Unexpected visual token shape: {tuple(visual_tokens.shape)}")


__all__ = ["WanVideoPolicy"]
