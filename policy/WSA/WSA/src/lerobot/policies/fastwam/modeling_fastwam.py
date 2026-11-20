from __future__ import annotations

import logging
from collections import deque
from pathlib import Path
from typing import Any

import torch
from huggingface_hub import hf_hub_download
from huggingface_hub.errors import HfHubHTTPError
from torch import Tensor

from lerobot.datasets.utils import serialize_dict, write_json
from lerobot.policies.pretrained import PreTrainedPolicy

from .configuration_fastwam import FastWAMConfig
from .core.data.lerobot.utils.normalizer import SingleFieldLinearNormalizer, load_dataset_stats_from_json
from .core.models.wan22.fastwam import FastWAM
from .core.models.wan22.fastwam_idm import FastWAMIDM
from .core.models.wan22.fastwam_joint import FastWAMJoint


class FastWAMPolicy(PreTrainedPolicy):
    config_class = FastWAMConfig
    name = "fastwam"
    _save_prefixes = ("model.mot.", "model.proprio_encoder.")
    _stats_filename = "stats.json"

    def __init__(self, config: FastWAMConfig):
        super().__init__(config)
        config.validate_features()
        self.config = config
        self._action_denorm_specs: list[dict[str, Any]] = []
        self._action_stats_source: str | None = None
        self._action_stats_payload: dict[str, Any] | None = None
        self.model = self._build_model(config)
        if config.native_checkpoint_path:
            logging.info("Loading FastWAM native checkpoint from %s", config.native_checkpoint_path)
            self.model.load_checkpoint(config.native_checkpoint_path)
        self._maybe_load_action_postprocess_from_config()
        self.reset()

    @staticmethod
    def _variant_to_class(variant: str):
        mapping = {
            "fastwam": FastWAM,
            "fastwam_joint": FastWAMJoint,
            "fastwam_idm": FastWAMIDM,
        }
        return mapping[variant]

    def _build_model(self, config: FastWAMConfig):
        model_cls = self._variant_to_class(config.variant)
        return model_cls.from_wan22_pretrained(
            device=config.device,
            torch_dtype=config.torch_dtype,
            model_id=config.model_id,
            tokenizer_model_id=config.tokenizer_model_id,
            tokenizer_max_len=config.tokenizer_max_len,
            load_text_encoder=config.load_text_encoder,
            proprio_dim=config.proprio_dim,
            redirect_common_files=config.redirect_common_files,
            video_dit_config=config.video_dit_config,
            action_dit_config=config.action_dit_config,
            action_dit_pretrained_path=config.action_dit_pretrained_path,
            skip_dit_load_from_pretrain=config.skip_dit_load_from_pretrain,
            mot_checkpoint_mixed_attn=config.mot_checkpoint_mixed_attn,
            video_train_shift=float(config.video_scheduler.get("train_shift", 5.0)),
            video_infer_shift=float(config.video_scheduler.get("infer_shift", 5.0)),
            video_num_train_timesteps=int(config.video_scheduler.get("num_train_timesteps", 1000)),
            action_train_shift=float(config.action_scheduler["train_shift"]),
            action_infer_shift=float(config.action_scheduler["infer_shift"]),
            action_num_train_timesteps=int(config.action_scheduler["num_train_timesteps"]),
            loss_lambda_video=float(config.lambda_video),
            loss_lambda_action=float(config.lambda_action),
        )

    def get_state_dict_to_save(self) -> dict[str, Tensor] | None:
        state_dict = self.state_dict()
        return {
            key: value
            for key, value in state_dict.items()
            if key.startswith(self._save_prefixes)
        }

    def _save_pretrained(self, save_directory: Path) -> None:
        self._save_pretrained_artifacts(save_directory)
        if self._action_stats_payload is not None:
            stats_payload = {"fastwam": self._action_stats_payload}
            write_json(serialize_dict(stats_payload), save_directory / self._stats_filename)

    @classmethod
    def from_pretrained(
        cls,
        pretrained_name_or_path: str | Path,
        *,
        config: FastWAMConfig | None = None,
        force_download: bool = False,
        resume_download: bool | None = None,
        proxies: dict | None = None,
        token: str | bool | None = None,
        cache_dir: str | Path | None = None,
        local_files_only: bool = False,
        revision: str | None = None,
        strict: bool = False,
        **kwargs,
    ):
        policy = super().from_pretrained(
            pretrained_name_or_path=pretrained_name_or_path,
            config=config,
            force_download=force_download,
            resume_download=resume_download,
            proxies=proxies,
            token=token,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
            revision=revision,
            strict=strict,
            **kwargs,
        )
        policy._maybe_load_action_postprocess_from_pretrained(
            pretrained_name_or_path=pretrained_name_or_path,
            force_download=force_download,
            resume_download=resume_download,
            proxies=proxies,
            token=token,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
            revision=revision,
        )
        return policy

    @staticmethod
    def _extract_fastwam_stats(stats_payload: dict[str, Any]) -> dict[str, Any]:
        if "fastwam" in stats_payload:
            stats_payload = stats_payload["fastwam"]
        if "action" not in stats_payload:
            raise ValueError("FastWAM stats payload must contain an `action` section.")
        return stats_payload

    @staticmethod
    def _infer_stat_dim(stats_for_key: dict[str, Any]) -> int:
        for stat_name in (
            "global_min",
            "stepwise_min",
            "global_mean",
            "stepwise_mean",
            "min",
            "mean",
        ):
            stat_value = stats_for_key.get(stat_name)
            if stat_value is None:
                continue
            if isinstance(stat_value, torch.Tensor):
                if stat_value.ndim == 0:
                    return 1
                return int(stat_value.shape[-1])
            if isinstance(stat_value, list):
                return len(stat_value)
        raise ValueError(f"Unable to infer action dimension from stats keys: {list(stats_for_key.keys())}")

    def set_action_postprocess_from_stats(self, stats_payload: dict[str, Any]) -> None:
        stats_payload = self._extract_fastwam_stats(stats_payload)
        action_stats = stats_payload["action"]
        if not isinstance(action_stats, dict) or len(action_stats) == 0:
            raise ValueError("FastWAM action stats must be a non-empty dict.")

        use_stepwise = bool(self.config.action_norm_use_stepwise)
        default_mode = str(self.config.action_norm_default_mode)
        exception_mode = self.config.action_norm_exception_mode or {}
        action_exception_mode = exception_mode.get("action", {})

        denorm_specs: list[dict[str, Any]] = []
        for key, key_stats in action_stats.items():
            prefix = "stepwise_" if use_stepwise else "global_"
            selected_stats = {
                stat_name.removeprefix(prefix): value
                for stat_name, value in key_stats.items()
                if stat_name.startswith(prefix)
            }
            if len(selected_stats) == 0:
                raise ValueError(
                    f"Missing {prefix}* stats for FastWAM action key '{key}'. Available keys: {list(key_stats.keys())}"
                )

            normalizer = SingleFieldLinearNormalizer(
                stats=selected_stats,
                mode=action_exception_mode.get(key, default_mode),
            )
            denorm_specs.append(
                {
                    "key": key,
                    "dim": self._infer_stat_dim(key_stats),
                    "scale": normalizer.scale.detach().cpu().to(torch.float32),
                    "offset": normalizer.offset.detach().cpu().to(torch.float32),
                }
            )

        total_dim = sum(int(spec["dim"]) for spec in denorm_specs)
        if total_dim != int(self.config.action_dim):
            raise ValueError(
                f"FastWAM action stats dim mismatch: stats sum to {total_dim}, config.action_dim={self.config.action_dim}"
            )
        self._action_denorm_specs = denorm_specs
        self._action_stats_payload = stats_payload

    def _load_action_postprocess_from_stats_path(self, stats_path: str | Path) -> None:
        stats_path = Path(stats_path)
        if not stats_path.is_file():
            raise FileNotFoundError(f"FastWAM action stats file does not exist: {stats_path}")
        stats_payload = load_dataset_stats_from_json(str(stats_path))
        self.set_action_postprocess_from_stats(stats_payload)
        self._action_stats_source = str(stats_path)
        logging.info("Loaded FastWAM action postprocess stats from %s", stats_path)

    @classmethod
    def _resolve_pretrained_stats_path(
        cls,
        pretrained_name_or_path: str | Path,
        *,
        force_download: bool = False,
        resume_download: bool | None = None,
        proxies: dict | None = None,
        token: str | bool | None = None,
        cache_dir: str | Path | None = None,
        local_files_only: bool = False,
        revision: str | None = None,
    ) -> Path | None:
        model_id = str(pretrained_name_or_path)
        local_dir = Path(model_id)
        if local_dir.is_dir():
            candidate = local_dir / cls._stats_filename
            if candidate.is_file():
                return candidate
            return None

        try:
            stats_file = hf_hub_download(
                repo_id=model_id,
                filename=cls._stats_filename,
                revision=revision,
                cache_dir=cache_dir,
                force_download=force_download,
                proxies=proxies,
                resume_download=resume_download,
                token=token,
                local_files_only=local_files_only,
            )
        except HfHubHTTPError:
            return None
        return Path(stats_file)

    def _maybe_load_action_postprocess_from_config(self) -> None:
        if self._action_denorm_specs:
            return
        stats_path = getattr(self.config, "action_stats_path", None)
        if stats_path is None:
            return
        self._load_action_postprocess_from_stats_path(stats_path)

    def _maybe_load_action_postprocess_from_pretrained(
        self,
        *,
        pretrained_name_or_path: str | Path,
        force_download: bool = False,
        resume_download: bool | None = None,
        proxies: dict | None = None,
        token: str | bool | None = None,
        cache_dir: str | Path | None = None,
        local_files_only: bool = False,
        revision: str | None = None,
    ) -> None:
        if self._action_denorm_specs:
            return
        stats_path = self._resolve_pretrained_stats_path(
            pretrained_name_or_path=pretrained_name_or_path,
            force_download=force_download,
            resume_download=resume_download,
            proxies=proxies,
            token=token,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
            revision=revision,
        )
        if stats_path is None:
            logging.info(
                "FastWAM pretrained source %s has no %s; action outputs will stay normalized unless "
                "`policy.action_stats_path` is provided.",
                pretrained_name_or_path,
                self._stats_filename,
            )
            return
        self._load_action_postprocess_from_stats_path(stats_path)

    def _set_dit_only_train_mode(self) -> None:
        self.model.eval()
        self.model.requires_grad_(False)
        self.model.dit.train()
        self.model.dit.requires_grad_(True)
        proprio_encoder = getattr(self.model, "proprio_encoder", None)
        if proprio_encoder is not None:
            proprio_encoder.train()
            proprio_encoder.requires_grad_(True)

    def get_optim_params(self):
        trainable_params = list(self.model.dit.parameters())
        proprio_encoder = getattr(self.model, "proprio_encoder", None)
        if proprio_encoder is not None:
            trainable_params.extend(list(proprio_encoder.parameters()))
        return (param for param in trainable_params if param.requires_grad)

    def reset(self):
        self._action_queue: deque[Tensor] = deque()

    def train(self, mode: bool = True):
        super().train(mode)
        if mode:
            self._set_dit_only_train_mode()
        return self

    def _prepare_training_sample(self, batch: dict[str, Any]) -> dict[str, Any]:
        sample = dict(batch)
        if "context" in sample and "context_mask" in sample:
            return sample
        prompts = sample.get("prompt")
        if prompts is None:
            raise ValueError(
                "FastWAM training requires either `context/context_mask` in the batch or `prompt` with "
                "`policy.load_text_encoder=true`."
            )
        if not self.config.load_text_encoder:
            raise ValueError(
                "The batch has no cached text embeddings, but `policy.load_text_encoder=false`. "
                "Either precompute text embeddings or enable the text encoder."
            )
        if isinstance(prompts, str):
            prompts = [prompts]
        context, context_mask = self.model.encode_prompt(list(prompts))
        sample["context"] = context
        sample["context_mask"] = context_mask
        return sample

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict]:
        sample = self._prepare_training_sample(batch)
        loss, loss_dict = self.model.training_loss(sample)
        output = {
            "loss": float(loss.detach().item()),
            "loss_video": float(loss_dict["loss_video"]),
            "loss_action": float(loss_dict["loss_action"]),
        }
        return loss, output

    def _resolve_context_for_inference(
        self, batch: dict[str, Any], index: int
    ) -> tuple[str | None, Tensor | None, Tensor | None]:
        if "context" in batch and "context_mask" in batch:
            context = batch["context"][index : index + 1]
            context_mask = batch["context_mask"][index : index + 1]
            return None, context, context_mask
        prompts = batch.get("prompt")
        if prompts is None:
            return None, None, None
        if isinstance(prompts, str):
            return prompts, None, None
        return prompts[index], None, None

    @staticmethod
    def _resolve_input_image(batch: dict[str, Any]) -> Tensor:
        if "input_image" in batch:
            image = batch["input_image"]
        elif "video" in batch:
            video = batch["video"]
            if video.ndim != 5:
                raise ValueError(f"`video` must be [B,3,T,H,W] for inference, got {tuple(video.shape)}")
            image = video[:, :, 0, :, :]
        else:
            raise ValueError("FastWAM inference expects `input_image` or `video` in the batch.")
        if image.ndim == 3:
            image = image.unsqueeze(0)
        if image.ndim != 4:
            raise ValueError(f"`input_image` must be [B,3,H,W], got {tuple(image.shape)}")
        return image

    @staticmethod
    def _resolve_proprio(batch: dict[str, Any], index: int) -> Tensor | None:
        proprio = batch.get("proprio")
        if proprio is None:
            return None
        if proprio.ndim == 3:
            return proprio[index, 0]
        if proprio.ndim == 2:
            return proprio[index]
        if proprio.ndim == 1:
            return proprio
        raise ValueError(f"Unsupported proprio shape for inference: {tuple(proprio.shape)}")

    def _denormalize_action(self, action: Tensor) -> Tensor:
        if len(self._action_denorm_specs) == 0:
            return action

        squeeze_batch = False
        if action.ndim == 2:
            action = action.unsqueeze(0)
            squeeze_batch = True
        elif action.ndim != 3:
            raise ValueError(f"Expected FastWAM action to be [B,T,D] or [T,D], got {tuple(action.shape)}")

        action_f32 = action.to(dtype=torch.float32)
        parts = []
        start = 0
        for spec in self._action_denorm_specs:
            dim = int(spec["dim"])
            end = start + dim
            cur_action = action_f32[..., start:end]
            if cur_action.shape[-1] != dim:
                raise ValueError(
                    f"FastWAM action dim mismatch while denormalizing: expected slice dim {dim}, got {cur_action.shape[-1]}"
                )
            scale = spec["scale"].to(device=cur_action.device, dtype=cur_action.dtype)
            offset = spec["offset"].to(device=cur_action.device, dtype=cur_action.dtype)
            parts.append((cur_action - offset) / scale)
            start = end

        if start != action_f32.shape[-1]:
            raise ValueError(
                f"FastWAM denormalizer consumed {start} dims but action has {action_f32.shape[-1]} dims."
            )

        action_denorm = torch.cat(parts, dim=-1)
        if squeeze_batch:
            action_denorm = action_denorm.squeeze(0)
        return action_denorm

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Tensor], **kwargs) -> Tensor:
        del kwargs
        self.eval()
        input_image = self._resolve_input_image(batch)
        batch_size = input_image.shape[0]
        actions = []
        for index in range(batch_size):
            prompt, context, context_mask = self._resolve_context_for_inference(batch, index)
            proprio = self._resolve_proprio(batch, index)
            out = self.model.infer_action(
                prompt=prompt,
                input_image=input_image[index : index + 1],
                action_horizon=self.config.action_horizon,
                proprio=proprio,
                context=context,
                context_mask=context_mask,
                num_inference_steps=self.config.num_inference_steps,
            )
            actions.append(self._denormalize_action(out["action"]))
        return torch.stack(actions, dim=0)

    def select_action(self, batch: dict[str, Tensor], **kwargs) -> Tensor:
        del kwargs
        self.eval()
        if len(self._action_queue) == 0:
            actions = self.predict_action_chunk(batch)[:, : self.config.n_action_steps]
            self._action_queue.extend(actions.transpose(0, 1))
        return self._action_queue.popleft()
