from __future__ import annotations

import logging
from collections import deque
from pathlib import Path
from typing import Any

import torch
from huggingface_hub import hf_hub_download
from huggingface_hub.errors import HfHubHTTPError
from safetensors.torch import load_file as load_safetensor_file
from torch import Tensor

from lerobot.datasets.utils import serialize_dict, write_json
from lerobot.policies.names import WSA_LARGE
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.policies.utils import log_model_loading_keys
from lerobot.utils.utils import format_big_number

from .configuration_wsa_large import WSALargeConfig
from .core.data.lerobot.utils.normalizer import (
    SingleFieldLinearNormalizer,
    load_dataset_stats_from_json,
)
from .core.models.wan22.wsa_large import WSALarge
from .core.models.wan22.wsa_large_joint import WSALargeJoint
from .stats_adapter import ensure_wsa_large_stats_format


class WSALargePolicy(PreTrainedPolicy):
    config_class = WSALargeConfig
    name = WSA_LARGE
    _save_prefixes = ("model.mot.", "model.proprio_encoder.")
    _stats_filename = "stats.json"
    _allowed_shape_mismatch_keys = {
        "model.mot.mixtures.action.action_encoder.weight",
        "model.mot.mixtures.action.head.weight",
        "model.mot.mixtures.action.head.bias",
        "model.proprio_encoder.weight",
    }

    def __init__(self, config: WSALargeConfig, *, _defer_config_action_stats: bool = False):
        super().__init__(config)
        config.validate_features()
        self.config = config
        self._action_denorm_specs: list[dict[str, Any]] = []
        self._action_stats_source: str | None = None
        self._action_stats_payload: dict[str, Any] | None = None
        self._dit_trainable_params_configured = False
        self._dit_only_train_mode_active = False
        self.model = self._build_model(config)
        if config.native_checkpoint_path:
            logging.info("Loading WSA_Large native checkpoint from %s", config.native_checkpoint_path)
            self.model.load_checkpoint(config.native_checkpoint_path)
        if not _defer_config_action_stats:
            self._maybe_load_action_postprocess_from_config(strict=True)
        self.reset()

    @staticmethod
    def _variant_to_class(variant: str):
        mapping = {
            "wsa_large": WSALarge,
            "wsa_large_joint": WSALargeJoint,
        }
        return mapping[variant]

    def _build_model(self, config: WSALargeConfig):
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
            future_3d_config=config.future_3d_config,
            action_dit_pretrained_path=config.action_dit_pretrained_path,
            future_3d_pretrained_path=config.future_3d_pretrained_path,
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
            mask_action_dim_padding_loss=bool(config.mask_action_dim_padding_loss),
            loss_lambda_3d=float(config.lambda_3d),
            da3_model_path_or_name=config.da3_model_path_or_name,
            da3_code_root=config.da3_code_root,
            da3_teacher_process_res=int(config.da3_teacher_process_res),
            da3_teacher_layers=config.da3_teacher_layers,
            da3_layer_weights=config.da3_layer_weights,
            log_da3_teacher_timing=config.log_da3_teacher_timing,
            future_3d_view_attention_layout=config.future_3d_view_attention_layout,
        )

    def get_state_dict_to_save(self) -> dict[str, Tensor] | None:
        state_dict = self.state_dict()
        return {
            key: value
            for key, value in state_dict.items()
            if key.startswith(self._save_prefixes)
        }

    @classmethod
    def _load_as_safetensor(
        cls,
        model: "WSALargePolicy",
        model_file: str,
        map_location: str,
        strict: bool,
    ) -> "WSALargePolicy":
        state_dict = load_safetensor_file(model_file, device="cpu")
        model_state = model.state_dict()
        filtered_state_dict: dict[str, Tensor] = {}
        skipped_shape_keys: list[str] = []
        unexpected_shape_keys: list[str] = []

        for key, value in state_dict.items():
            target = model_state.get(key)
            if target is not None and tuple(value.shape) != tuple(target.shape):
                if key in cls._allowed_shape_mismatch_keys:
                    skipped_shape_keys.append(key)
                    continue
                unexpected_shape_keys.append(key)
                continue
            filtered_state_dict[key] = value

        if unexpected_shape_keys:
            preview = ", ".join(unexpected_shape_keys[:8])
            remaining = len(unexpected_shape_keys) - 8
            suffix = f", ... (+{remaining} more)" if remaining > 0 else ""
            raise RuntimeError(
                "Unexpected shape mismatch while loading WSA_Large pretrained weights: "
                f"{preview}{suffix}"
            )

        if skipped_shape_keys:
            preview = ", ".join(skipped_shape_keys[:8])
            remaining = len(skipped_shape_keys) - 8
            suffix = f", ... (+{remaining} more)" if remaining > 0 else ""
            logging.info(
                "Skipping WSA_Large shape-mismatched pretrained key(s): %s%s",
                preview,
                suffix,
            )

        missing_keys, unexpected_keys = model.load_state_dict(filtered_state_dict, strict=strict)
        missing_keys = [key for key in missing_keys if key not in skipped_shape_keys]
        missing_keys, unexpected_keys, expected_missing_keys = model.classify_model_loading_keys(
            list(missing_keys),
            list(unexpected_keys),
        )
        if expected_missing_keys:
            preview = expected_missing_keys[:8]
            preview_str = ", ".join(preview)
            remaining = len(expected_missing_keys) - len(preview)
            suffix = f", ... (+{remaining} more)" if remaining > 0 else ""
            logging.info(
                "Expected missing key(s) when loading model (new modules initialized separately): "
                f"{preview_str}{suffix}"
            )
        log_model_loading_keys(missing_keys, unexpected_keys)

        if map_location != "cpu":
            model.to(map_location)
        return model

    def classify_model_loading_keys(
        self, missing_keys: list[str], unexpected_keys: list[str]
    ) -> tuple[list[str], list[str], list[str]]:
        expected_missing = [
            key
            for key in missing_keys
            if not any(key.startswith(prefix) for prefix in self._save_prefixes)
        ]
        filtered_missing = [key for key in missing_keys if key not in expected_missing]
        return filtered_missing, unexpected_keys, expected_missing

    @staticmethod
    def _count_params(module: torch.nn.Module | None, *, trainable_only: bool = False) -> int:
        if module is None:
            return 0
        return sum(
            param.numel()
            for param in module.parameters()
            if not trainable_only or param.requires_grad
        )

    @staticmethod
    def _format_path(path: str | None) -> str:
        if not path:
            return "<none>"
        return str(path)

    def startup_summary(self) -> str:
        """Compact startup summary for training logs."""
        model = self.model
        future_3d = model.future_3d_expert
        mot = model.mot

        total_params = self._count_params(self)
        trainable_params = self._count_params(self, trainable_only=True)
        optimizer_params = sum(param.numel() for param in self.get_optim_params())
        video_params = self._count_params(model.video_expert)
        future_3d_params = self._count_params(future_3d)
        action_params = self._count_params(model.action_expert)
        da3_params = self._count_params(getattr(model, "da3_teacher", None))

        lines = [
            "=" * 60,
            f"Policy: {self.__class__.__name__}",
            f"Variant: {self.config.variant}",
            "",
            "Parameter statistics:",
            f"  - Total params        : {total_params} ({format_big_number(total_params)})",
            f"  - Trainable params    : {trainable_params} ({format_big_number(trainable_params)})",
            f"  - Optimizer params    : {optimizer_params} ({format_big_number(optimizer_params)})",
            f"  - Video expert params : {video_params} ({format_big_number(video_params)})",
            f"  - Future3D params     : {future_3d_params} ({format_big_number(future_3d_params)})",
            f"  - Action expert params: {action_params} ({format_big_number(action_params)})",
            "",
            "MoT:",
            f"  - Experts             : {', '.join(mot.expert_order)}",
            f"  - Layers              : {mot.num_layers}",
            f"  - Heads / head dim    : {mot.num_heads} / {mot.attn_head_dim}",
            f"  - Mixed checkpointing : {mot.mot_checkpoint_mixed_attn}",
            "",
            "Objectives:",
            f"  - Action / proprio dim: {self.config.action_dim} / {self.config.proprio_dim}",
            f"  - Lambda video/action/3D: {model.loss_lambda_video} / {model.loss_lambda_action} / {model.loss_lambda_3d}",
            f"  - Mask action pad loss: {getattr(model, 'mask_action_dim_padding_loss', False)}",
            "",
            "Future 3D:",
            f"  - FUTURE_3D_QUERY_MODE: {future_3d.query_mode}",
            f"  - Query noise scale   : {future_3d.query_noise_scale}",
            f"  - Query sigma range   : [{future_3d.query_noise_min_sigma}, {future_3d.query_noise_max_sigma}]",
            f"  - Query sigma source  : {future_3d.query_sigma_source}",
            f"  - Slot pos scale      : {future_3d.slot_pos_scale}",
            f"  - Query layers        : {future_3d.query_layer_indices}",
            f"  - Views               : {future_3d.da3_num_views}",
            f"  - Query tokens/view   : {future_3d.query_tokens_per_view}",
            f"  - DA3 tokens/view     : {future_3d.da3_tokens_per_view}",
            f"  - Attention layout    : {model.future_3d_view_attention_layout}",
            f"  - Target frame index  : {self.config.future_3d_target_index}",
            "",
            "DA3 teacher:",
            f"  - Source              : {self.config.da3_model_path_or_name}",
            f"  - Variant             : {self.config.da3_variant}",
            f"  - Process resolution  : {self.config.da3_teacher_process_res}",
            f"  - Teacher layers      : {model.da3_teacher_layers}",
            f"  - Params              : {da3_params} ({format_big_number(da3_params)})",
            "",
            "Initialization:",
            f"  - Action backbone     : {self._format_path(self.config.action_dit_pretrained_path)}",
            f"  - Future3D backbone   : {self._format_path(self.config.future_3d_pretrained_path)}",
            f"  - Native checkpoint   : {self._format_path(self.config.native_checkpoint_path)}",
            "=" * 60,
        ]
        return "\n".join(lines)

    def _save_pretrained(self, save_directory: Path) -> None:
        self._save_pretrained_artifacts(save_directory)
        if self._action_stats_payload is not None:
            stats_payload = {WSA_LARGE: self._action_stats_payload, "wsa_large": self._action_stats_payload}
            write_json(serialize_dict(stats_payload), save_directory / self._stats_filename)

    @classmethod
    def from_pretrained(
        cls,
        pretrained_name_or_path: str | Path,
        *,
        config: WSALargeConfig | None = None,
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
        init_kwargs = {"_defer_config_action_stats": True}
        if config is None:
            config = WSALargeConfig.from_pretrained(
                pretrained_name_or_path=pretrained_name_or_path,
                force_download=force_download,
                resume_download=resume_download,
                proxies=proxies,
                token=token,
                cache_dir=cache_dir,
                local_files_only=local_files_only,
                revision=revision,
                **kwargs,
            )
        else:
            init_kwargs.update(kwargs)
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
            **init_kwargs,
        )
        local_dir = Path(pretrained_name_or_path).expanduser()
        # Fine-tuning should honor task-specific action stats from config; inference clears this field
        # before calling from_pretrained so saved checkpoint stats still take precedence there.
        if getattr(config, "action_stats_path", None):
            policy._maybe_load_action_postprocess_from_config(
                base_dir=local_dir if local_dir.is_dir() else None,
                strict=True,
            )
        else:
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
            if not policy._action_denorm_specs:
                policy._maybe_load_action_postprocess_from_config(
                    base_dir=local_dir if local_dir.is_dir() else None,
                    strict=False,
                )
        return policy

    @staticmethod
    def _extract_wsa_large_stats(stats_payload: dict[str, Any]) -> dict[str, Any]:
        stats_payload = ensure_wsa_large_stats_format(
            stats_payload,
            shape_meta=None,
            require_state=False,
        )
        if "action" not in stats_payload:
            raise ValueError("WSA_Large stats payload must contain an `action` section.")
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
        stats_payload = self._extract_wsa_large_stats(stats_payload)
        action_stats = stats_payload["action"]
        if not isinstance(action_stats, dict) or len(action_stats) == 0:
            raise ValueError("WSA_Large action stats must be a non-empty dict.")

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
                    f"Missing {prefix}* stats for WSA_Large action key '{key}'. Available keys: {list(key_stats.keys())}"
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
        if total_dim > int(self.config.action_dim):
            raise ValueError(
                f"WSA_Large action stats dim mismatch: stats sum to {total_dim}, "
                f"which is larger than config.action_dim={self.config.action_dim}"
            )
        if total_dim < int(self.config.action_dim):
            logging.info(
                "WSA_Large action stats cover %d/%d dims; trailing policy dims are treated as padded output.",
                total_dim,
                self.config.action_dim,
            )
        self._action_denorm_specs = denorm_specs
        self._action_stats_payload = stats_payload

    def _load_action_postprocess_from_stats_path(self, stats_path: str | Path) -> None:
        stats_path = Path(stats_path)
        if not stats_path.is_file():
            raise FileNotFoundError(f"WSA_Large action stats file does not exist: {stats_path}")
        stats_payload = load_dataset_stats_from_json(str(stats_path))
        self.set_action_postprocess_from_stats(stats_payload)
        self._action_stats_source = str(stats_path)
        logging.info("Loaded WSA_Large action postprocess stats from %s", stats_path)

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

    @staticmethod
    def _candidate_config_stats_paths(stats_path: str | Path, base_dir: str | Path | None = None) -> list[Path]:
        raw_path = Path(stats_path).expanduser()
        if raw_path.is_absolute():
            return [raw_path]

        candidates = [raw_path]
        if base_dir is not None:
            base_path = Path(base_dir).expanduser()
            candidates.append(base_path / raw_path)
            candidates.extend(parent / raw_path for parent in list(base_path.parents)[:5])

        deduped: list[Path] = []
        seen: set[str] = set()
        for candidate in candidates:
            key = str(candidate)
            if key in seen:
                continue
            seen.add(key)
            deduped.append(candidate)
        return deduped

    def _maybe_load_action_postprocess_from_config(
        self,
        *,
        base_dir: str | Path | None = None,
        strict: bool = True,
    ) -> None:
        if self._action_denorm_specs:
            return
        stats_path = getattr(self.config, "action_stats_path", None)
        if stats_path is None:
            return

        candidates = self._candidate_config_stats_paths(stats_path, base_dir=base_dir)
        errors: list[str] = []
        for candidate in candidates:
            if not candidate.is_file():
                continue
            try:
                self._load_action_postprocess_from_stats_path(candidate)
                return
            except Exception as exc:
                if strict:
                    raise
                errors.append(f"{candidate}: {exc}")

        if errors:
            message = f"WSA_Large config action_stats_path is not usable. Last error: {errors[-1]}"
        else:
            message = (
                "WSA_Large config action_stats_path was not found. Tried: "
                + ", ".join(str(candidate) for candidate in candidates)
            )
        if strict:
            raise FileNotFoundError(message)
        logging.warning("%s; skipping config action stats fallback.", message)

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
                "WSA_Large pretrained source %s has no %s; action outputs will stay normalized unless "
                "`policy.action_stats_path` is provided.",
                pretrained_name_or_path,
                self._stats_filename,
            )
            return
        try:
            self._load_action_postprocess_from_stats_path(stats_path)
        except ValueError as exc:
            message = str(exc)
            if (
                "Cannot adapt framework stats" not in message
                and "WSA_Large stats payload must contain an `action` section" not in message
                and "WSA_Large action stats dim mismatch" not in message
            ):
                raise
            logging.warning(
                "Skipping automatic WSA_Large action stats load from %s: %s. "
                "Provide task-specific stats through `policy.action_stats_path` or the evaluation adapter.",
                stats_path,
                exc,
            )

    def _configure_dit_trainable_params_once(self) -> None:
        if self._dit_trainable_params_configured:
            return
        self.model.requires_grad_(False)
        self.model.dit.requires_grad_(True)
        proprio_encoder = getattr(self.model, "proprio_encoder", None)
        if proprio_encoder is not None:
            proprio_encoder.requires_grad_(True)
        self._dit_trainable_params_configured = True

    def _set_dit_only_train_mode(self) -> None:
        self._configure_dit_trainable_params_once()
        if self._dit_only_train_mode_active:
            return
        self.training = True
        self.model.eval()
        self.model.dit.train()
        proprio_encoder = getattr(self.model, "proprio_encoder", None)
        if proprio_encoder is not None:
            proprio_encoder.train()
        self._dit_only_train_mode_active = True

    def get_optim_params(self):
        trainable_params = list(self.model.dit.parameters())
        proprio_encoder = getattr(self.model, "proprio_encoder", None)
        if proprio_encoder is not None:
            trainable_params.extend(list(proprio_encoder.parameters()))
        return (param for param in trainable_params if param.requires_grad)

    def reset(self):
        self._action_queue: deque[Tensor] = deque()

    def train(self, mode: bool = True):
        if not mode:
            self._dit_only_train_mode_active = False
            return super().train(mode)
        self._set_dit_only_train_mode()
        return self

    def _prepare_training_sample(self, batch: dict[str, Any]) -> dict[str, Any]:
        sample = dict(batch)
        if "context" in sample and "context_mask" in sample:
            return sample
        prompts = sample.get("prompt")
        if prompts is None:
            raise ValueError(
                "WSA_Large training requires either `context/context_mask` in the batch or `prompt` with "
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

    def forward(self, batch: dict[str, Tensor], *, collect_metrics: bool = True) -> tuple[Tensor, dict]:
        sample = self._prepare_training_sample(batch)
        loss, loss_dict = self.model.training_loss(
            sample,
            collect_loss_dict=True,
            collect_detailed_loss_dict=collect_metrics,
            loss_dict_as_tensors=not collect_metrics,
        )
        if not collect_metrics:
            return loss, loss_dict
        output = {
            "loss": float(loss.detach().item()),
            "loss_video": float(loss_dict["loss_video"]),
            "loss_action": float(loss_dict["loss_action"]),
        }
        if "loss_3d" in loss_dict:
            output["loss_3d"] = float(loss_dict["loss_3d"])
        for key, value in loss_dict.items():
            if (
                key.startswith("loss_action_dim")
                or key.startswith("loss_3d_q")
                or key in {"loss_video_w", "loss_action_w", "loss_3d_w"}
                or key in {"time_3d_teacher_forward_s", "future_3d_query_sigma"}
            ):
                output[key] = float(value)
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
            raise ValueError("WSA_Large inference expects `input_image` or `video` in the batch.")
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
            raise ValueError(f"Expected WSA_Large action to be [B,T,D] or [T,D], got {tuple(action.shape)}")

        action_f32 = action.to(dtype=torch.float32)
        parts = []
        start = 0
        for spec in self._action_denorm_specs:
            dim = int(spec["dim"])
            end = start + dim
            cur_action = action_f32[..., start:end]
            if cur_action.shape[-1] != dim:
                raise ValueError(
                    f"WSA_Large action dim mismatch while denormalizing: expected slice dim {dim}, got {cur_action.shape[-1]}"
                )
            scale = spec["scale"].to(device=cur_action.device, dtype=cur_action.dtype)
            offset = spec["offset"].to(device=cur_action.device, dtype=cur_action.dtype)
            parts.append((cur_action - offset) / scale)
            start = end

        if start > action_f32.shape[-1]:
            raise ValueError(
                f"WSA_Large denormalizer needs {start} dims but action has {action_f32.shape[-1]} dims."
            )

        action_denorm = torch.cat(parts, dim=-1)
        if squeeze_batch:
            action_denorm = action_denorm.squeeze(0)
        return action_denorm

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Tensor], **kwargs) -> Tensor:
        self.eval()
        input_image = self._resolve_input_image(batch)
        batch_size = input_image.shape[0]
        seed = kwargs.pop("seed", batch.get("seed", None))
        del kwargs
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
                seed=seed,
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
