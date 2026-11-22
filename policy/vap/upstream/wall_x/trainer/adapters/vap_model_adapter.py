"""FSDP trainer adapter for native Video-Action Pretraining."""

from __future__ import annotations

from collections import defaultdict
from itertools import islice

import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm

from wall_x.model.vap.backbone import load_wan22_5b_video_base
from wall_x.model.vap.block import VAPBlock
from wall_x.model.vap.data import IVAVAChunkLocalActionNormalizer
from wall_x.model.vap.data import rebase_normalized_relative_action_chunks
from wall_x.model.vap.data import rebase_relative_action_chunks
from wall_x.model.vap.training import VAPTrainingModel
from wall_x.trainer.adapters.base_adapter import ModelAdapter
from wall_x.trainer.adapters.vga_model_adapter import VGAAdapter
from wall_x.trainer.utils import move_batch_to_device


def _read_horizon(config, name: str):
    if config is None:
        return None
    if isinstance(config, dict):
        return config.get(name)
    return getattr(config, name, None)


class _LimitedValidationLoader:
    """Expose at most ``max_batches`` without materializing the val loader.

    The x2robot validation loader can contain a large fraction of the full
    corpus.  VAP loss validation is intended as a frequent health signal, so
    it must honor ``model.validation.num_val_samples`` instead of traversing
    the complete validation split every time.
    """

    def __init__(self, loader, max_batches: int) -> None:
        if max_batches <= 0:
            raise ValueError("validation num_val_samples must be positive")
        self.loader = loader
        self.max_batches = int(max_batches)

    def __len__(self) -> int:
        return min(len(self.loader), self.max_batches)

    def __iter__(self):
        return islice(iter(self.loader), self.max_batches)


def _masked_action_error_sums(
    predicted: torch.Tensor,
    target: torch.Tensor,
    dof_mask: torch.Tensor,
    *,
    action_valid_mask: torch.Tensor | None,
    action_steps_per_chunk: int,
) -> dict[str, torch.Tensor]:
    """Return additive normalized action-error statistics.

    Keeping numerators and denominators additive makes the final metric exact
    across variable-length tail windows, batch sizes and distributed ranks.
    """

    if predicted.ndim != 3 or target.shape != predicted.shape:
        raise ValueError(
            "predicted/target actions must share [B,T,D], got "
            f"{tuple(predicted.shape)} and {tuple(target.shape)}"
        )
    if dof_mask.shape != predicted.shape:
        raise ValueError(
            f"dof_mask {tuple(dof_mask.shape)} != actions {tuple(predicted.shape)}"
        )
    mask = dof_mask.to(device=predicted.device, dtype=torch.float32)
    if action_valid_mask is not None:
        valid = action_valid_mask.to(device=predicted.device)
        if valid.ndim == 2:
            valid = valid.unsqueeze(-1)
        if valid.ndim != 3 or valid.shape[:2] != predicted.shape[:2]:
            raise ValueError(
                "action_valid_mask must be [B,T] or [B,T,D], got "
                f"{tuple(valid.shape)} for actions {tuple(predicted.shape)}"
            )
        if valid.shape[2] not in (1, predicted.shape[2]):
            raise ValueError(
                f"action_valid_mask last dim must be 1 or {predicted.shape[2]}"
            )
        mask = mask * valid.to(dtype=mask.dtype)
    if bool((mask < 0).any().item()):
        raise ValueError("action metric mask contains negative weights")

    diff = predicted.float() - target.to(device=predicted.device, dtype=torch.float32)
    valid_entries = mask > 0
    if bool((~torch.isfinite(diff[valid_entries])).any().item()):
        raise ValueError("non-finite open-loop action error in valid entries")

    result: dict[str, torch.Tensor] = {}
    slices = {
        "all": slice(0, predicted.shape[1]),
        "a0": slice(0, action_steps_per_chunk),
        "a1": slice(action_steps_per_chunk, 2 * action_steps_per_chunk),
    }
    for name, time_slice in slices.items():
        local_diff = diff[:, time_slice]
        local_mask = mask[:, time_slice]
        result[f"{name}_abs_sum"] = (local_diff.abs() * local_mask).sum()
        result[f"{name}_sq_sum"] = (local_diff.square() * local_mask).sum()
        result[f"{name}_weight"] = local_mask.sum()
    return result


class VAPAdapter(VGAAdapter):
    """Reuse VGA data/FSDP plumbing while replacing the model and loss."""

    MODEL_TYPE = "vap"
    _ACTION_KEYWORDS = (
        "action_encoder",
        "action_head",
        "action_q",
        "action_k",
        "action_v",
        "action_o",
        "action_norm_q",
        "action_norm_k",
        "action_norm1",
        "action_norm2",
        "action_norm3",
        "action_ffn",
        "action_modulation",
        "action_time",
        "action_identity",
        "state_film",
        "first_frame_memory_action_gate",
    )
    _VIDEO_KEYWORDS = (
        "video_q",
        "video_k",
        "video_v",
        "video_o",
        "video_norm_q",
        "video_norm_k",
        "video_norm1",
        "video_norm2",
        "video_norm3",
        "video_ffn",
        "video_modulation",
        "video_time",
        "video_head",
        "video_patch_embedding",
        "text_embedding",
        "context_k",
        "context_v",
        "context_norm_k",
        "video_identity",
        "global_time_anchor",
        "first_frame_memory",
        "first_frame_memory_video_gate",
    )

    def _validate_config_contract(
        self, *, validate_checkpoint_initialization: bool = True
    ) -> None:
        model = self.cfg.model
        if (
            validate_checkpoint_initialization
            and getattr(model, "text_encoder_backend", "t5") in {"qwen35_cache", "qwen35_online"}
        ):
            from wall_x.model.vap.text_conditioning import text_init_missing_keys

            required = text_init_missing_keys({
                "model": vars(model), "checkpoint": vars(self.cfg.checkpoint),
            })
            allowed = self.cfg.checkpoint.model_init_allowed_missing_keys
            if not set(required).issubset(allowed):
                raise ValueError(
                    "T5->Qwen model-only initialization requires explicit new "
                    "projector missing-key allowance; configure the matching Qwen text backend"
                )
        video_only = getattr(model, "training_mode", "paired_ivava") == "video_only"
        dataset_type = self.cfg.data.dataset_type
        if bool(getattr(model, "first_frame_memory_enabled", False)):
            if dataset_type != "x2robot_v2":
                raise ValueError(
                    "first-frame memory currently requires data.dataset_type="
                    "x2robot_v2 so the true episode-first observation can be decoded"
                )
            if getattr(model, "training_mode", "paired_ivava") != "paired_ivava":
                raise ValueError("first-frame memory requires paired_ivava training")
            validation = getattr(model, "validation", None) or {}
            if bool(validation.get("action_openloop", False)):
                raise ValueError(
                    "first-frame memory currently supports diffusion-loss "
                    "validation only; set model.validation.action_openloop=false"
                )
        expected_video = model.num_interleave_chunks * model.video_frames_per_chunk
        expected_action = model.num_interleave_chunks * model.action_steps_per_chunk
        expected_interval = model.action_steps_per_chunk // model.video_frames_per_chunk
        if self.cfg.task.action_horizon != expected_action:
            raise ValueError(
                f"task.action_horizon={self.cfg.task.action_horizon}; "
                f"VAP expects {expected_action}"
            )
        action_horizon_flow = getattr(self.cfg.task, "action_horizon_flow", None)
        if (
            action_horizon_flow is not None
            and int(action_horizon_flow) != expected_action
        ):
            raise ValueError(
                f"task.action_horizon_flow={action_horizon_flow}; "
                f"VAP expects {expected_action}"
            )
        if dataset_type == "x2robot_v2":
            horizon_fields = ("future_length",)
            interval_field = "future_interval"
        else:
            horizon_fields = ("image_horizon", "future_image_length")
            interval_field = "future_image_interval"
        for field in horizon_fields:
            horizon = getattr(self.cfg.data, field, None)
            if horizon is not None and int(horizon) != expected_video:
                raise ValueError(
                    f"data.{field}={horizon}; VAP expects {expected_video}"
                )
        if not video_only:
            interval = getattr(self.cfg.data, interval_field, None)
            if interval is not None and int(interval) != expected_interval:
                raise ValueError(
                    f"data.{interval_field}={interval}; VAP expects "
                    f"{expected_interval} action steps per sampled video frame"
                )
            raw_indices = (self.cfg._raw_data or {}).get("future_image_indices")
            if raw_indices is not None:
                expected_indices = list(
                    range(expected_interval, expected_action + 1, expected_interval)
                )
                if [int(value) for value in raw_indices] != expected_indices:
                    raise ValueError(
                        "data.future_image_indices must exactly match the two "
                        f"fixed windows: {expected_indices}"
                    )
        if model.action_dim != self.cfg.action_dim:
            raise ValueError(
                f"model.action_dim={model.action_dim} != task dof sum "
                f"{self.cfg.action_dim}"
            )

    def load_processor(self, action_statistic_dof):
        self._validate_config_contract()
        result = super().load_processor(action_statistic_dof)
        processor = result.get("processor")
        if (
            processor is not None
            and self.cfg.data.dataset_type == "x2robot_v1"
            and getattr(processor, "normalizer_action", None) is not None
        ):
            processor.normalizer_action = IVAVAChunkLocalActionNormalizer(
                processor.normalizer_action,
                self.cfg.task.dof_config,
                action_steps_per_chunk=self.cfg.model.action_steps_per_chunk,
                num_chunks=self.cfg.model.num_interleave_chunks,
            )
        data_config = result.get("data_config")
        if data_config is not None:
            video_only = (
                getattr(self.cfg.model, "training_mode", "paired_ivava") == "video_only"
            )
            expected_video = (
                self.cfg.model.num_interleave_chunks
                * self.cfg.model.video_frames_per_chunk
            )
            expected_action = (
                self.cfg.model.num_interleave_chunks
                * self.cfg.model.action_steps_per_chunk
            )
            expected_interval = (
                self.cfg.model.action_steps_per_chunk
                // self.cfg.model.video_frames_per_chunk
            )
            action_horizon = _read_horizon(data_config, "action_horizon")
            for field in ("image_horizon", "future_image_length"):
                horizon = _read_horizon(data_config, field)
                if horizon is not None and int(horizon) != expected_video:
                    raise ValueError(
                        f"VAP data {field}={horizon}; expected {expected_video} "
                        f"({self.cfg.model.num_interleave_chunks} x "
                        f"{self.cfg.model.video_frames_per_chunk})"
                    )
            if action_horizon is not None and int(action_horizon) != expected_action:
                raise ValueError(
                    f"VAP data action_horizon={action_horizon}; expected "
                    f"{expected_action} ({self.cfg.model.num_interleave_chunks} "
                    f"x {self.cfg.model.action_steps_per_chunk})"
                )
            if not video_only:
                interval = _read_horizon(data_config, "future_image_interval")
                if interval is not None and int(interval) != expected_interval:
                    raise ValueError(
                        f"VAP data future_image_interval={interval}; expected "
                        f"{expected_interval}"
                    )
        return result

    def create_model(
        self,
        processor,
        tokenizer_mixin,
        model_config,
        *,
        validate_checkpoint_initialization: bool = True,
    ):
        del processor, tokenizer_mixin
        self._validate_config_contract(
            validate_checkpoint_initialization=validate_checkpoint_initialization
        )
        if model_config.state_film_enabled and self.cfg.propri_dim <= 0:
            raise ValueError(
                "state_film_enabled requires a non-empty task.agent_pos_config"
            )
        pipeline = load_wan22_5b_video_base(
            model_config,
            device="cpu",
            torch_dtype=torch.bfloat16,
        )
        model = VAPTrainingModel.from_wan_pipeline(
            pipeline,
            model_config,
            action_dim=self.cfg.action_dim,
            state_dim=self.cfg.propri_dim,
        )
        if getattr(model_config, "training_mode", "paired_ivava") == "video_only":
            action_parameters = []
            for name, parameter in model.named_parameters():
                if any(keyword in name for keyword in self._ACTION_KEYWORDS):
                    parameter.requires_grad_(False)
                    action_parameters.append(name)
            if not action_parameters:
                raise RuntimeError(
                    "video_only did not find any Action-expert parameters to freeze"
                )
        return model

    def load_weights(self, model, normalizer_action, normalizer_propri, **kwargs):
        del normalizer_action, normalizer_propri, kwargs
        return model

    def get_transformer_layer_cls(self):
        return VAPBlock

    def _view_ids(self, batch, device: torch.device) -> torch.Tensor:
        from wall_x.utils.constant import PHYSICAL_VIEW_IDS

        next_obs = batch.get("next_obs", {})
        view_keys = sorted(
            (key for key in next_obs if key.startswith("view") and key[4:].isdigit()),
            key=lambda key: int(key[4:]),
        )
        camera_names = batch.get("camera_names")
        if not isinstance(camera_names, list) or not camera_names:
            raise ValueError("VAP batch requires camera_names for physical view IDs")
        if not isinstance(camera_names[0], list):
            raise ValueError("camera_names must be List[List[str]]")
        suffixes = [int(key[4:]) for key in view_keys]
        zero_based = min(suffixes) == 0
        slot_indices = suffixes if zero_based else [index - 1 for index in suffixes]
        mapping = self.cfg.model.physical_view_ids or PHYSICAL_VIEW_IDS
        rows = []
        for names in camera_names:
            selected = []
            for slot in slot_indices:
                if slot >= len(names):
                    raise ValueError(f"camera_names={names} has no slot index {slot}")
                name = names[slot]
                if name not in mapping:
                    raise ValueError(f"camera {name!r} has no physical_view_id")
                selected.append(int(mapping[name]))
            rows.append(selected)
        return torch.tensor(rows, dtype=torch.long, device=device)

    def _prepare_x2robot_v2_actions(self, batch):
        """Ensure x2robot_v2 actions use VAP's chunk-local coordinate frame."""

        if getattr(self.cfg.model, "training_mode", "paired_ivava") == "video_only":
            return self._prepare_video_only_placeholders(batch)
        if self.cfg.data.dataset_type != "x2robot_v2":
            return batch
        action = batch.get("action")
        dof_mask = batch.get("dof_mask")
        if not torch.is_tensor(action) or not torch.is_tensor(dof_mask):
            raise TypeError("x2robot_v2 VAP batch requires tensor action/dof_mask")
        normalizer = getattr(self, "normalizer_action", None)
        if normalizer is None:
            raise RuntimeError("VAPAdapter.normalizer_action is unset")

        from wall_x.model.core.action.normalizer import (
            resolve_normalizer_dataset_names,
        )

        dataset_names = resolve_normalizer_dataset_names(
            batch.get("dataset_name"),
            normalizer,
            action.shape[0],
            source="x2robot_v2 VAP batch dataset_name",
        )
        normalizer.to(action.device)
        prepared = dict(batch)
        chunks_already_local = bool(
            (getattr(self.cfg, "_raw_data", None) or {}).get(
                "action_chunks_already_local", False
            )
        )
        action_steps_per_chunk = int(self.cfg.model.action_steps_per_chunk)
        num_chunks, remainder = divmod(int(action.shape[1]), action_steps_per_chunk)
        if remainder or num_chunks < int(self.cfg.model.num_interleave_chunks):
            raise ValueError(
                "x2robot_v2 action horizon must contain an integer number of "
                "model-sized chunks: "
                f"T={action.shape[1]}, chunk={action_steps_per_chunk}, "
                f"native_window={self.cfg.model.num_interleave_chunks}"
            )
        unnorm_action = batch.get("unnorm_action")
        if torch.is_tensor(unnorm_action):
            if unnorm_action.shape != action.shape:
                raise ValueError(
                    "x2robot_v2 unnorm_action must match action shape: "
                    f"{tuple(unnorm_action.shape)} != {tuple(action.shape)}"
                )
            raw = unnorm_action.to(device=action.device)
            local = (
                raw
                if chunks_already_local
                else rebase_relative_action_chunks(
                    raw,
                    self.cfg.task.dof_config,
                    action_steps_per_chunk=action_steps_per_chunk,
                    num_chunks=num_chunks,
                )
            )
            valid = dof_mask.to(device=local.device).bool()
            if bool((~torch.isfinite(local[valid])).any().item()):
                raise ValueError("non-finite chunk-local action in an active DOF")
            local_clean = torch.nan_to_num(local, nan=0.0, posinf=0.0, neginf=0.0)
            normalized = normalizer.normalize_data(local_clean, dataset_names)
            prepared["action"] = torch.where(
                valid,
                normalized,
                torch.zeros_like(normalized),
            )
            prepared["unnorm_action"] = local
        else:
            if chunks_already_local:
                # Dataset-side normalization already consumed the same
                # chunk-local physical labels and stats. Preserve it exactly.
                return prepared
            # Legacy manifests omitted unnorm_action. They are safe only when
            # the dataset-side affine transform was not clipped: otherwise the
            # original sample-relative A1 cannot be reconstructed before its
            # chunk-local rebase.
            if bool(getattr(normalizer, "clamp_normalized", True)):
                raise RuntimeError(
                    "clipped x2robot_v2 VAP normalization requires the VGA "
                    "epilogue to emit unnorm_action (params.keep_unnorm=true)"
                )
            prepared["action"] = rebase_normalized_relative_action_chunks(
                action,
                normalizer,
                dataset_names,
                self.cfg.task.dof_config,
                action_steps_per_chunk=action_steps_per_chunk,
                num_chunks=num_chunks,
                dof_mask=dof_mask,
            )
        return prepared

    def _prepare_video_only_placeholders(self, batch):
        """Supply shape-only Action/proprio tensors for isolated Stage 1.

        Pure-video sources are allowed to omit every action/proprio field.
        The model's video-only attention layout makes these placeholders
        invisible to Video queries and the loss masks them completely.  This
        method intentionally runs before any x2robot_v2 normalization/rebase,
        because a non-robot video has no normalizer identity.
        """

        next_obs = batch.get("next_obs")
        obs = batch.get("obs")
        if not isinstance(next_obs, dict) or not isinstance(obs, dict):
            raise TypeError("video-only VAP batch requires mapping obs/next_obs")
        view_keys = sorted(
            (key for key in next_obs if key.startswith("view") and key[4:].isdigit()),
            key=lambda key: int(key[4:]),
        )
        if not view_keys:
            raise ValueError("video-only VAP batch has no next_obs viewN tensors")
        reference = next_obs[view_keys[0]]
        if not torch.is_tensor(reference) or reference.ndim != 5:
            raise ValueError("video-only next_obs view must be [B,T,H,W,C]")
        batch_size = int(reference.shape[0])
        action_steps = int(self.cfg.model.num_interleave_chunks) * int(
            self.cfg.model.action_steps_per_chunk
        )
        action_dim = int(self.cfg.action_dim)
        propri_dim = sum(
            int(value) for value in self.cfg.task.agent_pos_config.values()
        )
        prepared = dict(batch)
        prepared_obs = dict(obs)
        prepared_obs["agent_pos"] = torch.zeros(
            batch_size,
            1,
            propri_dim,
            device=reference.device,
            dtype=torch.float32,
        )
        prepared["obs"] = prepared_obs
        prepared["action"] = torch.zeros(
            batch_size,
            action_steps,
            action_dim,
            device=reference.device,
            dtype=torch.float32,
        )
        prepared["dof_mask"] = torch.zeros_like(prepared["action"])
        prepared["action_valid_mask"] = torch.zeros(
            batch_size,
            action_steps,
            device=reference.device,
            dtype=torch.float32,
        )
        return prepared

    def forward(self, model, batch, **kwargs):
        global_step = kwargs.get("global_step")
        batch = self._prepare_x2robot_v2_actions(batch)
        parameter = next(model.parameters())
        return model(
            batch,
            view_ids=self._view_ids(batch, parameter.device),
            global_step=global_step,
        )

    def extract_loss(self, outputs):
        return outputs["total_loss"]

    def collect_output_stats(
        self,
        outputs,
        step_stats,
        reduce_tensor_fn,
        true_gather_fn,
        tokenizer_mixin=None,
    ):
        del true_gather_fn, tokenizer_mixin
        keys = {
            "video_loss",
            "action_loss",
            "action_optimized_loss",
            "action_xm_best_of_k",
            "action_xm_candidate0_loss",
            "action_xm_selected_loss",
            "action_xm_improvement",
            "action_xm_nonzero_winner_fraction",
            "video_sigma",
            "action_sigma",
            "action_prefix_noise_fraction",
            "action_prefix_noise_rms",
            "action_prefix_embedding_dropout_sample_fraction",
            "action_prefix_embedding_dropout_channel_fraction",
            "action_soar_active",
            "action_soar_weight",
            "action_soar_correction_loss",
            "action_soar_a0_loss",
            "action_soar_a1_loss",
            "action_soar_aux_sigma",
            "action_soar_rollout_rms",
        }
        for chunk in range(int(self.cfg.model.num_interleave_chunks)):
            keys.update(
                {
                    f"video_sigma_v{chunk}",
                    f"ar_stage_a{chunk}_loss",
                    f"ar_stage_v{chunk}_loss",
                    f"ar_stage_a{chunk}_fraction",
                    f"ar_stage_v{chunk}_fraction",
                }
            )
        # Every value below enters a distributed collective.  All ranks must
        # therefore visit the keys in exactly the same order; iterating the
        # set directly can silently mix unrelated metrics across ranks because
        # Python hash order is process-local.
        for key in sorted(keys):
            value = outputs.get(key)
            if value is not None:
                step_stats[key] = reduce_tensor_fn(value.detach()).item()

    def init_validation(self, normalizer_action, normalizer_propri):
        del normalizer_action, normalizer_propri

    def run_validation(self, **kwargs):
        val_config = self.cfg.model.validation or {}
        max_batches = int(val_config.get("num_val_samples", 2))
        val_dataloader = _LimitedValidationLoader(kwargs["val_dataloader"], max_batches)
        if not bool(val_config.get("action_openloop", False)):
            kwargs["val_dataloader"] = val_dataloader
            return ModelAdapter.run_validation(self, **kwargs)

        model = kwargs["model"]
        rank = int(kwargs["rank"])
        device = kwargs["device"]
        reduce_fn = kwargs["reduce_fn"]
        autocast_context = kwargs["autocast_context"]
        wandb_logger = kwargs["logger"]
        global_step = int(kwargs["global_step"])
        sampling_model = model.module if isinstance(model, DDP) else model
        seed = int(val_config.get("action_openloop_seed", 0))
        use_kv_cache = bool(val_config.get("action_openloop_use_kv_cache", True))

        scalar_sums: dict[str, torch.Tensor] = {}
        scalar_weight = torch.zeros((), device=device, dtype=torch.float32)
        action_sums: dict[str, torch.Tensor] = defaultdict(
            lambda: torch.zeros((), device=device, dtype=torch.float32)
        )

        model.eval()
        pbar = tqdm(
            val_dataloader,
            desc="Validating VAP loss + action open-loop",
            total=len(val_dataloader),
            disable=rank != 0,
        )
        try:
            with torch.no_grad():
                for batch_index, raw_batch in enumerate(pbar):
                    raw_batch = move_batch_to_device(raw_batch, device)
                    batch = self._prepare_x2robot_v2_actions(raw_batch)
                    batch_size = int(batch["action"].shape[0])
                    view_ids = self._view_ids(batch, next(model.parameters()).device)

                    flow_generator = torch.Generator(device=device)
                    flow_generator.manual_seed(seed + rank * 1000003 + batch_index)
                    with autocast_context():
                        outputs = model(
                            batch,
                            view_ids=view_ids,
                            generator=flow_generator,
                        )
                    for output_key, log_key in (
                        ("total_loss", "val_loss"),
                        ("video_loss", "val_video_flow_loss"),
                        ("action_loss", "val_action_flow_loss"),
                    ):
                        value = outputs.get(output_key)
                        if value is not None:
                            weighted = value.detach().float() * batch_size
                            scalar_sums[log_key] = (
                                scalar_sums.get(log_key, torch.zeros_like(weighted))
                                + weighted
                            )
                    scalar_weight += batch_size

                    action_generator = torch.Generator(device=device)
                    action_generator.manual_seed(
                        seed + 500000003 + rank * 1000003 + batch_index
                    )
                    sampled = sampling_model.sample_actions_from_batch(
                        batch,
                        view_ids=view_ids,
                        generator=action_generator,
                        use_kv_cache=use_kv_cache,
                    )
                    batch_sums = _masked_action_error_sums(
                        sampled["predicted_actions"],
                        sampled["target_actions"],
                        sampled["dof_mask"],
                        action_valid_mask=batch.get("action_valid_mask"),
                        action_steps_per_chunk=self.cfg.model.action_steps_per_chunk,
                    )
                    for key, value in batch_sums.items():
                        action_sums[key] = action_sums[key] + value.detach().float()
        finally:
            model.train()

        # Sum first, divide once: this remains exact when the final validation
        # windows have fewer valid action steps or ranks see different masks.
        global_scalar_weight = reduce_fn(scalar_weight, average=False)
        if global_scalar_weight.item() <= 0:
            raise RuntimeError("VAP validation produced no samples")
        log_dict = {
            key: (reduce_fn(value, average=False) / global_scalar_weight).item()
            for key, value in scalar_sums.items()
        }
        for prefix in ("all", "a0", "a1"):
            weight = reduce_fn(action_sums[f"{prefix}_weight"], average=False)
            if weight.item() <= 0:
                continue
            mae = reduce_fn(action_sums[f"{prefix}_abs_sum"], average=False) / weight
            mse = reduce_fn(action_sums[f"{prefix}_sq_sum"], average=False) / weight
            metric_prefix = "val_action_openloop"
            suffix = "" if prefix == "all" else f"_{prefix}"
            log_dict[f"{metric_prefix}{suffix}_l1"] = mae.item()
            log_dict[f"{metric_prefix}{suffix}_mse"] = mse.item()
            log_dict[f"{metric_prefix}{suffix}_l2"] = mse.sqrt().item()
        log_dict["val_action_openloop_valid_values"] = reduce_fn(
            action_sums["all_weight"], average=False
        ).item()

        if rank == 0:
            if self.logger is not None:
                self.logger.info(
                    "[VAP Val] Step %d: %s (normalized action; L2=RMSE; kv_cache=%s)",
                    global_step,
                    log_dict,
                    use_kv_cache,
                )
            if wandb_logger is not None and hasattr(wandb_logger, "log"):
                wandb_logger.log(log_dict, step=global_step)
        return log_dict


__all__ = ["VAPAdapter"]
