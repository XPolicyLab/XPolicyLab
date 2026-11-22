import dataclasses
import logging
import os
from abc import ABC, abstractmethod
from collections import defaultdict

import torch
import torch.distributed as dist
from tqdm import tqdm

from wall_x.trainer.utils import move_batch_to_device

logger = logging.getLogger(__name__)


def _distributed_union_keys(local_keys):
    keys = sorted(str(key) for key in local_keys)
    if not (dist.is_available() and dist.is_initialized()):
        return keys
    world_size = dist.get_world_size()
    if world_size <= 1:
        return keys
    gathered = [None for _ in range(world_size)]
    dist.all_gather_object(gathered, keys)
    union = set()
    for rank_keys in gathered:
        if isinstance(rank_keys, (list, tuple)):
            union.update(str(key) for key in rank_keys)
    return sorted(union)


def _channel_stats_device(*mappings):
    for mapping in mappings:
        if isinstance(mapping, dict):
            for value in mapping.values():
                if torch.is_tensor(value):
                    return value.device
    if torch.cuda.is_available():
        return torch.device("cuda", torch.cuda.current_device())
    return torch.device("cpu")


def load_x2r_data_config(cfg):
    """Build an ``X2RDataConfig`` from a typed ``TrainConfig``.

    Dispatches on ``cfg.data.dataset_config_path``:

    - If set and the file exists (Path A — external data YAML, either
      nested TrainConfig schema or flat VGA 2509 schema), load that file
      and use its content. The nested case re-enters through
      ``build_x2r_data_config``; the flat case falls back to
      ``get_data_configs`` + ``X2RDataConfig.from_yaml_dict``.
    - Otherwise (Path C — pure TrainConfig), delegate to
      ``build_x2r_data_config(cfg)`` directly.

    Non-x2robot backends (lerobot, x2robot_v2) build their own dataset
    lazily inside the adapter's fast-path and never consume an
    ``X2RDataConfig``. Return ``None`` so the legacy build code (which
    reads x2robot-v2-only fields like ``image_history_length``) is not
    invoked on mismatched configs; accidental v1-style access downstream
    will fail loudly rather than silently using stale state.
    """
    dataset_type = getattr(cfg.data, "dataset_type", "x2robot")
    if dataset_type in ("x2robot_v2", "lerobot"):
        return None

    # x2robot_dataset (v1) is deprecated and absent from inference-only envs.
    # VGA / online inference never consumes the returned X2RDataConfig (the
    # adapter reads dof/keys from cfg.task and normalizers from config stats),
    # so when the package is unavailable, skip building it entirely. Probe with
    # find_spec rather than importing, so x2robot_dataset is never imported.
    import importlib.util

    if importlib.util.find_spec("x2robot_dataset") is None:
        # Expected in inference-only envs (the package is deprecated and not
        # installed there), so info-level — avoid triggering ops alerts.
        logger.info(
            "x2robot_dataset (v1) not installed; skipping X2RDataConfig build "
            "(dataset_type=%s). Inference does not consume it.",
            dataset_type,
        )
        return None

    dcp = getattr(cfg.data, "dataset_config_path", None) or (cfg._raw_data or {}).get(
        "dataset_config_path"
    )
    # A string dataset_config_path is treated as an external config file
    # (typed TrainConfig or legacy flat YAML). A list is the x2robot_dataset
    # source-list form and must be passed through to X2RDataConfig unchanged.
    if isinstance(dcp, (str, os.PathLike)) and os.path.exists(dcp):
        from wall_x.config.loader import load_config

        typed_err = None
        try:
            ext_cfg = load_config(dcp)
            return build_x2r_data_config(ext_cfg)
        except Exception as err:
            # Typed TrainConfig load failed — the YAML may be a legacy flat
            # schema (VGA 2509 style) rather than nested. Keep the typed
            # error so the fallback can chain it if it also fails.
            typed_err = err
            logger.info(
                "load_x2r_data_config: typed TrainConfig load of %s failed "
                "(%s: %s); falling back to legacy flat schema.",
                dcp,
                type(err).__name__,
                err,
            )

        from wall_x.data import data_backend
        from wall_x.model.vga.openloop_visualization import get_data_configs
        import yaml

        try:
            with open(dcp) as f:
                ext_yaml = yaml.safe_load(f)
            ext_yaml["data"] = get_data_configs(ext_yaml.get("data", {}))
            return data_backend().load_x2r_data_config_from_yaml_dict(ext_yaml)
        except Exception as legacy_err:
            logger.warning(
                "load_x2r_data_config: both typed and legacy load of %s "
                "failed. Typed error: %s: %s. Legacy error: %s: %s.",
                dcp,
                type(typed_err).__name__,
                typed_err,
                type(legacy_err).__name__,
                legacy_err,
            )
            raise RuntimeError(
                f"Could not build X2RDataConfig from {dcp!r}: neither "
                f"typed TrainConfig schema nor legacy flat schema parsed "
                f"successfully. See logs for both underlying errors."
            ) from legacy_err

    return build_x2r_data_config(cfg)


def build_x2r_data_config(cfg):
    """Construct X2RDataConfig from typed TrainConfig fields (Path C).

    Builds the yaml-shaped dict that ``X2RDataConfig.from_yaml_dict``
    expects, then hands off to ``from_yaml_dict`` so its internal default
    handling (which differs from X2RDataConfig's dataclass field defaults)
    is preserved bit-for-bit.

    Path A (``cfg.data.dataset_config_path`` set) is handled by
    ``load_x2r_data_config`` which reads the external YAML and dispatches
    back into here (nested) or via the v1 backend's
    ``load_x2r_data_config_from_yaml_dict`` (flat).
    """
    from wall_x.data import data_backend

    data_section = dict(cfg._raw_data or {})

    # Phase 1 data-runtime fields: ensure they appear in data section even
    # when not set in yaml, so downstream reads the typed default instead of
    # falling back to X2RDataConfig/from_yaml_dict's own (possibly different)
    # default. Other cfg.data fields are left to from_yaml_dict's handling,
    # preserving bit-for-bit legacy behavior.
    _PHASE1_TYPED_FIELDS = (
        "image_history_length",
        "image_horizon",
        "image_history_interval",
        "obs_history_stride",
        "action_history_length",
        "event_level",
    )
    for name in _PHASE1_TYPED_FIELDS:
        # Some backend-specific configs do not carry every runtime field; skip
        # missing values and only mirror fields that the typed config exposes.
        val = getattr(cfg.data, name, None)
        if val is not None:
            data_section.setdefault(name, val)

    task_dict = dataclasses.asdict(cfg.task)
    for key in (
        "dof_config",
        "agent_pos_config",
        "action_horizon",
        "action_horizon_flow",
        "ar_dof_config",
        "use_state_string_representation",
    ):
        if task_dict.get(key) is not None:
            data_section.setdefault(key, task_dict[key])

    data_section.setdefault("batch_size_per_gpu", cfg.hyperparams.batch_size_per_gpu)
    data_section.setdefault("batch_size", cfg.hyperparams.batch_size_per_gpu)

    # VGA trainer passes fixed_resolution/height/width via CLI args into the
    # top-level config dict; in typed TrainConfig they live under ``model``.
    # Surface them so X2RDataConfig.from_yaml_dict picks them up identically.
    model_cfg = getattr(cfg, "model", None)
    for key in ("fixed_resolution", "height", "width"):
        val = getattr(model_cfg, key, None)
        if val is not None:
            data_section.setdefault(key, val)

    yaml_dict = {
        "model_type": cfg.model_type,
        "data": data_section,
        **task_dict,
    }
    if data_section.get("seed") is not None:
        yaml_dict["seed"] = data_section["seed"]

    # X2RDataConfig reads ``qwen_vl_act_config_path`` from the top-level
    # yaml dict and uses it to derive ``padding_side`` (right for
    # flash_attention_2, left otherwise). Without it the WallxDataProcessor
    # collator throws AttributeError on ``self.config.padding_side`` mid-run.
    # Typed TrainConfig stores the model config json under cfg.model.config_path.
    model_cfg_path = getattr(getattr(cfg, "model", None), "config_path", None)
    if model_cfg_path:
        yaml_dict.setdefault("qwen_vl_act_config_path", model_cfg_path)

    # Resolve dataset_path. Top-level ``cfg.dataset_path`` wins for legacy
    # yaml compatibility; new yamls put it under ``data.dataset_path`` so
    # all three backends keep their data-source pointer co-located in the
    # ``data:`` section.
    ds_path = cfg.dataset_path
    if ds_path is None:
        ds_path = getattr(cfg.data, "dataset_path", None)
    if ds_path is not None:
        yaml_dict["dataset_path"] = ds_path

    return data_backend().load_x2r_data_config_from_yaml_dict(yaml_dict)


class ModelAdapter(ABC):
    """Pure strategy that encapsulates model-type-specific behavior.

    The adapter is stateless w.r.t. the trainer: every method declares
    its inputs as parameters and communicates results via return values.
    The trainer orchestrates calls and manages its own state.

    Abstract methods (subclasses MUST implement):
        load_processor          -- load processor / tokenizer artifacts
        build_model_config      -- build model-specific config object
        create_model            -- instantiate the model
        load_weights            -- load pretrained weights & set normalizers
        get_transformer_layer_cls -- layer classes for FSDP auto-wrap
        load_dataset            -- build dataset & training dataloader
        forward                 -- run a forward pass
        extract_loss            -- extract scalar loss from model outputs

    Concrete methods with default behaviour (override when needed):
        load_step_and_epoch     -- restore global_step / epoch from checkpoint
        log_model_info          -- log model-specific info (no-op)
        collect_output_stats    -- collect auxiliary loss / accuracy stats (no-op)
        collect_param_norms     -- compute per-group param L2 norms (no-op)
        get_fwd_flops           -- profile forward FLOPs via DeepSpeed
        get_output_field        -- retrieve a named field from outputs (static)
    """

    def __init__(self, *, cfg=None, logger=None, model_type=None):
        self.cfg = cfg
        self.logger = logger

    # ---- processor ----
    @abstractmethod
    def load_processor(self, action_statistic_dof):
        """Load processor and associated tokenizer artifacts.

        Reads model/data config from ``self.cfg``.

        Args:
            action_statistic_dof: normalizer statistics dict.

        Returns:
            dict with keys:
                "processor", "data_config", "tokenizer_mixin",
                "train_action_tokenizer", "val_action_tokenizer",
                "action_mapper", "num_added_tokens"
        """
        ...

    # ---- model ----
    @abstractmethod
    def create_model(self, processor, tokenizer_mixin, model_config):
        """Instantiate the model (before weight loading / FSDP wrap).

        Reads model config from ``self.cfg``.
        """
        ...

    @abstractmethod
    def build_model_config(self):
        """Build the model-specific config object from ``self.cfg.model``."""
        ...

    @abstractmethod
    def load_weights(self, model, normalizer_action, normalizer_propri, **kwargs):
        """Load pretrained weights, resize embeddings, set normalizers, etc."""
        ...

    # ---- FSDP wrapping ----
    @abstractmethod
    def get_transformer_layer_cls(self):
        """Return layer classes for FSDP transformer_auto_wrap_policy."""
        ...

    # ---- sequence parallelism ----
    def sp_sharded_param_prefixes(self) -> tuple[str, ...]:
        """FQN prefixes of params trained on the SP sequence shard.

        These params see only this rank's sequence chunk under sequence
        parallelism, so their grads are partial and get summed across the SP
        group (see reduce_sp_partial_grads); params on the gathered full
        sequence are excluded. Default: none (model has no SP-sharded params).
        """
        return ()

    # ---- dataset ----
    @abstractmethod
    def load_dataset(self, data_config, processor, rank, world_size, **kwargs):
        """Load dataset and build the training dataloader.

        Args:
            data_config: parsed data config from load_processor().
            processor:   processor instance.
            rank:        current process rank.
            world_size:  total number of processes.
            **kwargs:    extra model-specific args.

        Returns:
            (dataset, train_dataloader, train_num) tuple.
        """
        ...

    def load_step_and_epoch(self, checkpoint_path: str, is_incomplete_epoch: bool):
        global_step = 0
        start_epoch = 0

        global_step_path = os.path.join(checkpoint_path, "global_step.pth")
        if os.path.exists(global_step_path):
            global_step = torch.load(global_step_path)["global_step"]

        current_epoch_path = os.path.join(checkpoint_path, "current_epoch.pth")
        if os.path.exists(current_epoch_path):
            start_epoch = torch.load(current_epoch_path)["current_epoch"]
            if not is_incomplete_epoch:
                start_epoch = start_epoch + 1

        return {"global_step": global_step, "start_epoch": start_epoch}

    # ---- forward / loss ----
    @abstractmethod
    def forward(self, model, batch, **kwargs):
        """Run a forward pass.

        Args:
            model: the (possibly wrapped) model.
            batch: dict of tensors already on device.
            **kwargs: extra context (e.g. global_step, mode).

        Returns:
            Raw model outputs (dict or object).
        """
        ...

    @abstractmethod
    def extract_loss(self, outputs):
        """Extract the scalar training loss from model outputs.

        Returns:
            torch.Tensor (scalar).
        """
        ...

    # ---- optional hooks (override when needed) ----
    def log_model_info(self, model, log_fn):
        """Log model-specific info (e.g. attention implementation).

        Args:
            model:  the unwrapped model.
            log_fn: callable(str) for logging.
        """
        pass

    def collect_output_stats(
        self,
        outputs,
        step_stats,
        reduce_tensor_fn,
        true_gather_fn,
        tokenizer_mixin=None,
    ):
        """Collect model-output-specific statistics into step_stats.

        Extracts auxiliary losses (cross_entropy_loss, flow_loss),
        per-dataset channel losses, and action accuracy metrics from
        the model outputs and merges them into *step_stats* in-place.

        This is an optional hook — subclasses override it when their
        output format carries extra fields.  The default implementation
        is a no-op.

        Args:
            outputs:          raw model outputs (dict or object).
            step_stats:       dict being built by the trainer; mutated in-place.
            reduce_tensor_fn: callable(tensor, average=True) → scalar tensor,
                              wraps dist.all_reduce for the current parallelism.
            true_gather_fn:   callable(tensor) → scalar | None,
                              gathers a value to rank-0 (returns None on
                              non-main ranks or when input is None).
            tokenizer_mixin:  optional ActionTokenizerMixin used to retrieve
                              extra accuracy keys (e.g. RVQ layer accuracy).
        """
        pass

    def collect_param_norms(
        self, model, step_stats, device, reduce_tensor_fn, params_sharded=False
    ):
        """Compute per-group parameter L2 norms and merge into step_stats.

        Groups parameters by name keyword:
          - "visual"  → visual_param_norm
          - "action"  → action_expert_param_norm
          - (others)  → org_vlm_param_norm
          - all       → total_param_norm

        When parameters are sharded across ranks (FSDP FULL_SHARD /
        HYBRID_SHARD), the local squared-sum is partial and must be
        all_reduce'd (sum, not avg) before taking the square root.
        When parameters are replicated (DDP, or FSDP with SHARD_GRAD_OP /
        NO_SHARD), all_reduce would inflate the result by world_size.

        This is an optional hook — the default implementation is a no-op.

        Args:
            model:            the (possibly FSDP/DDP-wrapped) model.
            step_stats:       dict being built by the trainer; mutated in-place.
            device:           torch.device for tensor allocation.
            reduce_tensor_fn: callable(tensor, average=False) → tensor,
                              wraps dist.all_reduce for the current parallelism.
            params_sharded:   True when parameters are actually sharded across
                              ranks (FSDP FULL_SHARD / HYBRID_SHARD).
        """
        pass

    def collect_grad_norms(
        self, model, step_stats, device, reduce_tensor_fn, params_sharded=False
    ):
        """Compute per-component gradient L2 norms and merge into step_stats.

        Must be called while gradients are still available (before
        optimizer.zero_grad).  Under FSDP each rank holds a gradient shard,
        so local squared-sums must be all_reduce'd (sum, not avg) before
        taking the square root.

        Default is a no-op; subclasses override for model-specific groupings.
        """
        pass

    def get_fwd_flops(self, autocast_context, model, batch, global_step):
        """Get the FLOPs of the forward pass."""
        from deepspeed.profiling.flops_profiler import FlopsProfiler

        model.eval()
        prof = None
        raw_model = model.module if hasattr(model, "module") else model
        checkpoint_attrs = {}
        for attr in (
            "use_gradient_checkpointing",
            "use_gradient_checkpointing_offload",
        ):
            if hasattr(raw_model, attr):
                checkpoint_attrs[attr] = getattr(raw_model, attr)
                setattr(raw_model, attr, False)
        try:
            prof = FlopsProfiler(model)
            prof.start_profile()
            with autocast_context(), torch.no_grad():
                _ = self.forward(model, batch, global_step=global_step)
            prof.print_model_profile(profile_step=global_step, detailed=False)
            flops = prof.get_total_flops()
            self.flops = flops
            return flops
        finally:
            for attr, value in checkpoint_attrs.items():
                setattr(raw_model, attr, value)
            if prof is not None:
                prof.end_profile()
            model.train()

    @staticmethod
    def get_output_field(outputs, key, default=None):
        """Retrieve a named field from model outputs (dict or object)."""
        if isinstance(outputs, dict):
            return outputs.get(key, default)
        return getattr(outputs, key, default)

    # ---- console logging fields ----
    def console_fields(self, tokenizer_mixin=None) -> list:
        """Per-step console fields as (stat_key, pretty_label, fmt_spec) triples.

        MetricsLogger renders them in order, skipping any key not present
        in step_stats. The default returns the pre-refactor union of VGA +
        VLA fields; rendering order matches the original ``training_log``
        for bit-for-bit console parity across model families.

        Subclasses MAY override to prune the default union; fields absent
        from step_stats at runtime are already silently skipped by the
        logger, so the default "union" is safe for all adapters.
        """
        fields = [
            ("video_loss", "vid_loss", ".6f"),
            ("action_loss", "act_loss", ".6f"),
            ("action_accuracy", "accuracy", ".4f"),
            ("flow_loss", "flow_loss", ".6f"),
        ]
        if tokenizer_mixin is not None:
            for key in tokenizer_mixin.get_accuracy_keys():
                if key != "action_accuracy":
                    short_key = key.replace("action_accuracy_", "acc_")
                    fields.append((key, short_key, ".4f"))
        fields.extend(
            [
                ("action_grad_norm", "act_gnorm", ".4f"),
                ("video_grad_norm", "vid_gnorm", ".4f"),
            ]
        )
        return fields

    # ---- distribution hooks ----
    def convert_to_mix_precision_hint(
        self,
        model,
        *,
        device,
        use_fsdp: bool | None = None,
        backend: str | None = None,
        log_fn=None,
    ):
        """Prepare *model* dtype + placement before FSDP/DDP wrap.

        FSDP path: cast to fp32 on device; MixedPrecision policy handles the
        bf16 compute cast at forward time.  DDP path: delegate to the model's
        ``convert_to_mix_precision`` method (manual bf16 conversion, since
        DDP has no equivalent of MixedPrecision), then move to device. Pure
        DMuon has its own forward dtype policy and must not rely on FSDP2
        MixedPrecision hooks.

        VGAAdapter overrides this to skip the full-model-to-GPU move under
        FSDP: WAN's ``convert_to_fsdp`` places each sub-module individually
        so only 1/world_size of each block lives on GPU post-wrap.
        """
        _log = log_fn or (lambda _msg: None)
        if backend is None:
            backend = "fsdp2" if use_fsdp else "ddp"

        if backend == "fsdp2":
            model.to(dtype=torch.float32)
            _log("FSDP mode: all params fp32, MixedPrecision handles bf16 compute")
            model.to(device)
        elif backend == "dmuon":
            if hasattr(model, "convert_to_pure_dmuon_mix_precision"):
                model.convert_to_pure_dmuon_mix_precision()
                _log(
                    "Pure DMuon mode: model applied explicit live forward dtype policy"
                )
            else:
                model.to(dtype=torch.float32)
                _log(
                    "Pure DMuon mode: all params fp32; no model-specific live dtype "
                    "policy is defined"
                )
            model.to(device)
        else:
            model.convert_to_mix_precision()
            _log("DDP mode: params converted to bf16 (no MixedPrecision policy)")
            model.to(device)

    # ---- validation ----
    def run_validation(
        self,
        *,
        model,
        val_dataloader,
        rank,
        world_size,
        device,
        autocast_context,
        reduce_fn,
        gather_fn,
        logger,
        global_step,
        output_path,
        tokenizer_mixin=None,
    ):
        """Default validation: forward-pass loss + adapter.collect_output_stats.

        Used by VLA and PI0. VGAAdapter overrides with the full video-gen /
        openloop / action-metric pipeline.

        ``output_path`` is accepted for interface uniformity but not used
        in this default.
        """
        del output_path
        model.eval()
        log_dict = defaultdict(float)
        pbar = tqdm(
            val_dataloader,
            desc="Validating",
            total=len(val_dataloader),
            disable=rank != 0,
        )

        for batch in pbar:
            batch = move_batch_to_device(batch, device)
            with autocast_context():
                outputs = self.forward(model, batch, global_step=0)

            loss = outputs["total_loss"] if "total_loss" in outputs else outputs["loss"]
            log_dict["val_loss"] += reduce_fn(loss.detach()).item()

            val_ce_loss = self.get_output_field(outputs, "cross_entropy_loss")
            if val_ce_loss is not None:
                log_ce_loss = gather_fn(val_ce_loss)
                if log_ce_loss is not None and rank == 0:
                    log_dict["val_cross_entropy_loss"] += log_ce_loss
            val_flow_loss = self.get_output_field(outputs, "flow_loss")
            if val_flow_loss is not None:
                log_flow_loss = gather_fn(val_flow_loss)
                if log_flow_loss is not None and rank == 0:
                    log_dict["val_flow_loss"] += log_flow_loss

            for k, v in collect_channel_loss_stats(
                outputs,
                prefix="val_",
                tokenizer_mixin=tokenizer_mixin,
            ).items():
                log_dict[k] += v

        log_dict = {k: v / len(val_dataloader) for k, v in log_dict.items()}

        if rank == 0:
            if self.logger is not None:
                self.logger.info(f"[FSDP Val] Step {global_step}: {log_dict}")
            if logger is not None and hasattr(logger, "log"):
                logger.log(log_dict, step=global_step)

            val_info = f"[Validation] step {global_step}"
            if "val_loss" in log_dict:
                val_info += f" | val_loss {log_dict['val_loss']:.4f}"
            if "val_cross_entropy_loss" in log_dict:
                val_info += f" | val_ce_loss {log_dict['val_cross_entropy_loss']:.4f}"
            if "val_flow_loss" in log_dict:
                val_info += f" | val_flow_loss {log_dict['val_flow_loss']:.6f}"
            if "val_action_accuracy" in log_dict:
                val_info += f" | val_accuracy {log_dict['val_action_accuracy']:.4f}"
            if tokenizer_mixin is not None:
                for key in tokenizer_mixin.get_accuracy_keys():
                    if key != "action_accuracy":
                        val_key = f"val_{key}"
                        if val_key in log_dict:
                            short_key = key.replace("action_accuracy_", "acc_")
                            val_info += f" | val_{short_key} {log_dict[val_key]:.4f}"
            if self.logger is not None:
                self.logger.info(val_info)

        model.train()
        return log_dict

    def init_validation(self, normalizer_action, normalizer_propri):
        """Optional pre-training-loop setup for validation (e.g. VGA openloop
        dataset construction).  Default is a no-op."""
        del normalizer_action, normalizer_propri

    # ---- prediction / inference ----
    def predict(
        self,
        prediction_type: str,
        *,
        model,
        val_dataloader,
        rank,
        world_size,
        device,
        processor,
        tokenizer_mixin,
        logger,
        current_step,
        max_iteration=None,
        save_dir=None,
        max_samples=None,
    ):
        """Run an inference loop for the given ``prediction_type``.

        Subclasses implement the dispatch table (flow_action / ar_action /
        dllm_action / text / ...). Default raises — only VLAdapter has
        meaningful predict paths today.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not support predict(prediction_type=...)"
        )

    # ---- optimizer-related defaults ----
    @property
    def default_action_lr_keywords(self) -> list[str]:
        """Keyword list used to identify action-expert parameters.

        The trainer splits parameters into two groups when
        ``action_expert_learning_rate`` is set: names containing any of these
        keywords go into the action-LR group. Subclasses override this when
        the architecture uses different naming conventions.
        """
        return [
            "action_preprocessor",
            "moe.experts.1",
            "qkv_proj_experts.1",
            "o_proj_experts.1",
            "input_layernorms.1",
            "post_attention_layernorms.1",
            "model.norms.1",
        ]


def collect_channel_loss_stats(outputs, prefix="", tokenizer_mixin=None):
    """Gather per-dataset channel losses + accuracy metrics via dist.all_reduce.

    Mirrors the legacy fsdp_trainer._collect_channel_loss_stats so training
    and validation stats come out with identical keys.  Returns {} when the
    model output dict lacks ``channel_loss_dict``.
    """
    stats = {}

    def output_field(name):
        return (
            outputs.get(name)
            if isinstance(outputs, dict)
            else getattr(outputs, name, None)
        )

    channel_loss_dict = output_field("channel_loss_dict") or {}
    channel_loss_count_dict = output_field("channel_loss_count_dict") or {}
    if not channel_loss_dict and not (dist.is_available() and dist.is_initialized()):
        return stats

    local_channel_keys = sorted(channel_loss_dict.keys())
    if dist.is_available() and dist.is_initialized():
        gathered_channel_keys = [None] * dist.get_world_size()
        dist.all_gather_object(gathered_channel_keys, local_channel_keys)
        channel_keys = sorted(
            {key for rank_keys in gathered_channel_keys for key in (rank_keys or [])}
        )
    else:
        channel_keys = local_channel_keys

    metric_ref = None
    for value in list(channel_loss_dict.values()) + list(
        channel_loss_count_dict.values()
    ):
        if torch.is_tensor(value):
            metric_ref = value
            break
    if metric_ref is None:
        for name in ("loss", "total_loss", "flow_loss", "cross_entropy_loss"):
            value = output_field(name)
            if torch.is_tensor(value):
                metric_ref = value
                break

    def zero_metric_tensor():
        if metric_ref is None:
            raise RuntimeError(
                "Cannot reduce channel metrics without a tensor device reference "
                "on this rank"
            )
        return torch.zeros((), device=metric_ref.device, dtype=torch.float32)

    def channel_metric_tensor(key, source_dict):
        value = source_dict.get(key)
        if value is None:
            return zero_metric_tensor()
        return value.detach().clone().to(dtype=torch.float32)

    def reduce_metric_tensor(tensor):
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        return tensor

    for dataset_name_i in channel_keys:
        count_tensor = channel_metric_tensor(dataset_name_i, channel_loss_count_dict)
        loss_tensor = channel_metric_tensor(dataset_name_i, channel_loss_dict)
        reduce_metric_tensor(count_tensor)
        reduce_metric_tensor(loss_tensor)
        count_sum = count_tensor.item()
        if count_sum >= 0.5:
            stats[f"{prefix}channel_loss_{dataset_name_i}"] = (
                loss_tensor.item() / count_sum
            )
    if "action_accuracy" in channel_keys and tokenizer_mixin is not None:
        acc_tensor = channel_metric_tensor("action_accuracy", channel_loss_dict)
        reduce_metric_tensor(acc_tensor)
        world_size = dist.get_world_size() if dist.is_initialized() else 1
        stats[f"{prefix}action_accuracy"] = acc_tensor.item() / world_size
        for key in tokenizer_mixin.get_accuracy_keys():
            if key != "action_accuracy" and key in channel_keys:
                rvq_acc_tensor = channel_metric_tensor(key, channel_loss_dict)
                reduce_metric_tensor(rvq_acc_tensor)
                stats[f"{prefix}{key}"] = rvq_acc_tensor.item() / world_size
    return stats
