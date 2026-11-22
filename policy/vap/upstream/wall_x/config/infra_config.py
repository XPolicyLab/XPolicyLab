"""Infrastructure config: distributed runtime, logging, checkpoints, debug, and serving."""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ModelServiceConfig:
    """Model-service push config (used by CkptPusher — trainer pushes new actor
    weights back to serving)."""

    model_service_url: str = ""
    push_max_retries: int = 3


@dataclass
class RLDataServiceConfig:
    """RLDataSource ctor kwargs for RL off-policy ingest (see ADR-0007).

    Collects RLDataSource's 5 kwargs under the yaml ``cfg.serving.data`` block so
    ``train_rl.py::_build_trainer`` can pass them straight to ``RLDataSource()``.
    (Episode physical location comes from the data service's discover ``data_uri``,
    not configured here — the trainer does not couple to storage layout.)"""

    data_service_url: str = ""
    tag: str = ""
    trainer_encoder_version: str = ""
    trainer_pi0_model_hash: str = ""
    trainer_data_pipeline_version: str = ""


@dataclass
class ServingConfig:
    """Unified config for the two serving counterparts the trainer talks to.

    ``model`` = model service (trainer → serving actor-weight push); ``data`` =
    data service (robot → DS upload, trainer ← DS ingest). Both are serving
    components the training process talks to over HTTP, so they are grouped under
    ``serving:``."""

    model: ModelServiceConfig = field(default_factory=ModelServiceConfig)
    data: RLDataServiceConfig = field(default_factory=RLDataServiceConfig)


@dataclass
class ShardedBackendConfig:
    """Shared 1D/2D sharding layout options."""

    sharding_strategy: str = "full_shard"
    hsdp_replicate_size: Optional[int] = None

    @property
    def reshard_after_forward(self) -> bool:
        return self.sharding_strategy not in ("shard_grad_op", "_hybrid_shard_zero2")


@dataclass
class FSDPBackendConfig(ShardedBackendConfig):
    """PyTorch FSDP2-specific distributed backend options."""

    backward_prefetch: str = "backward_pre"
    cpu_offload: bool = False
    use_orig_params: bool = True
    limit_all_gathers: bool = True
    forward_prefetch: bool = False
    sync_module_states: bool = True
    save_policy: str = "full"
    reduce_dtype: str = "bf16"


@dataclass
class DMuonBackendConfig(ShardedBackendConfig):
    """Pure DMuon distributed backend options."""

    sharding_strategy: str = "shard_grad_op"


@dataclass
class DDPBackendConfig:
    """PyTorch DistributedDataParallel backend options."""

    find_unused_parameters: bool = False
    broadcast_buffers: bool = True
    bucket_cap_mb: int = 25


@dataclass
class DistributedConfig:
    # Public distributed backend. DMuon means pure DMuon; FSDP2+DMuon dedicated
    # params is intentionally not a separate backend. DDP is kept as a
    # lightweight debug/small-model backend.
    backend: str = "fsdp2"
    sp_size: int = 1
    fsdp: FSDPBackendConfig = field(default_factory=FSDPBackendConfig)
    dmuon: DMuonBackendConfig = field(default_factory=DMuonBackendConfig)
    ddp: DDPBackendConfig = field(default_factory=DDPBackendConfig)
    # Mixed precision
    use_mixed_precision: bool = True
    bf16: bool = True
    use_amp: bool = False
    use_gradient_checkpointing: bool = False
    use_gradient_checkpointing_offload: bool = False
    # Partial recompute: number of leading transformer layers to checkpoint.
    # None = checkpoint all layers (legacy behavior when use_gradient_checkpointing=True).
    # N in [0, num_layers] = checkpoint only the first N layers, keep the rest's
    # activations resident in HBM. Clamped to [0, num_layers] at apply time.
    # Ignored when use_gradient_checkpointing=False.
    gradient_checkpoint_num_layers: Optional[int] = None
    use_selective_recompute: bool = False
    # DDP fallback
    find_unused_parameters: bool = False
    broadcast_buffers: bool = True
    bucket_cap_mb: int = 25

    @property
    def use_fsdp(self) -> bool:
        """Compatibility property for code paths that require FSDP2 wrappers."""

        return self.backend == "fsdp2"


@dataclass
class Pi0DistributedConfig(DistributedConfig):
    """PI0/PI05-only distributed knobs (split out of the base config so other
    models' schema stays clean — MR !556 review).

    Consumed only by ``pi0_pytorch.convert_to_fsdp`` /
    ``distribution_strategy``; built for ``model_type in {pi0, pi05}`` (see
    ``loader._build_distributed_config``). For other models these keys are
    absent from ``dataclasses.asdict(distributed)``, so ``build_strategy``'s
    ``if k in config`` forward skips them and the strategy uses its defaults.
    """

    # Checkpoint the SigLIP vision encoder (paligemma.vision_tower). True
    # (default) preserves legacy behavior — embed_image goes through
    # torch.utils.checkpoint, SigLIP forward recomputes during backward. False
    # skips ckpt on the vision call site: vision activation stays in HBM
    # (+~1-2 GiB per rank for batched 3-cam bs/gpu=32) and the long
    # `paligemma.vision_tower` recompute block disappears from the backward
    # timeline. Independent of gradient_checkpoint_num_layers (which only
    # controls the 18-layer joint transformer). Ignored when
    # use_gradient_checkpointing=False.
    gradient_checkpoint_vision_tower: bool = True
    # 把 root 上的 non-layer module（vision_tower / embed_tokens /
    # multi_modal_projector / suffix adapter / final norms）从 FSDP2 nested
    # shard 中拆出走 DDP 风格（params + grad 全副本，每 step backward 后
    # all_reduce 同步 grad）。motivation: 这些模块的 unshard all_gather 在
    # step 开头占 nccl schedule 大头，去掉换 DDP 通信简化 timeline。代价：
    # 每 rank +~1.8 GB params bf16 + ~11 GB AdamW 状态（无 ZeRO-1 时）。
    replicate_non_layer_modules: bool = False
    # 在 vision_tower 的 FSDP pre_forward 里显式 prefetch joint_layer[0]
    # 的 all_gather。默认 FSDP2 prefetch 是 i→i+1 chain，joint_layer[0] 没有
    # 上游帮它 prefetch，必须等进入它自己的 forward 才发 all_gather → 卡在
    # vision_tower 之后。开启后 joint_layer[0] all_gather 在 copy stream 上
    # 跟 vision_tower compute 并行，timeline 上的"vision_tower 完 → 等
    # joint_layer[0] all_gather"那段 gap 消失。HBM 代价：joint_layer[0] 的
    # unsharded params 多驻留 (vision_tower forward 时段)，~30 MB bf16 量级
    # 可忽略。要求 vision_tower 走 FSDP shard 路径（与 replicate_non_layer_modules
    # 互斥；后者为 True 时此 toggle no-op）。
    fsdp_forward_prefetch_joint0_with_vision: bool = True


@dataclass
class LoggingConfig:
    log_name: str = "exp"
    log_project: str = "wallx"
    log_entity: Optional[str] = None
    use_wandb: bool = True
    wandb_offline: bool = False
    log_interval: int = 1
    save_interval: int = 1000
    val_interval: int = 4000
    epoch_save_interval: int = 1
    enable_validation: bool = True
    gc_interval_steps: int = 1000
    ignore_until_interval: int = 0
    # 前几步 step_time / samples_per_sec 不写 wandb：cuDNN benchmark warm-up、
    # dataloader 首次预取、lazy init 让 step 0~N 远比稳态慢，写进去会把 y 轴拉爆。
    # 控制台 line 仍然显示原值，便于现场看到 warm-up 发生。
    perf_log_skip_steps: int = 5
    param_norm_interval: int = 100
    # Rolling-window for smoothing per-step training metrics displayed on the
    # console (and reused by tqdm). 1 = raw per-step (historical behavior).
    # 10 = DZ-style 10-step rolling average — diffusion losses are dominated
    # by timestep-sampling noise per step; the smoothing only changes display,
    # not training. Independent of log_interval (which buffers for wandb).
    loss_log_smooth_window: int = 1


@dataclass
class CheckpointConfig:
    """Checkpoint save and resume options.

    ``resume_from`` is shorthand for setting every component-specific resume
    path to the same checkpoint. Component-specific fields take precedence.
    """

    save_path: str = "./ckpt"
    validate_first: bool = False
    # Shorthand path for all components.
    resume_from: Optional[str] = None
    # Component-specific resume paths.
    resume_model: Optional[str] = None
    resume_optimizer: Optional[str] = None
    resume_scheduler: Optional[str] = None
    resume_ema: Optional[str] = None
    resume_rng: Optional[str] = None
    resume_data: Optional[str] = None
    resume_step: Optional[str] = None
    # When True, passes try_harder=True to reshape_compatible_state_dict so that
    # action-preprocessor weights are padded/sliced to match the current model shape.
    try_harder: bool = False
    # Require an exact key/shape match for single-file model-only warm starts.
    # Full directory resume has its own distributed checkpoint validation.
    strict_model_load: bool = False
    # Explicit model parameters that may be absent from a single-file warm-start
    # checkpoint. The loader fills only these keys from the target model's own
    # initialization before performing an otherwise-strict load. Keep this list
    # empty for architecture-exact warm starts.
    model_init_allowed_missing_keys: list[str] = field(default_factory=list)
    # CkptPusher: path to the JSON log of already-pushed weight versions.
    weight_version_persistence_path: str = ""

    def get_resume_path(self, component: str) -> Optional[str]:
        """Return the resume path for one checkpoint component."""
        specific = getattr(self, f"resume_{component}", None)
        if specific is not None:
            return specific
        return self.resume_from


@dataclass
class DebugConfig:
    profile: bool = False
    profile_save_path: str = "./profile"
    profile_wait_iters: int = 1
    profile_warmup_iters: int = 1
    profile_active_iters: int = 3
    # Number of profile cycles to record. Default 1 = capture once and stop.
    # torch.profiler.schedule's own default is 0 = repeat forever, which was
    # the silent behavior when this field was missing (the yaml key was being
    # dropped by the typed loader). Each cycle dumps a ~500MB trace per worker
    # and adds a multi-second stall, so 1 is the right default for ad-hoc runs.
    profile_repeat: int = 1
    # When True, traces are gzip-compressed by torch.profiler's
    # tensorboard_trace_handler. Output files become *.pt.trace.json.gz
    # (~10× smaller). Perfetto / chrome://tracing accept .gz directly.
    profile_use_gzip: bool = False
    # torch.profiler.profile() kwargs. Defaults tuned for "post-hoc analysis
    # via visualize_wallx" rather than the legacy "tensorboard + py-spy stack"
    # workflow:
    #   - with_stack=False saves a lot of trace size and CPU overhead BUT it
    #     drops the `nn.Module: <ClassName>` annotations (which `with_stack`
    #     inferred indirectly from python frames). wall-x compensates by
    #     emitting explicit record_function zones — see wall_x/utils/profile_zones.py.
    #   - with_modules=True has no effect in eager mode (PyTorch limits it
    #     to TorchScript), so the explicit-zone approach is the only path.
    #   - record_shapes keeps GEMM input shapes (visualize uses it for FLOPS)
    #   - profile_memory off by default — overlaps with enable_memory_snapshot
    profile_with_stack: bool = False
    profile_with_modules: bool = False  # no-op in eager mode — keep off
    profile_record_shapes: bool = True
    profile_memory: bool = False
    # When True, only rank 0 records a trace. Avoids dumping N copies of the
    # same timeline on multi-GPU runs (16-rank × ~80MB each = 1.3GB per cycle).
    profile_only_rank0: bool = True
    # CUDA memory snapshot. When True, rank 0 records the allocator history
    # starting at step 0 and dumps a .pickle at step `memory_snapshot_until_step`.
    # Output drops into `memory_snapshot_dir/memory_snapshot.pickle`; open it at
    # https://pytorch.org/memory_viz to see per-allocation timeline + peak +
    # which Python frame owned each block.
    enable_memory_snapshot: bool = False
    memory_snapshot_dir: str = "./memory_snapshot"
    # Recording starts in trainer __init__ (covers model load + FSDP wrap +
    # optimizer creation); dump fires right after global_step == until_step.
    # 3 is enough to see steady-state per-step alloc pattern without making
    # the .pickle huge.
    memory_snapshot_until_step: int = 3
    memory_snapshot_max_entries: int = 100000
    show_time_details: bool = False
    visualize_sample: bool = False
    save_debug_batch_path: Optional[str] = None
    nvtx: bool = False
    # Formula MFU uses the local FLOPs estimate and measured step time.
    enable_mfu: bool = False
    # DeepSpeed FlopsProfiler is expensive and runs an extra forward at step 0.
    enable_mfu_profile: bool = False
