import torch

from wall_x.trainer.adapters.base_adapter import load_x2r_data_config
from wall_x.trainer.adapters.base_adapter import ModelAdapter
from wall_x.trainer.metrics import DistributedNormAccumulator
from wall_x.trainer.metrics.mfu import resolve_gpu_specs
from wall_x.trainer.metrics.mfu import vga_mfu


class VGAAdapter(ModelAdapter):
    """Adapter for VGA (Video Generation Action) models."""

    _ACTION_KEYWORDS = (
        "action_transformer",
        "action_conditioner",
        "action_encoder",
        "action_decoder",
    )
    _VIDEO_KEYWORDS = ("dit", "vace", "motion_controller")

    @property
    def default_action_lr_keywords(self) -> list[str]:
        return list(self._ACTION_KEYWORDS)

    @staticmethod
    def _param_name_matches_keyword_segment(name, keywords):
        """True if dotted *name* has a segment exactly equal to some keyword."""
        segments = name.split(".")
        return any(kw in segments for kw in keywords)

    def __init__(self, *, cfg=None, logger=None, model_type=None):
        super().__init__(cfg=cfg, logger=logger, model_type=model_type)
        self._mfu_batch_info = None
        self._timers_ready = False  # set True once CUDA event hooks are installed
        self._start_ev = {}  # {comp_name: torch.cuda.Event}
        self._end_ev = {}  # {comp_name: torch.cuda.Event}
        self._timing_hooks = []  # registered hook handles (for future removal)
        self._multi_call_evs = {}  # {comp_name: [(start_ev, end_ev), ...]}
        self._component_timing_active = False
        self._enable_module_mfu = False
        self._enable_gemm_mfu = False
        self._last_post_train_phase = None

    # ---- distribution hook ----
    def convert_to_mix_precision_hint(
        self,
        model,
        *,
        device,
        use_fsdp: bool | None = None,
        backend: str | None = None,
        log_fn=None,
    ):
        """WAN FSDP: do NOT move the full model to GPU here.

        ``convert_to_fsdp`` places each sub-module individually before
        FSDP-wrapping it, so only 1/world_size of each sub-module's params
        end up on GPU. Moving the full model first would cause peak OOM on
        large models (e.g. 14B).

        DDP path falls back to the base implementation.
        """
        _log = log_fn or (lambda _msg: None)
        if backend is None:
            backend = "fsdp2" if use_fsdp else "ddp"

        if backend == "fsdp2":
            _log(
                "FSDP mode (WAN): pure bf16, convert_to_fsdp handles "
                "per-module device placement to avoid peak OOM"
            )
            return
        super().convert_to_mix_precision_hint(
            model,
            device=device,
            use_fsdp=use_fsdp,
            backend=backend,
            log_fn=log_fn,
        )

    # ---- validation ----
    def init_validation(self, normalizer_action, normalizer_propri):
        """Prepare VGA-specific objects needed by run_validation()."""
        import argparse

        model_cfg = self.cfg.model
        val_config = model_cfg.validation or {}

        self._val_inference_args = argparse.Namespace(
            num_frames=val_config.get("num_frames"),
            height=val_config.get("height"),
            width=val_config.get("width"),
            num_inference_steps=val_config.get("num_inference_steps"),
            cfg_scale=val_config.get("cfg_scale"),
            negative_prompt=val_config.get("negative_prompt", ""),
            seed=val_config.get("seed"),
            physical_view_ids=val_config.get("physical_view_ids", None),
        )
        self._val_num_samples = val_config.get("num_val_samples", 1)
        policy = model_cfg.policy or {}
        self._val_action_len = policy.get("action_len", 40)
        self._val_dataset_openloop = None

        # openloop needs an X2RDataConfig; None (lerobot / x2robot_v2) skips it
        val_data_config = load_x2r_data_config(self.cfg)
        self._val_dataload_config = val_data_config
        if val_data_config is not None and not val_data_config.event_level:
            from wall_x.model.vga.openloop_visualization import OpenLoopDataset

            openloop_cfg = val_config.get("openloop", {})
            episodes_path = openloop_cfg.get("episodes_path")
            if episodes_path:
                self._val_dataset_openloop = OpenLoopDataset(
                    val_data_config,
                    None,
                    None,
                    normalizer_action,
                    normalizer_propri,
                    episodes_path,
                    openloop_cfg.get("episode_index", 0),
                )

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
        """VGA video-gen + action-metric + openloop validation.

        The 4 legacy vga_*.py utility modules still take an accelerate-style
        ``accelerator`` object; we build a minimal shim here so those modules
        don't need to change in this step. A follow-up PR will port them to
        (rank, world_size, device) directly and drop this shim.
        """
        del reduce_fn, gather_fn, tokenizer_mixin  # not used for VGA path
        raise NotImplementedError(
            "VGA training validation is not included in the eval-only adapter."
        )
        from wall_x.trainer.adapters.vga_validation import (
            validate_and_log,
            validate_and_log_openloop,
            validate_video,
            validate_video_action,
        )

        if self.normalizer_action is None:
            raise RuntimeError("VGAAdapter.normalizer_action is unset")

        accelerator = type(
            "_Acc",
            (),
            {
                "process_index": rank,
                "num_processes": world_size,
                "device": device,
                "is_main_process": rank == 0,
            },
        )()

        model.eval()
        try:
            with autocast_context():
                _inner = model.module if hasattr(model, "module") else model
                common_kwargs = dict(
                    model=_inner,
                    dataloader_val=val_dataloader,
                    dataset_val_openloop=self._val_dataset_openloop,
                    accelerator=accelerator,
                    wandb_run=logger,
                    global_step=global_step,
                    num_samples=self._val_num_samples,
                    inference_args=self._val_inference_args,
                    output_path=output_path,
                    normalizer_action=self.normalizer_action,
                    interval=self._val_action_len,
                )
                dof = self._val_dataload_config.dof_config or {}
                agent_pos = self._val_dataload_config.agent_pos_config or {}

                if getattr(_inner, "enable_action", False) and not getattr(
                    _inner, "action_as_condition", False
                ):
                    validate_and_log(
                        **common_kwargs,
                        dof_config=dof,
                        agent_pos_config=agent_pos,
                    )
                    if self._val_dataset_openloop is not None:
                        validate_and_log_openloop(
                            **common_kwargs,
                            dof_config=dof,
                            agent_pos_config=agent_pos,
                        )
                elif getattr(_inner, "action_as_condition", False):
                    validate_video_action(
                        **common_kwargs,
                        dof_config=dof,
                        agent_pos_config=agent_pos,
                    )
                else:
                    validate_video(
                        **common_kwargs,
                        event_level=self._val_dataload_config.event_level,
                    )
        finally:
            model.train()

    # ---- processor ----
    def load_processor(self, action_statistic_dof):
        from wall_x.data import data_backend

        backend = data_backend()
        # backends without a VGDataProcessor (lerobot) get a no-op processor dict
        if not backend.supports("make_vg_processor"):
            return {
                "processor": None,
                "data_config": None,
                "tokenizer_mixin": None,
                "train_action_tokenizer": None,
                "val_action_tokenizer": None,
                "action_mapper": None,
                "num_added_tokens": None,
            }

        data_config = load_x2r_data_config(self.cfg)

        normalizer_action = getattr(self, "normalizer_action", None)
        normalizer_propri = getattr(self, "normalizer_propri", None)
        if normalizer_action is None or normalizer_propri is None:
            from wall_x.model.core.action.normalizer import create_normalizers

            norm_cfg = self.cfg.data.normalizer_config or {}
            normalizer_action, normalizer_propri, _ = create_normalizers(
                {
                    "dof_config": data_config["dof_config"],
                    "agent_pos_config": data_config["agent_pos_config"],
                    "normalizer_config": norm_cfg,
                },
                action_statistic_dof=action_statistic_dof,
            )

        processor = backend.make_vg_processor(
            data_config,
            normalizer_action=normalizer_action,
            normalizer_propri=normalizer_propri,
        )

        return {
            "processor": processor,
            "data_config": data_config,
            "tokenizer_mixin": None,
            "train_action_tokenizer": None,
            "val_action_tokenizer": None,
            "action_mapper": None,
            "num_added_tokens": None,
        }

    # ---- model ----
    def build_model_config(self):
        return self.cfg.model

    def _init_scheduler_training_mode(self, model):
        if hasattr(model, "pipe") and hasattr(model.pipe, "scheduler"):
            model.pipe.scheduler.set_timesteps(1000, training=True)
            if getattr(model.pipe, "separate_flow_matching", False):
                model.pipe.action_scheduler.set_timesteps(1000, training=True)
        return model

    def create_model(self, processor, tokenizer_mixin, model_config):
        self._enable_module_mfu = self.cfg.debug.enable_mfu

        from wall_x.model.vga.modeling.modeling_wan_act import WanTrainingModule

        d = self.cfg.data
        model = WanTrainingModule(
            self.cfg.model,
            distributed_config=self.cfg.distributed,
            image_history_length=d.image_history_length,
            obs_history_stride=d.obs_history_stride,
            event_level=d.event_level,
        )
        return self._init_scheduler_training_mode(model)

    def load_weights(self, model, normalizer_action, normalizer_propri, **kwargs):
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        elif (
            hasattr(model, "get_input_embeddings")
            and model.get_input_embeddings() is not None
        ):

            def _make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)

            model.get_input_embeddings().register_forward_hook(
                _make_inputs_require_grad
            )

        return model

    # ---- FSDP wrapping ----
    def get_transformer_layer_cls(self):
        return None

    # ---- sequence parallelism ----
    def sp_sharded_param_prefixes(self) -> tuple[str, ...]:
        # pipe.dit.blocks.* run on the Ulysses sequence shard; head, embeddings
        # and action_transformer run on the gathered full sequence.
        return ("pipe.dit.blocks.",)

    # ---- dataset ----
    def load_dataset(self, data_config, processor, rank, world_size, **kwargs):
        from wall_x.data import data_backend

        backend = data_backend()
        if backend.supports("build_vga_train_data"):
            resume_state = {
                key: kwargs[key]
                for key in ("sampler_state_path", "dataset_state")
                if kwargs.get(key) is not None
            }
            return backend.build_vga_train_data(
                self.cfg,
                rank,
                world_size,
                self.logger,
                resume_state=resume_state or None,
            )

        dataset = backend.make_vg_dynamic_dataset(
            data_config=data_config,
            processor=processor,
            rank=rank,
            world_size=world_size,
        )
        train_num = dataset.global_train_iters.value
        train_dataloader = dataset.get_train_dataloader()

        if rank == 0 and self.logger is not None:
            batch_size = data_config["batch_size"]
            global_batch_size = batch_size * world_size
            train_frames = train_num * batch_size * world_size
            self.logger.info(
                "\n"
                + "=" * 50
                + "\nData Loading Configuration:\n"
                + f"✦ RANK: {rank}\n"
                + f"✦ WORLD SIZE: {world_size}\n"
                + f"✦ ORIGINAL BATCH SIZE: {global_batch_size}\n"
                + f"✦ BATCH SIZE PER DEVICE: {global_batch_size} // {world_size} = {batch_size}\n"
                + "\nProcessing Configuration:\n"
                + f"✦ ACTION HORIZON: {data_config['action_horizon']}\n"
                + f"✦ ACTION HISTORY LENGTH: {data_config['action_history_length']}\n"
                + f"✦ IMAGE HORIZON: {data_config['image_horizon']}\n"
                + f"✦ IMAGE HISTORY LENGTH: {data_config['image_history_length']}\n"
                + f"✦ LEFT PADDING: {data_config['left_padding']}\n"
                + f"✦ RIGHT PADDING: {data_config['right_padding']}\n"
                + "\nDataset:\n"
                + f"✦ TOTAL FRAMES: {train_frames}\n"
                + f"✦ TOTAL BATCHES: {train_num}\n"
                + "=" * 50
            )

        return dataset, train_dataloader, train_num

    # ---- forward / loss ----
    def forward(self, model, batch, **kwargs):
        if (
            not self._timers_ready
            and self._enable_module_mfu
            and torch.cuda.is_available()
        ):
            try:
                _m = model.module if hasattr(model, "module") else model
                self._setup_component_timers(_m)
            except Exception:
                self._timers_ready = True  # suppress further attempts

        global_step = kwargs.get("global_step", 0)
        self._record_mfu_batch_info(batch)
        self._component_timing_active = bool(self._enable_module_mfu)
        try:
            outputs = model(batch, global_step)
        finally:
            self._component_timing_active = False
        self._record_mfu_batch_info(batch)
        self._last_post_train_phase = (
            outputs.get("post_train_phase") if isinstance(outputs, dict) else None
        )
        return outputs

    def extract_loss(self, outputs):
        return outputs["total_loss"]

    @staticmethod
    def log_attention_implementation(logger, model):
        pass

    def collect_output_stats(
        self,
        outputs,
        step_stats,
        reduce_tensor_fn,
        true_gather_fn,
        tokenizer_mixin=None,
    ):
        """Collect VGA-specific sub-losses from the model output dict."""
        if not isinstance(outputs, dict):
            return

        phase = outputs.get("post_train_phase")
        if phase == "video":
            keys = ("video_loss",)
        elif phase == "action":
            keys = (
                "action_loss",
                "dct_loss",
                "second_head_loss",
                "constraint_total",
            )
        else:
            keys = (
                "video_loss",
                "action_loss",
                "dct_loss",
                "second_head_loss",
                "constraint_total",
            )

        for key in keys:
            val = outputs.get(key)
            if val is not None and isinstance(val, torch.Tensor):
                step_stats[key] = reduce_tensor_fn(val.detach()).item()

    @staticmethod
    def _iter_dmuon_owned(model, attr: str):
        """Yield ``(fqn, tensor)`` for DMuon dedicated params on the owner rank.

        DMuon's ``dedicate_params`` swaps the live module attribute for a
        0-numel placeholder (see dmuon/param.py:95-121), so
        ``named_parameters()`` skips these entries entirely. The actual
        weight lives on ``dp._owned_data`` (attr='_owned_data') and the
        reduced gradient on ``dp._reduced_grad`` (attr='_reduced_grad'),
        both only populated on the owner rank. Without this accounting,
        action/video norms on FSDP2+DMuon under-count the big attention /
        FFN projection weights that are precisely the ones DMuon dedicates.

        Non-DMuon models yield nothing; non-owner ranks yield nothing. The
        subsequent all-reduce sums one rank's contribution per dedicated
        param → correct global norm.

        For ``attr='_reduced_grad'`` this also waits on pending DMuon
        async reduces — the attribute is None until ``wait_all_reduces``
        runs, and we may be called before the optimizer step (which
        otherwise triggers it).
        """
        from wall_x.trainer.optimizer.dmuon import is_dmuon_model

        if not is_dmuon_model(model):
            return
        # Imports below need the external dmuon package; if the model is
        # dmuon-wrapped but the package fails to import, surface that loudly
        # rather than silently skipping accounting for dedicated params.
        from dmuon.utils import get_dedicated_params, wait_all_reduces

        if attr == "_reduced_grad":
            wait_all_reduces(model)
        module_to_fqn = {id(mod): name for name, mod in model.named_modules()}
        for dp in get_dedicated_params(model):
            if not dp.is_owner:
                continue
            tensor = getattr(dp, attr, None)
            if tensor is None or tensor.numel() == 0:
                continue
            prefix = module_to_fqn.get(id(dp.module), "")
            fqn = f"{prefix}.{dp.param_name}" if prefix else dp.param_name
            yield fqn, tensor.detach()

    def _component_for_name(self, name, include_other=False):
        """Return the metric group for a parameter FQN."""
        if self._param_name_matches_keyword_segment(name, self._ACTION_KEYWORDS):
            return "action"
        if self._param_name_matches_keyword_segment(name, self._VIDEO_KEYWORDS):
            return "video"
        if include_other:
            return "other"
        return None

    def collect_grad_norms(
        self, model, step_stats, device, reduce_tensor_fn, params_sharded=False
    ):
        """Compute action / video gradient L2 norms for VGA models.

        Groups:
          "action" → action_transformer, action_conditioner, action_encoder,
                     action_decoder
          "video"  → dit, vace, motion_controller
        """
        del params_sharded
        norms = DistributedNormAccumulator(
            device=device, reduce_tensor_fn=reduce_tensor_fn
        )
        for name, param in model.named_parameters():
            g = param.grad
            if g is None or g.numel() == 0:
                continue
            norms.add(self._component_for_name(name), g)

        # DMuon dedicated params: owner holds the reduced grad (full shape);
        # non-owners contribute 0 through the owner-only accumulator path.
        for fqn, g in self._iter_dmuon_owned(model, "_reduced_grad"):
            norms.add_owner_only(self._component_for_name(fqn), g)

        phase = self._last_post_train_phase
        if phase != "action":
            step_stats["video_grad_norm"] = norms.norm("video").item()
        if phase != "video":
            step_stats["action_grad_norm"] = norms.norm("action").item()

    def collect_param_norms(
        self, model, step_stats, device, reduce_tensor_fn, params_sharded=False
    ):
        """Compute per-component parameter L2 norms for VGA models.

        Groups (same as collect_grad_norms):
          "action" → action_transformer, action_conditioner, action_encoder,
                     action_decoder
          "video"  → dit, vace, motion_controller
          all      → total
        """
        del params_sharded
        norms = DistributedNormAccumulator(
            device=device, reduce_tensor_fn=reduce_tensor_fn
        )
        for name, param in model.named_parameters():
            if not param.requires_grad or param.numel() == 0:
                continue
            norms.add(self._component_for_name(name, include_other=True), param)

        # DMuon dedicated params: owner holds the full fp32 master weight on
        # `_owned_data`; non-owners contribute 0 through the owner-only path.
        for fqn, p in self._iter_dmuon_owned(model, "_owned_data"):
            norms.add_owner_only(self._component_for_name(fqn, include_other=True), p)

        action_sq = norms.squared_norm("action")
        video_sq = norms.squared_norm("video")
        other_sq = norms.squared_norm("other")
        total_sq = action_sq + video_sq + other_sq
        step_stats["action_param_norm"] = action_sq.sqrt().item()
        step_stats["video_param_norm"] = video_sq.sqrt().item()
        step_stats["total_param_norm"] = total_sq.sqrt().item()

    # ---- MFU: formulas live in wall_x.trainer.metrics.mfu.vga_mfu; the adapter
    # keeps runtime state (batch-geometry recording, CUDA-event timers) and
    # diagnostics logging. ----

    def _setup_component_timers(self, model):
        """Register CUDA-event timing for each sub-component's forward time."""
        pipe = model.pipe if hasattr(model, "pipe") else model.module.pipe

        self._timing_hooks = getattr(self, "_timing_hooks", [])
        self._start_ev = {}
        self._end_ev = {}
        self._multi_call_evs = {}

        def _record_if_active(event):
            if getattr(self, "_component_timing_active", False):
                event.record()

        text_encoder = getattr(pipe, "text_encoder", None)
        if text_encoder is not None:
            se = torch.cuda.Event(enable_timing=True)
            ee = torch.cuda.Event(enable_timing=True)
            self._start_ev["text_encoder"] = se
            self._end_ev["text_encoder"] = ee
            self._timing_hooks.append(
                text_encoder.register_forward_pre_hook(
                    lambda m, i, _se=se: _record_if_active(_se)
                )
            )
            self._timing_hooks.append(
                text_encoder.register_forward_hook(
                    lambda m, i, o, _ee=ee: _record_if_active(_ee)
                )
            )

        vae = getattr(pipe, "vae", None)
        if vae is not None and hasattr(vae, "encode"):
            self._multi_call_evs["vae"] = []
            orig_encode = vae.encode

            def _timed_encode(
                *args, _evs=self._multi_call_evs, _orig=orig_encode, **kwargs
            ):
                if not getattr(self, "_component_timing_active", False):
                    return _orig(*args, **kwargs)
                se = torch.cuda.Event(enable_timing=True)
                ee = torch.cuda.Event(enable_timing=True)
                se.record()
                result = _orig(*args, **kwargs)
                ee.record()
                _evs["vae"].append((se, ee))
                return result

            vae.encode = _timed_encode

        image_encoder = getattr(pipe, "image_encoder", None)
        if image_encoder is not None and hasattr(image_encoder, "encode_image"):
            self._multi_call_evs["image_encoder"] = []
            orig_encode_image = image_encoder.encode_image

            def _timed_encode_image(
                *args, _evs=self._multi_call_evs, _orig=orig_encode_image, **kwargs
            ):
                if not getattr(self, "_component_timing_active", False):
                    return _orig(*args, **kwargs)
                se = torch.cuda.Event(enable_timing=True)
                ee = torch.cuda.Event(enable_timing=True)
                se.record()
                result = _orig(*args, **kwargs)
                ee.record()
                _evs["image_encoder"].append((se, ee))
                return result

            image_encoder.encode_image = _timed_encode_image

        dit = getattr(pipe, "dit", None)
        if dit is not None and hasattr(dit, "blocks") and len(dit.blocks) > 0:
            se = torch.cuda.Event(enable_timing=True)
            ee = torch.cuda.Event(enable_timing=True)
            self._start_ev["dit"] = se
            self._end_ev["dit"] = ee
            self._timing_hooks.append(
                dit.blocks[0].register_forward_pre_hook(
                    lambda m, i, _se=se: _record_if_active(_se)
                )
            )
            self._timing_hooks.append(
                dit.blocks[-1].register_forward_hook(
                    lambda m, i, o, _ee=ee: _record_if_active(_ee)
                )
            )

        action_transformer = getattr(pipe, "action_transformer", None)
        if action_transformer is not None:
            se = torch.cuda.Event(enable_timing=True)
            ee = torch.cuda.Event(enable_timing=True)
            self._start_ev["action_transformer"] = se
            self._end_ev["action_transformer"] = ee
            self._timing_hooks.append(
                action_transformer.register_forward_pre_hook(
                    lambda m, i, _se=se: _record_if_active(_se)
                )
            )
            self._timing_hooks.append(
                action_transformer.register_forward_hook(
                    lambda m, i, o, _ee=ee: _record_if_active(_ee)
                )
            )

        self._timers_ready = True

    def _read_component_times(self):
        """Read CUDA event elapsed times for the current step.

        Returns:
            (times, call_counts) where
              times:       {name: total_seconds}
              call_counts: {name: int}  — only for multi-call components
        """
        times = {}
        call_counts = {}
        for name in self._start_ev:
            try:
                elapsed_ms = self._start_ev[name].elapsed_time(self._end_ev[name])
                if elapsed_ms > 0:
                    times[name] = elapsed_ms * 1e-3
            except Exception:
                pass
        for name, ev_list in self._multi_call_evs.items():
            total_ms = 0.0
            n_calls = 0
            for se, ee in ev_list:
                try:
                    elapsed_ms = se.elapsed_time(ee)
                    if elapsed_ms > 0:
                        total_ms += elapsed_ms
                        n_calls += 1
                except Exception:
                    pass
            if total_ms > 0:
                times[name] = total_ms * 1e-3
                call_counts[name] = n_calls
            ev_list.clear()
        return times, call_counts

    def _record_mfu_batch_info(self, batch):
        """Remember the actual video/latent geometry used by the latest VGA forward."""
        if not isinstance(batch, dict):
            return
        videos = batch.get("video")
        info = vga_mfu.batch_info_from_videos(videos)
        if info is None:
            info = vga_mfu.batch_info_from_raw_batch(batch, self.cfg)
        if info is None:
            return
        if info != self._mfu_batch_info:
            self._mfu_batch_info = info

    def _compute_flops_per_step(self, model):
        """Delegate to the pure formula module with this adapter's runtime state."""
        return vga_mfu.compute_flops_per_step(
            model, self.cfg, self._mfu_batch_info, logger=self.logger
        )

    # ----------------------------------------------------------------
    #  Per-GEMM CUDA-event timing
    # ----------------------------------------------------------------

    def _setup_gemm_timers(self, model):
        """Register CUDA-event hooks on every GEMM-producing module."""
        if getattr(self, "_gemm_timers_ready", False):
            return
        self._gemm_timers_ready = True

        from collections import defaultdict

        pipe = model.pipe if hasattr(model, "pipe") else model.module.pipe

        self._gemm_fwd_ev = defaultdict(list)
        self._gemm_bwd_ev = defaultdict(list)
        self._gemm_timer_hooks = []
        self._gemm_attn_children = {}

        trainable_ids = {id(p) for p in pipe.dit.parameters() if p.requires_grad}

        def _is_trainable(module):
            return any(id(p) in trainable_ids for p in module.parameters())

        def _hook(key, module, trainable=None):
            if trainable is None:
                trainable = _is_trainable(module)

            se = torch.cuda.Event(enable_timing=True)
            ee = torch.cuda.Event(enable_timing=True)
            self._gemm_fwd_ev[key].append((se, ee))

            self._gemm_timer_hooks.append(
                module.register_forward_pre_hook(lambda m, i, _s=se: _s.record())
            )
            self._gemm_timer_hooks.append(
                module.register_forward_hook(lambda m, i, o, _e=ee: _e.record())
            )

            if trainable:
                bse = torch.cuda.Event(enable_timing=True)
                bee = torch.cuda.Event(enable_timing=True)
                self._gemm_bwd_ev[key].append((bse, bee))

                self._gemm_timer_hooks.append(
                    module.register_full_backward_pre_hook(
                        lambda m, go, _s=bse: _s.record()
                    )
                )
                self._gemm_timer_hooks.append(
                    module.register_full_backward_hook(
                        lambda m, gi, go, _e=bee: _e.record()
                    )
                )

        def _hook_attn(total_key, attn_module, child_keys, trainable=None):
            _hook(total_key, attn_module, trainable=trainable)
            self._gemm_attn_children[total_key] = list(child_keys)

        dit = pipe.dit
        for block in dit.blocks:
            _hook("dit.self_attn.q", block.self_attn.q)
            _hook("dit.self_attn.k", block.self_attn.k)
            _hook("dit.self_attn.v", block.self_attn.v)
            _hook("dit.self_attn.o", block.self_attn.o)
            _hook_attn(
                "dit.self_attn._total",
                block.self_attn,
                [
                    "dit.self_attn.q",
                    "dit.self_attn.k",
                    "dit.self_attn.v",
                    "dit.self_attn.o",
                ],
            )

            if hasattr(block, "view_attn"):
                _hook("dit.view_attn.q", block.view_attn.q)
                _hook("dit.view_attn.k", block.view_attn.k)
                _hook("dit.view_attn.v", block.view_attn.v)
                _hook("dit.view_attn.o", block.view_attn.o)
                _hook("dit.view_attn.proj", block.view_projector)
                _hook_attn(
                    "dit.view_attn._total",
                    block.view_attn,
                    [
                        "dit.view_attn.q",
                        "dit.view_attn.k",
                        "dit.view_attn.v",
                        "dit.view_attn.o",
                    ],
                )

            _hook("dit.cross_attn.q", block.cross_attn.q)
            _hook("dit.cross_attn.k_text", block.cross_attn.k)
            _hook("dit.cross_attn.v_text", block.cross_attn.v)
            _hook("dit.cross_attn.o", block.cross_attn.o)
            cross_children = [
                "dit.cross_attn.q",
                "dit.cross_attn.k_text",
                "dit.cross_attn.v_text",
                "dit.cross_attn.o",
            ]
            if hasattr(block.cross_attn, "k_img"):
                _hook("dit.cross_attn.k_img", block.cross_attn.k_img)
                _hook("dit.cross_attn.v_img", block.cross_attn.v_img)
                cross_children += ["dit.cross_attn.k_img", "dit.cross_attn.v_img"]
            _hook_attn(
                "dit.cross_attn._total",
                block.cross_attn,
                cross_children,
            )

            _hook("dit.ffn.up", block.ffn[0])
            _hook("dit.ffn.down", block.ffn[2])

        at = getattr(pipe, "action_transformer", None)
        if at is not None and hasattr(at, "blocks"):
            for blk in at.blocks:
                _hook("act.self_attn.q", blk.self_attn.q, trainable=False)
                _hook("act.self_attn.k", blk.self_attn.k, trainable=False)
                _hook("act.self_attn.v", blk.self_attn.v, trainable=False)
                _hook("act.self_attn.o", blk.self_attn.o, trainable=False)
                _hook_attn(
                    "act.self_attn._total",
                    blk.self_attn,
                    [
                        "act.self_attn.q",
                        "act.self_attn.k",
                        "act.self_attn.v",
                        "act.self_attn.o",
                    ],
                    trainable=False,
                )

                _hook("act.cross_attn.q", blk.cross_attn.q, trainable=False)
                _hook("act.cross_attn.k", blk.cross_attn.k, trainable=False)
                _hook("act.cross_attn.v", blk.cross_attn.v, trainable=False)
                _hook("act.cross_attn.o", blk.cross_attn.o, trainable=False)
                _hook_attn(
                    "act.cross_attn._total",
                    blk.cross_attn,
                    [
                        "act.cross_attn.q",
                        "act.cross_attn.k",
                        "act.cross_attn.v",
                        "act.cross_attn.o",
                    ],
                    trainable=False,
                )

                _hook("act.ffn.up", blk.ffn[0], trainable=False)
                _hook("act.ffn.down", blk.ffn[2], trainable=False)

        te = getattr(pipe, "text_encoder", None)
        if te is not None and hasattr(te, "blocks"):
            try:
                for blk in te.blocks:
                    _hook("text.attn.q", blk.attn.q, trainable=False)
                    _hook("text.attn.k", blk.attn.k, trainable=False)
                    _hook("text.attn.v", blk.attn.v, trainable=False)
                    _hook("text.attn.o", blk.attn.o, trainable=False)
                    _hook_attn(
                        "text.attn._total",
                        blk.attn,
                        ["text.attn.q", "text.attn.k", "text.attn.v", "text.attn.o"],
                        trainable=False,
                    )
                    _hook("text.ffn.gate", blk.ffn.gate, trainable=False)
                    _hook("text.ffn.fc1", blk.ffn.fc1, trainable=False)
                    _hook("text.ffn.fc2", blk.ffn.fc2, trainable=False)
            except Exception:
                pass

        ie = getattr(pipe, "image_encoder", None)
        if ie is not None:
            try:
                vit = ie.model.visual
                _hook("clip.patch_embed", vit.patch_embedding, trainable=False)
                for blk in vit.transformer:
                    _hook("clip.attn.qkv", blk.attn.to_qkv, trainable=False)
                    _hook("clip.attn.o", blk.attn.proj, trainable=False)
                    _hook_attn(
                        "clip.attn._total",
                        blk.attn,
                        ["clip.attn.qkv", "clip.attn.o"],
                        trainable=False,
                    )
                    _hook("clip.ffn.fc1", blk.mlp[0], trainable=False)
                    _hook("clip.ffn.fc2", blk.mlp[2], trainable=False)
            except Exception:
                pass

    def _read_gemm_times(self):
        """Read measured per-GEMM times from CUDA events.

        Returns:
            dict: ``{timer_key: {"fwd_s": float, "bwd_s": float}}``
        """
        result = {}

        def _sum_events(ev_list):
            total_ms = 0.0
            for se, ee in ev_list:
                try:
                    total_ms += se.elapsed_time(ee)
                except Exception:
                    pass
            return total_ms

        for key, ev_list in self._gemm_fwd_ev.items():
            result.setdefault(key, {})["fwd_s"] = _sum_events(ev_list) / 1e3

        for key, ev_list in self._gemm_bwd_ev.items():
            result.setdefault(key, {})["bwd_s"] = _sum_events(ev_list) / 1e3

        for total_key, children in self._gemm_attn_children.items():
            kernel_key = total_key.replace("._total", "._kernel")
            if total_key not in result:
                continue

            for phase in ("fwd_s", "bwd_s"):
                parent_t = result[total_key].get(phase, 0)
                child_t = sum(result.get(ck, {}).get(phase, 0) for ck in children)
                result.setdefault(kernel_key, {})[phase] = max(0.0, parent_t - child_t)

        res_comp_times, _ = self._read_component_times()
        vae_time = res_comp_times.get("vae", 0)
        if vae_time and vae_time > 0:
            info = getattr(self, "_last_mfu_flops_info", None)
            if info is not None:
                vae_gemm_flops = {
                    gm["timer_key"]: gm["total_flops"]
                    for gm in info.get("gemm_table", [])
                    if (gm.get("timer_key") or "").startswith("vae.")
                }
                total_vae_flops = sum(vae_gemm_flops.values())
                if total_vae_flops > 0:
                    for vk, vf in vae_gemm_flops.items():
                        result.setdefault(vk, {})["fwd_s"] = (
                            vae_time * vf / total_vae_flops
                        )

        return result

    def compute_mfu(self, model, step_time_seconds):
        """Compute Model FLOPs Utilization (MFU) for VGA DiT training.

        FLOPs are computed for this rank's local batch, so MFU divides by one
        GPU's peak FLOPs. The returned num_gpus is informational only.
        """
        if step_time_seconds <= 0:
            return None

        if not self._enable_module_mfu and not self._enable_gemm_mfu:
            return None

        try:
            info = self._compute_flops_per_step(model)
            self._last_mfu_flops_info = info
            if self._enable_gemm_mfu:
                self._setup_gemm_timers(model)
            flops_total = info["flops_per_step"]
            profiled_fwd_flops = getattr(self, "flops", None)

            gpu_specs = resolve_gpu_specs()
            gpu_name = gpu_specs.gpu_name
            peak_flops = gpu_specs.peak_flops
            mem_bw_gbs = gpu_specs.mem_bw_gbs
            ops_byte_ratio = gpu_specs.ops_byte_ratio

            import torch.distributed as dist

            num_gpus = (
                dist.get_world_size()
                if dist.is_available() and dist.is_initialized()
                else 1
            )

            mfu_total = flops_total / (step_time_seconds * peak_flops)

            if (
                profiled_fwd_flops is not None
                and self.logger
                and not getattr(self, "_profile_mfu_compare_logged", False)
            ):
                fwd_formula = info.get("flops_forward_total", 0)
                ratio = profiled_fwd_flops / fwd_formula if fwd_formula else 0.0
                self.logger.info(
                    "[MFU] forward_flops_profile={:.3f}T  "
                    "forward_flops_formula={:.3f}T  raw_ratio={:.3f}  "
                    "shape=B{}xV{}xT{}x{}x{} latent={} source={}".format(
                        profiled_fwd_flops / 1e12,
                        fwd_formula / 1e12,
                        ratio,
                        info.get("batch_eff", 0) // max(info.get("num_views", 1), 1),
                        info.get("num_views", 0),
                        info.get("num_frames", 0),
                        info.get("height", 0),
                        info.get("width", 0),
                        info.get("latent_dims"),
                        info.get("latent_source"),
                    )
                )
                self._profile_mfu_compare_logged = True

            comp_times = {}
            call_counts = {}
            mfu_dit = mfu_text = mfu_vae = mfu_clip = mfu_action = None

            if self._enable_module_mfu:
                comp_times, call_counts = self._read_component_times()

                expected_vae_calls = info.get("num_vae_calls", 2)
                vae_calls = call_counts.get("vae", expected_vae_calls)
                clip_calls = call_counts.get("image_encoder", 1)
                flops_vae_total = info["flops_vae"] * vae_calls
                flops_clip_total = info["flops_clip"] * clip_calls

                flops_total_adjusted = (
                    flops_total
                    + info["flops_vae"] * (vae_calls - expected_vae_calls)
                    + info["flops_clip"] * (clip_calls - 1)
                )
                mfu_total = flops_total_adjusted / (step_time_seconds * peak_flops)

                def _comp_mfu(flops, comp_name, time_multiplier=1.0):
                    t = comp_times.get(comp_name)
                    if t and t > 0:
                        return flops / (t * time_multiplier * peak_flops)
                    return None

                mfu_dit = _comp_mfu(
                    info["flops_dit_total"],
                    "dit",
                    time_multiplier=info.get("dit_train_multiplier", 3.0),
                )
                mfu_text = _comp_mfu(info["flops_text"], "text_encoder")
                mfu_vae = _comp_mfu(flops_vae_total, "vae")
                mfu_clip = _comp_mfu(flops_clip_total, "image_encoder")
                mfu_action = _comp_mfu(info["flops_action"], "action_transformer")

                def _pct(v):
                    return "{:.2f}%".format(v * 100) if v is not None else "n/a"

                def _time_str(comp_name):
                    t = comp_times.get(comp_name, 0)
                    n = call_counts.get(comp_name, 0)
                    total_ms = t * 1e3
                    if n > 1:
                        return "{:.1f}ms (avg {:.1f}ms x{})".format(
                            total_ms, total_ms / n, n
                        )
                    return "{:.1f}ms".format(total_ms)

                if self.logger:
                    self.logger.info(
                        "[MFU] overall={overall}  |  "
                        "text_enc={te} (fwd {te_t})  "
                        "vae={vae} (fwd {vae_t})  "
                        "clip_vit={clip} (fwd {clip_t})  "
                        "dit={dit} (fwd {dit_t}, total~{dit_mult:.0f}xfwd)  "
                        "act_tf={atf} (fwd {atf_t})  "
                        "|  step={step:.3f}s  {ng}x{gpu} @ {pk:.0f} TFLOPs/s".format(
                            overall=_pct(mfu_total),
                            te=_pct(mfu_text),
                            te_t=_time_str("text_encoder"),
                            vae=_pct(mfu_vae),
                            vae_t=_time_str("vae"),
                            clip=_pct(mfu_clip),
                            clip_t=_time_str("image_encoder"),
                            dit=_pct(mfu_dit),
                            dit_t=_time_str("dit"),
                            dit_mult=info.get("dit_train_multiplier", 3.0),
                            atf=_pct(mfu_action),
                            atf_t=_time_str("action_transformer"),
                            step=step_time_seconds,
                            ng=num_gpus,
                            gpu=gpu_name,
                            pk=peak_flops / 1e12,
                        )
                    )

            gemm_table = info.get("gemm_table", [])
            gemm_mfu_list = []
            if self._enable_gemm_mfu:
                try:
                    gemm_times = self._read_gemm_times()
                    gemm_flops_sum = sum(g["total_flops"] for g in gemm_table)
                    for gm in gemm_table:
                        ai = gm["arith_intensity"]
                        flops_pct = (
                            gm["total_flops"] / gemm_flops_sum * 100
                            if gemm_flops_sum > 0
                            else 0
                        )

                        tk = gm.get("timer_key")
                        tp = gm.get("timer_phase", "fwd")
                        tf = gm.get("timer_frac", 1.0)
                        measured_s = None
                        gemm_mfu_val = None
                        if tk and tk in gemm_times:
                            raw_s = gemm_times[tk].get(f"{tp}_s", 0)
                            measured_s = raw_s * tf
                            if measured_s > 0:
                                gemm_mfu_val = gm["total_flops"] / (
                                    measured_s * peak_flops
                                )

                        gemm_mfu_list.append(
                            {
                                **gm,
                                "gemm_mfu": gemm_mfu_val,
                                "measured_s": measured_s,
                                "flops_pct": flops_pct,
                                "bound_type": (
                                    "compute" if ai >= ops_byte_ratio else "memory"
                                ),
                            }
                        )
                    gemm_mfu_list.sort(key=lambda x: x["total_flops"], reverse=True)

                    if self.logger:
                        self.logger.info(
                            "[MFU-GEMM] {:40s} | {:>30s} | {:>10s} | {:>8s} "
                            "| {:>9s} | {:>7s} | {:>7s} | {:>7s}".format(
                                "name",
                                "shape (M, N, K)",
                                "GFLOPs",
                                "FLOPs%",
                                "time(ms)",
                                "MFU%",
                                "AI",
                                "bound",
                            )
                        )
                        for gm in gemm_mfu_list:
                            t_ms = (
                                "{:>9.3f}".format(gm["measured_s"] * 1e3)
                                if gm["measured_s"] is not None
                                else "      n/a"
                            )
                            mfu_s = (
                                "{:>6.1f}%".format(gm["gemm_mfu"] * 100)
                                if gm["gemm_mfu"] is not None
                                else "    n/a"
                            )
                            self.logger.info(
                                "[MFU-GEMM] {:40s} | {:>30s} | {:>10.2f} | {:>7.2f}% "
                                "| {:>9s} | {:>7s} | {:>7.1f} | {:>7s}".format(
                                    gm["name"],
                                    "({}, {}, {})".format(gm["M"], gm["N"], gm["K"]),
                                    gm["total_flops"] / 1e9,
                                    gm["flops_pct"],
                                    t_ms,
                                    mfu_s,
                                    gm["arith_intensity"],
                                    gm["bound_type"],
                                )
                            )
                        total_measured = sum(
                            g["measured_s"]
                            for g in gemm_mfu_list
                            if g["measured_s"] is not None
                        )
                        overall_gemm_mfu = (
                            gemm_flops_sum / (total_measured * peak_flops) * 100
                            if total_measured > 0
                            else 0
                        )
                        self.logger.info(
                            "[MFU-GEMM] {:40s} | {:>30s} | {:>10.2f} | {:>7.1f}% "
                            "| {:>8.3f}s | {:>6.1f}%".format(
                                "TOTAL (all GEMMs)",
                                "",
                                gemm_flops_sum / 1e9,
                                100.0,
                                total_measured,
                                overall_gemm_mfu,
                            )
                        )
                except Exception as gemm_err:
                    if self.logger:
                        self.logger.info(f"[MFU-GEMM] timing not available: {gemm_err}")

            return {
                "mfu": mfu_total,
                "flops_per_step_T": flops_total / 1e12,
                "flops_forward_T": info["flops_forward_total"] / 1e12,
                "profile_fwd_flops_T": (
                    profiled_fwd_flops / 1e12
                    if profiled_fwd_flops is not None
                    else None
                ),
                "mfu_text_encoder": mfu_text,
                "mfu_vae": mfu_vae,
                "mfu_clip_vit": mfu_clip,
                "mfu_dit": mfu_dit,
                "mfu_action_transformer": mfu_action,
                "flops_text_T": info["flops_text"] / 1e12,
                "flops_vae_T": info["flops_vae"] * info.get("num_vae_calls", 2) / 1e12,
                "flops_clip_T": info["flops_clip"] / 1e12,
                "flops_dit_T": info["flops_dit_total"] / 1e12,
                "flops_action_T": info["flops_action"] / 1e12,
                "fwd_time_text_s": comp_times.get("text_encoder"),
                "fwd_time_vae_s": comp_times.get("vae"),
                "fwd_time_clip_s": comp_times.get("image_encoder"),
                "fwd_time_dit_s": comp_times.get("dit"),
                "fwd_time_action_s": comp_times.get("action_transformer"),
                "peak_gpu_tflops": peak_flops / 1e12,
                "mem_bw_gbs": mem_bw_gbs,
                "ops_byte_ratio": ops_byte_ratio,
                "step_time": step_time_seconds,
                "gpu_name": gpu_name,
                "num_gpus": num_gpus,
                "gemm_mfu": gemm_mfu_list,
                **info,
            }
        except Exception as e:
            if self.logger:
                self.logger.info(f"[MFU] compute error: {e}")
            return None
