"""Reviewed native DeepSpeed ZeRO-2 with the portable trainer's update semantics.

This module owns platform execution only.  Loss denominators, RNG replay,
coverage, sampling, model hooks, and the LR scheduler stay in the trainer.
DeepSpeed's FP32 master parameters intentionally differ from BF16 AdamW
states; FP32 reference parity does not establish BF16/NPU numerical parity.
"""
from copy import deepcopy
from pathlib import Path
import inspect
import json
import math
import os
import torch
import torch.distributed as dist
from .bounded_reduction import DEFAULT_REDUCTION_ELEMENTS, finite_gradients
from .ordered_zero2 import OrderedGradientReduction
SCHEMA = 'cloudrobo_wm.portable.zero2.v2'
SUPPORTED_DEEPSPEED = ('0.19.0',)
DOMAINS = ()
GROUP_KEY = 'cloudrobo_wm_group'

def _reviewed_deepspeed_version():
    import deepspeed
    version = deepspeed.__version__.split('+')[0]
    if version not in SUPPORTED_DEEPSPEED:
        raise RuntimeError('Reviewed DeepSpeed versions are ' + ', '.join(SUPPORTED_DEEPSPEED) + '; installed ' + deepspeed.__version__)
    return version

def _backward_has_scaling_control(engine_type):
    function=inspect.unwrap(engine_type.backward)
    for _ in range(4):
        if 'scale_wrt_gas' in inspect.signature(function).parameters:
            return True
        captured=inspect.getclosurevars(function).nonlocals.get('func')
        if not inspect.isfunction(captured):return False
        function=captured
    return False

class MasterAdamW(torch.optim.AdamW):
    """AdamW over native FP32 master partitions with an explicit shared group."""

    def __init__(self, groups, **kwargs):
        super().__init__(groups, **kwargs)
        self._domain_active = {domain: True for domain in DOMAINS}
        self._platform_group_steps = {}

    def set_domain_presence(self, active):
        if set(active) != set(DOMAINS):
            raise ValueError('Explicit presence for both portable domains is required')
        self._domain_active = {name: bool(active[name]) for name in DOMAINS}

    def step(self, closure=None):
        if closure is not None:
            raise ValueError('Portable ZeRO-2 does not support optimizer closures')
        for group in self.param_groups:
            domain = group.get(GROUP_KEY)
            if domain not in ('shared', *DOMAINS):
                raise RuntimeError('ZeRO optimizer lost its portable group identity')
            if domain != 'shared' and (not self._domain_active[domain]):
                for parameter in group['params']:
                    parameter.grad = None
            if any((p.grad is not None for p in group['params'])):
                self._platform_group_steps[domain] = self._platform_group_steps.get(domain, 0) + 1
        return super().step()

def make_optimizer(model, cfg, backend):
    parameters = [p for p in model.parameters() if p.requires_grad]
    return MasterAdamW([dict(params=parameters, **{GROUP_KEY: 'shared'})], lr=cfg['learning_rate'], betas=tuple(cfg.get('betas', (0.9, 0.95))), weight_decay=cfg.get('weight_decay', 0.01), foreach=False, fused=False)

def zero2_config(cfg, backend, *, microbatch, accumulation, world):
    if min(microbatch, accumulation, world) < 1:
        raise ValueError('Positive microbatch, external accumulation, and world are required')
    value = {}
    if cfg.get('deepspeed_config'):
        value = json.loads(Path(cfg['deepspeed_config']).read_text())
    allowed = {'train_batch_size', 'train_micro_batch_size_per_gpu', 'gradient_accumulation_steps', 'gradient_clipping', 'bf16', 'fp16', 'zero_optimization', 'steps_per_print', 'wall_clock_breakdown', 'checkpoint', 'zero_allow_untested_optimizer', 'data_types', 'communication_data_type'}
    if set(value) - allowed:
        raise ValueError('Unreviewed DeepSpeed configuration keys: ' + str(sorted(set(value) - allowed)))
    zero = value.setdefault('zero_optimization', {})
    allowed_zero = {'stage', 'offload_optimizer', 'offload_param', 'overlap_comm', 'contiguous_gradients', 'elastic_checkpoint', 'round_robin_gradients', 'reduce_bucket_size', 'allgather_bucket_size', 'reduce_scatter'}
    if set(zero) - allowed_zero:
        raise ValueError('Unreviewed ZeRO configuration keys: ' + str(sorted(set(zero) - allowed_zero)))
    for name in ('bf16', 'fp16'):
        if set(value.get(name, {})) - {'enabled'}:
            raise ValueError('Unreviewed DeepSpeed precision settings')
    if set(value.get('checkpoint', {})) - {'use_node_local_storage'}:
        raise ValueError('Unreviewed DeepSpeed checkpoint settings')
    data_types = value.get('data_types', {})
    if set(data_types) - {'grad_accum_dtype'} or data_types.get('grad_accum_dtype', 'fp32') != 'fp32':
        raise ValueError('Portable ZeRO-2 requires explicitly reviewed FP32 gradient accumulation')
    if zero.get('stage', 2) != 2:
        raise ValueError('The portable platform adapter supports ZeRO stage 2 only')
    for key in ('offload_optimizer', 'offload_param'):
        if set(zero.get(key, {})) - {'device'}:
            raise ValueError('Unreviewed optimizer/parameter offload settings')
        if zero.get(key, {}).get('device', 'none') != 'none':
            raise ValueError('Portable ZeRO-2 supports no CPU/NVMe offload')
    for key in ('overlap_comm', 'elastic_checkpoint', 'round_robin_gradients'):
        if zero.get(key, False):
            raise ValueError('Unsupported portable ZeRO-2 option: ' + key)
    if value.get('communication_data_type', cfg.get('communication_dtype','fp32')) not in ('fp32','bf16'):
        raise ValueError('Communication must be explicitly fp32 or bf16; accumulation stays FP32')
    if not zero.get('contiguous_gradients', True) or not zero.get('reduce_scatter', True):
        raise ValueError('ZeRO-2 requires contiguous gradients and reduction with partitioning')
    if value.get('optimizer') or value.get('scheduler') or value.get('fp16', {}).get('enabled'):
        raise ValueError('Optimizer/scheduler and FP16 scaling must not be changed by DeepSpeed JSON')
    value.update(train_micro_batch_size_per_gpu=int(microbatch), gradient_accumulation_steps=1, train_batch_size=int(microbatch * world), gradient_clipping=float(cfg.get('max_grad_norm', 1.0)), bf16=dict(enabled=True), fp16=dict(enabled=False), data_types=dict(grad_accum_dtype='fp32'), communication_data_type=cfg.get('communication_dtype','fp32'), zero_allow_untested_optimizer=True, steps_per_print=10 ** 12, wall_clock_breakdown=False)
    zero.update(stage=2, offload_optimizer=dict(device='none'), offload_param=dict(device='none'), overlap_comm=False, contiguous_gradients=True, reduce_scatter=True, elastic_checkpoint=False, round_robin_gradients=False)
    zero.setdefault('reduce_bucket_size', DEFAULT_REDUCTION_ELEMENTS)
    zero.setdefault('allgather_bucket_size', 200000000)
    value['checkpoint'] = dict(use_node_local_storage=cfg.get('storage', 'shared') == 'node_local')
    return value

def _collective_flag(value, device, op):
    tensor = torch.tensor(int(value), dtype=torch.int32, device=device)
    if dist.is_initialized():
        dist.all_reduce(tensor, op=op)
    return bool(tensor.item())

class Zero2Adapter:
    """An external accumulation boundary around a native DeepSpeed engine."""

    def __init__(self, engine, optimizer, model, *, accumulation, configuration, control_group=None, gradient_order=None, trace_root=None):
        self.engine, self.optimizer = (engine, optimizer)
        self.model = model
        self.accumulation = int(accumulation)
        self.configuration = deepcopy(configuration)
        self.device = next((p for p in model.parameters() if p.requires_grad)).device
        self._expected = self._backwards = 0
        self._open = False
        self._partition_gradients = None
        self._touched = set()
        self._touch_handles = []
        for parameter in model.parameters():
            if parameter.requires_grad:
                parameter_id = id(parameter)

                def touch(gradient, identity=parameter_id):
                    self._touched.add(identity)
                    return None
                self._touch_handles.append(parameter.register_hook(touch))
        self._validate_layout()
        self.ordered_reduction = OrderedGradientReduction(engine.optimizer, model, control_group, plan=gradient_order, trace_root=trace_root)
        self._memory_report('initialized')

    def _memory_report(self, stage, *, error=None):
        """Bounded diagnostics for two smoke steps and any backward failure."""
        if self.device.type != 'npu' or (int(self.engine.global_steps) >= 2 and error is None):
            return
        record = dict(event='zero2_memory', stage=stage, rank=dist.get_rank() if dist.is_initialized() else 0, optimizer_steps=int(self.engine.global_steps), gradient_accumulation_dtype='float32', communication_dtype=self.configuration['communication_data_type'])
        record['ordered_reduction'] = self.ordered_reduction.diagnostics
        try:
            api = torch.npu
            record.update(allocated_bytes=int(api.memory_allocated(self.device)), reserved_bytes=int(api.memory_reserved(self.device)), peak_allocated_bytes=int(api.max_memory_allocated(self.device)))
            if hasattr(api, 'mem_get_info'):
                free, total = api.mem_get_info(self.device)
                record.update(driver_free_bytes=int(free), driver_total_bytes=int(total))
            if self._partition_gradients is not None:
                gradients = self._gradients()
                record.update(local_gradient_elements=sum((g.numel() for g in gradients)), local_gradient_bytes=sum((g.numel() * g.element_size() for g in gradients)), actual_gradient_dtypes=sorted({str(g.dtype) for g in gradients}))
        except Exception as exc:
            record['measurement_error'] = type(exc).__name__ + ': ' + str(exc)
        if error is not None:
            record['backward_error'] = type(error).__name__ + ': ' + str(error)
        print('CLOUDROBO_WM_ZERO2: ' + json.dumps(record), flush=True)

    def _validate_layout(self):
        engine, wrapper = (self.engine, self.engine.optimizer)
        if int(engine.zero_optimization_stage()) != 2 or engine.gradient_accumulation_steps() != 1:
            raise RuntimeError('Native engine no longer matches external-accumulation ZeRO-2')
        if wrapper.optimizer is not self.optimizer:
            raise RuntimeError('DeepSpeed replaced the master AdamW')
        if (type(wrapper).__module__, type(wrapper).__name__) != ('deepspeed.runtime.zero.stage_1_and_2', 'DeepSpeedZeroOptimizer'):
            raise RuntimeError('Expected the native ZeRO-2 optimizer')
        if wrapper.cpu_offload or not wrapper.partition_gradients or wrapper.overlap_comm or wrapper.has_moe_layers:
            raise RuntimeError('Unexpected offload/unpartitioned/overlap/MoE optimizer path')
        if not wrapper.contiguous_gradients or not wrapper.reduce_scatter:
            raise RuntimeError('Native contiguous-gradient partitioning is required')
        if wrapper.gradient_accumulation_dtype != torch.float32 or wrapper.communication_data_type != {'fp32':torch.float32,'bf16':torch.bfloat16}[self.configuration['communication_data_type']]:
            raise RuntimeError('ZeRO-2 communication or FP32 accumulation differs from the declared contract')
        if float(wrapper.loss_scale) != 1.0:
            raise RuntimeError('External portable accumulation requires unscaled BF16/FP32 gradients')
        self.layout = 'zero2'
        masters, low = (wrapper.single_partition_of_fp32_groups, wrapper.bit16_groups)
        if len(masters) != len(self.optimizer.param_groups) or len(low) != len(masters):
            raise RuntimeError('ZeRO master group count changed')
        for master, group in zip(masters, self.optimizer.param_groups):
            if len(group['params']) != 1 or group['params'][0] is not master or master.dtype != torch.float32:
                raise RuntimeError('AdamW group must own its exact FP32 ZeRO master partition')
        if not getattr(self, '_bucket_checked', False):
            from deepspeed.runtime.zero.stage_1_and_2 import split_half_float_double
            sample = low[0][0].detach().new_ones(1)
            buckets = split_half_float_double([sample])
            if sum((len(bucket) for bucket in buckets)) != 1 or not any((tensor is sample for bucket in buckets for tensor in bucket)):
                raise RuntimeError('Native gradient buckets omit this device/dtype')
            self._bucket_checked = True
        self._masters, self._low = (masters, low)

    def begin_update(self, batches_count):
        if self._open or int(batches_count) != batches_count or (not 0 < batches_count <= self.accumulation):
            raise ValueError('Invalid external ZeRO update boundary/count')
        self.engine.optimizer.zero_grad(set_to_none=True)
        self._touched.clear()
        self._partition_gradients = None
        self._expected, self._backwards, self._open = (int(batches_count), 0, True)
        self.engine.set_gradient_accumulation_boundary(False)

    def __call__(self, *args, **kwargs):
        if not self._open:
            raise RuntimeError('begin_update must precede the replay forward')
        self.engine.set_gradient_accumulation_boundary(True)
        return self.engine(*args, **kwargs)

    def backward(self, loss, *, is_last):
        if not self._open or self._backwards >= self._expected:
            raise RuntimeError('Backward outside the declared external update')
        if bool(is_last) != (self._backwards + 1 == self._expected):
            raise ValueError('External microbatch boundary differs from its declared update')
        self.engine.set_gradient_accumulation_boundary(True)
        self.ordered_reduction.begin()
        self._memory_report('before_backward')
        try:
            self.engine.backward(loss, scale_wrt_gas=False)
            self.ordered_reduction.finish()
        except Exception as error:
            self._memory_report('backward_failed', error=error)
            raise
        self._capture_fp32_partitions()
        self._memory_report('after_backward_fp32_shards')
        self._backwards += 1

    @torch.no_grad()
    def _capture_fp32_partitions(self):
        wrapper = self.engine.optimizer
        if set(wrapper.averaged_gradients) != set(range(len(self._masters))):
            raise RuntimeError('Native ZeRO-2 reduction did not produce all local gradient groups')
        first = self._partition_gradients is None
        if first:
            self._partition_gradients = {}
        for i, master in enumerate(self._masters):
            incoming = wrapper.averaged_gradients[i]
            if not isinstance(incoming, (tuple, list)) or not incoming or any((g is None for g in incoming)):
                raise RuntimeError('Incomplete local gradient partition')
            if sum((g.numel() for g in incoming)) != master.numel():
                raise RuntimeError('Local gradient elements differ from the FP32 master shard')
            if first:
                self._partition_gradients[i] = [g.detach().to(dtype=torch.float32, copy=True) for g in incoming]
            else:
                saved = self._partition_gradients[i]
                if len(saved) != len(incoming) or any((a.shape != b.shape for a, b in zip(saved, incoming))):
                    raise RuntimeError('Local gradient partition layout changed between microbatches')
                for accumulator, gradient in zip(saved, incoming):
                    accumulator.add_(gradient.float())
        wrapper.averaged_gradients.update({i: list(values) for i, values in self._partition_gradients.items()})

    def _presence(self):
        # This recipe trains one shared population and has no domain groups.
        return {}

    def _gradients(self):
        if self._partition_gradients is None:
            raise RuntimeError('No reduced FP32 local gradient shards')
        gradients = [g for i in range(len(self._masters)) for g in self._partition_gradients[i]]
        if any((g.dtype != torch.float32 or g.device != self.device for g in gradients)):
            raise RuntimeError('Gradient shards must actually be FP32 on the training device')
        return gradients

    def _mask_absent(self, active):
        for i, group in enumerate(self.optimizer.param_groups):
            domain = group[GROUP_KEY]
            if domain != 'shared' and (not active[domain]):
                for gradient in self._partition_gradients[i]:
                    gradient.zero_()

    def _check_active_participation(self, active):
        missing = []
        names = {id(parameter): name for name, parameter in self.model.named_parameters()}
        for group, parameters in zip(self.optimizer.param_groups, self._low):
            domain = group[GROUP_KEY]
            if domain == 'shared' or active[domain]:
                missing.extend((names.get(id(p), '<unknown>') for p in parameters if id(p) not in self._touched))
        if not _collective_flag(not missing, self.device, dist.ReduceOp.MIN):
            raise RuntimeError('An active trainable parameter did not participate in this update; flat AdamW must not silently decay it: ' + str(missing[:8]))

    def _finite_norm_before_step(self):
        gradients = self._gradients()
        squares = torch.zeros((), dtype=torch.float32, device=self.device)
        for gradient in gradients:
            squares += torch.linalg.vector_norm(gradient.float()).square()
        if dist.is_initialized():
            dist.all_reduce(squares)
        finite = bool(torch.isfinite(squares.sqrt()))
        if not _collective_flag(finite, self.device, dist.ReduceOp.MIN):
            raise FloatingPointError('Nonfinite ZeRO gradient norm; optimizer/cursor not advanced')

    def step(self):
        if not self._open or self._backwards != self._expected:
            raise RuntimeError('Cannot step an incomplete external update')
        self._validate_layout()
        from contextlib import nullcontext
        timer = getattr(self, '_phase_timer', None)
        section = lambda name: timer.section(name) if timer is not None else nullcontext()
        with section('domain_and_participation'):
            active = self._presence()
            self._check_active_participation(active)
            self.optimizer.set_domain_presence(active)
            self._mask_absent(active)
        gradients = self._gradients()
        with section('gradient_validation'):
            finite = finite_gradients(gradients, self.configuration['zero_optimization']['reduce_bucket_size'])
            if not _collective_flag(finite, self.device, dist.ReduceOp.MIN):
                raise FloatingPointError('Nonfinite ZeRO gradient; optimizer/cursor not advanced')
            self._finite_norm_before_step()
        before = (int(self.engine.global_steps), int(self.engine.skipped_steps))
        self.engine.set_gradient_accumulation_boundary(True)
        self.engine.optimizer.averaged_gradients.update({i: list(values) for i, values in self._partition_gradients.items()})
        with section('native_optimizer_and_gather'):
            self.engine.step()
        if (int(self.engine.global_steps), int(self.engine.skipped_steps)) != (before[0] + 1, before[1]):
            raise RuntimeError('DeepSpeed skipped/repeated an optimizer update; cursor must not commit')
        norm = float(self.engine.get_global_grad_norm())
        if not math.isfinite(norm):
            raise FloatingPointError('DeepSpeed produced a nonfinite gradient norm')
        self._memory_report('after_optimizer_step')
        self._open = False
        self._partition_gradients = None
        return norm

    @property
    def contract(self):
        return dict(version=SCHEMA, deepspeed=_reviewed_deepspeed_version(), accumulation='external_no_loss_rescale', external_accumulation=self.accumulation, master_dtype='float32', gradient_accumulation_dtype='float32', groups=[g[GROUP_KEY] for g in self.optimizer.param_groups], configuration=self.configuration, wrapper_layout=self.layout, zero_stage=2, reduced_gradient_partition_dtype='float32', communication_dtype=self.configuration['communication_data_type'], cpu_dtype_bucket_compatibility=False, ordered_reduction=self.ordered_reduction.contract, initialization='cloudrobo_wm.native_initialization.fenced.v1', initialization_max_pending_parameters=32)

def create_engine(wrapped, optimizer, cfg, backend, microbatch, accumulation, world):
    expected_accelerator = 'npu'
    if backend.kind != 'npu':
        raise ValueError('Unsupported native accelerator')
    selected = os.environ.get('DS_ACCELERATOR')
    if selected is not None and selected != expected_accelerator:
        raise RuntimeError('DS_ACCELERATOR differs from the explicitly requested backend')
    os.environ.setdefault('DS_ACCELERATOR', expected_accelerator)
    import deepspeed
    from deepspeed.accelerator import get_accelerator
    if get_accelerator().device_name() != expected_accelerator:
        raise RuntimeError('DeepSpeed selected a different native accelerator')
    _reviewed_deepspeed_version()
    if not dist.is_initialized() or dist.get_world_size() != int(world):
        raise RuntimeError('Initialize the native torch process group before ZeRO-2, including world=1')
    if not isinstance(optimizer, MasterAdamW):
        raise TypeError('ZeRO-2 requires master AdamW groups')
    if not _backward_has_scaling_control(deepspeed.DeepSpeedEngine):
        raise RuntimeError('DeepSpeed cannot disable its own loss accumulation scaling')
    model = wrapped.model
    if any((p.device.type != expected_accelerator for p in model.parameters() if p.requires_grad)):
        raise ValueError('Trainable parameter devices differ from the native platform backend')
    target_dtype = torch.bfloat16
    if any((p.dtype != target_dtype for p in model.parameters() if p.requires_grad)):
        raise ValueError('ZeRO-2 must not silently change original trainable parameter precision')
    buffers = {name: buffer.detach().clone() for name, buffer in model.named_buffers() if buffer.is_floating_point()}
    frozen = {name: (parameter, parameter.detach()) for name, parameter in model.named_parameters() if not parameter.requires_grad and parameter.is_floating_point()}
    if world > 1 and any((saved.device.type == 'cpu' for _, saved in frozen.values())):
        if getattr(backend, 'cpu_group', None) is None:
            raise ValueError('Offloaded frozen CPU parameters require the explicit Gloo CPU control group')
    configuration = zero2_config(cfg, backend, microbatch=microbatch, accumulation=accumulation, world=world)
    from .initialization import admit,progress,MAX_PENDING_PARAMETERS,PROTOCOL
    plan=admit(wrapped,backend,getattr(backend,'cpu_group',None),cfg['output_root'])
    if not hasattr(wrapped,'fenced_initialization'):
        raise TypeError('Native engine requires the owned startup-fenced wrapper')
    with wrapped.fenced_initialization(backend.synchronize):
        engine, installed, _, scheduler = deepspeed.initialize(model=wrapped, optimizer=optimizer, model_parameters=[p for p in model.parameters() if p.requires_grad], config=configuration, dist_init_required=False)
    backend.synchronize()
    progress(cfg['output_root'],'native_engine_constructed',parameter_plan_sha256=plan,completed_fences=wrapped._initialization_fences)
    if scheduler is not None or installed is not engine.optimizer:
        raise RuntimeError('DeepSpeed must not create a scheduler or a second optimizer')
    current_buffers = dict(model.named_buffers())
    for name, saved in buffers.items():
        current_buffers[name].data = saved
    current_parameters = dict(model.named_parameters())
    progress(cfg['output_root'],'restoring_frozen_parameters')
    pending=0
    for name, (original, saved) in frozen.items():
        parameter = current_parameters[name]
        if parameter is not original:
            raise RuntimeError('Native engine replaced a frozen parameter object')
        parameter.data = saved
        if world > 1:
            cpu_group = getattr(backend, 'cpu_group', None) if parameter.device.type == 'cpu' else None
            dist.broadcast(parameter.data, src=0, group=cpu_group)
            pending+=1
            if pending==MAX_PENDING_PARAMETERS:
                backend.synchronize();pending=0
    backend.synchronize()
    progress(cfg['output_root'],'frozen_parameters_synchronized',parameter_plan_sha256=plan)
    adapter = Zero2Adapter(engine, optimizer, model, accumulation=accumulation, configuration=configuration, control_group=getattr(backend, 'cpu_group', None), gradient_order=cfg.get('gradient_order_path'), trace_root=str(Path(cfg['output_root']) / 'gradient_order') if cfg.get('trace_gradient_order') else None)
    optimizer._domain_active = dict.fromkeys(DOMAINS, True)
    progress(cfg['output_root'],'native_optimizer_ready',parameter_plan_sha256=plan)
    return adapter
