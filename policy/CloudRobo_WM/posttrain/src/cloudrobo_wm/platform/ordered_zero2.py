"""Order native ZeRO-2 gradient submission independently of local branches.

Native hooks still own gradient accumulation and the backward epilogue. Only
their submission order is scheduled here; native reduction, partition slicing,
clipping and optimizer math are unchanged. No third-party class is patched.
"""
import hashlib
import json
import torch
import torch.distributed as dist

PROTOCOL = 'cloudrobo_wm.zero2.ordered_reduction.v1'


class OrderedGradientReduction:
    def __init__(self, native, model, control_group, *, plan=None, trace_root=None):
        if (native.use_grad_accum_attribute or not native.partition_gradients
                or native.overlap_comm or native.cpu_offload):
            raise ValueError('Ordered reduction requires the reviewed native ZeRO-2 layout')
        self.native = native
        self._process = native.process_gradients
        self._epilogue = native.independent_gradient_partition_epilogue
        groups = {id(p): i for i, values in enumerate(native.bit16_groups) for p in values}
        native_indices = {id(p): j for values in native.bit16_groups for j, p in enumerate(values)}
        registered = [(name, p) for name, p in model.named_parameters() if p.requires_grad]
        if (len(groups) != len(registered) or len(groups) != sum(map(len, native.bit16_groups))
                or set(groups) != {id(p) for _, p in registered}):
            raise ValueError('Ordered gradient plan differs from trainable native parameters')
        self.order = [(name, groups[id(p)], p) for name, p in reversed(registered)]
        self.order_name = 'reverse_model_registration.v1'
        if plan is not None:
            from pathlib import Path
            record = json.loads(Path(plan).read_text())
            names = record.get('parameters', [])
            by_name = {name: (name, group, p) for name, group, p in self.order}
            identity = {name: list(p.shape) for name, _, p in self.order}
            if (record.get('version') != 'cloudrobo_wm.gradient_order.v1' or
                    record.get('shapes') != identity or len(names) != len(by_name) or set(names) != set(by_name)):
                raise ValueError('Gradient order plan differs from exact trainable parameters')
            self.order = [by_name[name] for name in names]
            self.order_name = record.get('origin','measured_readiness.v1')
        self.trace_root = trace_root
        self.ready_names = []
        self.trace_updates = 0
        self.positions = {id(p): j for j, (_, _, p) in enumerate(self.order)}
        if any(p.dtype != torch.bfloat16 for _, _, p in self.order):
            raise ValueError('Ordered native gradients must retain BF16 leaf precision')
        manifest = dict(protocol=PROTOCOL, order=self.order_name,
                        bucket_elements=int(native.reduce_bucket_size),
                        communication_dtype=str(native.communication_data_type),
                        partition_elements=[int(n) for n in native.partition_size],
                        partition_counts=[int(n) for n in native.partition_count],
                        parameters=[dict(name=name, group=i, shape=list(p.shape),
                                         native_index=native_indices[id(p)],
                                         dtype=str(p.dtype), elements=p.numel())
                                    for name, i, p in self.order])
        encoded = json.dumps(manifest, sort_keys=True, separators=(',', ':')).encode()
        self.signature = hashlib.sha256(encoded).hexdigest()
        self.total_bytes = sum(p.numel()*p.element_size() for _, _, p in self.order)
        self.parameter_count = len(self.order)
        # The existing CPU control group avoids device-dependent object transport.
        if dist.get_world_size() > 1:
            if control_group is None or dist.get_backend(control_group) != 'gloo':
                raise ValueError('Ordered gradient admission requires the CPU Gloo control group')
            peers = [None] * dist.get_world_size(group=control_group)
            dist.all_gather_object(peers, self.signature, group=control_group)
            if any(value != self.signature for value in peers):
                raise ValueError('Ordered ZeRO-2 parameter/bucket plan differs across ranks')
        self._active = False
        self._ready = set()
        self._seen = set()
        self._cursor = self._pending_bytes = self._peak_pending_bytes = 0
        self._epilogues = 0
        self._missing = []
        native.process_gradients = self.ready
        native.independent_gradient_partition_epilogue = self.epilogue

    @property
    def contract(self):
        return dict(protocol=PROTOCOL, order=self.order_name,
                    plan_sha256=self.signature, parameter_count=self.parameter_count,
                    pending_dtype='bfloat16', maximum_pending_bytes=self.total_bytes)

    @property
    def diagnostics(self):
        return dict(plan_sha256=self.signature, submitted_parameters=self._cursor,
                    pending_bytes=self._pending_bytes,
                    peak_pending_bytes=self._peak_pending_bytes,
                    maximum_pending_bytes=self.total_bytes,
                    zero_filled_parameters=len(self._missing), epilogues=self._epilogues)

    def begin(self):
        if self._active or self._ready:
            raise RuntimeError('Previous ordered gradient backward did not finish')
        self._active = True
        self.ready_names = []
        self._ready.clear()
        self._seen.clear()
        self._cursor = self._pending_bytes = self._peak_pending_bytes = 0
        self._epilogues = 0
        self._missing = []

    def ready(self, parameter, group):
        if not self._active:
            raise RuntimeError('Gradient hook outside the declared backward')
        index = self.positions.get(id(parameter))
        if index is None or self.order[index][1] != group or index in self._seen:
            raise RuntimeError('Unexpected/repeated gradient hook in ordered ZeRO-2 backward')
        if parameter.grad is None or parameter.grad.dtype != torch.bfloat16:
            raise RuntimeError('Ordered reduction expects the native BF16 leaf gradient')
        self.ready_names.append(self.order[index][0])
        self._seen.add(index)
        self._ready.add(index)
        self._pending_bytes += parameter.numel()*parameter.element_size()
        self._peak_pending_bytes = max(self._peak_pending_bytes, self._pending_bytes)
        self._drain()

    def _drain(self):
        while self._cursor in self._ready:
            index = self._cursor
            _, group, parameter = self.order[index]
            self._ready.remove(index)
            self._process(parameter, group)
            self._pending_bytes -= parameter.numel()*parameter.element_size()
            self._cursor += 1

    def epilogue(self):
        if not self._active or self._epilogues:
            raise RuntimeError('Ordered reduction requires one non-reentrant backward phase')
        # An unused local parameter still occupies its fixed communication slot.
        # The adapter's independent participation gate continues to reject an
        # untouched active parameter before AdamW. This does not mark it touched.
        while self._cursor < self.parameter_count:
            if self._cursor not in self._ready:
                name, _, parameter = self.order[self._cursor]
                if parameter.grad is not None:
                    raise RuntimeError('Unobserved gradient cannot be treated as an absent branch')
                parameter.grad = torch.zeros_like(parameter)
                self._missing.append(name)
                self._ready.add(self._cursor)
                self._pending_bytes += parameter.numel()*parameter.element_size()
                self._peak_pending_bytes = max(self._peak_pending_bytes, self._pending_bytes)
            self._drain()
        self._epilogue()
        self._epilogues += 1
        self._active = False

    def finish(self):
        if self._active or self._epilogues != 1 or self._ready or self._pending_bytes:
            raise RuntimeError('Native backward did not finish the ordered gradient plan')
        if self._cursor != self.parameter_count:
            raise RuntimeError('Ordered backward omitted trainable parameters')
        self.write_trace()

    def write_trace(self):
        if self.trace_root is None or self.trace_updates >= 2:
            return
        from pathlib import Path
        from .io import atomic_json as _atomic_json
        root = Path(self.trace_root)
        root.mkdir(parents=True, exist_ok=True)
        rank = dist.get_rank() if dist.is_initialized() else 0
        record = dict(version='cloudrobo_wm.gradient_readiness.v1', rank=rank,
                      world_size=dist.get_world_size() if dist.is_initialized() else 1,
                      update=self.trace_updates, plan_sha256=self.signature,
                      shapes={name:list(p.shape) for name,_,p in self.order},
                      ready=self.ready_names, missing=self._missing, diagnostics=self.diagnostics)
        _atomic_json(root / f'rank{rank:06d}-update{self.trace_updates}.json', record)
        self.trace_updates += 1
