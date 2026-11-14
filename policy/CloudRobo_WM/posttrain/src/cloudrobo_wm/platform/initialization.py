"""Bound startup-only device work without changing model or optimizer math."""
from contextlib import contextmanager
from pathlib import Path
import datetime
import hashlib
import json
import torch
import torch.distributed as dist
from .io import atomic_json

MAX_PENDING_PARAMETERS = 32
PROTOCOL = 'cloudrobo_wm.native_initialization.fenced.v1'


def progress(root, stage, **values):
    rank = dist.get_rank() if dist.is_initialized() else 0
    atomic_json(Path(root)/'ranks'/f'rank{rank:06d}.INITIALIZATION.json',
                dict(protocol=PROTOCOL, rank=rank, stage=stage,
                     utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), **values))


class InitializationFencedModule(torch.nn.Module):
    """Keep native parameter iteration bounded only during engine construction.

    Native engine broadcasts iterate this owned wrapper. A completed group is
    drained before the next group is yielded. No external class is modified.
    Normal forward/backward and optimizer iterations use ordinary iteration.
    """
    def __init__(self):
        super().__init__()
        self._initialization_sync = None
        self._initialization_fences = 0

    def named_parameters(self, prefix='', recurse=True, remove_duplicate=True):
        iterator = super().named_parameters(prefix=prefix, recurse=recurse,
                                            remove_duplicate=remove_duplicate)
        synchronize = self._initialization_sync
        if synchronize is None:
            yield from iterator
            return
        pending = 0
        for item in iterator:
            yield item
            pending += 1
            if pending == MAX_PENDING_PARAMETERS:
                synchronize()
                self._initialization_fences += 1
                pending = 0
        if pending:
            synchronize()
            self._initialization_fences += 1

    @contextmanager
    def fenced_initialization(self, synchronize):
        if self._initialization_sync is not None:
            raise RuntimeError('Nested native initialization is forbidden')
        self._initialization_sync = synchronize
        try:
            yield
        finally:
            self._initialization_sync = None


def admit(model, backend, control_group, root):
    progress(root, 'draining_model_initialization')
    backend.synchronize()
    plan = [dict(name=name, shape=list(p.shape), dtype=str(p.dtype),
                 device=p.device.type, trainable=p.requires_grad)
            for name, p in model.named_parameters()]
    signature = hashlib.sha256(json.dumps(plan, sort_keys=True).encode()).hexdigest()
    world = dist.get_world_size()
    peers = [None] * world
    dist.all_gather_object(peers, signature, group=control_group)
    if len(set(peers)) != 1:
        raise ValueError('Cross-rank initialization parameter order differs')
    progress(root, 'native_collective_probe', parameter_plan_sha256=signature)
    probe = torch.full((32,), 0.25, device=backend.device, dtype=torch.bfloat16)
    dist.broadcast(probe, src=0)
    backend.synchronize()
    if not torch.equal(probe.cpu(), torch.full((32,), 0.25, dtype=torch.bfloat16)):
        raise ValueError('Native initialization broadcast probe differs')
    count = torch.ones(1, device=backend.device, dtype=torch.float32)
    dist.all_reduce(count)
    backend.synchronize()
    if count.item() != world:
        raise ValueError('Native initialization reduction probe differs')
    dist.barrier(group=control_group)
    progress(root, 'native_engine_construction', parameter_plan_sha256=signature,
             maximum_pending_parameters=MAX_PENDING_PARAMETERS)
    return signature
