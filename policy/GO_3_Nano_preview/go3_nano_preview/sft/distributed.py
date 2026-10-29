"""Collective setup and coordinated side effects for the SFT runner."""
from datetime import timedelta
import os

import torch
import torch.distributed as dist


def initialize(timeout_seconds, *, backend='nccl'):
    if timeout_seconds <= 0:
        raise ValueError('nccl_timeout must be positive')
    timeout = timedelta(seconds=timeout_seconds)
    if backend == 'nccl':
        # DeepSpeed creates additional groups with timeout=None.
        from torch.distributed import distributed_c10d
        distributed_c10d.default_pg_nccl_timeout = timeout
    if not dist.is_initialized():
        kwargs = {'device_id': torch.device('cuda', int(os.environ['LOCAL_RANK']))} if backend == 'nccl' else {}
        dist.init_process_group(backend, timeout=timeout, **kwargs)
    return dist.new_group(backend='gloo', timeout=timeout)


def on_rank_zero(function, control_group):
    """Propagate a main-rank failure before peers enter their next collective."""
    error = None
    status = [None]
    if dist.get_rank() == 0:
        try:
            function()
        except Exception as exc:
            error = exc
            status[0] = f'{type(exc).__name__}: {exc}'
    dist.broadcast_object_list(status, src=0, group=control_group)
    if status[0] is not None:
        raise RuntimeError(f'Rank 0 operation failed: {status[0]}') from error


def on_all_ranks(function, control_group):
    """Run local preparation and report errors on every rank before proceeding."""
    result = None
    error = None
    try:
        result = function()
    except Exception as exc:
        error = exc
    statuses = [None] * dist.get_world_size()
    message = None if error is None else f'rank {dist.get_rank()}: {type(error).__name__}: {error}'
    dist.all_gather_object(statuses, message, group=control_group)
    failures = [s for s in statuses if s is not None]
    if failures:
        raise RuntimeError('Distributed preparation failed: ' + '; '.join(failures)) from error
    return result


def mean_losses(losses, device):
    names = ('vgen', 'ae', 'cotrain')
    values = torch.tensor([losses[name] for name in names], dtype=torch.float64, device=device)
    dist.all_reduce(values, op=dist.ReduceOp.SUM)
    values /= dist.get_world_size()
    return dict(zip(names, values.cpu().tolist()))
