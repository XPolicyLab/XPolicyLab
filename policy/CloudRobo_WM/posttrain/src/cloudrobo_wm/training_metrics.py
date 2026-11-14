"""Detached scalar telemetry; model losses and optimizer state are untouched."""
import math
import numbers
from collections.abc import Mapping

import torch
import torch.distributed as dist


METRICS_PROTOCOL = 'cloudrobo_wm.as2.metrics.global_rank_mean.v1'
METRICS_CONTRACT = {
    'protocol': METRICS_PROTOCOL,
    'aggregation': 'rank_mean_after_model_global_valid_count_normalization',
    'dtype': 'float32',
    'collective': 'one_packed_sum_then_divide_by_group_world_size',
    'model_global_diagnostics': 'mean_preserves_existing_global_scale',
    'metrics_jsonl': 'global_rank_mean',
    'global_status_json': 'global_rank_mean',
    'rank_status_json': 'rank_local_losses_with_model_global_diagnostics',
    'seconds': 'rank_local_compute_time_excludes_telemetry_and_saving',
}
IDENTITY_FIELDS = frozenset({
    'step', 'epoch', 'rank', 'world', 'process_nonce', 'seconds', 'learning_rate',
    'consumed_windows', 'skipped_updates', 'replacements', 'metrics_protocol',
    'metrics_scope', 'metric_names',
})
RESERVED_METRIC_NAMES = IDENTITY_FIELDS | {'loss', 'gradnorm'}


def _scalar(value, device, name):
    if torch.is_tensor(value):
        if value.numel() != 1 or value.is_complex():
            raise ValueError('Telemetry must contain real scalars: ' + name)
        return value.detach().to(device=device, dtype=torch.float32).reshape(())
    if not isinstance(value, numbers.Real):
        raise TypeError('Telemetry must contain real scalars: ' + name)
    return torch.tensor(value, device=device, dtype=torch.float32)


@torch.no_grad()
def reduce_training_metrics(loss, metrics, grad_norm, *, group=None):
    """Return detached rank means, using one FP32 collective on the loss device.

    The model already scales local loss numerators by the global valid count.
    Their rank mean is the global loss, as in the original scalar logger.
    Diagnostics already reduced inside the model retain their existing scale.
    Identity, cursor, scheduler and timing metadata are appended by the caller.
    All ranks execute this function with the same model-metric schema/order.
    """
    if not torch.is_tensor(loss) or loss.numel() != 1 or not loss.is_floating_point():
        raise ValueError('Loss telemetry requires one real floating-point tensor')
    if not isinstance(metrics, Mapping) or any(not isinstance(key, str) for key in metrics):
        raise TypeError('Model telemetry requires named scalar metrics')
    if RESERVED_METRIC_NAMES.intersection(metrics):
        raise ValueError('Model telemetry cannot contain identity or progress metadata')
    names = sorted(metrics)
    packed = torch.stack([
        _scalar(loss, loss.device, 'loss'),
        _scalar(grad_norm, loss.device, 'gradnorm'),
        *[_scalar(metrics[name], loss.device, name) for name in names],
    ])
    if dist.is_available() and dist.is_initialized():
        world = dist.get_world_size(group=group)
        if world < 1:
            raise ValueError('Telemetry rank is outside the selected process group')
        dist.all_reduce(packed, op=dist.ReduceOp.SUM, group=group)
        packed.div_(world)
    result = packed.cpu().tolist()
    invalid = [name for name, value in zip(['loss', 'gradnorm', *names], result)
               if not math.isfinite(value)]
    if invalid:
        raise FloatingPointError('Nonfinite global telemetry: ' + ','.join(invalid))
    return result[0], dict(zip(names, result[2:])), result[1]


def metric_records(local_loss, local_metrics, local_grad_norm, global_loss,
                   global_metrics, global_grad_norm, metadata):
    """Build explicitly scoped local diagnostics and global log rows."""
    if set(local_metrics) != set(global_metrics):
        raise ValueError('Local and global telemetry schemas differ')
    if not isinstance(metadata, Mapping) or set(metadata) - IDENTITY_FIELDS:
        raise ValueError('Unexpected telemetry identity/progress metadata')
    if RESERVED_METRIC_NAMES.intersection(local_metrics):
        raise ValueError('Model metrics collide with telemetry metadata')
    local = {
        key: float(value.detach()) if torch.is_tensor(value) else float(value)
        for key, value in local_metrics.items()
    }
    local.update(metadata)
    local_norm = float(local_grad_norm.detach()) if torch.is_tensor(local_grad_norm) else float(local_grad_norm)
    local.update(loss=float(local_loss.detach()), gradnorm=local_norm,
                 metrics_protocol=METRICS_PROTOCOL,
                 metrics_scope='rank_local_losses_with_model_global_diagnostics')
    global_row = dict(global_metrics)
    global_row.update(metadata)
    global_row.update(loss=global_loss, gradnorm=global_grad_norm,
                      metrics_protocol=METRICS_PROTOCOL, metrics_scope='global_rank_mean',
                      metric_names=sorted(global_metrics))
    return local, global_row
