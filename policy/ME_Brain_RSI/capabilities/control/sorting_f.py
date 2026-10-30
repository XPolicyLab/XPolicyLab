"""Extend the actual retained sorting policy without changing other tasks."""
from coap.tasks.sorting_r07_d import transform as retained_transform
from coap.tasks.sorting_r04_bridge import run as bridge
from coap.tasks import sorting_initial_color as initialcolor


def transform(ctx, replay):
    """Prepare local AST additions before the retained stack is applied once."""
    prepared = replay
    prepared = initialcolor.prepare_core(prepared, refine_demonstrations=False)
    core = retained_transform(ctx, prepared)
    return core


def run(ctx):
    """Preserve retained timing records and clean up this task-local context."""
    original_class = type(ctx)
    execution = bridge(ctx, transform)
    try:
        for action in execution:
            partition = getattr(ctx.memory, 'initial_partition', None)
            records = getattr(ctx, 'sorting_close_timing_records', None)
            if isinstance(partition, dict) and records is not None:
                partition['close_timing'] = list(records)
            yield action
    finally:
        execution.close()
        ctx.__class__ = original_class
