"""Extend the actual retained sorting policy without changing other tasks."""
from coap.tasks.sorting_r09_c import transform as retained_transform
from coap.tasks.sorting_r04_bridge import run as bridge
from coap.tasks import sorting_car_route_alignment as alignment
from coap.tasks import sorting_demo_boundary_fragments as crossfragment
from coap.tasks import sorting_demo_fragments as demo
from coap.tasks import sorting_demo_color_bridge as democolorbridge
from coap.tasks import sorting_phone_side_role as phonepartition


def transform(ctx, replay):
    """Prepare local AST additions before the retained stack is applied once."""
    prepared = replay
    prepared = phonepartition.prepare_core(prepared)
    prepared = demo.prepare_core(prepared)
    prepared = alignment.prepare_core(prepared)
    core = retained_transform(ctx, prepared)
    core = democolorbridge.finish_core(ctx, core)
    core = crossfragment.finish_core(ctx, core)
    core = phonepartition.finish_core(ctx, core)
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
