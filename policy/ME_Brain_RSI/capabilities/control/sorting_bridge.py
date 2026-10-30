"""Preserve the retained aligned-view initializer around a private memory transform."""
import types
from coap.tasks import sorting_r03_views as parent_views


def run(ctx, transform):
    """Apply a sorting-only inner transform without changing the retained outer program."""
    outer=parent_views.parent_D.run
    namespace=dict(outer.__globals__)
    namespace['memory']=types.SimpleNamespace(run=transform(ctx,namespace['memory'].run))
    private=types.FunctionType(outer.__code__,namespace,outer.__name__,outer.__defaults__,outer.__closure__)
    private.__kwdefaults__=outer.__kwdefaults__
    yield from parent_views.run_with_core(ctx,private)
