"""Bind private observer aliases to the frozen resident CUDA implementation."""
from importlib import import_module
from coap.tasks import classify_r29_language_semantic_history_cuda_device as device

def _resolve_module(module):
    """Accept either a fully qualified import path or an imported module."""
    return import_module(module) if isinstance(module, str) else module

def enable_for_observers(observer_modules):
    """Enable CUDA first, then bind every explicit observer and its text cache."""
    metadata = dict(device.enable())
    dino = import_module('coap.tasks.sorting_r03_dino')
    dino._encode = device._encode
    observers = [_resolve_module(module) for module in observer_modules]
    cleared = []
    for observer in observers:
        observer._encode = device._encode
        text_vectors = getattr(observer, '_text_vectors', None)
        if text_vectors is not None:
            text_vectors.cache_clear()
            cleared.append(observer.__name__)
    metadata.update(adapter='classify_r28_cuda_observers', dino_producer='coap.tasks.sorting_r03_dino', observer_modules=[observer.__name__ for observer in observers], cleared_text_cache_modules=cleared)
    return metadata

def run_with_observers(ctx, policy_module, observer_modules, scene=None):
    """Delegate the retained generator after binding its explicit observer modules."""
    ctx.memory['classify_device'] = enable_for_observers(observer_modules)
    policy = _resolve_module(policy_module)
    return (yield from policy.run(ctx, scene))
