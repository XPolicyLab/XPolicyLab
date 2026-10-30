"""Run one retained language overlay with the audited FP32 CUDA observers."""


def run(ctx, scene=None):
    """Preserve the retained generator protocol after explicit observer binding."""
    from coap.tasks.classify_r29_language_semantic_history_cuda_observers import run_with_observers
    return (yield from run_with_observers(ctx,
        policy_module='coap.tasks.classify_r32_language_source_material_policy',
        observer_modules=['coap.tasks.classify_r07_n2_material_centroid', 'coap.tasks.classify_r25_b_language'], scene=scene))
