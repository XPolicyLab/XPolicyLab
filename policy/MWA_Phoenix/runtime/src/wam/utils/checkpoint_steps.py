"""Checkpoint progress is descriptive metadata, never a compatibility gate."""


def checkpoint_steps(payload):
    """Read actual saved counters; absent metadata stays unknown (None)."""
    metadata = payload.get('joint96_metadata') or {}
    initialization = (metadata.get('model_paths') or {}).get('posttrain_initialization') or {}
    contract = (metadata.get('soft_prompt') or {}).get('training_contract') or {}
    return {
        'step': payload.get('step'),
        'pretrain_step': initialization.get('step', contract.get('pretrain_step')),
    }
