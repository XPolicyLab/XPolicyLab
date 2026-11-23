def bounded_sequence(steps):
    """Run named checked subskills until one needs the agent's attention.

    Each (name, callable) returns a dict with status='verified' only after its
    own observation checks. No automatic replay. Simulator BaseException termination
    propagates. Ordinary failures return evidence, never success or episode end.
    """
    completed = []
    for name, function in steps:
        try:
            evidence = function()
        except Exception as exc:
            return {'status': 'needs_recovery', 'completed': completed, 'failed_step': name, 'error': f'{type(exc).__name__}: {exc}', 'held_object': 'unknown; inspect'}
        if not isinstance(evidence, dict) or evidence.get('status') != 'verified':
            return {'status': 'needs_recovery', 'completed': completed, 'failed_step': name, 'evidence': evidence, 'held_object': 'unknown; inspect'}
        completed.append({'step': name, 'evidence': evidence})
    return {'status': 'subtask_verified', 'completed': completed, 'task_success': 'unknown; agent must continue until the episode ends'}
