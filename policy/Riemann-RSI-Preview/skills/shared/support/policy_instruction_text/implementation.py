@staticmethod
def _task_text(obs: dict) -> str:
    task = obs.get('instruction') or obs.get('instructions') or ''
    if isinstance(task, (list, tuple)):
        task = task[0] if len(task) else ''
    return str(task)
