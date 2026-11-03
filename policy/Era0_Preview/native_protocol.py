"""Read official per-task evaluation counts without importing Isaac Sim."""
from pathlib import Path
import subprocess
import sys
import yaml


def native_counts(robodojo_root, tasks):
    path = Path(robodojo_root) / 'task/RoboDojo/config/_task.yml'
    spec = yaml.safe_load(path.read_text())
    result = {}
    for task in tasks:
        if not (path.parent / (task + '.yml')).is_file():
            raise ValueError('Missing official task config: ' + task)
        count = spec['tasks'].get(task, {}).get('eval_nums', spec.get('common', {}).get('eval_nums', 50))
        if type(count) is not int or count <= 0:
            raise ValueError('Invalid official eval_nums: ' + task)
        result[task] = count
    return result


def default_tasks(robodojo_root):
    """Use the SDK benchmark's default all-dimensions runnable inventory."""
    script = Path(robodojo_root) / 'scripts/internal/task_inventory.py'
    result = subprocess.run(
        [sys.executable, str(script), '--only-runnable', '--dimension', 'all'],
        check=True, capture_output=True, text=True,
    )
    tasks = result.stdout.splitlines()
    if not tasks:
        raise ValueError('The SDK default task inventory is empty')
    return tasks
