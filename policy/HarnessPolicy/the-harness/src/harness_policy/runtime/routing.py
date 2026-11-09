"""Explicit ablation task routes. The default agent never calls task routing."""


def validate_task_routing(value, policies):
    """Validate {default: policy, tasks: {base task: 'own' or the same policy}}.

    One policy/checkpoint for all tasks; random variants must inherit the base.
    No scores, observations, task-name heuristics or per-task model selection.
    """
    if value is None:
        return
    if not isinstance(value, dict) or set(value) != {"default", "tasks"}:
        raise ValueError("runtime.task_routing requires default and tasks")
    policy = value["default"]
    if not isinstance(policy, str) or policy not in policies:
        raise ValueError("runtime.task_routing.default must name an enabled policy")
    tasks = value["tasks"]
    if not isinstance(tasks, dict) or any(
        not isinstance(task, str) or not task or task.endswith("_random")
        or route not in ("own", policy) for task, route in tasks.items()
    ):
        raise ValueError("runtime.task_routing.tasks must map base tasks to own or the default policy")


def task_route(routing, task):
    """None without an explicit ablation table; otherwise the fixed route."""
    if routing is None:
        return None
    base = str(task).removesuffix("_random")
    return routing["tasks"].get(base, routing["default"])
