"""Select the registered task implementation."""
def task_program(task, diagnostic_mode=None):
    if diagnostic_mode is not None:
        raise ValueError("Diagnostic programs are outside evaluation routing")
    from b88r42wq.tasks.long_horizon import fill_egg_holder
    return fill_egg_holder
