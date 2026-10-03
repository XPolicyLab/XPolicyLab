"""RoboDojo isolated-environment actor limit; execution remains in core."""

from PhysicalRSI_core.infra.batch import run_batch


def run_actors(jobs, output, *, num_envs=1, cancelled=None):
    if type(num_envs) is not int or not 1 <= num_envs <= 10:
        raise ValueError("Execution requires 1 to 10 independent environment actors")
    return run_batch(jobs, output, workers=num_envs, cancelled=cancelled)
