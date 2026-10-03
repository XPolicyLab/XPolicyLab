"""Serial and parallel composition of the same versioned capability contract."""

from dataclasses import asdict

from PhysicalRSI_core.contracts import Contract, Operation
from PhysicalRSI_core.infra.executor import Executor
from PhysicalRSI_core.infra.storage import digest


def identity(operation: Operation) -> dict:
    return dict(
        name=operation.name,
        revision=operation.revision,
        input=asdict(operation.input),
        output=asdict(operation.output),
        effects=sorted(operation.effects),
    )


def sequence(name: str, *steps: Operation) -> Operation:
    if not steps:
        raise ValueError("A sequence needs at least one operation")
    for previous, following in zip(steps, steps[1:]):
        if previous.output != following.input:
            raise ValueError(
                f"Incompatible contracts: {previous.name} → {following.name}"
            )

    def execute(value, context):
        for step in steps:
            value = step(value, context)
        return value

    return Operation(
        name,
        digest({"kind": "sequence", "steps": [identity(step) for step in steps]}),
        steps[0].input,
        steps[-1].output,
        execute,
        frozenset().union(*(step.effects for step in steps)),
        tuple(steps),
    )


def parallel(
    name: str, *branches: Operation, executor: Executor | None = None
) -> Operation:
    """Fan out one input into an ordered tuple, only with disjoint effects.

    Each branch receives a deep copy. A following explicit join Operation can
    consume the tuple contract. Nested compositions retain their effect sets.
    """
    if not branches:
        raise ValueError("A parallel composition needs branches")
    if any(branch.input != branches[0].input for branch in branches):
        raise ValueError("Parallel input contracts differ")
    seen = set()
    for branch in branches:
        if seen.intersection(branch.effects):
            raise ValueError("Parallel branches have conflicting effects")
        seen.update(branch.effects)
    runtime = executor if executor is not None else Executor(len(branches))
    output = Contract("tuple:" + digest([asdict(branch.output) for branch in branches]))

    def execute(value, context):
        from copy import deepcopy

        return tuple(
            runtime.map(
                lambda branch: branch(deepcopy(value), context),
                branches,
                context=context,
            )
        )

    return Operation(
        name,
        digest(
            {"kind": "parallel", "branches": [identity(branch) for branch in branches]}
        ),
        branches[0].input,
        output,
        execute,
        frozenset(seen),
        tuple(branches),
    )
