"""Task skill_selection is a versioned meta-skill composed with immutable memory."""

from PhysicalRSI_core.contracts import Operation
from PhysicalRSI_core.infra.storage import digest


def skill_selection_skill(name, snapshot, experts):
    """Bind every skill_choice to an exact executable revision; dispatch once per episode.

    This constructor does not qualify a skill_choice. Publish the resulting complete
    harness only through Self-Harness comparison and lineage commit.
    """
    skill_choices = snapshot.read()["skill_choices"]
    if not skill_choices:
        raise ValueError("Skill_selection memory is empty")
    bound = {}
    for task, skill_choice in skill_choices.items():
        operation = experts[skill_choice["name"]]
        if operation.revision != skill_choice["revision"]:
            raise ValueError("Skill_selection expert revision changed: " + skill_choice["name"])
        bound[task] = operation
    first = next(iter(bound.values()))
    if any(
        op.input != first.input or op.output != first.output for op in bound.values()
    ):
        raise ValueError("Skill_selection experts require identical episode contracts")

    def execute(request, context):
        task = request.get("task")
        if task not in bound:
            raise ValueError("No committed skill_choice for task: " + str(task))
        return bound[task](request, context)

    children = tuple(dict.fromkeys((op.name, op.revision) for op in bound.values()))
    return Operation(
        name,
        digest(dict(memory=snapshot.revision, experts=children)),
        first.input,
        first.output,
        execute,
        effects=frozenset().union(*(op.effects for op in bound.values())),
        children=tuple(bound.values()),
    )
