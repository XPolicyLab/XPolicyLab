"""Compile frozen memory into a sequence of registered executable operations."""

from PhysicalRSI_core.contracts import Operation
from PhysicalRSI_core.infra.storage import digest
from .composition import sequence


def compile_program(snapshot, program_id, registry):
    """Bind explicit operation revisions; memory cannot import or evaluate code."""
    memory = snapshot.read()
    if memory.get('schema') != 'physicalrsi.skill-program-memory/v1':
        raise ValueError('Versioned skill program memory required')
    program = memory['programs'][program_id]
    steps = program.get('steps')
    if not isinstance(steps, list) or not steps:
        raise ValueError('A memory program needs executable steps')
    operations = []
    for step in steps:
        if not isinstance(step, dict) or set(step) != {'operation', 'revision'}:
            raise ValueError('Each step must bind an operation and revision')
        operation = registry[step['operation']]
        if (not isinstance(operation, Operation)
                or operation.name != step['operation']
                or operation.revision != step['revision']):
            raise ValueError('Memory operation identity differs from the registry')
        operations.append(operation)
    compiled = sequence(program_id, *operations)

    def execute(value, context):
        snapshot.read()
        return compiled(value, context)

    return Operation(
        program_id,
        digest({'memory': snapshot.revision, 'program': compiled.revision}),
        compiled.input, compiled.output, execute, compiled.effects, (compiled,),
    )
