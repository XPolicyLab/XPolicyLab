"""Execute class-scoped operation memories through registered methods."""
from pathlib import Path as _MemoryPath
import json as _memory_json
import tempfile as _memory_tempfile
from Lxm6wuec_core.contracts import Context as _MemoryContext, Contract as _MemoryContract, Operation as _MemoryOperation
from Lxm6wuec.Embodied_Harness.memory.store import MemoryStore as _SkillMemoryStore
from Lxm6wuec.Embodied_Harness.skills.memory_program import compile_program as _compile_memory_program
from lxm6wuec_structured_blocks.plan_transfer import apply_launch_plan as _apply_launch_plan

def execute_blocks(instance, source_file, owner, method, packet):
    cache_name = '_structured_memory_' + owner
    runtime = getattr(instance, cache_name, None)
    if runtime is None:
        document = _memory_json.loads(_MemoryPath(source_file).with_suffix('.memory.json').read_text())
        memory = _apply_launch_plan(document['classes'][owner])
        directory = _memory_tempfile.TemporaryDirectory(prefix='lxm6wuec-memory-')
        snapshot = _SkillMemoryStore(directory.name).snapshot(memory)
        registry = {}
        for entry in document['operations'][owner]:

            def invoke(value, context, entry=entry):
                instance._structured_operation_trace.append(entry['operation'])
                return getattr(instance, entry['method'])(value)
            registry[entry['operation']] = _MemoryOperation(entry['operation'], entry['revision'], _MemoryContract(entry['input']), _MemoryContract(entry['output']), invoke, frozenset({'episode-state'}))
        runtime = (directory, snapshot, {name: _compile_memory_program(snapshot, name, registry) for name in memory['programs']})
        setattr(instance, cache_name, runtime)
        if not hasattr(instance, '_structured_operation_trace'):
            instance._structured_operation_trace = []
    result = runtime[2][method](packet, _MemoryContext('skill-execution'))
    return result.get('result')
