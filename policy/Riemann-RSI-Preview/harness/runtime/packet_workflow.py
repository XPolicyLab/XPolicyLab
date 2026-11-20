"""Run declared per-observation programs with ordinary Python skill exports."""
import importlib
import json
from pathlib import Path
from .declarative_workflow import execute, load_bindings


class ActionFrame:
    def __init__(self, owner, arguments):
        self.owner=owner
        self.values=dict(arguments)

    def update(self, namespace, names):
        self.values.update({name:namespace[name] for name in names if name in namespace})


def document(filename):
    value=json.loads(Path(filename).read_text())
    if value.get('schema')!='riemann-current-observation-programs-v1':
        raise ValueError('A current-observation policy requires a declared program document')
    return value


def run_program(owner, name, arguments):
    graph=owner._workflow['programs'][name]
    cache=owner.__dict__.setdefault('_workflow_exports',{})
    if name not in cache:
        cache[name]=load_bindings(graph,owner._workflow['package'])
    bindings,errors=cache[name]
    return execute(graph,ActionFrame(owner,arguments),bindings,errors)


def instantiate_plan(document, name, arguments):
    """Preserve numeric targets by reference; only the stage skeleton is JSON."""
    def expand(value):
        if isinstance(value,dict):
            if set(value)=={'argument'}:
                return arguments[value['argument']]
            return {key:expand(item) for key,item in value.items()}
        if isinstance(value,list):return [expand(item) for item in value]
        return value
    return expand(document['plans'][name])
