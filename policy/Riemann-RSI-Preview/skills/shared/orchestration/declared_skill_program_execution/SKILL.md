---
name: declared_skill_program_execution
description: "Loads explicit local skill exports and exception classes, validates a frozen rule-only declarative workflow, and interprets declared calls, branches, bounded loops, try/catch, raise, and return without evaluating Python from JSON. Bound skills own motion."
---

# Declared skill program execution

## Function

Loads explicit local skill exports and exception classes, validates a frozen rule-only declarative workflow, and interprets declared calls, branches, bounded loops, try/catch, raise, and return without evaluating Python from JSON. Bound skills own motion.

## Inputs

- graph dict with schema 'riemann-declarative-skill-workflow-v1', mode 'rule_only', wam_reselection False, positive int official_action_horizon, int max_operations in [1,4096], bindings, errors, and steps
- package string used to import package + '.' + skills.<part>.<implementation>
- bindings mapping names to callables and errors mapping names to Exception subclasses
- context passed as the first argument to each bound skill call

## Outputs

- load_bindings returns (bindings, errors) after validation
- execute returns the workflow return value, or None when no return is executed
- InvalidWorkflow is raised for schema, binding, node, limit, unresolved-variable, or operation-budget violations

## Callable interface

Implementation: [implementation.py](implementation.py).

- `InvalidWorkflow: exception or data type`
- `load_bindings(graph, package)`
- `validate(graph, bindings, errors)`
- `execute(graph, context, bindings, errors)`

## Constraints

- Binding specs must be skills.<part>.<implementation>:<public_symbol> with exactly three module parts and a non-underscore symbol
- Workflow nesting depth <=16, total nodes <=1024, for_each max_items 1..64, and operation count <= max_operations; for_each requires a list/tuple within the declared bound
- try/catch only handles declared Exception subclasses; return uses an internal _Returned(BaseException)
- Executor only interprets declared control flow; skills own motion
