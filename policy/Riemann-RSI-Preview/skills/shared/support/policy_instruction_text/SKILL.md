---
name: policy_instruction_text
description: "Returns the public task instruction from obs, preferring 'instruction' over 'instructions' and reducing a list or tuple to its first element. Missing or empty values become an empty string."
---

# Policy instruction text

## Function

Returns the public task instruction from obs, preferring 'instruction' over 'instructions' and reducing a list or tuple to its first element. Missing or empty values become an empty string.

## Inputs

- obs: mapping with optional 'instruction' or 'instructions'
- instruction value may be scalar or list/tuple; only the first list/tuple element is read

## Outputs

- str: selected instruction, or '' when absent or empty

## Callable interface

Implementation: [implementation.py](implementation.py).

- `_task_text(obs: dict)`

## Constraints

- Implemented as the Model._task_text static method; it performs no task selection and uses no hidden layout information
- If both keys exist, 'instruction' wins
- A list/tuple is not concatenated; an empty list/tuple yields ''
- The selected value is converted with str()
