---
name: resolve_garment_fold
description: "Selects one next garment fold skill from a fixed phase-to-skill map when the instruction matches fold garment, the recorded phase has fewer than 3 attempts, and recent matching failures are below the uncertainty threshold. It returns a resolver contract result with the chosen skill, side parameter, and current part and phase relation, or an empty non-applicable result."
---

# Resolve garment fold

## Function

Selects one next garment fold skill from a fixed phase-to-skill map when the instruction matches fold garment, the recorded phase has fewer than 3 attempts, and recent matching failures are below the uncertainty threshold. It returns a resolver contract result with the chosen skill, side parameter, and current part and phase relation, or an empty non-applicable result.

## Inputs

- api exposing instruction() and command_memory()
- memory list of prior skill events with skill and execution_feedback
- maximum_uncertain_attempts, accepted only when exactly 2

## Outputs

- resolver dict with applicable, operations, relation, and reason
- operations containing one skill name and parameters.side as [side] from the first route arm or [auto]

## Callable interface

Implementation: [implementation.py](implementation.py).

- `resolve(api, memory, maximum_uncertain_attempts=2)`

## Constraints

- empty result when instruction or object match fails, phase is not in NEXT, attempts are at least 3, or maximum_uncertain_attempts is not 2
- failure scan covers reversed memory[-32:], ignores skills not in ACTIONS, and stops at a different skill or completed feedback
- at least 2 matching noncompleted failures suppress another command
- a completed command phase is not official task success
