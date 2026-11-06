---
name: current_semantic_descriptions
description: "Normalizes a bounded public semantic description and expands known common names into up to three alternative text queries. Appends an object-name variant when the normalized description ends with ' object'."
---

# Current semantic descriptions

## Function

Normalizes a bounded public semantic description and expands known common names into up to three alternative text queries. Appends an object-name variant when the normalized description ends with ' object'.

## Inputs

- description: string of 1-128 printable characters; control characters or empty normalized text raise ValueError

## Outputs

- list of 1-3 natural-language query strings; known alternatives return three names

## Callable interface

Implementation: [implementation.py](implementation.py).

- `run(description)`

## Constraints

- underscores become spaces and whitespace is collapsed before matching; matching is by normalized lower-case text only
- no task ID, asset name, layout, historical image, coordinate, size, template, object state, or score is used
- returned names only change current RGB queries and do not grant geometric qualification
- a generic term not detected by the model is not evidence that the object is absent
