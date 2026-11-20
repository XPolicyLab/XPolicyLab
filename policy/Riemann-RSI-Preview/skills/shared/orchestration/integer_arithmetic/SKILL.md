---
name: integer_arithmetic
description: "calculate performs bounded integer arithmetic with nonnegative subtraction and exact, nonzero division. inconsistent_claim scans public planner goal and reason text for explicit arithmetic equations and flags any equation whose stated result does not match calculate."
---

# Integer arithmetic

## Function

calculate performs bounded integer arithmetic with nonnegative subtraction and exact, nonzero division. inconsistent_claim scans public planner goal and reason text for explicit arithmetic equations and flags any equation whose stated result does not match calculate.

## Inputs

- calculate: left_operand and right_operand must be exact int instances; operation must be one of plus, minus, multiplication, or division.
- inconsistent_claim: instruction must be a str and request must be a dict.
- inconsistent_claim request: goal must be a str of at most 160 characters and reason a str of at most 320 characters; these fields are read only after an arithmetic keyword match.
- Expression text: unsigned 1-5 digit left and right operands, an operation phrase or symbol, an =/equals/is equal to marker, and an optional-minus 1-5 digit result.

## Outputs

- calculate returns an int sum, difference, product, or exact integer quotient; it returns None for negative subtraction, zero or non-exact division, or an unsupported operation.
- inconsistent_claim returns False when instruction lacks equation/arithmetic/calculation/operator wording or when no contradictory explicit expression is found.
- inconsistent_claim returns True when a matched expression's calculated value is None or differs from the stated result.
- inconsistent_claim raises ValueError for invalid instruction/request types or invalid goal/reason types or length limits.

## Callable interface

Implementation: [implementation.py](implementation.py).

- `calculate(left_operand, operation, right_operand)`
- `inconsistent_claim(instruction, request)`

## Constraints

- Subtraction is permitted only when left_operand >= right_operand; division uses integer // only when right_operand is nonzero and left_operand % right_operand == 0.
- The expression regex is case-insensitive, excludes matches adjacent to word characters or dots, treats left/right operands as unsigned, and permits a negative result.
- Matched operation text is lowercased and whitespace-normalized before OPERATIONS lookup; only expressions with the =/equals/is equal to result form are checked.
- The functions do not read environments, objects, hidden layouts, evaluator results, or robot poses, and they do not issue actions or declare success.
