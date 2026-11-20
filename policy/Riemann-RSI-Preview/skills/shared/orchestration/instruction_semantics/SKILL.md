---
name: instruction_semantics
description: "Parses bounded public instruction text into canonical action, object, term, and relation facets using a finite synonym/inflection vocabulary and conservative relation heuristics. matches, prior_terms, and resolver_contract expose the parsed facets and annotate resolver applicability."
---

# Instruction semantics

## Function

Parses bounded public instruction text into canonical action, object, term, and relation facets using a finite synonym/inflection vocabulary and conservative relation heuristics. matches, prior_terms, and resolver_contract expose the parsed facets and annotate resolver applicability.

## Inputs

- instruction string or None (None becomes empty), maximum 32768 characters
- matches keyword filters actions, all_actions, objects, terms, relations, any_objects, any_terms, term_groups, exclude_objects
- resolver_contract instruction_predicate callable; decorate resolver callable with signature resolver(api, memory, maximum_uncertain_attempts=2)

## Outputs

- parse returns frozen InstructionSemantics with frozenset terms, actions, objects, relations
- matches returns bool; prior_terms returns set of words including trailing-s variants
- resolver_contract returns decorator; decorated resolver returns result dict with applicable bool

## Callable interface

Implementation: [implementation.py](implementation.py).

- `canonical_word(word)`
- `InstructionSemantics: exception or data type`
- `parse(instruction)`
- `cached_parse(instruction)`
- `matches(instruction, *, actions=(), all_actions=(), objects=(), terms=(), relations=(), any_objects=(), any_terms=(), term_groups=(), exclude_objects=())`
- `prior_terms(instruction)`
- `resolver_contract(instruction_predicate)`

## Constraints

- Normalizes NFKC, casefolds, replaces curly apostrophe; tokenizes with regex; caches at most 8 parses
- Negated actions within the next 12 words up to punctuation/but/instead are excluded; special cases add align for line-up wording and pour for empty bottle-to-cup wording
- Raises ExecutionError for non-str or >32768 characters
- Conservative finite vocabulary, not arbitrary NLU; unsupported wording remains uncertain and a semantic match does not prove an executable pose; resolver predicate must not read perception, task identifiers, priors, execution state, or official outcomes
