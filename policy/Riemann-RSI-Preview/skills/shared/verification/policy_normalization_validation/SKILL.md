---
name: policy_normalization_validation
description: "Authenticates a normalization file by comparing the SHA-256 of its bytes with the expected digest. Returns None on a match and raises ValueError containing the path and actual digest on a mismatch."
---

# Policy normalization validation

## Function

Authenticates a normalization file by comparing the SHA-256 of its bytes with the expected digest. Returns None on a match and raises ValueError containing the path and actual digest on a mismatch.

## Inputs

- path: path string passed to pathlib.Path(path).read_bytes()
- expected_sha256: expected SHA-256 hex digest string

## Outputs

- None when the computed digest equals expected_sha256
- ValueError containing the mismatch message, path, and actual digest otherwise

## Callable interface

Implementation: [implementation.py](implementation.py).

- `_verify_norm_stats(path: str, expected_sha256: str)`

## Constraints

- Reads the entire file bytes with pathlib.Path(path).read_bytes(); hashlib is imported inside the function
- Uses exact string comparison of hashlib.sha256(...).hexdigest() to expected_sha256 and does not normalize or validate digest format
- Performs no model construction and does not mutate external state
