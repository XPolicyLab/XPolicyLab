---
name: verified_geometry_assets
description: "Resolves declared public v23 resources under an optional root, enforcing root/member symlink and containment checks, then verifies each member's byte length and SHA256 against the manifest. Returns a mapping from declared member names to verified Paths."
---

# Verified geometry assets

## Function

Resolves declared public v23 resources under an optional root, enforcing root/member symlink and containment checks, then verifies each member's byte length and SHA256 against the manifest. Returns a mapping from declared member names to verified Paths.

## Inputs

- declaration_path: path passed to rsi.resource_locations.load_manifest
- root: optional public resource root; defaults to declaration['default_root']

## Outputs

- dict mapping each declared member name to a verified pathlib.Path
- Raises ValueError for symlinked root/member, invalid member path, escape, or byte/hash mismatch

## Callable interface

Implementation: [implementation.py](implementation.py).

- `resolve(root=None, *, declaration_path)`

## Constraints

- Only members whose first path part is 'skill' are accepted; absolute paths and '..' are rejected
- The root must not be a symlink; no path component of a declared member may be a symlink
- Each file's len(content) must equal record['bytes'] and its sha256 hex must equal record['source_sha256']
- Loads only external public v23 resources; does not import original policy or SDK code or create an evaluation environment
