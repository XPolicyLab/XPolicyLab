---
name: declared_resource_bundle
description: "Resolves a named public resource for an explicit task directory and verifies every declared member in that task's resources.json before returning the bundle root."
---

# Declared resource bundle

## Function

Resolves a named public resource for an explicit task directory and verifies every declared member in that task's resources.json before returning the bundle root.

## Inputs

- task: canonical task directory path
- name: optional member name passed to resource(Path(task), name)

## Outputs

- asset(task, name) returns the existing verified resource path
- asset_root(task) returns the located verified bundle root after checking all manifest['files'] members

## Callable interface

Implementation: [implementation.py](implementation.py).

- `asset(task, name)`
- `asset_root(task)`

## Constraints

- Reads the explicit task's resources.json through load_manifest and the common logical resource locator
- Does not read simulator state, select a task, change resource precedence, or infer object poses
- Manifest, resource, or path errors propagate; no fallback verification is added
