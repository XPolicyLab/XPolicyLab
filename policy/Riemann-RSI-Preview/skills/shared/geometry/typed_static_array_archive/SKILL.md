---
name: typed_static_array_archive
description: "Serializes a tree of public static model values to arrays.npz plus a hash-bearing manifest.json, and loads one named tree after authenticating the manifest and archive. Arrays are stored with dtype, shape, and finite-value checks; loaded arrays and containers are fresh copies."
---

# Typed static array archive

## Function

Serializes a tree of public static model values to arrays.npz plus a hash-bearing manifest.json, and loads one named tree after authenticating the manifest and archive. Arrays are stored with dtype, shape, and finite-value checks; loaded arrays and containers are fresh copies.

## Inputs

- write(directory, trees, *, source_root, source_files, provenance): a new directory, named trees, public source root, source file identities, and provenance metadata
- trees may contain dictionaries with string keys, lists, tuples, primitives, and finite numeric arrays; source paths are stored relative to source_root
- load(manifest_path, tree_name, *, expected_manifest_sha256, source_root=None): manifest path, named tree, expected manifest SHA-256, and optional source-root rebinding

## Outputs

- write returns the manifest path and its SHA-256 digest and creates arrays.npz and manifest.json
- load returns the decoded named tree with fresh mutable arrays and containers
- source_path values are rebound under source_root when supplied, otherwise represented as public-cad:<relative>

## Callable interface

Implementation: [implementation.py](implementation.py).

- `digest(path)`
- `write(directory, trees, *, source_root, source_files, provenance)`
- `load(manifest_path, tree_name, *, expected_manifest_sha256, source_root=None)`

## Constraints

- Numeric arrays must have dtype kind in biufc and all finite values; non-finite arrays raise ValueError.
- write requires a non-existing directory and string dictionary keys; unsupported values raise TypeError or ValueError.
- load rejects symlinks, absolute or parent-traversing archive paths, schema or flag mismatch, changed archive or manifest hashes, and changed array membership, dtype, or shape.
- The archive uses allow_pickle=False and does not import pxr; only public static data is accepted, not observations, scene layouts, historical frames, or simulator state.
