---
name: curobo_thread_capture
description: "Installs a child-process view of torch/cuda for the pinned cuRobo cuda_graph_util module so graph capture defaults to thread_local error checking while forwarding all other attributes and original graph calls. report() summarizes the pinned module hash and observed capture modes."
---

# Curobo thread capture

## Function

Installs a child-process view of torch/cuda for the pinned cuRobo cuda_graph_util module so graph capture defaults to thread_local error checking while forwarding all other attributes and original graph calls. report() summarizes the pinned module hash and observed capture modes.

## Inputs

- official_root path containing third_party/curobo official source
- view returned by install, for report()

## Outputs

- view with torch/cuda wrappers; module.torch is replaced by the view
- report dict with scope, module_sha256, default_capture_error_mode, capture_modes_observed, and unchanged/equivalence flags

## Callable interface

Implementation: [implementation.py](implementation.py).

- `install(official_root)`
- `report(view)`

## Constraints

- install accepts only official_root/third_party/curobo/curobo/_src/util/cuda_graph_util.py with SHA-256 404358daccb304e8f1632653ec653a6c50b143a25313ecdeac58a3533e805447 and module.torch still identical to imported torch.
- graph() changes only a default capture_error_mode of global to thread_local, records the mode, and leaves capture function, stream, pool, replay, kernels, inputs, and streams unchanged.
- Changes only the pinned module's torch binding in the child process; no official source files or global torch binding are changed.
- No numerical or performance equivalence is claimed.
