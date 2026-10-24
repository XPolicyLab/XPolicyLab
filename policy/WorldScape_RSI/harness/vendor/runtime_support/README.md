# Retained runtime source

Copied from `WorldScape_RSI_20261015/support/harness_runtime` of the supplied
20261015 delivery. Original file notices are retained. The CuRobo source files
carry NVIDIA Apache-2.0 notices; its license is included in `curobo/LICENSE`.
This is a Python source overlay, not a complete CUDA dependency installation.

Migration changes: `native_runtime/env/planner_manager/curobo_planner.py` uses
relative imports for `global_configs` and `utils.transformer`. Trailing blank lines and YAML trailing whitespace are normalized for Git
submission; the delivered algorithms are unchanged. The policy's robot-only
planner subclass imports this private package; it does not load a scene model.

Runtime source files are included in the Harness frozen-release manifest;
upstream test files and caches are not inference assets. Refer to the policy's
INSTALLATION.md for external robot assets and GPU dependencies.
