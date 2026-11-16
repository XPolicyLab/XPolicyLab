# Execution-skill dependencies

OrimaPI uses the public policy interfaces of OpenPI and XPolicyLab. Their
Apache-2.0 license texts are retained here. Official source code, licenses and
additional notices are preserved in the pinned Git submodules under third_party/.

OrimaPI implements its own lifecycle, explicit state/action layouts, standard
RGB image decoding, parameter validation, per-environment observation ownership,
and local code-program interface. Model architectures and training utilities
remain in OpenPI; simulation code remains in RoboDojo. Upstream files are not
patched by OrimaPI.

Weights and task programs must be provided explicitly. A working adapter or
successful load does not establish simulator or physical-task performance.
