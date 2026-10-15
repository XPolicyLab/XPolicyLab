# Retained source components

Vendored Agent/Skill implementation and Responses conversion originate from the
team-supplied archive recorded in `SOURCE_MAP.json`. Provider-profile loading is
retained from PR #162's `vlm_client.py`. WAM invokes the original adjacent WSP
model; its implementation and checkpoint transforms are not copied or replaced.

Native planner/CuRobo support sources retain their original notices. Shared RGB
helpers and WS protocol are consumed from XPolicyLab. Launch/evaluation scripts
originate from `demo_policy`; the server enforces one active trial per process.
The current policy README defines the active contract over historical vendor docs.

The delivery's native planner and CuRobo support tree are now included under
`vendor/runtime_support/`. CuRobo retains its NVIDIA SPDX Apache-2.0 notices and
an Apache-2.0 license copy. Native runtime provenance is the supplied archive
recorded in SOURCE_MAP.json. Only the native planner's two local import paths
were changed to package-relative imports during migration; the CuRobo source
algorithms are unchanged; trailing blank lines and YAML whitespace are normalized. CUDA dependencies and robot assets remain external.
