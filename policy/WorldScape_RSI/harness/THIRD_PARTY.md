# Retained source components

Vendored Agent/Skill implementation and Responses conversion originate from the
team-supplied archive recorded in `SOURCE_MAP.json`. Provider-profile loading is
retained from PR #162's `vlm_client.py`. WAM invokes the bundled WSP
model with the public instruction and a uniform checkpoint-owned history capacity.
The adapter is revised in this delivery; checkpoint weights and transforms are unchanged.

Native planner/CuRobo support sources retain their original notices. Shared RGB
helpers and WS protocol are consumed from XPolicyLab. Launch/evaluation scripts
originate from `demo_policy`; the server enforces one active trial per process.
The current policy README defines the active contract over historical vendor docs.

The delivery's native planner and CuRobo support tree are now included under
`vendor/runtime_support/`. CuRobo retains its NVIDIA SPDX Apache-2.0 notices and
an Apache-2.0 license copy. Native runtime provenance is the supplied archive
recorded in SOURCE_MAP.json. The distributed native subset contains only the robot planner, its global constants
and relative transformer dependency. Unused simulator, camera, reward and scene
sources (including camera_view.py) are excluded. Unused CuRobo examples, including
their separate sample-scene YAML files, are also excluded. This is a packaging
exclusion, not a relicense or removal of their notices. Original delivery copies
remain in the private audit archive. The retained CuRobo library algorithms and
Apache-2.0 notices are unchanged; historical whitespace normalization remains
recorded in SOURCE_MAP.json. CUDA dependencies and robot assets remain external.
