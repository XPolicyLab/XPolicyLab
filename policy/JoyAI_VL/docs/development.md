# Development record

## 2026-11-21 — Model API access

The installation examples now use the JD Cloud ModelService endpoint and the
`joyai_vl_testing` serving model ID. API keys are available by email from either
team contact listed in the README. The model egress proxy generates a fresh
`Trace-Id` per HTTP request and records it in its request logs. The model request
body, inference settings other than the serving model ID, and robot API routing
are unchanged.

## 2026-11-19 — API-driven Agent release

Source: the JoyAI-VL XPolicyLab adapter prepared at commit
`d3720b053ef394d16a07a2471bff26c7b7022df5`.

This variant removes task skills, task notes, cached task recipes, task-specific tool
directories and prebuilt task manuals. The task-card assembler consumes the instruction
and current robot observations. All tasks use one generic tool registry. The system
prompt contains robot API documentation and general execution instructions; stored
task procedures and benchmark scoring tolerances have been removed.

At the initial adapter commit, the API request loop, model inference parameters other
than the skills flag, shared SAM3 service, robot controller and perception geometry
retained their source bytes.
`metadata/variant_manifest.json` records source hashes and the file-level changes.

Evaluation records for the source release, which used task skills, are maintained
with that release. Current adapter checks are recorded in [validation](validation.md).

## 2026-11-19 — Image transport and deployment validation

Internal camera and tool PNGs now use a shared Pillow codec. Image tools keep RGB
arrays and use the corresponding RGB-to-HSV/Lab conversions. SAM3 mask decoding
uses grayscale PNG; the service, geometric calculations and motion planning are
unchanged. Grayscale conversion uses Pillow luminance, which can differ from the
previous PNG decoder by one intensity level. Tests cover official legacy/marked
JPEG input, internal camera images, tool pictures, colour names and mask pixels.

A full process debug run found that the fallback Docker network could have an
empty gateway after the default address pools were exhausted. Fallback creation
now specifies its gateway. The launcher reports a missing gateway before starting
an agent. Policy startup also puts the selected XPolicyLab checkout before an
external RoboDojo checkout in the Python import path.

The debug fixture uses synthetic observations and a local fixed-response model API.
It exercises the real policy server, agent container, SAM3 and the official debug
client. The fixture is kept under `tests/`.
After these changes, 22 CPU tests and both ten-episode process checks passed.
Validation commands, environment versions and results are recorded in
`docs/validation.md`; per-file hashes are in the variant and release manifests.
