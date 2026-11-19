# Development record

## 2026-11-19 — API-driven Agent release

Source: the JoyAI-VL XPolicyLab adapter prepared at commit
`d3720b053ef394d16a07a2471bff26c7b7022df5`.

This variant removes task skills, task notes, cached task recipes, task-specific tool
directories and prebuilt task manuals. The task-card assembler consumes the instruction
and current robot observations. All tasks use one generic tool registry. The system
prompt contains robot API documentation and general execution instructions; stored
task procedures and benchmark scoring tolerances have been removed.

The API request loop, model inference parameters other than the skills flag, shared
SAM3 service, robot controller and perception geometry retain their source bytes.
`metadata/variant_manifest.json` records source hashes and the file-level changes.

The previously evaluated JoyAI-VL release includes task skills; its results are recorded
with that release. This variant has passed 14 CPU checks using synthetic inputs and
mocked services. Simulator evaluation: not run.
