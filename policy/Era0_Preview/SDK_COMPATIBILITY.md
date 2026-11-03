# Local RoboDojo compatibility bundled with Era0-Preview

This package carries the relevant local evaluation changes inside the policy.
It does not require copying a modified RoboDojo checkout or editing installed
SDK files. `simulator_entry.py` selects hash-checked source snapshots for this
process, including after the SDK's `os.execv` recovery restart.

## Execution and rendering are independent

| Setting | Default | Behavior |
| --- | --- | --- |
| `--execution-profile local` / `ERA0_EXECUTION_PROFILE=local` | Yes | Local IK protection, requested random-task concurrency, local client diagnostics/videos, portable output/resume paths. |
| `--execution-profile official` / `ERA0_EXECUTION_PROFILE=official` | No | Audited official client execution, including its IK behavior and random-task concurrency cap. Only output-root and numeric-seed path plumbing is added. |
| `--render-mode legacy` / `ROBODOJO_EVAL_LEGACY_RENDER=1` | No; requires `HY_OBS_LAG=0` | Skip zero-delay Kit injection and render once before capture. |
| `--render-mode sdk` / `ROBODOJO_EVAL_LEGACY_RENDER=0` | Yes | Use the installed SDK's camera synchronization without overrides. |

For the current official execution/render combination, select **both**
`--execution-profile official --render-mode sdk`. Changing the rendering option
alone leaves local IK protection enabled. Legacy rendering and execution
protection must be disclosed when reporting scores; acceptance of compatibility
settings is for the benchmark maintainers to determine.

Neither profile rolls back physics, reset-drive fixes, rewards, task layouts,
or simulator assets. Joint/EE action encoding, coordinate frame, normalization,
and action chunk selection remain the policy's original contract.

## IK protection

The local profile copies the local `IKGuard` and its real execution call site.
For each arm, before control interpolation, it compares the IK solution with
**live measured joints**. It rejects that arm's entire target when:

- the solver does not return `status == "Success"`;
- target joints are missing, nonnumeric, incorrectly shaped, NaN or Inf;
- any raw joint displacement exceeds **0.5 rad** by default.

The difference is deliberately not wrapped modulo 2π: the controller
interpolates raw joint coordinates. A rejected arm receives an explicit hold at
its measured position through every interpolation step. Other arms and grippers
continue their normal actions. The ordinary action-step budget is still
consumed; this rejection does not discard a trajectory or select another seed.
If the measured joints themselves are invalid, `UnStableError` is raised before
any control sequence in that batch is pushed; the SDK's existing unstable-env
handling then applies.

`ik_guard_config.json` records the actual threshold and behavior next to the
task result. Each episode's `_result.json` detail includes `ik_rejections` per
arm. Counters reset per episode. Logs print the first five rejections, then
every hundredth. The guard can affect executed motions and scores; it is not
merely diagnostic. The official execution profile does not install it.

## Local change audit

Comparison: official RoboDojo `f57e1633ffed59761bc8e568cfe7530c4b33381c`
against local `fcd09556957a834233e09e78704e0db43a705c3e`.
`sdk_sources/change_inventory.json` lists the bundled evaluation runtime changes.

| Local changes | Packaging decision |
| --- | --- |
| `src/eval_client/ik_guard.py`, IK handling in `eval_env.py` | Enabled in the local profile; original implementation and regression tests copied. |
| Output-root and seed directory handling in `main.py` / `eval_env.py` | Included in both profiles so result discovery and resume-manifest reader/writer agree on a clean official SDK. |
| Random-task concurrency in `utils/pipeline_utils.py` | Local keeps requested parallel count (driver requests 10); official retains SDK clutter cap. No changes to native trajectory totals. |
| Timing/progress logs, subtask trace and three-camera video in the local client | Included in local profile. Subtask text is added only to recorded video copies, never policy observations. `ffmpeg` is needed for combined videos; failure retains per-camera files. |
| Local WebSocket client construction | Preserved in local profile; official profile retains official construction. Does not alter observations/actions. |
| Replay-only render/camera-disable flags in `base_env.py` / `task_env.py` | Diff archived in `other_local_changes.patch` with machine paths redacted; not activated or overlaid. The full driver removes inherited disable flags and sample-ratio overrides. |
| `stack_blocks_xshift2m` task/config extension | Excluded. The default task set comes from the installed SDK all-dimensions inventory. |
| Local launch scripts, retries, reports and asset repair utilities | Policy launchers, SDK in-process recovery and packaged summary provide the evaluation path. Machine-specific scripts and automatic asset repair are not executed. |
| `Assets` link | Not copied; use official installation/assets. |
| Curobo submodule revision | Recorded below; no library, robot-asset or example replacement. |

`sdk_sources/local/` stores exact local snapshots;
`sdk_sources/base/` stores the three audited official files;
`sdk_sources/official/` stores the official profile with output plumbing only.
Reviewable local-versus-official diffs are in `sdk_sources/audit_patches/`.
The RoboDojo source license is retained in `sdk_sources/LICENSE.RoboDojo`.

The source loader accepts the audited official/local file hashes recorded in
`sdk_sources/manifest.json`. Unknown host versions or modified bundled snapshots
fail before simulator startup; they require compatibility review instead of
silently replacing newer code. This validates the overlaid files, not every
simulator dependency. `evaluation_contract.json` and camera provenance also
record the execution profile, host/overlay hashes and SDK revision.

The local Curobo pin is `895c6517243f8cb091c73c018c8167192d39599a`;
the official pin is `d17b54ce32cba095c0b000c4c58777075d11de0e`.
Their `curobo/_src` trees have no diff, including the Warp compatibility code.
Other revision differences concern examples/configs/assets; their diffstat is
archived in `sdk_sources/curobo_revision_diff.txt`. No Curobo patch is applied.

## Validation and limits

The 25 CPU tests cover IK rejection/hold behavior using the actual packaged
execution methods, per-arm/gripper independence, invalid measurements, clean
official SDK acceptance, unknown-version rejection, matching resume paths, and
real Python `execv` restart with overlay reinstallation in both profiles. The
existing fake-simulator driver tests cover scheduling and reporting for all
three seeds. Metadata previews retain 54 tasks / 6,300 native trajectories.

These checks do not establish real simulator performance or score equivalence.
A separate two-run simulator smoke test on `press_by_number` used policy lag 1, SDK rendering/physics and the official execution profile, with ten seed-0 episodes per run (Scores 90 and 80). Full-suite score equivalence remains unverified.

Policy-side `HY_OBS_LAG` defaults to 1. SDK rendering is the default, including
when lag is explicitly disabled. Positive lag cannot be combined with legacy
rendering; legacy rendering requires explicit `HY_OBS_LAG=0` and
`ROBODOJO_EVAL_LEGACY_RENDER=1` (driver: `--obs-lag 0 --render-mode legacy`). Image delay is per
observation update and environment; state and language remain current.
