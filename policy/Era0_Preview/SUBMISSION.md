# Era0-Preview submission checklist

The [model repository](https://huggingface.co/roboterax/Era0-5B-Preview-RoboDojo) provides model weights, inference assets and exact normalization. Evaluation code is maintained in the XPolicyLab policy directory.

- Contributor: **RobotEra Team**; organization: **RobotEra**.
- Model type: **VLA**; evaluation type: **simulation**; release scope: **evaluation only**.
- Contact: **Liming Zheng**, **liming.zheng@robotera.com**.
- Technical report: planned for public release within two weeks, by **2026-11-18**. No broader artifact-release commitment is implied.
- Model display name: **Era0-Preview**. Import/policy directory: **Era0_Preview**.
- Scope: eval-only, RoboDojo / arx_x5 / ee, single and vectorized environments.
- Complete protocol: installed SDK default all-task inventory, seeds 0/1/2, native counts; no local task extensions.
- Runtime: included under `runtime/`, self-contained inference code with no policy symlink.
- Download: `bash download_checkpoint.sh`; pinned repository/revision and weight/norm hashes are in [checkpoint_release.json](checkpoint_release.json). The published model assets use a flat layout including `norm_stats.pkl`.
- Preserve the selected checkpoint's `config.json`, `hy_xvla_adapter.json`, processor/tokenizer files, all safetensors, and exact normalization pickle.
- Preserve the saved time-RoPE configuration. Do not switch action coordinates, normalization, chunk horizon, color order or prompt placement.
- Defaults use policy-side image delay (`HY_OBS_LAG=1`) with SDK rendering, and retain local IK protection. For the audited official execution/render combination, select `--execution-profile official --render-mode sdk`, or `ERA0_EXECUTION_PROFILE=official ROBODOJO_EVAL_LEGACY_RENDER=0` on standard entries. Both profiles preserve SDK-native output/resume and video paths. The official profile executes the selected SDK files without path changes. Setting the rendering switch alone retains IK protection. Disclose compatibility settings; see [SDK_COMPATIBILITY.md](SDK_COMPATIBILITY.md).
- Missing trajectories remain incomplete, including unstable `make_toast_random`; no 520-episode acceptance shortcut.

Code distribution: use the XPolicyLab `feature/era0_preview` branch directly. No code archive is required.

Pending before an official submission:

1. Plain and encoded debug loops passed on the released weights. Attach the successful log tails in [DEBUG_VERIFICATION.md](DEBUG_VERIFICATION.md).
2. Attach download instructions, environment details, protocol/mode provenance, and the measured results below to the standard XPolicyLab PR template. A complete local three-seed run is not stated as a prerequisite to opening the PR; the existing multi-task campaign is incomplete and uses compatibility settings.

## Test evidence for the submission PR

The official [Testing and PR standard](../../CONTRIBUTING.md#testing) asks for test evidence and a checkpoint download command. Copy the measured results and limitations into the PR's Testing and Limitations sections; debug evidence is recorded in [DEBUG_VERIFICATION.md](DEBUG_VERIFICATION.md).

| Test | Recorded result | Scope / limitation |
| --- | --- | --- |
| Local multi-task simulation, official SDK summary | Partial Overview Score **33.44 ± 0.42**; SR **28.35 ± 0.45%**; 122/126 complete task–seed cells | 6,225/6,300 episodes available; incomplete cells excluded; mixed SDK versions, legacy rendering and local IK protection; historical reference only |
| Packaged adapter simulation, GPU 1 | `press_by_number`, seed 0: **9/10** successes, Score **90** | Policy image lag 1, SDK rendering/physics, official execution profile |
| Packaged adapter simulation, GPU 2 | `press_by_number`, seed 0: **8/10** successes, Score **80** | Same settings, independent ten-episode run |
| Clean policy environment installation | PASS | Fresh venv, source-built CUDA extensions, dependency/import and GPU checks; see [DEBUG_VERIFICATION.md](DEBUG_VERIFICATION.md) |
| Plain-image debug loop | PASS | 10 rounds × 20 action steps; 10 environments; `[MAIN] eval finished`, no tracebacks |
| Encoded-image debug loop | PASS | `DEBUG_OBS_ENCODED=1`; same round/step counts; successful final log, no tracebacks |
| Complete native three-seed run with packaged official profile | Pending | Existing local reference results do not establish this result |

The [evaluation report](EVALUATION_RESULTS.md) is the unmodified official SDK summary of the historical results, retaining the historical policy label `Hy_XVLA_RoboDojo`. Only directory indices were arranged for SDK discovery; result JSON and videos were unchanged. It includes 122/126 completed task–seed cells and excludes incomplete cells. There are 75 missing episodes: `make_toast_random` seeds 0/1/2 lack 10/25/25, and `store_laptop_and_headphones_random` seed 2 lacks 15. The SDK media check reports mismatches in all 158 checked result directories because the historical videos are combined views rather than separately named camera videos. The campaign combines older-SDK results with newer-SDK legacy-render additions and local IK protection; it is not a uniform run of the submitted official execution profile. Do not label it as an official leaderboard result or complete native evaluation.

Include this checkpoint download command in the PR:

```bash
cd policy/Era0_Preview
bash download_checkpoint.sh
```

The [public model card](https://huggingface.co/roboterax/Era0-5B-Preview-RoboDojo#local-reference-results) also documents the local reference results. The packaged current-chain single-task tests support adapter functionality, not full-suite equivalence to that campaign.

## Official CLI smoke test

Two native simulator tasks completed through the official CLI, with two seed-0 episodes each, valid result records and all twelve camera videos. See [SIMULATOR_SMOKE_TEST.md](SIMULATOR_SMOKE_TEST.md) for measured results and the launch command. This does not replace a complete three-seed evaluation.

## Code validation

The local package was checked with:

```bash
python -m py_compile policy/Era0_Preview/model.py policy/Era0_Preview/deploy.py
python -m compileall -q policy/Era0_Preview
for script in policy/Era0_Preview/*.sh; do bash -n "$script"; done
```

The plain and encoded debug loops, plus focused observation-delay regression, passed; see [DEBUG_VERIFICATION.md](DEBUG_VERIFICATION.md). A complete three-seed simulator run with the packaged official execution profile remains pending.

Additional local checks: isolated package model import, local checkpoint processor load without downloading weights, 54-task/6,300-trial preview against existing checkpoint metadata, and official RoboDojo `eval` / native `benchmark` dry-run resolution. A separate simulator smoke test completed two independent ten-episode `press_by_number`, seed-0 runs with policy image lag 1, SDK rendering/physics and the official execution profile: Scores 90 and 80. This checks one task only and does not establish full-suite equivalence.

Policy image delay is enabled by default: `HY_OBS_LAG=1` on the policy server,
or `--obs-lag 1` in the bundled all-task driver. Set `HY_OBS_LAG=0` or
`--obs-lag 0` to disable it while retaining SDK rendering. This preserves current state/language and requires
SDK rendering. The smoke test above used this option.

SDK integration: the result report comes exclusively from the installed SDK `scripts/internal/summarize_result.py`, writing `eval_result/RoboDojo/_summary.md`. The policy does not ship a result summarizer. Video names and directories are the SDK defaults. A CPU-only integration fixture with 300 result records and 900 placeholder camera filenames passed the official summary inventory check (zero mismatches); this is not a simulator score.
