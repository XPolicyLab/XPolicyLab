# Era0-Preview submission checklist

The [model repository](https://huggingface.co/roboterax/Era0-5B-Preview-RoboDojo) provides model weights, inference assets and exact normalization. Evaluation code is maintained in the XPolicyLab policy directory.

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

1. Run the official plain and encoded debug loops on the selected weights and attach their successful log tails. These checks are not yet recorded as passing.
2. Attach download instructions, environment details, protocol/mode provenance, and the measured results below to the standard XPolicyLab PR template. A complete local three-seed run is not stated as a prerequisite to opening the PR; the existing multi-task campaign is incomplete and uses compatibility settings.

## Test evidence for the submission PR

The official [Testing and PR standard](../../CONTRIBUTING.md#testing) asks for test evidence and a checkpoint download command. Copy the measured results and limitations into the PR's Testing and Limitations sections; do not mark pending debug checks as passed.

| Test | Recorded result | Scope / limitation |
| --- | --- | --- |
| Local multi-task simulation | Overview Score **33.16 ± 0.35**; SR **27.96 ± 0.47%**; 1,716 successes / 6,225 episodes | Seeds 0/1/2; native target 6,300; mixed SDK versions, legacy rendering and local IK protection; incomplete reference result |
| Packaged adapter simulation, GPU 1 | `press_by_number`, seed 0: **9/10** successes, Score **90** | Policy image lag 1, SDK rendering/physics, official execution profile |
| Packaged adapter simulation, GPU 2 | `press_by_number`, seed 0: **8/10** successes, Score **80** | Same settings, independent ten-episode run |
| Plain-image debug loop | Pending | Attach log tail reaching `[MAIN] eval finished` without tracebacks |
| Encoded-image debug loop | Pending | Repeat with `DEBUG_OBS_ENCODED=1`; attach successful log tail |
| Complete native three-seed run with packaged official profile | Pending | Existing local reference results do not establish this result |

The [evaluation report](EVALUATION_RESULTS.md) contains capability scores, per-task/per-seed results and the 75 missing trials. The multi-task campaign combines older-SDK results with newer-SDK legacy-render additions and is not a uniform SDK run. Its local Overview uses available episodes without a minimum per-task threshold; do not label it as an official leaderboard result or as complete native evaluation.

Include this checkpoint download command in the PR:

```bash
cd policy/Era0_Preview
bash download_checkpoint.sh
```

The [public model card](https://huggingface.co/roboterax/Era0-5B-Preview-RoboDojo#local-reference-results) also documents the local reference results. The packaged current-chain single-task tests support adapter functionality, not full-suite equivalence to that campaign.

## Code validation

The local package was checked with:

```bash
python -m unittest discover -s policy/Era0_Preview/tests -v
python -m compileall -q policy/Era0_Preview
for script in policy/Era0_Preview/*.sh; do bash -n "$script"; done
```

The CPU tests exercise the real subprocess queue and SDK summary invocation with a fake simulator, including all three seeds, native-count forwarding, environment-subsampling removal, missing/incorrect result rejection, and observation/action shapes. They also cover actual packaged IK call sites, hold behavior, clean/unknown SDK versions, native result/video/resume paths and real `execv` restart with profile reinstallation. The plain/encoded debug loops and full simulator suite remain pending.

Additional local checks: isolated package model import, local checkpoint processor load without downloading weights, 54-task/6,300-trial preview against existing checkpoint metadata, and official RoboDojo `eval` / native `benchmark` dry-run resolution. A separate simulator smoke test completed two independent ten-episode `press_by_number`, seed-0 runs with policy image lag 1, SDK rendering/physics and the official execution profile: Scores 90 and 80. This checks one task only and does not establish full-suite equivalence.

Policy image delay is enabled by default: `HY_OBS_LAG=1` on the policy server,
or `--obs-lag 1` in the bundled all-task driver. Set `HY_OBS_LAG=0` or
`--obs-lag 0` to disable it while retaining SDK rendering. This preserves current state/language and requires
SDK rendering. The smoke test above used this option.

SDK integration: the result report comes exclusively from the installed SDK `scripts/internal/summarize_result.py`, writing `eval_result/RoboDojo/_summary.md`. The policy does not ship a result summarizer. Video names and directories are the SDK defaults. A CPU-only integration fixture with 300 result records and 900 placeholder camera filenames passed the official summary inventory check (zero mismatches); this is not a simulator score.
