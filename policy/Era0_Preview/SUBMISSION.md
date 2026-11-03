# Era0-Preview submission checklist

The [model and normalization](https://huggingface.co/roboterax/Era0-Preview-5B-RoboDojo/tree/eval-v1) are published. The model-upload PR was merged; no XPolicyLab code PR or cloud evaluation request has been submitted.

- Model display name: **Era0-Preview**. Import/policy directory: **Era0_Preview**.
- Scope: eval-only, RoboDojo / arx_x5 / ee, single and vectorized environments.
- Complete protocol: installed SDK default all-task inventory, seeds 0/1/2, native counts; no local task extensions.
- Runtime: included under `runtime/`, self-contained inference code with no policy symlink.
- Download: `bash download_checkpoint.sh`; pinned repository/revision and weight/norm hashes are in [checkpoint_release.json](checkpoint_release.json). The publication uses a flat layout and includes `norm_stats.pkl`.
- Preserve the selected checkpoint's `config.json`, `hy_xvla_adapter.json`, processor/tokenizer files, all safetensors, and exact normalization pickle.
- Preserve the saved time-RoPE configuration. Do not switch action coordinates, normalization, chunk horizon, color order or prompt placement.
- Defaults use policy-side image delay (`HY_OBS_LAG=1`) with SDK rendering, and retain local IK protection. For the audited official execution/render combination, select `--execution-profile official --render-mode sdk`, or `ERA0_EXECUTION_PROFILE=official ROBODOJO_EVAL_LEGACY_RENDER=0` on standard entries. The official profile adds only output/resume path plumbing. Setting the rendering switch alone retains IK protection. Disclose compatibility settings; see [SDK_COMPATIBILITY.md](SDK_COMPATIBILITY.md).
- Missing trajectories remain incomplete, including unstable `make_toast_random`; no 520-episode acceptance shortcut.

Publication completed: model, normalization and inference assets; revision `eval-v1`; 16 uploaded files verified.

Pending before an official submission:

1. Run the official plain and encoded debug loops on the selected weights; complete the full three-seed evaluation.
2. Attach download instructions, environment details, protocol/mode provenance, and measured results to the standard XPolicyLab PR template.

## Code validation

The local package was checked with:

```bash
python -m unittest discover -s policy/Era0_Preview/tests -v
python -m compileall -q policy/Era0_Preview
for script in policy/Era0_Preview/*.sh; do bash -n "$script"; done
```

The CPU tests exercise the real subprocess queue and report generation with a fake simulator, including all three seeds, native-count forwarding, environment-subsampling removal, missing/incorrect result rejection, and observation/action shapes. They also cover actual packaged IK call sites, hold behavior, clean/unknown SDK versions, consistent resume paths and real `execv` restart with profile reinstallation. The plain/encoded debug loops and full simulator suite remain pending.

Additional local checks: isolated package model import, local checkpoint processor load without downloading weights, 54-task/6,300-trial preview against existing checkpoint metadata, and official RoboDojo `eval` / native `benchmark` dry-run resolution. A separate simulator smoke test completed two independent ten-episode `press_by_number`, seed-0 runs with policy image lag 1, SDK rendering/physics and the official execution profile: Scores 90 and 80. This checks one task only and does not establish full-suite equivalence.

Policy image delay is enabled by default: `HY_OBS_LAG=1` on the policy server,
or `--obs-lag 1` in the bundled all-task driver. Set `HY_OBS_LAG=0` or
`--obs-lag 0` to disable it while retaining SDK rendering. This preserves current state/language and requires
SDK rendering. The smoke test above used this option.
