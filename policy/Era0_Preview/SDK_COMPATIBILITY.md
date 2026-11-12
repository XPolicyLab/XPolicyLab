# RoboDojo SDK compatibility for Era0-Preview

The submitted adapter targets RoboDojo revision `f57e1633ffed59761bc8e568cfe7530c4b33381c`. The selected SDK owns evaluation results, videos, resume paths, task counts and score aggregation. No installed SDK file is modified. Keep XPolicyLab as an actual checkout inside the SDK workspace: the shared dimension helper does not support an external-checkout symlink.

## Execution profiles

- `official`: run the selected SDK's validated `src/eval_client/main.py`, `eval_env.py` and `utils/pipeline_utils.py` without source changes.
- `local` (default): the same audited SDK plus the optional IK rejection/hold guard in `eval_env.py`. This profile adds IK diagnostics only; it does not replace video handling, change random-task concurrency or relocate outputs.

Select the profile with `--execution-profile official` in the three-seed driver, or `ERA0_EXECUTION_PROFILE=official` on standard entries. The installed SDK files and optional guard sources are checked against `sdk_sources/manifest.json`; unknown versions fail before simulator startup. `sdk_sources/base/` and `sdk_sources/official/` match the audited SDK exactly. `sdk_sources/local/` extends that base with IK protection; its reviewable diff is under `sdk_sources/audit_patches/`. The SDK license is retained.

## Official results and videos

Both profiles use the native directory:

```text
eval_result/RoboDojo/<task>/Era0_Preview/arx_x5/
  <seed>_ckpt_name=<name>,action_type=ee/<run_id>/
    _result.json
    episode_<index>_cam_head_<outcome>.mp4
    episode_<index>_cam_left_wrist_<outcome>.mp4
    episode_<index>_cam_right_wrist_<outcome>.mp4
```

The SDK also owns temporary stream handling and resume manifests. No custom root, numeric-only seed directory, text overlay or combined video is added. Use a path-safe logical checkpoint name; pass any external weight directory separately through `ERA0_CHECKPOINT_PATH`.

From the SDK root, `bash scripts/robodojo.sh summarize` calls the installed `scripts/internal/summarize_result.py`, producing `eval_result/RoboDojo/_summary.md`. It selects results, computes all scores and checks video inventories using its own implementation. The three-seed driver invokes that same script; no policy-local summarizer is distributed. Driver `--output` stores launch logs and protocol/completeness metadata only.

## IK protection

In the local profile, each arm's IK result is compared with its live measured joints before control interpolation. Solver failure, missing/invalid targets or a raw per-joint displacement above 0.5 rad rejects that arm's target and holds its measured position through interpolation. Other arms and grippers proceed, and the normal action-step budget is consumed. Joint differences are not wrapped modulo 2π. Invalid measured joints raise the SDK's `UnStableError` before a batch is pushed.

`ik_guard_config.json` and per-episode `ik_rejections` record guard behavior. Counters reset per episode. This can affect executed actions and scores; the official profile does not install it.

## Image timing

The policy always consumes current observations and uses native SDK rendering. No camera synchronization overrides or image-delay queues are installed. Physics, rewards and layouts are unchanged.

## Verification limits

CPU tests cover real guard call sites, native save/resume paths, identical SDK video methods, SDK source validation, subprocess scheduling and restart. A separate integration fixture was accepted by the actual SDK summarizer with zero result/video filename mismatches. Fixture video files are placeholders, not simulator output. Real plain/encoded debug loops and full three-seed simulator evaluation remain required before an official submission.
