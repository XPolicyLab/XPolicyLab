# Prepared revision checks

This is a local preparation record, not a published release or a new benchmark result. Earlier debug and simulation reports apply to the previous assets.

- PASS: independent copies of weights and normalization; SHA256 equality to the selected source files. No hard links or source edits.
- PASS: checkpoint configuration and processor load offline using the packaged runtime; tensor names, dimensions and dtypes match the existing architecture.
- PASS: finite normalization arrays with the expected state/action dimensions.
- PASS: 9/1/1 history, interval 40, chunk cadence and initial valid-mask behavior; wrists retain only the current observation.
- PASS: current-image policy input with zero added observation lag.
- PASS: time-RoPE coefficients exactly match the selected assets. Local progress metadata is replaced by a saturated equivalent specification; effective maximum phase remains 1.0 rad.
- PASS: previous assets are rejected by the new compatibility check, preventing accidental use with the new defaults.
- PASS: Python syntax, shell syntax and evaluation plan previews.
- Native preview: all SDK default tasks, seeds 0/1/2, 6,300 expected episodes.
- Diagnostic preview: supplied 54-task inventory, seed 0, ten episodes per task, 540 expected episodes. This is labelled partial and does not change the native default.
- PASS: real-weight simulator debug on two GPUs with the audited official SDK, current observations and SDK rendering: `press_by_number` and `stack_bowls`, seed 0, two episodes each. Both completed with 2/2 successes, Score 100, exit 0 and six camera videos each. This verifies the inference/transport/simulator path, not full-suite performance.
- NOT RUN: complete native three-seed simulator evaluation of these prepared assets.

The existing audited SDK snapshot passes source validation. A separate customized local SDK has additional output, video and random-task concurrency changes; its hashes are intentionally rejected. No extra host version was accepted and no SDK files were edited. A full score comparison requires a selected, consistent SDK and actual evaluation.

The weights and normalization remain outside Git under `checkpoints/`; `checkpoint_release.json` identifies the local prepared directory and checksums. The earlier public asset metadata is retained in `checkpoint_release_published.json`. Default remote downloading is disabled while the new revision is unpublished.
