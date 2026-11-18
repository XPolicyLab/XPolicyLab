# Evaluation package checks

No model files are included in Git.

- PASS: weight and normalization file integrity; independent asset copies.
- PASS: offline model configuration and processor loading; compatible tensor structure.
- PASS: finite normalization arrays with the required dimensions.
- PASS: current RGB/state/instruction forwarding, reordered environment indices and reset.
- PASS: image history, valid-mask behavior and saved temporal configuration.
- PASS: native evaluation launch previews and Python/shell syntax checks.
- PASS: two-GPU simulator smoke test on `press_by_number` and `stack_bowls`, seed 0, two episodes each. Both completed with 2/2 successes, Score 100 and six camera videos each. This is limited functional evidence.
- PASS: plain/encoded RGB debug, complete history accumulation over 34 chunks, environment reordering, reset, finite actions and action dimensions.

## Current native evaluation

The [unmodified official SDK summary](EVALUATION_RESULTS.md) reports partial Overview Score **43.00 ± 0.68**, SR **37.61 ± 0.88%**, and **123/126** complete task–seed cells. Saved episode count is **6,255/6,300**. Only `make_toast_random` remains incomplete: seeds 0/1/2 each lack 15 episodes. Missing trajectories are not scored as failures.

The SDK video inventory check reports mismatches in all **159** checked directories because the available videos combine camera views rather than using separate camera files. This is a partial local result.
