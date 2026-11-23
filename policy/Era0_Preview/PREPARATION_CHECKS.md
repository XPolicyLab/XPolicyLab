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

The [official SDK summary](EVALUATION_RESULTS.md) reports **Score 44.54 ± 0.91**, **SR 39.18 ± 1.13%**, with **122/126** task–seed cells complete and **6,215/6,300** episodes saved. These are partial local results; incomplete cells are excluded.

Incomplete task–seed cells:

- seed 0: `make_toast_random` (5/25)
- seed 0: `store_laptop_and_headphones` (0/25)
- seed 1: `make_toast_random` (5/25)
- seed 2: `make_toast_random` (5/25)

Results use the official SDK summarizer. Missing trajectories are not scored as failures.
