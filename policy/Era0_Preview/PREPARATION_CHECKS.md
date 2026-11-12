# Evaluation package checks

Matching model assets are prepared locally and have not been published. No model files are included in Git.

- PASS: weight and normalization file integrity; independent asset copies.
- PASS: offline model configuration and processor loading; compatible tensor structure.
- PASS: finite normalization arrays with the required dimensions.
- PASS: current RGB/state/instruction forwarding, reordered environment indices and reset.
- PASS: image history, valid-mask behavior and saved temporal configuration.
- PASS: native evaluation launch previews and Python/shell syntax checks.
- PASS: two-GPU simulator smoke test on `press_by_number` and `stack_bowls`, seed 0, two episodes each. Both completed with 2/2 successes, Score 100 and six camera videos each. This is limited functional evidence.
- PASS: removal of camera synchronization overrides and image-delay queues. CPU checks verify that SDK camera synchronization functions retain their original identities; rendering/delay CLI switches have been removed.

## Current native evaluation

The [unmodified official SDK summary](EVALUATION_RESULTS.md) reports partial Overview Score **38.36 ± 0.55**, SR **33.11 ± 0.27%**, and **123/126** complete task–seed cells. Saved episode count is **6,265/6,300**. Only `make_toast_random` remains incomplete: seeds 0/1/2 lack 5/20/10 episodes. Missing trajectories are not scored as failures.

All **159** complete result directories passed the SDK video inventory check. This does not certify incomplete directories or constitute a complete native evaluation.

The code cleanup was checked with CPU fixtures and launch previews; a new simulator run was not launched solely for that cleanup. Earlier-release debug reports do not establish current full-suite performance.
