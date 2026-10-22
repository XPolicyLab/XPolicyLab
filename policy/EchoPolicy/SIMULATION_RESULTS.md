# RoboDojo simulation results — seed 0

**Official-algorithm success rate: 13.45%; score: 19.11/100.**

Self-run evaluation completed on 2026-10-08: **42 base tasks, 54 runnable task variants, 2,100 records (278 successes / 1,822 failures), and 6,300 camera videos verified**. This replaces the earlier 10-episode `cover_blocks` smoke result as the main simulation evidence for this PR.

Only **seed 0** was evaluated. The contributor has confirmed with the RoboDojo maintainers that one complete seed is acceptable for the self-test results in this PR. All tasks reach their native per-seed record counts; seeds 1/2 and cross-seed standard deviations are outside this agreed PR self-test scope. These are self-run results, not an official cloud-verified leaderboard listing. See the [evaluation guide](https://robodojo-benchmark.com/doc/usage/quick-evaluation/) and [official submission workflow](https://robodojo-benchmark.com/doc/usage/robodojo-submission/) for separate publication/evaluation procedures.

## Metric calculation

- Non-Generalization tasks: 50 records each. The 12 Generalization tasks: 25 standard + 25 random records each. All 2,100 selected records enter the calculation.
- Per-task success rate = successes / 50 × 100; score = sum of the 50 recorded scores / 50 × 100.
- Average task metrics within each of the five capability dimensions, then average the five dimension values equally.
- The pooled episode success fraction is 278/2,100 = 13.2381%; it is a separate descriptive statistic, not the official dimension-weighted 13.45%. No cross-seed standard deviation is reported.
- The unchanged [official summarizer](https://github.com/RoboDojo-Benchmark/RoboDojo/blob/cb2a37d58dbbeb2777b0c2b30e97ecb8ab91095c/scripts/internal/summarize_result.py) was rerun on an isolated view containing exactly 46 original runs + 8 supplemented runs. Its SHA-256 matches the evaluated version: `9ec361af587e1788d83d3cf8670e05b817ff252d33c33829a6337dc16ae370d2`. It returned 42 seed-0 task values and 0 result/video mismatches.

| Dimension | Tasks | Success rate (%) | Score /100 |
|---|---:|---:|---:|
| Generalization | 12 | 13.50 | 19.75 |
| Precision | 8 | 11.75 | 18.75 |
| Long-Horizon | 8 | 20.75 | 33.35 |
| Memory | 6 | 18.00 | 19.43 |
| Open | 8 | 3.25 | 4.25 |
| **Equal mean of dimensions** | **42** | **13.45** | **19.11** |

## Evaluated configuration and reproducibility scope

- RoboDojo source baseline: [`726e9aabfaa642203722eb126f5eaf0f37f3e1ad`](https://github.com/RoboDojo-Benchmark/RoboDojo/commit/726e9aabfaa642203722eb126f5eaf0f37f3e1ad). Latest upstream checked on 2026-10-09: `cb2a37d58dbbeb2777b0c2b30e97ecb8ab91095c`; the scoring script is byte-identical.
- `arx_x5`, joint actions, seed 0; official RoboDojo Pi05 checkpoint step `59999`, `pi05_base_aloha_full_sim_arx-x5_seed_0`, `assets/arx_x5_sim/norm_stats.json`.
- Provider model identifier `gemini-3.8-flash`, thinking `medium`; fixed Pi05 batch 160, 16 candidates/environment, JAX memory fraction 0.9.
- Four policy servers served up to eight simulator clients (at most two per policy); 10 environments/client. Observations transferred every 10 action steps with lossless compression and exact batched forward kinematics.
- **Evaluated deployment differs from the PR source base `c1a2cca`.** Deployment changes include transport batching/compression/session handling, exact batched FK and API retry handling. The machine-readable artifact records all 35 evaluated source hashes and the 9 differences from that PR base. Six simulator runtime hashes are also recorded. This documentation-only update does not assert that an unmodified PR checkout reproduces the measured results; reproducing the campaign requires those deployment changes. No benchmark task scoring or success criteria were changed.
- The original sweep took 102.041 hours; supplementation took 9.088 hours. The calendar span from first evaluation launch to final supplementation was 118.807 hours, including interruptions and gaps. These are campaign wall times, not a controlled throughput benchmark.

## Layout replacement scope

The final selection keeps all 2,070 original records and appends 30 sequential extra layouts through the official replacement mechanism. Eight deficient runs were continued in a separate result scope; the final summary uses the supplemented copy once, never both original and copy. All saved successes and failures were retained. Original unstable/PhysX-abandoned layouts were not replayed. This is **not full coverage of the original canonical layout IDs**.

Original exclusions and saved replacement IDs below are set-level disclosures, **not one-to-one causal pairings**. Additional excluded extra IDs are listed separately. Infrastructure session failures and reset-loop-consumed IDs were not counted as physical exclusions.

| Runnable task | Original excluded IDs | Saved extra IDs | Additional excluded extras |
|---|---|---|---|
| deposit_coin | 23 | 50 | — |
| fill_egg_holder | 5, 9, 39, 41, 46 | 50, 51, 52, 53, 54 | 56 |
| imitate_sorting_sequence | 4, 8, 17, 36, 48 | 50, 51, 52, 53, 54 | — |
| make_toast | 3, 6, 10, 11, 14, 15, 18 | 25, 26, 27, 28, 29, 30, 31 | 46 |
| make_toast_random | 13, 14, 16, 18, 20, 21 | 25, 27, 28, 29, 30, 32 | 26, 31 |
| play_tic_tac_toe | 0, 26, 33 | 50, 51, 52 | — |
| pour_liquid_into_cup_random | 8 | 25 | — |
| store_laptop_and_headphones | 2, 4 | 25, 26 | — |

The 30 supplemental episodes alone had 1 success (3.33%) and a record-weighted mean score of 12.33/100. This subset statistic does not use the five-dimension overall aggregation.

## Per-task results

Generalization rows combine their standard and random halves. All rows contain 50 records.

| Task | Successes /50 | Success rate (%) | Score /100 |
|---|---:|---:|---:|
| align_blocks | 0 | 0.00 | 0.00 |
| arrange_largest_number | 2 | 4.00 | 10.90 |
| build_tower | 12 | 24.00 | 34.80 |
| classify_objects | 18 | 36.00 | 48.40 |
| classify_objects_by_language | 0 | 0.00 | 3.00 |
| cover_blocks | 36 | 72.00 | 77.50 |
| deposit_coin | 1 | 2.00 | 9.60 |
| fasten_screws | 5 | 10.00 | 19.60 |
| fill_egg_holder | 0 | 0.00 | 8.60 |
| fill_pen_holder | 6 | 12.00 | 35.00 |
| fold_clothes | 24 | 48.00 | 54.00 |
| general_pickup | 13 | 26.00 | 26.00 |
| hang_mugs | 1 | 2.00 | 10.50 |
| imitate_sorting_sequence | 1 | 2.00 | 5.10 |
| insert_key | 0 | 0.00 | 14.40 |
| insert_tubes | 16 | 32.00 | 45.60 |
| make_kong | 8 | 16.00 | 16.00 |
| make_toast | 1 | 2.00 | 11.50 |
| match_and_pick_from_conveyor | 11 | 22.00 | 22.00 |
| organize_table | 2 | 4.00 | 44.50 |
| pack_objects_into_box | 5 | 10.00 | 29.80 |
| pick_from_conveyor_by_image | 0 | 0.00 | 0.00 |
| play_Xylophone | 0 | 0.00 | 0.00 |
| play_stacking_toy | 0 | 0.00 | 0.00 |
| play_tic_tac_toe | 0 | 0.00 | 15.80 |
| plug_in_charger | 1 | 2.00 | 2.00 |
| pour_balls_into_vase | 12 | 24.00 | 24.00 |
| pour_by_language | 0 | 0.00 | 0.00 |
| pour_liquid_into_cup | 11 | 22.00 | 22.00 |
| press_by_number | 5 | 10.00 | 10.00 |
| push_T | 1 | 2.00 | 2.00 |
| put_bottles_into_dustbin | 49 | 98.00 | 98.50 |
| solve_equation | 0 | 0.00 | 0.00 |
| sort_nesting_dolls_by_size | 3 | 6.00 | 6.00 |
| stack_blocks | 5 | 10.00 | 17.50 |
| stack_blocks_by_language | 0 | 0.00 | 4.00 |
| stack_bowls | 23 | 46.00 | 53.20 |
| store_laptop_and_headphones | 2 | 4.00 | 13.60 |
| store_tools_in_toolbox | 0 | 0.00 | 1.00 |
| swap_T | 0 | 0.00 | 0.00 |
| swap_blocks | 1 | 2.00 | 2.00 |
| sweep_blocks | 3 | 6.00 | 6.00 |

## Evidence and checks

- [Machine-readable results](results/seed0-20261008.json) include all 2,100 per-record layout IDs, success flags and scores, per-run original result hashes, per-task/dimension metrics and evaluated source hashes. Private endpoints, credentials and machine paths are omitted.
- 6,318 original result/video/log files were rehashed unchanged, including 2,070 original records and 6,210 videos. The 810 copied original videos were also SHA-verified unchanged.
- All 90 new videos passed ffprobe metadata/duration and SHA checks. The combined 6,300-video inventory has one head, left-wrist and right-wrist video per record, with no duplicates or missing files. Original metadata checks were reused only after byte-equivalence verification; full frame-by-frame decoding was not performed.
- 465 layout asset hashes, six simulator runtime files and 35 policy source files were verified. Raw videos and infrastructure incident logs remain in the evaluation archive and are not embedded in this PR.
- Requirements reviewed on 2026-10-09: [XPolicyLab contribution standard](https://github.com/XPolicyLab/XPolicyLab/blob/53c5da5a01bc720d1cf0e6ab6ed212e8cef0eda2/CONTRIBUTING.md), [RoboDojo Quick Evaluation](https://robodojo-benchmark.com/doc/usage/quick-evaluation/), and [RoboDojo Submission](https://robodojo-benchmark.com/doc/usage/robodojo-submission/).
- Earlier 35/42-task partial official metrics (16.03% / 21.62) are superseded for current reporting; their task coverage differs from this 42/42-task result and they are not a controlled before/after policy comparison.
