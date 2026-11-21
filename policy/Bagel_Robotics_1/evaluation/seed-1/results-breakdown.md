# Bagel-Robotics-1 — complete reported results

Checkpoint 1000, **evaluation seed 1**. This companion report contains the previously reported five-suite totals and all 42 task results. The tables below are a readable companion to the unchanged [native summary](summary.md); all 42 tasks have been verified against retained per-episode native records. Success means full task completion; partial scores measure credited progress, including in failed episodes.

All 2,100 reported outcomes were checked against retained native result files and completion records: 1,350 from Node 2, plus 450 Generalization outcomes and 300 reported Memory outcomes from the retained Nebius archive. The Memory archive contains 299 valid originals and a separately labeled repaired variant.

| Suite | Successes / reported outcomes | Success rate | Mean partial /100 |
|---|---:|---:|---:|
| Precision | 126/400 | 31.50% | 40.43 |
| Memory | 173/300 | 57.67% | 58.00 |
| Long Horizon | 137/400 | 34.25% | 53.00 |
| Generalization | 183/600 | 30.50% | 36.82 |
| Open | 16/400 | 4.00% | 4.48 |
| **Equal-weight five-suite average** | **635/2,100 total** | **31.58%** | **38.55** |

The overall score is the mean of the five suite scores, not successes divided by all episodes.

## Scene selection and sources

- Memory: 299 original valid results plus the separately repaired sorting-sequence layout 47 (failure, partial 0). Original-only totals: 173/299 successes (57.86%), partial 58.19/100. The original layout remains unscored.
- Long Horizon: 398 originals plus supplied egg-holder layouts 50 and 51, replacing unscored originals 15 and 20. The replacements scored 0 and 0.25, both failures.
- Open: 399 originals plus supplied pickup layout 50, replacing unscored original 5. The replacement failed with partial 0.
- Generalization: 25 standard plus 25 randomized episodes per task. All 12 tasks were checked against native records: three from Node 2 and nine from the retained Nebius archive. The combined standard/random total is 600 outcomes.

“Original” refers to the frozen task selections. Deposit Coin uses supplied layouts 50–53 instead of 4, 8, 39 and 42; Imitate Sorting Sequence uses supplied layouts 50–53 instead of 18, 33, 34 and 44. These selections were fixed before the saved runs.

## Task-level results

### Precision

| Task | Successes / reported outcomes | Success rate | Mean partial /100 |
|---|---:|---:|---:|
| Build tower | 32/50 | 64.00% | 73.60 |
| Deposit coin | 11/50 | 22.00% | 26.80 |
| Fasten screws | 18/50 | 36.00% | 66.80 |
| Insert key | 0/50 | 0.00% | 15.00 |
| Insert tubes | 32/50 | 64.00% | 75.20 |
| Play xylophone | 0/50 | 0.00% | 0.00 |
| Plug in charger | 2/50 | 4.00% | 4.00 |
| Pour balls into vase | 31/50 | 62.00% | 62.00 |

### Memory

| Task | Successes / reported outcomes | Success rate | Mean partial /100 |
|---|---:|---:|---:|
| Cover blocks | 50/50 | 100.00% | 100.00 |
| Imitate sorting sequence | 0/50 | 0.00% | 2.00 |
| Match and pick from conveyor | 46/50 | 92.00% | 92.00 |
| Press by number | 50/50 | 100.00% | 100.00 |
| Swap T | 11/50 | 22.00% | 22.00 |
| Swap blocks | 16/50 | 32.00% | 32.00 |

### Long Horizon

| Task | Successes / reported outcomes | Success rate | Mean partial /100 |
|---|---:|---:|---:|
| Classify objects | 16/50 | 32.00% | 44.50 |
| Fill egg holder | 2/50 | 4.00% | 13.80 |
| Fill pen holder | 11/50 | 22.00% | 43.90 |
| Make kong | 43/50 | 86.00% | 86.00 |
| Organize table | 9/50 | 18.00% | 64.50 |
| Play stacking toy | 6/50 | 12.00% | 32.00 |
| Play tic tac toe | 1/50 | 2.00% | 40.50 |
| Put bottles into dustbin | 49/50 | 98.00% | 98.80 |

### Generalization

| Task | Successes / reported outcomes | Success rate | Mean partial /100 |
|---|---:|---:|---:|
| Arrange largest number | 2/50 | 4.00% | 11.00 |
| Fold clothes | 36/50 | 72.00% | 74.40 |
| Hang mugs | 14/50 | 28.00% | 38.50 |
| Make toast | 22/50 | 44.00% | 52.50 |
| Pack objects into box | 6/50 | 12.00% | 35.40 |
| Pour liquid into cup | 22/50 | 44.00% | 44.00 |
| Push T | 1/50 | 2.00% | 2.00 |
| Sort nesting dolls by size | 10/50 | 20.00% | 20.00 |
| Stack blocks | 22/50 | 44.00% | 50.00 |
| Stack bowls | 26/50 | 52.00% | 58.90 |
| Store laptop and headphones | 20/50 | 40.00% | 51.20 |
| Sweep blocks | 2/50 | 4.00% | 4.00 |

### Open

| Task | Successes / reported outcomes | Success rate | Mean partial /100 |
|---|---:|---:|---:|
| Align blocks | 0/50 | 0.00% | 0.00 |
| Classify objects by language | 0/50 | 0.00% | 2.60 |
| General pickup | 12/50 | 24.00% | 24.00 |
| Pick from conveyor by image | 0/50 | 0.00% | 0.00 |
| Pour by language | 0/50 | 0.00% | 0.00 |
| Solve equation | 0/50 | 0.00% | 0.00 |
| Stack blocks by language | 4/50 | 8.00% | 9.20 |
| Store tools in toolbox | 0/50 | 0.00% | 0.00 |

Checkpoint identity: `790bffc9464fa7a026aa43b217aec2033d186895cad643166daa0bcc01914d5c`.

## Reading the native summary

[summary.md](summary.md) is the unchanged output of `bash scripts/robodojo.sh summarize` at [RoboDojo revision f57e163](https://github.com/RoboDojo-Benchmark/RoboDojo/tree/f57e1633ffed59761bc8e568cfe7530c4b33381c), generated from the retained native records. [verification.json](verification.json) records its SHA-256 and the completed checks.

All 42 tasks are reported for seed 1. The native summary shows 42/126 (33.3%) because its denominator includes seeds 0, 1 and 2; the other seed columns are blank. Its `±0.00` reflects one available seed, not measured variation across seeds. The native script's rounding can differ by 0.01 from the readable tables above; the underlying episode scores agree. The summary's `with_labeled_replacements_and_repair` suffix identifies the reported scene selection; the model name is Bagel-Robotics-1.
