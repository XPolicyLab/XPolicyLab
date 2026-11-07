# Bagel-Robotics-1 results

Bagel-Robotics-1 / AdaPrune Final, checkpoint 1000. Success means full task completion; partial scores average credited progress over all reported outcomes, including failures. Partial values here are out of 100. Generalization combines standard and randomized episodes; other suites use their recorded standard scene selections.

| Suite | Successful / reported outcomes | Success rate | Mean partial /100 |
|---|---:|---:|---:|
| Precision | 126/400 | 31.50% | 40.43 |
| Memory† | 173/300 | 57.67% | 58.00 |
| Long Horizon‡ | 137/400 | 34.25% | 53.00 |
| Generalization | 183/600 | 30.50% | 36.82 |
| Open§ | 16/400 | 4.00% | 4.48 |
| **Equal-weight five-suite average** | **635/2,100 total; not the averaging formula** | **31.58%** | **38.55** |

## Original scenes, repairs and replacements

† Memory has 299 valid original results: 173 successes, **57.86% success and 58.19/100 partial**. Original sorting-sequence episode 47 remains unscored. Its separately repaired variant is a valid failure with score zero; the headline includes that variant, giving 300 reported outcomes. The repaired variant is not an unchanged-original result.

‡ Long Horizon includes 398 original results plus egg-holder replacements 50 and 51 for invalid original layouts 15 and 20. Both replacements failed the task, with partial scores 0 and 0.25. Original-only totals: 137/398 successes, 53.2035/100 partial.

§ Open includes 399 original results plus pickup replacement 50 for invalid original layout 5. That replacement failed with partial zero. Original-only totals: 16/399 successes, 4.4862/100 partial.

Generalization combines standard and randomized episodes over all 12 tasks (600 results). The nine user-supplied completed tasks contribute 131/450 successes; the three node-2 tasks contribute 52/150. The original stack-block layout 6 and laptop/headphones layout 22 succeeded on retry; these are original scenes, not replacements. Cancellation of duplicate queue entries does not discard the separately supplied completed results.

These totals contain **2,096 original results, three replacements and one repaired variant**. The overall score gives each suite equal weight, not each episode. Organizer acceptance of local substitutions or repairs is not assumed.

## Sources and scope

Precision, Open and Long Horizon retain the original receipt hashes and replacement evidence in `evidence/local-results.json`. The completed five-suite calculation is in `evidence/five-suite-results.json`.

Memory and nine Generalization tasks come from the user's supplied completed task summaries. Their raw episode logs were not independently re-audited during this packaging update. Stack blocks, store laptop/headphones and sort nesting dolls use node-2 receipts verified in the preceding results review, including successful original-layout retries. No new multi-seed estimate is implied.

These results belong to the frozen research runner. Packaged-adapter debug, smoke and matched-action validation are reported separately in `VALIDATION.md`; research scores do not substitute for those tests. This is not an official leaderboard entry.

## Task-level breakdown

### Precision

| Task | Successful / reported outcomes | Success rate | Mean partial /100 |
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

| Task | Successful / reported outcomes | Success rate | Mean partial /100 |
|---|---:|---:|---:|
| Cover blocks | 50/50 | 100.00% | 100.00 |
| Imitate sorting sequence | 0/50 | 0.00% | 2.00 |
| Match and pick from conveyor | 46/50 | 92.00% | 92.00 |
| Press by number | 50/50 | 100.00% | 100.00 |
| Swap T | 11/50 | 22.00% | 22.00 |
| Swap blocks | 16/50 | 32.00% | 32.00 |

### Long Horizon

| Task | Successful / reported outcomes | Success rate | Mean partial /100 |
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

| Task | Successful / reported outcomes | Success rate | Mean partial /100 |
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

| Task | Successful / reported outcomes | Success rate | Mean partial /100 |
|---|---:|---:|---:|
| Align blocks | 0/50 | 0.00% | 0.00 |
| General pickup | 12/50 | 24.00% | 24.00 |
| Stack blocks by language | 4/50 | 8.00% | 9.20 |
| Solve equation | 0/50 | 0.00% | 0.00 |
| Classify objects by language | 0/50 | 0.00% | 2.60 |
| Pick from conveyor by image | 0/50 | 0.00% | 0.00 |
| Store tools in toolbox | 0/50 | 0.00% | 0.00 |
| Pour by language | 0/50 | 0.00% | 0.00 |

Checkpoint identity: `790bffc9464fa7a026aa43b217aec2033d186895cad643166daa0bcc01914d5c`.
