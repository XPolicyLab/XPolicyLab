# Suite and task scores by evaluation seed

**Bagel-Robotics-1, checkpoint 1000.** Scores below include the labeled replacements and repaired variant documented in [the selection appendix](replacements-and-repairs.md). All partial scores are out of 100; the CSV stores native 0–1 scores.

## Suite scores

| Seed | Suite | Scored / planned | Successes | Failures | Success rate (%) | Partial /100 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 0 | Precision | 400/400 | 144 | 256 | 36.00 | 43.50 |
| 0 | Memory | 300/300 | 161 | 139 | 53.67 | 53.87 |
| 0 | Generalization | 600/600 | 158 | 442 | 26.33 | 33.52 |
| 0 | Long Horizon | 400/400 | 145 | 255 | 36.25 | 53.39 |
| 0 | Open | 400/400 | 15 | 385 | 3.75 | 4.36 |
| **0** | **Equal-weight five-suite average** | **2,100/2,100** | **623** | **1477** | **31.20** | **37.73** |
| 1 | Precision | 400/400 | 126 | 274 | 31.50 | 40.42 |
| 1 | Memory | 300/300 | 173 | 127 | 57.67 | 58.00 |
| 1 | Generalization | 600/600 | 183 | 417 | 30.50 | 36.83 |
| 1 | Long Horizon | 400/400 | 137 | 263 | 34.25 | 53.00 |
| 1 | Open | 400/400 | 16 | 384 | 4.00 | 4.47 |
| **1** | **Equal-weight five-suite average** | **2,100/2,100** | **635** | **1465** | **31.58** | **38.55** |
| 2 | Precision | 400/400 | 139 | 261 | 34.75 | 43.38 |
| 2 | Memory | 300/300 | 175 | 125 | 58.33 | 58.82 |
| 2 | Generalization | 600/600 | 169 | 431 | 28.17 | 35.27 |
| 2 | Long Horizon | 400/400 | 142 | 258 | 35.50 | 52.70 |
| 2 | Open | 400/400 | 13 | 387 | 3.25 | 3.90 |
| **2** | **Equal-weight five-suite average** | **2,100/2,100** | **638** | **1462** | **32.00** | **38.81** |

## Three-seed suite averages

| Suite | Successes / reported outcomes (3 seeds) | Success rate (%) | Partial score /100 |
| --- | ---: | ---: | ---: |
| Precision | 409/1200 | 34.08 ± 1.90 | 42.43 ± 1.42 |
| Memory | 509/900 | 56.56 ± 2.06 | 56.89 ± 2.17 |
| Generalization | 510/1800 | 28.33 ± 1.71 | 35.20 ± 1.35 |
| Long Horizon | 424/1200 | 35.33 ± 0.82 | 53.03 ± 0.28 |
| Open | 44/1200 | 3.67 ± 0.31 | 4.25 ± 0.25 |
| **Five-suite average** | **6,300 outcomes** | **31.59 ± 0.33** | **38.36 ± 0.46** |

All ± values in this custom breakdown are population standard deviations across the three seed-level values. In the unchanged native [summary](summary.md), overview ± values use that definition, while per-task ± values use half the highest-to-lowest range. The native report also includes an episode/task-weighted final row; use its five-suite Overview for the headline.

## Task scores

Each task has 50 reported outcomes per seed. Generalization combines 25 standard and 25 randomized episodes. The success columns show count, denominator and percentage; failures equal the denominator minus successes.

### Precision

| Task | Seed 0 successes (%) | Seed 0 partial | Seed 1 successes (%) | Seed 1 partial | Seed 2 successes (%) | Seed 2 partial | Mean SR (%) ± SD | Mean partial ± SD |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Build tower | 40/50 (80.00%) | 85.60 | 32/50 (64.00%) | 73.60 | 34/50 (68.00%) | 75.40 | 70.67 ± 6.80 | 78.20 ± 5.28 |
| Deposit coin | 14/50 (28.00%) | 32.40 | 11/50 (22.00%) | 26.80 | 12/50 (24.00%) | 28.40 | 24.67 ± 2.49 | 29.20 ± 2.36 |
| Fasten screws | 24/50 (48.00%) | 72.20 | 18/50 (36.00%) | 66.80 | 20/50 (40.00%) | 68.20 | 41.33 ± 4.99 | 69.07 ± 2.29 |
| Insert key | 0/50 (0.00%) | 15.00 | 0/50 (0.00%) | 15.00 | 0/50 (0.00%) | 15.00 | 0.00 ± 0.00 | 15.00 ± 0.00 |
| Insert tubes | 36/50 (72.00%) | 82.80 | 32/50 (64.00%) | 75.20 | 30/50 (60.00%) | 74.00 | 65.33 ± 4.99 | 77.33 ± 3.90 |
| Play xylophone | 0/50 (0.00%) | 0.00 | 0/50 (0.00%) | 0.00 | 0/50 (0.00%) | 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 |
| Plug in charger | 6/50 (12.00%) | 12.00 | 2/50 (4.00%) | 4.00 | 4/50 (8.00%) | 8.00 | 8.00 ± 3.27 | 8.00 ± 3.27 |
| Pour balls into vase | 24/50 (48.00%) | 48.00 | 31/50 (62.00%) | 62.00 | 39/50 (78.00%) | 78.00 | 62.67 ± 12.26 | 62.67 ± 12.26 |

### Memory

| Task | Seed 0 successes (%) | Seed 0 partial | Seed 1 successes (%) | Seed 1 partial | Seed 2 successes (%) | Seed 2 partial | Mean SR (%) ± SD | Mean partial ± SD |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Cover blocks | 50/50 (100.00%) | 100.00 | 50/50 (100.00%) | 100.00 | 50/50 (100.00%) | 100.00 | 100.00 ± 0.00 | 100.00 ± 0.00 |
| Imitate sorting sequence | 1/50 (2.00%) | 3.20 | 0/50 (0.00%) | 2.00 | 0/50 (0.00%) | 2.90 | 0.67 ± 0.94 | 2.70 ± 0.51 |
| Match and pick from conveyor | 41/50 (82.00%) | 82.00 | 46/50 (92.00%) | 92.00 | 43/50 (86.00%) | 86.00 | 86.67 ± 4.11 | 86.67 ± 4.11 |
| Press by number | 49/50 (98.00%) | 98.00 | 50/50 (100.00%) | 100.00 | 50/50 (100.00%) | 100.00 | 99.33 ± 0.94 | 99.33 ± 0.94 |
| Swap t | 11/50 (22.00%) | 22.00 | 11/50 (22.00%) | 22.00 | 16/50 (32.00%) | 32.00 | 25.33 ± 4.71 | 25.33 ± 4.71 |
| Swap blocks | 9/50 (18.00%) | 18.00 | 16/50 (32.00%) | 32.00 | 16/50 (32.00%) | 32.00 | 27.33 ± 6.60 | 27.33 ± 6.60 |

### Generalization

| Task | Seed 0 successes (%) | Seed 0 partial | Seed 1 successes (%) | Seed 1 partial | Seed 2 successes (%) | Seed 2 partial | Mean SR (%) ± SD | Mean partial ± SD |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Arrange largest number | 3/50 (6.00%) | 10.60 | 2/50 (4.00%) | 11.00 | 0/50 (0.00%) | 7.60 | 3.33 ± 2.49 | 9.73 ± 1.52 |
| Fold clothes | 33/50 (66.00%) | 68.00 | 36/50 (72.00%) | 74.40 | 37/50 (74.00%) | 76.80 | 70.67 ± 3.40 | 73.07 ± 3.71 |
| Hang mugs | 15/50 (30.00%) | 41.40 | 14/50 (28.00%) | 38.50 | 14/50 (28.00%) | 40.60 | 28.67 ± 0.94 | 40.17 ± 1.22 |
| Make toast | 16/50 (32.00%) | 47.50 | 22/50 (44.00%) | 52.50 | 17/50 (34.00%) | 45.00 | 36.67 ± 5.25 | 48.33 ± 3.12 |
| Pack objects into box | 1/50 (2.00%) | 26.50 | 6/50 (12.00%) | 35.40 | 3/50 (6.00%) | 29.90 | 6.67 ± 4.11 | 30.60 ± 3.67 |
| Pour liquid into cup | 15/50 (30.00%) | 30.00 | 22/50 (44.00%) | 44.00 | 23/50 (46.00%) | 46.00 | 40.00 ± 7.12 | 40.00 ± 7.12 |
| Push t | 0/50 (0.00%) | 0.00 | 1/50 (2.00%) | 2.00 | 1/50 (2.00%) | 2.00 | 1.33 ± 0.94 | 1.33 ± 0.94 |
| Sort nesting dolls by size | 10/50 (20.00%) | 20.00 | 10/50 (20.00%) | 20.00 | 7/50 (14.00%) | 14.00 | 18.00 ± 2.83 | 18.00 ± 2.83 |
| Stack blocks | 18/50 (36.00%) | 43.50 | 22/50 (44.00%) | 50.00 | 23/50 (46.00%) | 52.00 | 42.00 ± 4.32 | 48.50 ± 3.63 |
| Stack bowls | 24/50 (48.00%) | 55.50 | 26/50 (52.00%) | 58.90 | 23/50 (46.00%) | 54.10 | 48.67 ± 2.49 | 56.17 ± 2.02 |
| Store laptop and headphones | 18/50 (36.00%) | 49.20 | 20/50 (40.00%) | 51.20 | 17/50 (34.00%) | 47.20 | 36.67 ± 2.49 | 49.20 ± 1.63 |
| Sweep blocks | 5/50 (10.00%) | 10.00 | 2/50 (4.00%) | 4.00 | 4/50 (8.00%) | 8.00 | 7.33 ± 2.49 | 7.33 ± 2.49 |

### Long Horizon

| Task | Seed 0 successes (%) | Seed 0 partial | Seed 1 successes (%) | Seed 1 partial | Seed 2 successes (%) | Seed 2 partial | Mean SR (%) ± SD | Mean partial ± SD |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Classify objects | 23/50 (46.00%) | 56.70 | 16/50 (32.00%) | 44.50 | 21/50 (42.00%) | 51.10 | 40.00 ± 5.89 | 50.77 ± 4.99 |
| Fill egg holder | 1/50 (2.00%) | 11.00 | 2/50 (4.00%) | 13.80 | 0/50 (0.00%) | 12.20 | 2.00 ± 1.63 | 12.33 ± 1.15 |
| Fill pen holder | 18/50 (36.00%) | 53.90 | 11/50 (22.00%) | 43.90 | 15/50 (30.00%) | 52.40 | 29.33 ± 5.73 | 50.07 ± 4.40 |
| Make kong | 42/50 (84.00%) | 84.00 | 43/50 (86.00%) | 86.00 | 44/50 (88.00%) | 88.00 | 86.00 ± 1.63 | 86.00 ± 1.63 |
| Organize table | 7/50 (14.00%) | 57.50 | 9/50 (18.00%) | 64.50 | 6/50 (12.00%) | 58.00 | 14.67 ± 2.49 | 60.00 ± 3.19 |
| Play stacking toy | 7/50 (14.00%) | 31.80 | 6/50 (12.00%) | 32.00 | 4/50 (8.00%) | 27.20 | 11.33 ± 2.49 | 30.33 ± 2.22 |
| Play tic tac toe | 0/50 (0.00%) | 35.80 | 1/50 (2.00%) | 40.50 | 3/50 (6.00%) | 33.90 | 2.67 ± 2.49 | 36.73 ± 2.77 |
| Put bottles into dustbin | 47/50 (94.00%) | 96.40 | 49/50 (98.00%) | 98.80 | 49/50 (98.00%) | 98.80 | 96.67 ± 1.89 | 98.00 ± 1.13 |

### Open

| Task | Seed 0 successes (%) | Seed 0 partial | Seed 1 successes (%) | Seed 1 partial | Seed 2 successes (%) | Seed 2 partial | Mean SR (%) ± SD | Mean partial ± SD |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Align blocks | 0/50 (0.00%) | 0.00 | 0/50 (0.00%) | 0.00 | 0/50 (0.00%) | 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 |
| Classify objects by language | 0/50 (0.00%) | 3.60 | 0/50 (0.00%) | 2.60 | 3/50 (6.00%) | 9.20 | 2.00 ± 2.83 | 5.13 ± 2.90 |
| General pickup | 13/50 (26.00%) | 26.00 | 12/50 (24.00%) | 24.00 | 8/50 (16.00%) | 16.00 | 22.00 ± 4.32 | 22.00 ± 4.32 |
| Pick from conveyor by image | 0/50 (0.00%) | 0.00 | 0/50 (0.00%) | 0.00 | 0/50 (0.00%) | 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 |
| Pour by language | 0/50 (0.00%) | 0.00 | 0/50 (0.00%) | 0.00 | 0/50 (0.00%) | 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 |
| Solve equation | 0/50 (0.00%) | 0.00 | 0/50 (0.00%) | 0.00 | 0/50 (0.00%) | 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 |
| Stack blocks by language | 2/50 (4.00%) | 4.80 | 4/50 (8.00%) | 9.20 | 2/50 (4.00%) | 6.00 | 5.33 ± 1.89 | 6.67 ± 1.86 |
| Store tools in toolbox | 0/50 (0.00%) | 0.50 | 0/50 (0.00%) | 0.00 | 0/50 (0.00%) | 0.00 | 0.00 ± 0.00 | 0.17 ± 0.24 |

## Generalization: standard versus randomized

Each split has 300 episodes per seed. The following rows preserve the 25-episode task halves.

| Seed | Split | Successes / scored | Success rate (%) | Partial /100 |
| --- | --- | ---: | ---: | ---: |
| 0 | Standard | 134/300 | 44.67 | 51.13 |
| 0 | Randomized | 24/300 | 8.00 | 15.90 |
| 1 | Standard | 148/300 | 49.33 | 54.88 |
| 1 | Randomized | 35/300 | 11.67 | 18.77 |
| 2 | Standard | 129/300 | 43.00 | 49.30 |
| 2 | Randomized | 40/300 | 13.33 | 21.23 |

<details>
<summary>Per-task standard and randomized results</summary>

| Task / split | Seed 0 successes (%) | Seed 0 partial | Seed 1 successes (%) | Seed 1 partial | Seed 2 successes (%) | Seed 2 partial |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Arrange largest number / standard | 3/25 (12.00%) | 16.60 | 1/25 (4.00%) | 13.00 | 0/25 (0.00%) | 8.60 |
| Arrange largest number / randomized | 0/25 (0.00%) | 4.60 | 1/25 (4.00%) | 9.00 | 0/25 (0.00%) | 6.60 |
| Fold clothes / standard | 17/25 (68.00%) | 70.40 | 20/25 (80.00%) | 80.80 | 16/25 (64.00%) | 68.00 |
| Fold clothes / randomized | 16/25 (64.00%) | 65.60 | 16/25 (64.00%) | 68.00 | 21/25 (84.00%) | 85.60 |
| Hang mugs / standard | 15/25 (60.00%) | 71.00 | 14/25 (56.00%) | 67.40 | 14/25 (56.00%) | 69.00 |
| Hang mugs / randomized | 0/25 (0.00%) | 11.80 | 0/25 (0.00%) | 9.60 | 0/25 (0.00%) | 12.20 |
| Make toast / standard | 14/25 (56.00%) | 71.00 | 18/25 (72.00%) | 79.00 | 15/25 (60.00%) | 67.00 |
| Make toast / randomized | 2/25 (8.00%) | 24.00 | 4/25 (16.00%) | 26.00 | 2/25 (8.00%) | 23.00 |
| Pack objects into box / standard | 0/25 (0.00%) | 29.40 | 4/25 (16.00%) | 42.60 | 2/25 (8.00%) | 34.60 |
| Pack objects into box / randomized | 1/25 (4.00%) | 23.60 | 2/25 (8.00%) | 28.20 | 1/25 (4.00%) | 25.20 |
| Pour liquid into cup / standard | 13/25 (52.00%) | 52.00 | 17/25 (68.00%) | 68.00 | 16/25 (64.00%) | 64.00 |
| Pour liquid into cup / randomized | 2/25 (8.00%) | 8.00 | 5/25 (20.00%) | 20.00 | 7/25 (28.00%) | 28.00 |
| Push t / standard | 0/25 (0.00%) | 0.00 | 1/25 (4.00%) | 4.00 | 1/25 (4.00%) | 4.00 |
| Push t / randomized | 0/25 (0.00%) | 0.00 | 0/25 (0.00%) | 0.00 | 0/25 (0.00%) | 0.00 |
| Sort nesting dolls by size / standard | 9/25 (36.00%) | 36.00 | 9/25 (36.00%) | 36.00 | 7/25 (28.00%) | 28.00 |
| Sort nesting dolls by size / randomized | 1/25 (4.00%) | 4.00 | 1/25 (4.00%) | 4.00 | 0/25 (0.00%) | 0.00 |
| Stack blocks / standard | 18/25 (72.00%) | 76.20 | 22/25 (88.00%) | 89.80 | 23/25 (92.00%) | 93.20 |
| Stack blocks / randomized | 0/25 (0.00%) | 10.80 | 0/25 (0.00%) | 10.20 | 0/25 (0.00%) | 10.80 |
| Stack bowls / standard | 24/25 (96.00%) | 96.60 | 23/25 (92.00%) | 93.20 | 21/25 (84.00%) | 86.40 |
| Stack bowls / randomized | 0/25 (0.00%) | 14.40 | 3/25 (12.00%) | 24.60 | 2/25 (8.00%) | 21.80 |
| Store laptop and headphones / standard | 17/25 (68.00%) | 78.40 | 17/25 (68.00%) | 76.80 | 12/25 (48.00%) | 60.80 |
| Store laptop and headphones / randomized | 1/25 (4.00%) | 20.00 | 3/25 (12.00%) | 25.60 | 5/25 (20.00%) | 33.60 |
| Sweep blocks / standard | 4/25 (16.00%) | 16.00 | 2/25 (8.00%) | 8.00 | 2/25 (8.00%) | 8.00 |
| Sweep blocks / randomized | 1/25 (4.00%) | 4.00 | 0/25 (0.00%) | 0.00 | 2/25 (8.00%) | 8.00 |

</details>
