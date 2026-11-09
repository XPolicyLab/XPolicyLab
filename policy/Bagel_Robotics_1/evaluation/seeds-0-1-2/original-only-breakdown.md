# Original-selection-only results

This view excludes the 28 later replacement layouts and the repaired Memory variant. It contains **6,271 scored original-selection outcomes**. Invalid original episodes remain unscored, not zero-valued failures. Seed 1 Precision retains its historical frozen selection, including the four earlier substitutions listed in [the appendix](replacements-and-repairs.md).

These tables divide by the number of scored outcomes shown. The unchanged native [original-selection summary](summary-original-selection.md) instead leaves incomplete task/seed cells blank and excludes them from its aggregates, so its coverage and averages differ.

| Seed | Suite | Scored / planned | Successes | Failures | Success rate (%) | Partial /100 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 0 | Precision | 395/400 | 143 | 252 | 36.20 | 43.80 |
| 0 | Memory | 293/300 | 161 | 132 | 54.95 | 55.14 |
| 0 | Generalization | 600/600 | 158 | 442 | 26.33 | 33.52 |
| 0 | Long Horizon | 397/400 | 145 | 252 | 36.52 | 53.69 |
| 0 | Open | 399/400 | 14 | 385 | 3.51 | 4.12 |
| 1 | Precision | 400/400 | 126 | 274 | 31.50 | 40.42 |
| 1 | Memory | 299/300 | 173 | 126 | 57.86 | 58.19 |
| 1 | Generalization | 600/600 | 183 | 417 | 30.50 | 36.83 |
| 1 | Long Horizon | 398/400 | 137 | 261 | 34.42 | 53.20 |
| 1 | Open | 399/400 | 16 | 383 | 4.01 | 4.49 |
| 2 | Precision | 398/400 | 139 | 259 | 34.92 | 43.59 |
| 2 | Memory | 296/300 | 175 | 121 | 59.12 | 59.56 |
| 2 | Generalization | 600/600 | 169 | 431 | 28.17 | 35.27 |
| 2 | Long Horizon | 397/400 | 142 | 255 | 35.77 | 53.01 |
| 2 | Open | 400/400 | 13 | 387 | 3.25 | 3.90 |

## Tasks affected by later replacements or repair

| Seed | Task | Original scored / planned | Successes | Failures | Success rate (%) | Partial /100 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 0 | Deposit coin | 45/50 | 13 | 32 | 28.89 | 33.78 |
| 0 | Fill egg holder | 47/50 | 1 | 46 | 2.13 | 10.85 |
| 0 | General pickup | 49/50 | 12 | 37 | 24.49 | 24.49 |
| 0 | Imitate sorting sequence | 43/50 | 1 | 42 | 2.33 | 3.60 |
| 1 | Fill egg holder | 48/50 | 2 | 46 | 4.17 | 13.85 |
| 1 | General pickup | 49/50 | 12 | 37 | 24.49 | 24.49 |
| 1 | Imitate sorting sequence | 49/50 | 0 | 49 | 0.00 | 2.04 |
| 2 | Deposit coin | 48/50 | 12 | 36 | 25.00 | 29.58 |
| 2 | Fill egg holder | 47/50 | 0 | 47 | 0.00 | 12.23 |
| 2 | Imitate sorting sequence | 46/50 | 0 | 46 | 0.00 | 2.83 |

All other task rows match the [reported task breakdown](results-breakdown.md). The [episode ledger](episode-scores.csv) can be filtered with `record_type=original` to reproduce this view.
