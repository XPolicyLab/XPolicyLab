# Bagel-Robotics-1: three-seed local results

Checkpoint 1000 (AdaPrune Final), evaluation seeds **0, 1 and 2**: five suites, 42 tasks, **6,300 reported outcomes**. Success means full task completion. Partial score measures credited progress and includes successful and failed episodes.

| Suite | Successes / reported outcomes (3 seeds) | Success rate (%) | Partial score /100 |
| --- | ---: | ---: | ---: |
| Precision | 409/1200 | 34.08 ± 1.90 | 42.43 ± 1.42 |
| Memory | 509/900 | 56.56 ± 2.06 | 56.89 ± 2.17 |
| Generalization | 510/1800 | 28.33 ± 1.71 | 35.20 ± 1.35 |
| Long Horizon | 424/1200 | 35.33 ± 0.82 | 53.03 ± 0.28 |
| Open | 44/1200 | 3.67 ± 0.31 | 4.25 ± 0.25 |
| **Five-suite average** | **6,300 outcomes** | **31.59 ± 0.33** | **38.36 ± 0.46** |

Suite and overall values are the mean ± population standard deviation across the three seeds, matching the official summary overview. Each seed's overall value weights the five suites equally. The success counts are provided for transparency; pooling all episodes is a different calculation.

The reported selection contains **6,271 valid original-selection outcomes, 28 labeled replacement layouts and one separately repaired Memory variant**. The exact mappings and original-only scores are linked below; the original invalid episodes remain unscored. Seed 1 Precision also retains four pre-existing substitutions in its frozen selection, documented in the appendix.

- [Unchanged RoboDojo-generated summary](evaluation/seeds-0-1-2/summary.md)
- [Individual-seed suite/task scores and standard/randomized breakdown](evaluation/seeds-0-1-2/results-breakdown.md)
- [Replacement, repair and completion-recovery details](evaluation/seeds-0-1-2/replacements-and-repairs.md)
- [Original-selection-only scores](evaluation/seeds-0-1-2/original-only-breakdown.md)
- [All 6,300 episode scores](evaluation/seeds-0-1-2/episode-scores.csv)
- [Source verification and summary regeneration](evaluation/seeds-0-1-2/README.md)

Checkpoint: [bageldotcom/Bagel-Robotics-1](https://huggingface.co/bageldotcom/Bagel-Robotics-1/tree/2651631c5d08819088475e8778676f5cf6d9afcf), pinned to `2651631c5d08819088475e8778676f5cf6d9afcf`. All runs use the same checkpoint, 32 predicted actions / 16 executed actions, ten Euler steps and 90/10 temporal blending. Source protocols preserve the evaluation/model seed for each record.

These are local evaluation results from the frozen research runtimes. Packaged-adapter validation is documented separately in [VALIDATION.md](VALIDATION.md). The earlier [seed-1 report](evaluation/seed-1/results-breakdown.md) is retained.
