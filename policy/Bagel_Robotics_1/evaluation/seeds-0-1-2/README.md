# Three-seed local evaluation package

**Bagel-Robotics-1 / checkpoint 1000 — seeds 0, 1 and 2.** This score package compiles 6,300 retained outcomes, covering five suites and 42 tasks. It includes the labeled scene substitutions and repaired variant documented below.

| File | Contents |
| --- | --- |
| [summary.md](summary.md) | Unchanged output from the official RoboDojo summarizer; all 126 task/seed cells filled for the reported selection. |
| [results-breakdown.md](results-breakdown.md) | Per-seed suite/task success counts, denominators, success rates, partial scores and standard/randomized Generalization split. |
| [replacements-and-repairs.md](replacements-and-repairs.md) | All later replacements, the repaired Memory episode, pre-existing Seed 1 Precision selections and completion recoveries. |
| [original-only-breakdown.md](original-only-breakdown.md) | All scored original-selection outcomes, using their actual denominators. |
| [summary-original-selection.md](summary-original-selection.md) | Native summary with later replacements and repair excluded; incomplete task/seed cells stay blank. |
| [episode-scores.csv](episode-scores.csv) | All 6,300 scored outcomes, native 0–1 partial scores, actual layout IDs, record types and protocol/completion hashes. |
| [unscored-episodes.csv](unscored-episodes.csv) | 29 unscored original identities and two failed reserve candidates; no fabricated zero scores. |
| [protocols.json](protocols.json) / [verification.json](verification.json) | Checkpoint/inference identities, source hashes, coverage and independent aggregate checks. |
| [SHA256SUMS](SHA256SUMS) | SHA-256 checksums for the public files in this directory. |

## Reading the scores

The headline is the mean of five equally weighted suite scores for each seed, then the mean across seeds. It is not the success count divided by all 6,300 episodes. Each Generalization task combines 25 standard and 25 randomized episodes; other tasks have 50 reported episodes per seed.

The native overview uses population standard deviation across seeds. Its task-table ± values use half the highest-to-lowest range. The custom breakdown uses population standard deviation consistently. The earlier Notion research report used sample standard deviation; means are identical, but those ± values are larger. These are measures of variation between seeds, not confidence intervals.

The native policy label explicitly identifies replacements and repair. It represents the same Bagel-Robotics-1 checkpoint. “Original selection” refers to the frozen campaign selection, including four earlier Seed 1 Precision substitutions; see the appendix.

## Checkpoint and evaluation identity

- Public checkpoint: [bageldotcom/Bagel-Robotics-1](https://huggingface.co/bageldotcom/Bagel-Robotics-1/tree/2651631c5d08819088475e8778676f5cf6d9afcf), revision `2651631c5d08819088475e8778676f5cf6d9afcf`.
- Evaluation lineage: frozen RoboDojo snapshot based on [`08b7ee4`](https://github.com/RoboDojo-Benchmark/RoboDojo/commit/08b7ee46034c3d0c8a389b2b7bfc138f4d55ee4d), with local runtime/reproducibility patches. All 34 recorded protocol manifests match the retained baseline's 55 task-source hashes; 23 also record and match all 55 configuration hashes. Eleven cloud protocol manifests do not include those configuration hashes.
- Summary script revision: [`f57e163`](https://github.com/RoboDojo-Benchmark/RoboDojo/commit/f57e1633ffed59761bc8e568cfe7530c4b33381c), verified byte-for-byte. Generating the summary at this revision does not change the benchmark version that produced the episodes.
- Inference: checkpoint 1000, BF16, ten Euler steps, shift 1, 32 predicted / 16 executed actions, 90% new / 10% previous aligned prediction. Evaluation/model seeds 0/1/2 are checked against each record's protocol. Native simulator reset uses the selected layout identity.
- This compilation joins the retained cloud archive, earlier verified seed-1 package and completed node-2 records. The 750 overlapping seed-1 outcomes agree and are included once. Seed 2 Open comes from the completed retained cloud campaign.

## Regenerate the summary

Use a separate RoboDojo checkout at the pinned summary revision; do not change an active evaluation checkout. No GPU, model weights or simulator installation is needed to summarize these scores.

```bash
git clone https://github.com/RoboDojo-Benchmark/RoboDojo.git /tmp/robodojo-summary-source
git -C /tmp/robodojo-summary-source checkout f57e1633ffed59761bc8e568cfe7530c4b33381c

# From this evaluation/seeds-0-1-2 directory:
sha256sum -c SHA256SUMS
python3 generate_summary.py --robodojo-root /tmp/robodojo-summary-source \
  --output-dir /tmp/bagel-robotics-1-reported-summary
cmp summary.md /tmp/bagel-robotics-1-reported-summary/summary.md

python3 generate_summary.py --robodojo-root /tmp/robodojo-summary-source \
  --view original --output-dir /tmp/bagel-robotics-1-original-summary
cmp summary-original-selection.md /tmp/bagel-robotics-1-original-summary/summary.md
```

The output directories must be new or empty. `generate_summary.py` verifies the two pinned script hashes, groups the published score fields into 162 native task/seed JSON files and invokes **`bash scripts/robodojo.sh summarize` unchanged**. Actual layout IDs and outcomes are preserved. These grouped JSONs are a score projection of the retained native records, not new simulator runs or byte-identical copies of the original one-episode files.

This score-only package omits rollout videos. The unmodified summarizer consequently reports missing videos in `summarize.log`; no placeholder videos are created. For model re-evaluation, use [REEVALUATION.md](../../REEVALUATION.md).
