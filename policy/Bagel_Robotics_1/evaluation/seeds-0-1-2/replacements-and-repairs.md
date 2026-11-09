# Layout replacements, repair and completion recovery

All reported outcomes retain their native success Boolean and partial score. Replacement scenes use supplied unused layouts; each is counted once in place of its unscored original. Failed startup attempts are not task failures and are not included in scored denominators.

## Outcome accounting

| Seed | Scored original selection | Later replacement layouts | Repaired variant | Reported outcomes |
| --- | ---: | ---: | ---: | ---: |
| 0 | 2084 | 16 | 0 | 2100 |
| 1 | 2096 | 3 | 1 | 2100 |
| 2 | 2091 | 9 | 0 | 2100 |
| **Total** | **6,271** | **28** | **1** | **6,300** |

## Complete replacement and repair ledger

Partial scores below are out of 100. All listed originals remain unscored in the unchanged-original view.

| Seed | Task | Original layout | Reported layout | Type | Full success | Partial /100 |
| --- | --- | ---: | ---: | --- | --- | ---: |
| 0 | Fill egg holder | 28 | 50 | Supplied replacement | No | 40.00 |
| 0 | Fill egg holder | 35 | 51 | Supplied replacement | No | 0.00 |
| 0 | Fill egg holder | 37 | 52 | Supplied replacement | No | 0.00 |
| 0 | Imitate sorting sequence | 4 | 50 | Supplied replacement | No | 0.00 |
| 0 | Imitate sorting sequence | 8 | 51 | Supplied replacement | No | 5.00 |
| 0 | Imitate sorting sequence | 17 | 52 | Supplied replacement | No | 0.00 |
| 0 | Imitate sorting sequence | 29 | 53 | Supplied replacement | No | 0.00 |
| 0 | Imitate sorting sequence | 36 | 54 | Supplied replacement | No | 0.00 |
| 0 | Imitate sorting sequence | 48 | 55 | Supplied replacement | No | 0.00 |
| 0 | Imitate sorting sequence | 49 | 56 | Supplied replacement | No | 0.00 |
| 0 | General pickup | 29 | 50 | Supplied replacement | Yes | 100.00 |
| 0 | Deposit coin | 36 | 51 | Supplied replacement | No | 0.00 |
| 0 | Deposit coin | 37 | 52 | Supplied replacement | Yes | 100.00 |
| 0 | Deposit coin | 41 | 53 | Supplied replacement | No | 0.00 |
| 0 | Deposit coin | 43 | 54 | Supplied replacement | No | 0.00 |
| 0 | Deposit coin | 31 | 55 | Supplied replacement | No | 0.00 |
| 1 | Fill egg holder | 15 | 50 | Supplied replacement | No | 0.00 |
| 1 | Fill egg holder | 20 | 51 | Supplied replacement | No | 25.00 |
| 1 | Imitate sorting sequence | 47 | 47 | Repaired same layout | No | 0.00 |
| 1 | General pickup | 5 | 50 | Supplied replacement | No | 0.00 |
| 2 | Fill egg holder | 13 | 50 | Supplied replacement | No | 25.00 |
| 2 | Fill egg holder | 18 | 52 | Supplied replacement | No | 0.00 |
| 2 | Fill egg holder | 40 | 53 | Supplied replacement | No | 10.00 |
| 2 | Imitate sorting sequence | 16 | 50 | Supplied replacement | No | 0.00 |
| 2 | Imitate sorting sequence | 24 | 51 | Supplied replacement | No | 15.00 |
| 2 | Imitate sorting sequence | 32 | 52 | Supplied replacement | No | 0.00 |
| 2 | Imitate sorting sequence | 43 | 53 | Supplied replacement | No | 0.00 |
| 2 | Deposit coin | 24 | 50 | Supplied replacement | No | 0.00 |
| 2 | Deposit coin | 33 | 51 | Supplied replacement | No | 0.00 |

Machine-readable records: [replacement/repair ledger](replacements-and-repairs.csv), [unscored originals and failed candidates](unscored-episodes.csv).

## Earlier Seed 1 Precision selection

The frozen Deposit Coin selection already contained supplied layouts **50, 51, 52 and 53**, with original IDs **4, 8, 39 and 42** omitted. These are four pre-existing selected substitutions, separate from the 28 later recovery replacements above. No pairwise mapping between these two sets is inferred. “Original selection” means the frozen manifest used by that run, not necessarily layouts 0–49. Their scored outcomes are:

| Selected layout | Full success | Partial /100 |
| --- | --- | ---: |
| 50 | Yes | 100.00 |
| 51 | No | 0.00 |
| 52 | No | 20.00 |
| 53 | No | 0.00 |

## Why replacements were used

- Deposit Coin, General Pickup and Fill Egg Holder originals failed native startup/stability validation before policy actions. Sorting Sequence originals failed the benchmark demonstration-validity check. Original attempts are retained in the source records.
- Seed 0 Deposit Coin reserve 50 failed all three startup attempts; reserve 55 was used for original layout 31. Reserve 50 contributes no score.
- Seed 2 Fill Egg Holder reserve 51 also failed startup and contributes no score. The scored replacements are 13→50, 18→52 and 40→53.

## Memory Seed 1: repaired Sorting Sequence layout 47

The original supplied demonstration was invalid. A separately recorded repair changes the demonstration timing, adding **1,431 physics frames at 0.004 seconds/frame (5.724 seconds)**. The policy step budget remains 1,600. This repaired variant completed as a **valid failure, partial score 0**, after 629 policy steps. It is not an unchanged-original result.

- Original demonstration SHA-256: `170c63d412e6168d6d71dd58d5870f39d3caf85836ca0bb79d441742e7f6083e`.
- Repaired demonstration SHA-256: `526c9c572d1e7a635114da6e8cf436fd4ad12451927e25e8940da0cd143a1182`.
- Repair protocol: `8c8f9894f60ac3a905644871b9675d9b6c87ba4e3f52e7228951b49ac1623d75`.

## Completed scores recovered after shutdown errors

| Seed | Task / layout | Retained native outcome | Recovery |
| --- | --- | --- | --- |
| 1 | Match and Pick from Conveyor / 30 | Success; partial 100/100; 388 actions | Native evaluation completed before simulator teardown exit −6. Completion was reconciled from retained evidence; no rerun or score change. |
| 2 | Insert Tubes / 25 | Failure; partial 40/100; 500 actions | Native evaluation completed before shutdown timeout. Completion was reconciled; the original error receipt and native score remain unchanged. |

## Retries of original scenes

The [retry ledger](retried-originals.csv) lists scored originals with more than one recorded attempt or a later attempt index: **80 for seed 0, 66 for seed 1 and 33 for seed 2**. They are counted once, with their original scene identity. Retained campaigns expose attempt history differently; this field is not a uniform count of extra evaluated episodes. Seed 2 Swap T layout 3, for example, completed on its second original attempt with failure and partial zero.

The [original-only table](original-only-breakdown.md) makes the effect of including substitutions explicit. This appendix records the local selection; it does not imply committee approval of substitutions.
