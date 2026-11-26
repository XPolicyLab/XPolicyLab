<!-- Local application evidence only. Produced by `bash robodojo.sh summarize`
on an isolated staging root of run `compliant_desens_s0_native_12w_20261125_2324`
(seed 0, --eval-num native). Policy directory was later renamed to HarnessTree;
numbers are unchanged. store_laptop_and_headphones was operator-forced to SR=0
after a mid-run hang (2026-11-26). Not a verified multi-seed leaderboard listing.
Protocol: https://robodojo-benchmark.com/leaderboard/protocol -->

# RoboDojo Evaluation Summary

## Overview (by capability dimension)

Progress † marks incomplete evaluation (126 = 42 tasks × 3 seeds). Cells show `mean ±std` across seeds; scores reflect completed cells only. Policies are sorted by overall Avg SR.

| Policy | Progress | Generalization SR (%) | Generalization Score | Precision SR (%) | Precision Score | Long-Horizon SR (%) | Long-Horizon Score | Memory SR (%) | Memory Score | Open SR (%) | Open Score | Avg SR (%) | Avg Score |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| HarnessTree | 33.3% † (42/126) | 65.50 ±0.00 | 70.22 ±0.00 | 82.25 ±0.00 | 84.76 ±0.00 | 60.00 ±0.00 | 79.67 ±0.00 | 78.33 ±0.00 | 79.28 ±0.00 | 58.50 ±0.00 | 63.16 ±0.00 | 68.92 ±0.00 | 75.42 ±0.00 |

## Generalization: Standard vs Random (25 episodes each)

Each cell shows `mean ±std` across seeds, averaged over the 12 Generalization tasks with completed data for that half. Δ = Random − Standard (mean SR). Sorted by Standard SR.

| Policy | Standard SR (%) | Standard Score | Random SR (%) | Random Score | Δ SR | Δ Score |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| HarnessTree | 70.00 ±0.00 | 73.75 ±0.00 | 61.00 ±0.00 | 66.70 ±0.00 | -9.00 | -7.05 |

### Per-task Standard vs Random

Additional view only: each of the 12 Generalization tasks is listed as the 25-episode standard split and its `_random` sibling. Columns match the per-policy all-task tables. Existing merged 50-episode scores are unchanged.

#### HarnessTree

| Task | Seed 0 SR (%) | Seed 0 Score | Seed 1 SR (%) | Seed 1 Score | Seed 2 SR (%) | Seed 2 Score | Avg SR (%) | Avg Score |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| <nobr>stack_bowls</nobr> | 84 | 86.40 |  |  |  |  |  |  |
| <nobr>stack_bowls_random</nobr> | 88 | 89.80 |  |  |  |  |  |  |
| <nobr>push_T</nobr> | 100 | 100.00 |  |  |  |  |  |  |
| <nobr>push_T_random</nobr> | 100 | 100.00 |  |  |  |  |  |  |
| <nobr>pack_objects_into_box</nobr> | 52 | 65.60 |  |  |  |  |  |  |
| <nobr>pack_objects_into_box_random</nobr> | 48 | 67.80 |  |  |  |  |  |  |
| <nobr>fold_clothes</nobr> | 100 | 100.00 |  |  |  |  |  |  |
| <nobr>fold_clothes_random</nobr> | 88 | 90.40 |  |  |  |  |  |  |
| <nobr>hang_mugs</nobr> | 44 | 59.40 |  |  |  |  |  |  |
| <nobr>hang_mugs_random</nobr> | 16 | 41.00 |  |  |  |  |  |  |
| <nobr>sweep_blocks</nobr> | 92 | 92.00 |  |  |  |  |  |  |
| <nobr>sweep_blocks_random</nobr> | 68 | 68.00 |  |  |  |  |  |  |
| <nobr>pour_liquid_into_cup</nobr> | 92 | 92.00 |  |  |  |  |  |  |
| <nobr>pour_liquid_into_cup_random</nobr> | 92 | 92.00 |  |  |  |  |  |  |
| <nobr>make_toast</nobr> | 36 | 48.00 |  |  |  |  |  |  |
| <nobr>make_toast_random</nobr> | 4 | 8.00 |  |  |  |  |  |  |
| <nobr>arrange_largest_number</nobr> | 96 | 97.60 |  |  |  |  |  |  |
| <nobr>arrange_largest_number_random</nobr> | 92 | 94.80 |  |  |  |  |  |  |
| <nobr>sort_nesting_dolls_by_size</nobr> | 44 | 44.00 |  |  |  |  |  |  |
| <nobr>sort_nesting_dolls_by_size_random</nobr> | 40 | 40.00 |  |  |  |  |  |  |
| <nobr>store_laptop_and_headphones</nobr> | 0 | 0.00 |  |  |  |  |  |  |
| <nobr>store_laptop_and_headphones_random</nobr> | 0 | 12.00 |  |  |  |  |  |  |
| <nobr>stack_blocks</nobr> | 100 | 100.00 |  |  |  |  |  |  |
| <nobr>stack_blocks_random</nobr> | 96 | 96.60 |  |  |  |  |  |  |
| **Avg Standard** | 70.0 | 73.75 |  |  |  |  | 70 ±0 | 73.75 ±0.00 |
| **Avg Random** | 61.0 | 66.70 |  |  |  |  | 61 ±0 | 66.70 ±0.00 |

## HarnessTree

| Task | Seed 0 SR (%) | Seed 0 Score | Seed 1 SR (%) | Seed 1 Score | Seed 2 SR (%) | Seed 2 Score | Avg SR (%) | Avg Score |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| <nobr>align_blocks</nobr> | 100 | 100.00 |  |  |  |  |  |  |
| <nobr>arrange_largest_number</nobr> | 94 | 96.20 |  |  |  |  |  |  |
| <nobr>build_tower</nobr> | 100 | 100.00 |  |  |  |  |  |  |
| <nobr>classify_objects</nobr> | 90 | 93.00 |  |  |  |  |  |  |
| <nobr>classify_objects_by_language</nobr> | 0 | 1.40 |  |  |  |  |  |  |
| <nobr>cover_blocks</nobr> | 100 | 100.00 |  |  |  |  |  |  |
| <nobr>deposit_coin</nobr> | 82 | 82.80 |  |  |  |  |  |  |
| <nobr>fasten_screws</nobr> | 96 | 98.00 |  |  |  |  |  |  |
| <nobr>fill_egg_holder</nobr> | 44 | 63.20 |  |  |  |  |  |  |
| <nobr>fill_pen_holder</nobr> | 40 | 63.50 |  |  |  |  |  |  |
| <nobr>fold_clothes</nobr> | 94 | 95.20 |  |  |  |  |  |  |
| <nobr>general_pickup</nobr> | 60 | 60.00 |  |  |  |  |  |  |
| <nobr>hang_mugs</nobr> | 30 | 50.20 |  |  |  |  |  |  |
| <nobr>imitate_sorting_sequence</nobr> | 72 | 77.70 |  |  |  |  |  |  |
| <nobr>insert_key</nobr> | 0 | 14.10 |  |  |  |  |  |  |
| <nobr>insert_tubes</nobr> | 92 | 95.20 |  |  |  |  |  |  |
| <nobr>make_kong</nobr> | 98 | 98.00 |  |  |  |  |  |  |
| <nobr>make_toast</nobr> | 20 | 28.00 |  |  |  |  |  |  |
| <nobr>match_and_pick_from_conveyor</nobr> | 88 | 88.00 |  |  |  |  |  |  |
| <nobr>organize_table</nobr> | 0 | 68.50 |  |  |  |  |  |  |
| <nobr>pack_objects_into_box</nobr> | 50 | 66.70 |  |  |  |  |  |  |
| <nobr>pick_from_conveyor_by_image</nobr> | 6 | 6.00 |  |  |  |  |  |  |
| <nobr>play_Xylophone</nobr> | 100 | 100.00 |  |  |  |  |  |  |
| <nobr>play_stacking_toy</nobr> | 90 | 94.80 |  |  |  |  |  |  |
| <nobr>play_tic_tac_toe</nobr> | 26 | 61.50 |  |  |  |  |  |  |
| <nobr>plug_in_charger</nobr> | 94 | 94.00 |  |  |  |  |  |  |
| <nobr>pour_balls_into_vase</nobr> | 94 | 94.00 |  |  |  |  |  |  |
| <nobr>pour_by_language</nobr> | 50 | 58.40 |  |  |  |  |  |  |
| <nobr>pour_liquid_into_cup</nobr> | 92 | 92.00 |  |  |  |  |  |  |
| <nobr>press_by_number</nobr> | 14 | 14.00 |  |  |  |  |  |  |
| <nobr>push_T</nobr> | 100 | 100.00 |  |  |  |  |  |  |
| <nobr>put_bottles_into_dustbin</nobr> | 92 | 94.90 |  |  |  |  |  |  |
| <nobr>solve_equation</nobr> | 94 | 94.00 |  |  |  |  |  |  |
| <nobr>sort_nesting_dolls_by_size</nobr> | 42 | 42.00 |  |  |  |  |  |  |
| <nobr>stack_blocks</nobr> | 98 | 98.30 |  |  |  |  |  |  |
| <nobr>stack_blocks_by_language</nobr> | 100 | 100.00 |  |  |  |  |  |  |
| <nobr>stack_bowls</nobr> | 86 | 88.10 |  |  |  |  |  |  |
| <nobr>store_laptop_and_headphones</nobr> | 0 | 6.00 |  |  |  |  |  |  |
| <nobr>store_tools_in_toolbox</nobr> | 58 | 85.50 |  |  |  |  |  |  |
| <nobr>swap_T</nobr> | 100 | 100.00 |  |  |  |  |  |  |
| <nobr>swap_blocks</nobr> | 96 | 96.00 |  |  |  |  |  |  |
| <nobr>sweep_blocks</nobr> | 80 | 80.00 |  |  |  |  |  |  |
| **Avg (per seed)** | 68.1 | 74.74 |  |  |  |  | 68 ±0 | 74.74 ±0.00 |

