# RoboDojo Evaluation Summary

## Overview (by capability dimension)

Progress † indicates partial seed coverage (126 = 42 tasks × 3 seeds). Cells  use `mean ±std` across seeds; scores reflect completed cells only. Policies are sorted by overall Avg SR.

| Policy | Progress | Generalization SR (%) | Generalization Score | Precision SR (%) | Precision Score | Long-Horizon SR (%) | Long-Horizon Score | Memory SR (%) | Memory Score | Open SR (%) | Open Score | Avg SR (%) | Avg Score |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| REAL_RSI | 33.3% † (42/126) | 41.17 ±0.00 | 46.74 ±0.00 | 74.00 ±0.00 | 75.93 ±0.00 | 56.00 ±0.00 | 71.70 ±0.00 | 75.67 ±0.00 | 77.57 ±0.00 | 62.75 ±0.00 | 69.60 ±0.00 | 61.92 ±0.00 | 68.31 ±0.00 |

## Generalization: Standard vs Random (25 episodes each)

Each cell shows `mean ±std` across seeds, averaged over the 12 Generalization tasks with completed data for that half. Δ = Random − Standard (mean SR). Sorted by Standard SR.

| Policy | Standard SR (%) | Standard Score | Random SR (%) | Random Score | Δ SR | Δ Score |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| REAL_RSI | 47.33 ±0.00 | 53.78 ±0.00 | 35.00 ±0.00 | 39.70 ±0.00 | -12.33 | -14.08 |

### Per-task Standard vs Random

Additional view only: each of the 12 Generalization tasks is listed as the 25-episode standard split and its `_random` sibling. Columns match the per-policy all-task tables. Existing merged 50-episode scores are unchanged.

#### REAL_RSI

| Task | Seed 0 SR (%) | Seed 0 Score | Seed 1 SR (%) | Seed 1 Score | Seed 2 SR (%) | Seed 2 Score | Avg SR (%) | Avg Score |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| <nobr>stack_bowls</nobr> | 84 | 86.40 |  |  |  |  |  |  |
| <nobr>stack_bowls_random</nobr> | 76 | 77.20 |  |  |  |  |  |  |
| <nobr>push_T</nobr> | 100 | 100.00 |  |  |  |  |  |  |
| <nobr>push_T_random</nobr> | 96 | 96.00 |  |  |  |  |  |  |
| <nobr>pack_objects_into_box</nobr> | 0 | 9.20 |  |  |  |  |  |  |
| <nobr>pack_objects_into_box_random</nobr> | 0 | 0.00 |  |  |  |  |  |  |
| <nobr>fold_clothes</nobr> | 48 | 56.80 |  |  |  |  |  |  |
| <nobr>fold_clothes_random</nobr> | 56 | 64.00 |  |  |  |  |  |  |
| <nobr>hang_mugs</nobr> | 4 | 21.40 |  |  |  |  |  |  |
| <nobr>hang_mugs_random</nobr> | 0 | 25.60 |  |  |  |  |  |  |
| <nobr>sweep_blocks</nobr> | 16 | 16.00 |  |  |  |  |  |  |
| <nobr>sweep_blocks_random</nobr> | 0 | 0.00 |  |  |  |  |  |  |
| <nobr>pour_liquid_into_cup</nobr> | 92 | 92.00 |  |  |  |  |  |  |
| <nobr>pour_liquid_into_cup_random</nobr> | 84 | 84.00 |  |  |  |  |  |  |
| <nobr>make_toast</nobr> | 12 | 24.00 |  |  |  |  |  |  |
| <nobr>make_toast_random</nobr> | 0 | 0.00 |  |  |  |  |  |  |
| <nobr>arrange_largest_number</nobr> | 48 | 65.60 |  |  |  |  |  |  |
| <nobr>arrange_largest_number_random</nobr> | 16 | 34.00 |  |  |  |  |  |  |
| <nobr>sort_nesting_dolls_by_size</nobr> | 72 | 72.00 |  |  |  |  |  |  |
| <nobr>sort_nesting_dolls_by_size_random</nobr> | 0 | 0.00 |  |  |  |  |  |  |
| <nobr>store_laptop_and_headphones</nobr> | 0 | 8.80 |  |  |  |  |  |  |
| <nobr>store_laptop_and_headphones_random</nobr> | 0 | 2.40 |  |  |  |  |  |  |
| <nobr>stack_blocks</nobr> | 92 | 93.20 |  |  |  |  |  |  |
| <nobr>stack_blocks_random</nobr> | 92 | 93.20 |  |  |  |  |  |  |
| **Avg Standard** | 47.3 | 53.78 |  |  |  |  | 47 ±0 | 53.78 ±0.00 |
| **Avg Random** | 35.0 | 39.70 |  |  |  |  | 35 ±0 | 39.70 ±0.00 |

## REAL_RSI

| Task | Seed 0 SR (%) | Seed 0 Score | Seed 1 SR (%) | Seed 1 Score | Seed 2 SR (%) | Seed 2 Score | Avg SR (%) | Avg Score |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| <nobr>align_blocks</nobr> | 96 | 96.00 |  |  |  |  |  |  |
| <nobr>arrange_largest_number</nobr> | 32 | 49.80 |  |  |  |  |  |  |
| <nobr>build_tower</nobr> | 96 | 96.40 |  |  |  |  |  |  |
| <nobr>classify_objects</nobr> | 78 | 85.00 |  |  |  |  |  |  |
| <nobr>classify_objects_by_language</nobr> | 50 | 62.80 |  |  |  |  |  |  |
| <nobr>cover_blocks</nobr> | 98 | 98.00 |  |  |  |  |  |  |
| <nobr>deposit_coin</nobr> | 50 | 52.40 |  |  |  |  |  |  |
| <nobr>fasten_screws</nobr> | 90 | 94.00 |  |  |  |  |  |  |
| <nobr>fill_egg_holder</nobr> | 18 | 43.20 |  |  |  |  |  |  |
| <nobr>fill_pen_holder</nobr> | 0 | 3.20 |  |  |  |  |  |  |
| <nobr>fold_clothes</nobr> | 52 | 60.40 |  |  |  |  |  |  |
| <nobr>general_pickup</nobr> | 74 | 74.00 |  |  |  |  |  |  |
| <nobr>hang_mugs</nobr> | 2 | 23.50 |  |  |  |  |  |  |
| <nobr>imitate_sorting_sequence</nobr> | 6 | 17.40 |  |  |  |  |  |  |
| <nobr>insert_key</nobr> | 64 | 69.40 |  |  |  |  |  |  |
| <nobr>insert_tubes</nobr> | 92 | 95.20 |  |  |  |  |  |  |
| <nobr>make_kong</nobr> | 98 | 98.00 |  |  |  |  |  |  |
| <nobr>make_toast</nobr> | 6 | 12.00 |  |  |  |  |  |  |
| <nobr>match_and_pick_from_conveyor</nobr> | 50 | 50.00 |  |  |  |  |  |  |
| <nobr>organize_table</nobr> | 12 | 72.50 |  |  |  |  |  |  |
| <nobr>pack_objects_into_box</nobr> | 0 | 4.60 |  |  |  |  |  |  |
| <nobr>pick_from_conveyor_by_image</nobr> | 18 | 18.00 |  |  |  |  |  |  |
| <nobr>play_Xylophone</nobr> | 94 | 94.00 |  |  |  |  |  |  |
| <nobr>play_stacking_toy</nobr> | 66 | 86.40 |  |  |  |  |  |  |
| <nobr>play_tic_tac_toe</nobr> | 100 | 100.00 |  |  |  |  |  |  |
| <nobr>plug_in_charger</nobr> | 22 | 22.00 |  |  |  |  |  |  |
| <nobr>pour_balls_into_vase</nobr> | 84 | 84.00 |  |  |  |  |  |  |
| <nobr>pour_by_language</nobr> | 86 | 86.00 |  |  |  |  |  |  |
| <nobr>pour_liquid_into_cup</nobr> | 88 | 88.00 |  |  |  |  |  |  |
| <nobr>press_by_number</nobr> | 100 | 100.00 |  |  |  |  |  |  |
| <nobr>push_T</nobr> | 98 | 98.00 |  |  |  |  |  |  |
| <nobr>put_bottles_into_dustbin</nobr> | 76 | 85.30 |  |  |  |  |  |  |
| <nobr>solve_equation</nobr> | 76 | 76.00 |  |  |  |  |  |  |
| <nobr>sort_nesting_dolls_by_size</nobr> | 36 | 36.00 |  |  |  |  |  |  |
| <nobr>stack_blocks</nobr> | 92 | 93.20 |  |  |  |  |  |  |
| <nobr>stack_blocks_by_language</nobr> | 96 | 96.00 |  |  |  |  |  |  |
| <nobr>stack_bowls</nobr> | 80 | 81.80 |  |  |  |  |  |  |
| <nobr>store_laptop_and_headphones</nobr> | 0 | 5.60 |  |  |  |  |  |  |
| <nobr>store_tools_in_toolbox</nobr> | 6 | 48.00 |  |  |  |  |  |  |
| <nobr>swap_T</nobr> | 100 | 100.00 |  |  |  |  |  |  |
| <nobr>swap_blocks</nobr> | 100 | 100.00 |  |  |  |  |  |  |
| <nobr>sweep_blocks</nobr> | 8 | 8.00 |  |  |  |  |  |  |
| **Avg (per seed)** | 59.3 | 65.81 |  |  |  |  | 59 ±0 | 65.81 ±0.00 |

