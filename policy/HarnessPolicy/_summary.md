# RoboDojo Evaluation Summary

## Overview (by capability dimension)

Progress † marks incomplete evaluation (126 = 42 tasks × 3 seeds). Cells show `mean ±std` across seeds; scores reflect completed cells only. Policies are sorted by overall Avg SR.

| Policy | Progress | Generalization SR (%) | Generalization Score | Precision SR (%) | Precision Score | Long-Horizon SR (%) | Long-Horizon Score | Memory SR (%) | Memory Score | Open SR (%) | Open Score | Avg SR (%) | Avg Score |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| HarnessPolicy | 33.3% † (42/126) | 30.33 ±0.00 | 38.07 ±0.00 | 30.00 ±0.00 | 37.53 ±0.00 | 45.50 ±0.00 | 60.09 ±0.00 | 47.33 ±0.00 | 48.20 ±0.00 | 16.00 ±0.00 | 16.71 ±0.00 | 33.83 ±0.00 | 40.12 ±0.00 |

## Generalization: Standard vs Random (25 episodes each)

Each cell shows `mean ±std` across seeds, averaged over the 12 Generalization tasks with completed data for that half. Δ = Random − Standard (mean SR). Sorted by Standard SR.

| Policy | Standard SR (%) | Standard Score | Random SR (%) | Random Score | Δ SR | Δ Score |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| HarnessPolicy | 34.67 ±0.00 | 42.48 ±0.00 | 26.00 ±0.00 | 33.67 ±0.00 | -8.67 | -8.80 |

### Per-task Standard vs Random

Additional view only: each of the 12 Generalization tasks is listed as the 25-episode standard split and its `_random` sibling. Columns match the per-policy all-task tables. Existing merged 50-episode scores are unchanged.

#### HarnessPolicy

| Task | Seed 0 SR (%) | Seed 0 Score | Seed 1 SR (%) | Seed 1 Score | Seed 2 SR (%) | Seed 2 Score | Avg SR (%) | Avg Score |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| <nobr>stack_bowls</nobr> | 88 | 88.67 |  |  |  |  |  |  |
| <nobr>stack_bowls_random</nobr> | 64 | 67.27 |  |  |  |  |  |  |
| <nobr>push_T</nobr> | 0 | 0.00 |  |  |  |  |  |  |
| <nobr>push_T_random</nobr> | 0 | 0.00 |  |  |  |  |  |  |
| <nobr>pack_objects_into_box</nobr> | 16 | 49.93 |  |  |  |  |  |  |
| <nobr>pack_objects_into_box_random</nobr> | 20 | 47.20 |  |  |  |  |  |  |
| <nobr>fold_clothes</nobr> | 80 | 82.93 |  |  |  |  |  |  |
| <nobr>fold_clothes_random</nobr> | 76 | 78.40 |  |  |  |  |  |  |
| <nobr>hang_mugs</nobr> | 0 | 13.60 |  |  |  |  |  |  |
| <nobr>hang_mugs_random</nobr> | 0 | 11.40 |  |  |  |  |  |  |
| <nobr>sweep_blocks</nobr> | 4 | 5.33 |  |  |  |  |  |  |
| <nobr>sweep_blocks_random</nobr> | 4 | 4.00 |  |  |  |  |  |  |
| <nobr>pour_liquid_into_cup</nobr> | 56 | 57.33 |  |  |  |  |  |  |
| <nobr>pour_liquid_into_cup_random</nobr> | 52 | 53.33 |  |  |  |  |  |  |
| <nobr>make_toast</nobr> | 16 | 24.00 |  |  |  |  |  |  |
| <nobr>make_toast_random</nobr> | 4 | 7.67 |  |  |  |  |  |  |
| <nobr>arrange_largest_number</nobr> | 12 | 27.40 |  |  |  |  |  |  |
| <nobr>arrange_largest_number_random</nobr> | 8 | 23.20 |  |  |  |  |  |  |
| <nobr>sort_nesting_dolls_by_size</nobr> | 28 | 28.00 |  |  |  |  |  |  |
| <nobr>sort_nesting_dolls_by_size_random</nobr> | 12 | 12.00 |  |  |  |  |  |  |
| <nobr>store_laptop_and_headphones</nobr> | 48 | 59.20 |  |  |  |  |  |  |
| <nobr>store_laptop_and_headphones_random</nobr> | 40 | 59.20 |  |  |  |  |  |  |
| <nobr>stack_blocks</nobr> | 68 | 73.33 |  |  |  |  |  |  |
| <nobr>stack_blocks_random</nobr> | 32 | 40.40 |  |  |  |  |  |  |
| **Avg Standard** | 34.7 | 42.48 |  |  |  |  | 35 ±0 | 42.48 ±0.00 |
| **Avg Random** | 26.0 | 33.67 |  |  |  |  | 26 ±0 | 33.67 ±0.00 |

## HarnessPolicy

| Task | Seed 0 SR (%) | Seed 0 Score | Seed 1 SR (%) | Seed 1 Score | Seed 2 SR (%) | Seed 2 Score | Avg SR (%) | Avg Score |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| <nobr>align_blocks</nobr> | 0 | 0.00 |  |  |  |  |  |  |
| <nobr>arrange_largest_number</nobr> | 10 | 25.30 |  |  |  |  |  |  |
| <nobr>build_tower</nobr> | 84 | 87.40 |  |  |  |  |  |  |
| <nobr>classify_objects</nobr> | 62 | 69.30 |  |  |  |  |  |  |
| <nobr>classify_objects_by_language</nobr> | 4 | 6.00 |  |  |  |  |  |  |
| <nobr>cover_blocks</nobr> | 100 | 100.00 |  |  |  |  |  |  |
| <nobr>deposit_coin</nobr> | 12 | 16.80 |  |  |  |  |  |  |
| <nobr>fasten_screws</nobr> | 30 | 48.00 |  |  |  |  |  |  |
| <nobr>fill_egg_holder</nobr> | 0 | 16.60 |  |  |  |  |  |  |
| <nobr>fill_pen_holder</nobr> | 38 | 58.50 |  |  |  |  |  |  |
| <nobr>fold_clothes</nobr> | 78 | 80.66 |  |  |  |  |  |  |
| <nobr>general_pickup</nobr> | 58 | 58.00 |  |  |  |  |  |  |
| <nobr>hang_mugs</nobr> | 0 | 12.50 |  |  |  |  |  |  |
| <nobr>imitate_sorting_sequence</nobr> | 2 | 6.50 |  |  |  |  |  |  |
| <nobr>insert_key</nobr> | 0 | 15.00 |  |  |  |  |  |  |
| <nobr>insert_tubes</nobr> | 22 | 42.40 |  |  |  |  |  |  |
| <nobr>make_kong</nobr> | 78 | 78.00 |  |  |  |  |  |  |
| <nobr>make_toast</nobr> | 10 | 15.83 |  |  |  |  |  |  |
| <nobr>match_and_pick_from_conveyor</nobr> | 56 | 56.00 |  |  |  |  |  |  |
| <nobr>organize_table</nobr> | 26 | 71.83 |  |  |  |  |  |  |
| <nobr>pack_objects_into_box</nobr> | 18 | 48.56 |  |  |  |  |  |  |
| <nobr>pick_from_conveyor_by_image</nobr> | 0 | 0.00 |  |  |  |  |  |  |
| <nobr>play_Xylophone</nobr> | 4 | 3.33 |  |  |  |  |  |  |
| <nobr>play_stacking_toy</nobr> | 0 | 0.20 |  |  |  |  |  |  |
| <nobr>play_tic_tac_toe</nobr> | 60 | 86.27 |  |  |  |  |  |  |
| <nobr>plug_in_charger</nobr> | 30 | 29.33 |  |  |  |  |  |  |
| <nobr>pour_balls_into_vase</nobr> | 58 | 58.00 |  |  |  |  |  |  |
| <nobr>pour_by_language</nobr> | 0 | 0.40 |  |  |  |  |  |  |
| <nobr>pour_liquid_into_cup</nobr> | 54 | 55.33 |  |  |  |  |  |  |
| <nobr>press_by_number</nobr> | 88 | 88.00 |  |  |  |  |  |  |
| <nobr>push_T</nobr> | 0 | 0.00 |  |  |  |  |  |  |
| <nobr>put_bottles_into_dustbin</nobr> | 100 | 100.00 |  |  |  |  |  |  |
| <nobr>solve_equation</nobr> | 46 | 46.00 |  |  |  |  |  |  |
| <nobr>sort_nesting_dolls_by_size</nobr> | 20 | 20.00 |  |  |  |  |  |  |
| <nobr>stack_blocks</nobr> | 50 | 56.86 |  |  |  |  |  |  |
| <nobr>stack_blocks_by_language</nobr> | 20 | 22.80 |  |  |  |  |  |  |
| <nobr>stack_bowls</nobr> | 76 | 77.97 |  |  |  |  |  |  |
| <nobr>store_laptop_and_headphones</nobr> | 44 | 59.20 |  |  |  |  |  |  |
| <nobr>store_tools_in_toolbox</nobr> | 0 | 0.50 |  |  |  |  |  |  |
| <nobr>swap_T</nobr> | 0 | 0.00 |  |  |  |  |  |  |
| <nobr>swap_blocks</nobr> | 38 | 38.67 |  |  |  |  |  |  |
| <nobr>sweep_blocks</nobr> | 4 | 4.67 |  |  |  |  |  |  |
| **Avg (per seed)** | 32.9 | 39.54 |  |  |  |  | 33 ±0 | 39.54 ±0.00 |

