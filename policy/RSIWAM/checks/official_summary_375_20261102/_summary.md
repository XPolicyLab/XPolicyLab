# RoboDojo Evaluation Summary

## Overview (by capability dimension)

Progress † marks incomplete evaluation (126 = 42 tasks × 3 seeds). Cells show `mean ±std` across seeds; scores reflect completed cells only. Policies are sorted by overall Avg SR.

| Policy | Progress | Generalization SR (%) | Generalization Score | Precision SR (%) | Precision Score | Long-Horizon SR (%) | Long-Horizon Score | Memory SR (%) | Memory Score | Open SR (%) | Open Score | Avg SR (%) | Avg Score |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| RSIWAM | 5.6% † (7/126) |  |  | 53.00 ±0.00 | 56.70 ±0.00 | 23.00 ±0.00 | 32.00 ±0.00 | 100.00 ±0.00 | 100.00 ±0.00 | 10.00 ±0.00 | 10.00 ±0.00 | 46.50 ±0.00 | 49.67 ±0.00 |

## Generalization: Standard vs Random (25 episodes each)

Each cell shows `mean ±std` across seeds, averaged over the 12 Generalization tasks with completed data for that half. Δ = Random − Standard (mean SR). Sorted by Standard SR.

| Policy | Standard SR (%) | Standard Score | Random SR (%) | Random Score | Δ SR | Δ Score |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| RSIWAM | 4.00 ±0.00 | 12.20 ±0.00 |  |  |  |  |

### Per-task Standard vs Random

Additional view only: each of the 12 Generalization tasks is listed as the 25-episode standard split and its `_random` sibling. Columns match the per-policy all-task tables. Existing merged 50-episode scores are unchanged.

#### RSIWAM

| Task | Seed 0 SR (%) | Seed 0 Score | Seed 1 SR (%) | Seed 1 Score | Seed 2 SR (%) | Seed 2 Score | Avg SR (%) | Avg Score |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| <nobr>stack_bowls</nobr> |  |  |  |  |  |  |  |  |
| <nobr>stack_bowls_random</nobr> |  |  |  |  |  |  |  |  |
| <nobr>push_T</nobr> |  |  |  |  |  |  |  |  |
| <nobr>push_T_random</nobr> |  |  |  |  |  |  |  |  |
| <nobr>pack_objects_into_box</nobr> |  |  |  |  |  |  |  |  |
| <nobr>pack_objects_into_box_random</nobr> |  |  |  |  |  |  |  |  |
| <nobr>fold_clothes</nobr> |  |  |  |  |  |  |  |  |
| <nobr>fold_clothes_random</nobr> |  |  |  |  |  |  |  |  |
| <nobr>hang_mugs</nobr> |  |  |  |  |  |  |  |  |
| <nobr>hang_mugs_random</nobr> |  |  |  |  |  |  |  |  |
| <nobr>sweep_blocks</nobr> |  |  |  |  |  |  |  |  |
| <nobr>sweep_blocks_random</nobr> |  |  |  |  |  |  |  |  |
| <nobr>pour_liquid_into_cup</nobr> |  |  |  |  |  |  |  |  |
| <nobr>pour_liquid_into_cup_random</nobr> |  |  |  |  |  |  |  |  |
| <nobr>make_toast</nobr> |  |  |  |  |  |  |  |  |
| <nobr>make_toast_random</nobr> |  |  |  |  |  |  |  |  |
| <nobr>arrange_largest_number</nobr> | 4 | 12.20 |  |  |  |  |  |  |
| <nobr>arrange_largest_number_random</nobr> |  |  |  |  |  |  |  |  |
| <nobr>sort_nesting_dolls_by_size</nobr> |  |  |  |  |  |  |  |  |
| <nobr>sort_nesting_dolls_by_size_random</nobr> |  |  |  |  |  |  |  |  |
| <nobr>store_laptop_and_headphones</nobr> |  |  |  |  |  |  |  |  |
| <nobr>store_laptop_and_headphones_random</nobr> |  |  |  |  |  |  |  |  |
| <nobr>stack_blocks</nobr> |  |  |  |  |  |  |  |  |
| <nobr>stack_blocks_random</nobr> |  |  |  |  |  |  |  |  |
| **Avg Standard** | 4.0 | 12.20 |  |  |  |  | 4 ±0 | 12.20 ±0.00 |
| **Avg Random** |  |  |  |  |  |  |  |  |

## RSIWAM

| Task | Seed 0 SR (%) | Seed 0 Score | Seed 1 SR (%) | Seed 1 Score | Seed 2 SR (%) | Seed 2 Score | Avg SR (%) | Avg Score |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| <nobr>align_blocks</nobr> | 0 | 0.00 |  |  |  |  |  |  |
| <nobr>arrange_largest_number</nobr> |  |  |  |  |  |  |  |  |
| <nobr>build_tower</nobr> | 88 | 90.20 |  |  |  |  |  |  |
| <nobr>classify_objects</nobr> | 46 | 55.50 |  |  |  |  |  |  |
| <nobr>classify_objects_by_language</nobr> |  |  |  |  |  |  |  |  |
| <nobr>cover_blocks</nobr> | 100 | 100.00 |  |  |  |  |  |  |
| <nobr>deposit_coin</nobr> | 18 | 23.20 |  |  |  |  |  |  |
| <nobr>fasten_screws</nobr> |  |  |  |  |  |  |  |  |
| <nobr>fill_egg_holder</nobr> | 0 | 8.50 |  |  |  |  |  |  |
| <nobr>fill_pen_holder</nobr> |  |  |  |  |  |  |  |  |
| <nobr>fold_clothes</nobr> |  |  |  |  |  |  |  |  |
| <nobr>general_pickup</nobr> | 20 | 20.00 |  |  |  |  |  |  |
| <nobr>hang_mugs</nobr> |  |  |  |  |  |  |  |  |
| <nobr>imitate_sorting_sequence</nobr> |  |  |  |  |  |  |  |  |
| <nobr>insert_key</nobr> |  |  |  |  |  |  |  |  |
| <nobr>insert_tubes</nobr> |  |  |  |  |  |  |  |  |
| <nobr>make_kong</nobr> |  |  |  |  |  |  |  |  |
| <nobr>make_toast</nobr> |  |  |  |  |  |  |  |  |
| <nobr>match_and_pick_from_conveyor</nobr> |  |  |  |  |  |  |  |  |
| <nobr>organize_table</nobr> |  |  |  |  |  |  |  |  |
| <nobr>pack_objects_into_box</nobr> |  |  |  |  |  |  |  |  |
| <nobr>pick_from_conveyor_by_image</nobr> |  |  |  |  |  |  |  |  |
| <nobr>play_Xylophone</nobr> |  |  |  |  |  |  |  |  |
| <nobr>play_stacking_toy</nobr> |  |  |  |  |  |  |  |  |
| <nobr>play_tic_tac_toe</nobr> |  |  |  |  |  |  |  |  |
| <nobr>plug_in_charger</nobr> |  |  |  |  |  |  |  |  |
| <nobr>pour_balls_into_vase</nobr> |  |  |  |  |  |  |  |  |
| <nobr>pour_by_language</nobr> |  |  |  |  |  |  |  |  |
| <nobr>pour_liquid_into_cup</nobr> |  |  |  |  |  |  |  |  |
| <nobr>press_by_number</nobr> |  |  |  |  |  |  |  |  |
| <nobr>push_T</nobr> |  |  |  |  |  |  |  |  |
| <nobr>put_bottles_into_dustbin</nobr> |  |  |  |  |  |  |  |  |
| <nobr>solve_equation</nobr> |  |  |  |  |  |  |  |  |
| <nobr>sort_nesting_dolls_by_size</nobr> |  |  |  |  |  |  |  |  |
| <nobr>stack_blocks</nobr> |  |  |  |  |  |  |  |  |
| <nobr>stack_blocks_by_language</nobr> |  |  |  |  |  |  |  |  |
| <nobr>stack_bowls</nobr> |  |  |  |  |  |  |  |  |
| <nobr>store_laptop_and_headphones</nobr> |  |  |  |  |  |  |  |  |
| <nobr>store_tools_in_toolbox</nobr> |  |  |  |  |  |  |  |  |
| <nobr>swap_T</nobr> |  |  |  |  |  |  |  |  |
| <nobr>swap_blocks</nobr> |  |  |  |  |  |  |  |  |
| <nobr>sweep_blocks</nobr> |  |  |  |  |  |  |  |  |
| **Avg (per seed)** | 38.9 | 42.49 |  |  |  |  | 39 ±0 | 42.49 ±0.00 |

