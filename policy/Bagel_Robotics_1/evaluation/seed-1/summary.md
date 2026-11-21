# RoboDojo Evaluation Summary

## Overview (by capability dimension)

Progress † marks incomplete evaluation (126 = 42 tasks × 3 seeds). Cells show `mean ±std` across seeds; scores reflect completed cells only. Policies are sorted by overall Avg SR.

| Policy | Progress | Generalization SR (%) | Generalization Score | Precision SR (%) | Precision Score | Long-Horizon SR (%) | Long-Horizon Score | Memory SR (%) | Memory Score | Open SR (%) | Open Score | Avg SR (%) | Avg Score |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Bagel-Robotics-1_with_labeled_replacements_and_repair | 33.3% † (42/126) | 30.50 ±0.00 | 36.82 ±0.00 | 31.50 ±0.00 | 40.42 ±0.00 | 34.25 ±0.00 | 53.00 ±0.00 | 57.67 ±0.00 | 58.00 ±0.00 | 4.00 ±0.00 | 4.47 ±0.00 | 31.58 ±0.00 | 38.55 ±0.00 |

## Generalization: Standard vs Random (25 episodes each)

Each cell shows `mean ±std` across seeds, averaged over the 12 Generalization tasks with completed data for that half. Δ = Random − Standard (mean SR). Sorted by Standard SR.

| Policy | Standard SR (%) | Standard Score | Random SR (%) | Random Score | Δ SR | Δ Score |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Bagel-Robotics-1_with_labeled_replacements_and_repair | 49.33 ±0.00 | 54.88 ±0.00 | 11.67 ±0.00 | 18.77 ±0.00 | -37.67 | -36.12 |

### Per-task Standard vs Random

Additional view only: each of the 12 Generalization tasks is listed as the 25-episode standard split and its `_random` sibling. Columns match the per-policy all-task tables. Existing merged 50-episode scores are unchanged.

#### Bagel-Robotics-1_with_labeled_replacements_and_repair

| Task | Seed 0 SR (%) | Seed 0 Score | Seed 1 SR (%) | Seed 1 Score | Seed 2 SR (%) | Seed 2 Score | Avg SR (%) | Avg Score |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| <nobr>stack_bowls</nobr> |  |  | 92 | 93.20 |  |  |  |  |
| <nobr>stack_bowls_random</nobr> |  |  | 12 | 24.60 |  |  |  |  |
| <nobr>push_T</nobr> |  |  | 4 | 4.00 |  |  |  |  |
| <nobr>push_T_random</nobr> |  |  | 0 | 0.00 |  |  |  |  |
| <nobr>pack_objects_into_box</nobr> |  |  | 16 | 42.60 |  |  |  |  |
| <nobr>pack_objects_into_box_random</nobr> |  |  | 8 | 28.20 |  |  |  |  |
| <nobr>fold_clothes</nobr> |  |  | 80 | 80.80 |  |  |  |  |
| <nobr>fold_clothes_random</nobr> |  |  | 64 | 68.00 |  |  |  |  |
| <nobr>hang_mugs</nobr> |  |  | 56 | 67.40 |  |  |  |  |
| <nobr>hang_mugs_random</nobr> |  |  | 0 | 9.60 |  |  |  |  |
| <nobr>sweep_blocks</nobr> |  |  | 8 | 8.00 |  |  |  |  |
| <nobr>sweep_blocks_random</nobr> |  |  | 0 | 0.00 |  |  |  |  |
| <nobr>pour_liquid_into_cup</nobr> |  |  | 68 | 68.00 |  |  |  |  |
| <nobr>pour_liquid_into_cup_random</nobr> |  |  | 20 | 20.00 |  |  |  |  |
| <nobr>make_toast</nobr> |  |  | 72 | 79.00 |  |  |  |  |
| <nobr>make_toast_random</nobr> |  |  | 16 | 26.00 |  |  |  |  |
| <nobr>arrange_largest_number</nobr> |  |  | 4 | 13.00 |  |  |  |  |
| <nobr>arrange_largest_number_random</nobr> |  |  | 4 | 9.00 |  |  |  |  |
| <nobr>sort_nesting_dolls_by_size</nobr> |  |  | 36 | 36.00 |  |  |  |  |
| <nobr>sort_nesting_dolls_by_size_random</nobr> |  |  | 4 | 4.00 |  |  |  |  |
| <nobr>store_laptop_and_headphones</nobr> |  |  | 68 | 76.80 |  |  |  |  |
| <nobr>store_laptop_and_headphones_random</nobr> |  |  | 12 | 25.60 |  |  |  |  |
| <nobr>stack_blocks</nobr> |  |  | 88 | 89.80 |  |  |  |  |
| <nobr>stack_blocks_random</nobr> |  |  | 0 | 10.20 |  |  |  |  |
| **Avg Standard** |  |  | 49.3 | 54.88 |  |  | 49 ±0 | 54.88 ±0.00 |
| **Avg Random** |  |  | 11.7 | 18.77 |  |  | 12 ±0 | 18.77 ±0.00 |

## Bagel-Robotics-1_with_labeled_replacements_and_repair

| Task | Seed 0 SR (%) | Seed 0 Score | Seed 1 SR (%) | Seed 1 Score | Seed 2 SR (%) | Seed 2 Score | Avg SR (%) | Avg Score |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| <nobr>align_blocks</nobr> |  |  | 0 | 0.00 |  |  |  |  |
| <nobr>arrange_largest_number</nobr> |  |  | 4 | 11.00 |  |  |  |  |
| <nobr>build_tower</nobr> |  |  | 64 | 73.60 |  |  |  |  |
| <nobr>classify_objects</nobr> |  |  | 32 | 44.50 |  |  |  |  |
| <nobr>classify_objects_by_language</nobr> |  |  | 0 | 2.60 |  |  |  |  |
| <nobr>cover_blocks</nobr> |  |  | 100 | 100.00 |  |  |  |  |
| <nobr>deposit_coin</nobr> |  |  | 22 | 26.80 |  |  |  |  |
| <nobr>fasten_screws</nobr> |  |  | 36 | 66.80 |  |  |  |  |
| <nobr>fill_egg_holder</nobr> |  |  | 4 | 13.80 |  |  |  |  |
| <nobr>fill_pen_holder</nobr> |  |  | 22 | 43.90 |  |  |  |  |
| <nobr>fold_clothes</nobr> |  |  | 72 | 74.40 |  |  |  |  |
| <nobr>general_pickup</nobr> |  |  | 24 | 24.00 |  |  |  |  |
| <nobr>hang_mugs</nobr> |  |  | 28 | 38.50 |  |  |  |  |
| <nobr>imitate_sorting_sequence</nobr> |  |  | 0 | 2.00 |  |  |  |  |
| <nobr>insert_key</nobr> |  |  | 0 | 15.00 |  |  |  |  |
| <nobr>insert_tubes</nobr> |  |  | 64 | 75.20 |  |  |  |  |
| <nobr>make_kong</nobr> |  |  | 86 | 86.00 |  |  |  |  |
| <nobr>make_toast</nobr> |  |  | 44 | 52.50 |  |  |  |  |
| <nobr>match_and_pick_from_conveyor</nobr> |  |  | 92 | 92.00 |  |  |  |  |
| <nobr>organize_table</nobr> |  |  | 18 | 64.50 |  |  |  |  |
| <nobr>pack_objects_into_box</nobr> |  |  | 12 | 35.40 |  |  |  |  |
| <nobr>pick_from_conveyor_by_image</nobr> |  |  | 0 | 0.00 |  |  |  |  |
| <nobr>play_Xylophone</nobr> |  |  | 0 | 0.00 |  |  |  |  |
| <nobr>play_stacking_toy</nobr> |  |  | 12 | 32.00 |  |  |  |  |
| <nobr>play_tic_tac_toe</nobr> |  |  | 2 | 40.50 |  |  |  |  |
| <nobr>plug_in_charger</nobr> |  |  | 4 | 4.00 |  |  |  |  |
| <nobr>pour_balls_into_vase</nobr> |  |  | 62 | 62.00 |  |  |  |  |
| <nobr>pour_by_language</nobr> |  |  | 0 | 0.00 |  |  |  |  |
| <nobr>pour_liquid_into_cup</nobr> |  |  | 44 | 44.00 |  |  |  |  |
| <nobr>press_by_number</nobr> |  |  | 100 | 100.00 |  |  |  |  |
| <nobr>push_T</nobr> |  |  | 2 | 2.00 |  |  |  |  |
| <nobr>put_bottles_into_dustbin</nobr> |  |  | 98 | 98.80 |  |  |  |  |
| <nobr>solve_equation</nobr> |  |  | 0 | 0.00 |  |  |  |  |
| <nobr>sort_nesting_dolls_by_size</nobr> |  |  | 20 | 20.00 |  |  |  |  |
| <nobr>stack_blocks</nobr> |  |  | 44 | 50.00 |  |  |  |  |
| <nobr>stack_blocks_by_language</nobr> |  |  | 8 | 9.20 |  |  |  |  |
| <nobr>stack_bowls</nobr> |  |  | 52 | 58.90 |  |  |  |  |
| <nobr>store_laptop_and_headphones</nobr> |  |  | 40 | 51.20 |  |  |  |  |
| <nobr>store_tools_in_toolbox</nobr> |  |  | 0 | 0.00 |  |  |  |  |
| <nobr>swap_T</nobr> |  |  | 22 | 22.00 |  |  |  |  |
| <nobr>swap_blocks</nobr> |  |  | 32 | 32.00 |  |  |  |  |
| <nobr>sweep_blocks</nobr> |  |  | 4 | 4.00 |  |  |  |  |
| **Avg (per seed)** |  |  | 30.2 | 37.45 |  |  | 30 ±0 | 37.45 ±0.00 |

