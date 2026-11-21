# RoboDojo Evaluation Summary

## Overview (by capability dimension)

Progress † marks incomplete evaluation (126 = 42 tasks × 3 seeds). Cells show `mean ±std` across seeds; scores reflect completed cells only. Policies are sorted by overall Avg SR.

| Policy | Progress | Generalization SR (%) | Generalization Score | Precision SR (%) | Precision Score | Long-Horizon SR (%) | Long-Horizon Score | Memory SR (%) | Memory Score | Open SR (%) | Open Score | Avg SR (%) | Avg Score |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Bagel-Robotics-1_original_selection | 92.1% † (116/126) | 28.33 ±1.71 | 35.20 ±1.35 | 34.98 ±2.48 | 43.68 ±2.30 | 40.10 ±1.10 | 58.84 ±0.43 | 67.73 ±2.66 | 67.73 ±2.66 | 1.65 ±1.15 | 2.29 ±1.15 | 34.56 ±0.79 | 41.55 ±0.82 |

## Generalization: Standard vs Random (25 episodes each)

Each cell shows `mean ±std` across seeds, averaged over the 12 Generalization tasks with completed data for that half. Δ = Random − Standard (mean SR). Sorted by Standard SR.

| Policy | Standard SR (%) | Standard Score | Random SR (%) | Random Score | Δ SR | Δ Score |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Bagel-Robotics-1_original_selection | 45.67 ±2.68 | 51.77 ±2.32 | 11.00 ±2.23 | 18.63 ±2.18 | -34.67 | -33.14 |

### Per-task Standard vs Random

Additional view only: each of the 12 Generalization tasks is listed as the 25-episode standard split and its `_random` sibling. Columns match the per-policy all-task tables. Existing merged 50-episode scores are unchanged.

#### Bagel-Robotics-1_original_selection

| Task | Seed 0 SR (%) | Seed 0 Score | Seed 1 SR (%) | Seed 1 Score | Seed 2 SR (%) | Seed 2 Score | Avg SR (%) | Avg Score |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| <nobr>stack_bowls</nobr> | 96 | 96.60 | 92 | 93.20 | 84 | 86.40 | 91 ±6 | 92.07 ±5.10 |
| <nobr>stack_bowls_random</nobr> | 0 | 14.40 | 12 | 24.60 | 8 | 21.80 | 7 ±6 | 20.27 ±5.10 |
| <nobr>push_T</nobr> | 0 | 0.00 | 4 | 4.00 | 4 | 4.00 | 3 ±2 | 2.67 ±2.00 |
| <nobr>push_T_random</nobr> | 0 | 0.00 | 0 | 0.00 | 0 | 0.00 | 0 ±0 | 0.00 ±0.00 |
| <nobr>pack_objects_into_box</nobr> | 0 | 29.40 | 16 | 42.60 | 8 | 34.60 | 8 ±8 | 35.53 ±6.60 |
| <nobr>pack_objects_into_box_random</nobr> | 4 | 23.60 | 8 | 28.20 | 4 | 25.20 | 5 ±2 | 25.67 ±2.30 |
| <nobr>fold_clothes</nobr> | 68 | 70.40 | 80 | 80.80 | 64 | 68.00 | 71 ±8 | 73.07 ±6.40 |
| <nobr>fold_clothes_random</nobr> | 64 | 65.60 | 64 | 68.00 | 84 | 85.60 | 71 ±10 | 73.07 ±10.00 |
| <nobr>hang_mugs</nobr> | 60 | 71.00 | 56 | 67.40 | 56 | 69.00 | 57 ±2 | 69.13 ±1.80 |
| <nobr>hang_mugs_random</nobr> | 0 | 11.80 | 0 | 9.60 | 0 | 12.20 | 0 ±0 | 11.20 ±1.30 |
| <nobr>sweep_blocks</nobr> | 16 | 16.00 | 8 | 8.00 | 8 | 8.00 | 11 ±4 | 10.67 ±4.00 |
| <nobr>sweep_blocks_random</nobr> | 4 | 4.00 | 0 | 0.00 | 8 | 8.00 | 4 ±4 | 4.00 ±4.00 |
| <nobr>pour_liquid_into_cup</nobr> | 52 | 52.00 | 68 | 68.00 | 64 | 64.00 | 61 ±8 | 61.33 ±8.00 |
| <nobr>pour_liquid_into_cup_random</nobr> | 8 | 8.00 | 20 | 20.00 | 28 | 28.00 | 19 ±10 | 18.67 ±10.00 |
| <nobr>make_toast</nobr> | 56 | 71.00 | 72 | 79.00 | 60 | 67.00 | 63 ±8 | 72.33 ±6.00 |
| <nobr>make_toast_random</nobr> | 8 | 24.00 | 16 | 26.00 | 8 | 23.00 | 11 ±4 | 24.33 ±1.50 |
| <nobr>arrange_largest_number</nobr> | 12 | 16.60 | 4 | 13.00 | 0 | 8.60 | 5 ±6 | 12.73 ±4.00 |
| <nobr>arrange_largest_number_random</nobr> | 0 | 4.60 | 4 | 9.00 | 0 | 6.60 | 1 ±2 | 6.73 ±2.20 |
| <nobr>sort_nesting_dolls_by_size</nobr> | 36 | 36.00 | 36 | 36.00 | 28 | 28.00 | 33 ±4 | 33.33 ±4.00 |
| <nobr>sort_nesting_dolls_by_size_random</nobr> | 4 | 4.00 | 4 | 4.00 | 0 | 0.00 | 3 ±2 | 2.67 ±2.00 |
| <nobr>store_laptop_and_headphones</nobr> | 68 | 78.40 | 68 | 76.80 | 48 | 60.80 | 61 ±10 | 72.00 ±8.80 |
| <nobr>store_laptop_and_headphones_random</nobr> | 4 | 20.00 | 12 | 25.60 | 20 | 33.60 | 12 ±8 | 26.40 ±6.80 |
| <nobr>stack_blocks</nobr> | 72 | 76.20 | 88 | 89.80 | 92 | 93.20 | 84 ±10 | 86.40 ±8.50 |
| <nobr>stack_blocks_random</nobr> | 0 | 10.80 | 0 | 10.20 | 0 | 10.80 | 0 ±0 | 10.60 ±0.30 |
| **Avg Standard** | 44.7 | 51.13 | 49.3 | 54.88 | 43.0 | 49.30 | 46 ±3 | 51.77 ±2.79 |
| **Avg Random** | 8.0 | 15.90 | 11.7 | 18.77 | 13.3 | 21.23 | 11 ±3 | 18.63 ±2.67 |

## Bagel-Robotics-1_original_selection

| Task | Seed 0 SR (%) | Seed 0 Score | Seed 1 SR (%) | Seed 1 Score | Seed 2 SR (%) | Seed 2 Score | Avg SR (%) | Avg Score |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| <nobr>align_blocks</nobr> | 0 | 0.00 | 0 | 0.00 | 0 | 0.00 | 0 ±0 | 0.00 ±0.00 |
| <nobr>arrange_largest_number</nobr> | 6 | 10.60 | 4 | 11.00 | 0 | 7.60 | 3 ±3 | 9.73 ±1.70 |
| <nobr>build_tower</nobr> | 80 | 85.60 | 64 | 73.60 | 68 | 75.40 | 71 ±8 | 78.20 ±6.00 |
| <nobr>classify_objects</nobr> | 46 | 56.70 | 32 | 44.50 | 42 | 51.10 | 40 ±7 | 50.77 ±6.10 |
| <nobr>classify_objects_by_language</nobr> | 0 | 3.60 | 0 | 2.60 | 6 | 9.20 | 2 ±3 | 5.13 ±3.30 |
| <nobr>cover_blocks</nobr> | 100 | 100.00 | 100 | 100.00 | 100 | 100.00 | 100 ±0 | 100.00 ±0.00 |
| <nobr>deposit_coin</nobr> |  |  | 22 | 26.80 |  |  |  |  |
| <nobr>fasten_screws</nobr> | 48 | 72.20 | 36 | 66.80 | 40 | 68.20 | 41 ±6 | 69.07 ±2.70 |
| <nobr>fill_egg_holder</nobr> |  |  |  |  |  |  |  |  |
| <nobr>fill_pen_holder</nobr> | 36 | 53.90 | 22 | 43.90 | 30 | 52.40 | 29 ±7 | 50.07 ±5.00 |
| <nobr>fold_clothes</nobr> | 66 | 68.00 | 72 | 74.40 | 74 | 76.80 | 71 ±4 | 73.07 ±4.40 |
| <nobr>general_pickup</nobr> |  |  |  |  | 16 | 16.00 |  |  |
| <nobr>hang_mugs</nobr> | 30 | 41.40 | 28 | 38.50 | 28 | 40.60 | 29 ±1 | 40.17 ±1.45 |
| <nobr>imitate_sorting_sequence</nobr> |  |  |  |  |  |  |  |  |
| <nobr>insert_key</nobr> | 0 | 15.00 | 0 | 15.00 | 0 | 15.00 | 0 ±0 | 15.00 ±0.00 |
| <nobr>insert_tubes</nobr> | 72 | 82.80 | 64 | 75.20 | 60 | 74.00 | 65 ±6 | 77.33 ±4.40 |
| <nobr>make_kong</nobr> | 84 | 84.00 | 86 | 86.00 | 88 | 88.00 | 86 ±2 | 86.00 ±2.00 |
| <nobr>make_toast</nobr> | 32 | 47.50 | 44 | 52.50 | 34 | 45.00 | 37 ±6 | 48.33 ±3.75 |
| <nobr>match_and_pick_from_conveyor</nobr> | 82 | 82.00 | 92 | 92.00 | 86 | 86.00 | 87 ±5 | 86.67 ±5.00 |
| <nobr>organize_table</nobr> | 14 | 57.50 | 18 | 64.50 | 12 | 58.00 | 15 ±3 | 60.00 ±3.50 |
| <nobr>pack_objects_into_box</nobr> | 2 | 26.50 | 12 | 35.40 | 6 | 29.90 | 7 ±5 | 30.60 ±4.45 |
| <nobr>pick_from_conveyor_by_image</nobr> | 0 | 0.00 | 0 | 0.00 | 0 | 0.00 | 0 ±0 | 0.00 ±0.00 |
| <nobr>play_Xylophone</nobr> | 0 | 0.00 | 0 | 0.00 | 0 | 0.00 | 0 ±0 | 0.00 ±0.00 |
| <nobr>play_stacking_toy</nobr> | 14 | 31.80 | 12 | 32.00 | 8 | 27.20 | 11 ±3 | 30.33 ±2.40 |
| <nobr>play_tic_tac_toe</nobr> | 0 | 35.80 | 2 | 40.50 | 6 | 33.90 | 3 ±3 | 36.73 ±3.30 |
| <nobr>plug_in_charger</nobr> | 12 | 12.00 | 4 | 4.00 | 8 | 8.00 | 8 ±4 | 8.00 ±4.00 |
| <nobr>pour_balls_into_vase</nobr> | 48 | 48.00 | 62 | 62.00 | 78 | 78.00 | 63 ±15 | 62.67 ±15.00 |
| <nobr>pour_by_language</nobr> | 0 | 0.00 | 0 | 0.00 | 0 | 0.00 | 0 ±0 | 0.00 ±0.00 |
| <nobr>pour_liquid_into_cup</nobr> | 30 | 30.00 | 44 | 44.00 | 46 | 46.00 | 40 ±8 | 40.00 ±8.00 |
| <nobr>press_by_number</nobr> | 98 | 98.00 | 100 | 100.00 | 100 | 100.00 | 99 ±1 | 99.33 ±1.00 |
| <nobr>push_T</nobr> | 0 | 0.00 | 2 | 2.00 | 2 | 2.00 | 1 ±1 | 1.33 ±1.00 |
| <nobr>put_bottles_into_dustbin</nobr> | 94 | 96.40 | 98 | 98.80 | 98 | 98.80 | 97 ±2 | 98.00 ±1.20 |
| <nobr>solve_equation</nobr> | 0 | 0.00 | 0 | 0.00 | 0 | 0.00 | 0 ±0 | 0.00 ±0.00 |
| <nobr>sort_nesting_dolls_by_size</nobr> | 20 | 20.00 | 20 | 20.00 | 14 | 14.00 | 18 ±3 | 18.00 ±3.00 |
| <nobr>stack_blocks</nobr> | 36 | 43.50 | 44 | 50.00 | 46 | 52.00 | 42 ±5 | 48.50 ±4.25 |
| <nobr>stack_blocks_by_language</nobr> | 4 | 4.80 | 8 | 9.20 | 4 | 6.00 | 5 ±2 | 6.67 ±2.20 |
| <nobr>stack_bowls</nobr> | 48 | 55.50 | 52 | 58.90 | 46 | 54.10 | 49 ±3 | 56.17 ±2.40 |
| <nobr>store_laptop_and_headphones</nobr> | 36 | 49.20 | 40 | 51.20 | 34 | 47.20 | 37 ±3 | 49.20 ±2.00 |
| <nobr>store_tools_in_toolbox</nobr> | 0 | 0.50 | 0 | 0.00 | 0 | 0.00 | 0 ±0 | 0.17 ±0.25 |
| <nobr>swap_T</nobr> | 22 | 22.00 | 22 | 22.00 | 32 | 32.00 | 25 ±5 | 25.33 ±5.00 |
| <nobr>swap_blocks</nobr> | 18 | 18.00 | 32 | 32.00 | 32 | 32.00 | 27 ±7 | 27.33 ±7.00 |
| <nobr>sweep_blocks</nobr> | 10 | 10.00 | 4 | 4.00 | 8 | 8.00 | 7 ±3 | 7.33 ±3.00 |
| **Avg (per seed)** | 31.3 | 38.49 | 31.8 | 39.32 | 32.1 | 39.29 | 32 ±0 | 39.03 ±0.41 |

