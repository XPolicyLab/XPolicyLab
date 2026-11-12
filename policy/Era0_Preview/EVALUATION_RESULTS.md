# RoboDojo Evaluation Summary

## Overview (by capability dimension)

Progress † marks incomplete evaluation (126 = 42 tasks × 3 seeds). Cells show `mean ±std` across seeds; scores reflect completed cells only. Policies are sorted by overall Avg SR.

| Policy | Progress | Generalization SR (%) | Generalization Score | Precision SR (%) | Precision Score | Long-Horizon SR (%) | Long-Horizon Score | Memory SR (%) | Memory Score | Open SR (%) | Open Score | Avg SR (%) | Avg Score |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Era0_Preview | 97.6% † (123/126) | 38.18 ±2.20 | 44.07 ±2.40 | 30.33 ±1.36 | 36.85 ±0.78 | 36.50 ±0.89 | 48.11 ±1.86 | 52.78 ±1.03 | 54.20 ±0.85 | 7.75 ±0.35 | 8.59 ±0.26 | 33.11 ±0.27 | 38.36 ±0.55 |

## Generalization: Standard vs Random (25 episodes each)

Each cell shows `mean ±std` across seeds, averaged over the 12 Generalization tasks with completed data for that half. Δ = Random − Standard (mean SR). Sorted by Standard SR.

| Policy | Standard SR (%) | Standard Score | Random SR (%) | Random Score | Δ SR | Δ Score |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Era0_Preview | 43.33 ±3.81 | 49.06 ±3.28 | 31.03 ±1.98 | 37.35 ±2.41 | -12.30 | -11.71 |

### Per-task Standard vs Random

Additional view only: each of the 12 Generalization tasks is listed as the 25-episode standard split and its `_random` sibling. Columns match the per-policy all-task tables. Existing merged 50-episode scores are unchanged.

#### Era0_Preview

| Task | Seed 0 SR (%) | Seed 0 Score | Seed 1 SR (%) | Seed 1 Score | Seed 2 SR (%) | Seed 2 Score | Avg SR (%) | Avg Score |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| <nobr>stack_bowls</nobr> | 100 | 100.00 | 100 | 100.00 | 96 | 96.60 | 99 ±2 | 98.87 ±1.70 |
| <nobr>stack_bowls_random</nobr> | 60 | 63.60 | 72 | 73.80 | 60 | 65.40 | 64 ±6 | 67.60 ±5.10 |
| <nobr>push_T</nobr> | 0 | 0.00 | 0 | 0.00 | 0 | 0.00 | 0 ±0 | 0.00 ±0.00 |
| <nobr>push_T_random</nobr> | 4 | 4.00 | 0 | 0.00 | 0 | 0.00 | 1 ±2 | 1.33 ±2.00 |
| <nobr>pack_objects_into_box</nobr> | 16 | 38.20 | 28 | 53.20 | 32 | 50.20 | 25 ±8 | 47.20 ±7.50 |
| <nobr>pack_objects_into_box_random</nobr> | 0 | 20.80 | 16 | 37.20 | 4 | 21.00 | 7 ±8 | 26.33 ±8.20 |
| <nobr>fold_clothes</nobr> | 84 | 85.60 | 88 | 90.40 | 96 | 96.80 | 89 ±6 | 90.93 ±5.60 |
| <nobr>fold_clothes_random</nobr> | 88 | 89.60 | 56 | 62.40 | 68 | 71.20 | 71 ±16 | 74.40 ±13.60 |
| <nobr>hang_mugs</nobr> | 20 | 29.60 | 36 | 44.80 | 40 | 51.60 | 32 ±10 | 42.00 ±11.00 |
| <nobr>hang_mugs_random</nobr> | 0 | 10.00 | 0 | 12.80 | 0 | 9.40 | 0 ±0 | 10.73 ±1.70 |
| <nobr>sweep_blocks</nobr> | 8 | 8.00 | 12 | 12.00 | 0 | 0.00 | 7 ±6 | 6.67 ±6.00 |
| <nobr>sweep_blocks_random</nobr> | 4 | 4.00 | 0 | 0.00 | 4 | 4.00 | 3 ±2 | 2.67 ±2.00 |
| <nobr>pour_liquid_into_cup</nobr> | 72 | 72.00 | 84 | 84.00 | 72 | 72.00 | 76 ±6 | 76.00 ±6.00 |
| <nobr>pour_liquid_into_cup_random</nobr> | 76 | 76.00 | 92 | 92.00 | 64 | 64.00 | 77 ±14 | 77.33 ±14.00 |
| <nobr>make_toast</nobr> | 12 | 24.00 | 32 | 38.00 | 20 | 28.00 | 21 ±10 | 30.00 ±7.00 |
| <nobr>make_toast_random</nobr> |  |  |  |  |  |  |  |  |
| <nobr>arrange_largest_number</nobr> | 12 | 21.60 | 16 | 23.40 | 24 | 32.40 | 17 ±6 | 25.80 ±5.40 |
| <nobr>arrange_largest_number_random</nobr> | 12 | 15.40 | 16 | 19.00 | 20 | 23.60 | 16 ±4 | 19.33 ±4.10 |
| <nobr>sort_nesting_dolls_by_size</nobr> | 44 | 44.00 | 36 | 36.00 | 36 | 36.00 | 39 ±4 | 38.67 ±4.00 |
| <nobr>sort_nesting_dolls_by_size_random</nobr> | 44 | 44.00 | 48 | 48.00 | 32 | 32.00 | 41 ±8 | 41.33 ±8.00 |
| <nobr>store_laptop_and_headphones</nobr> | 28 | 46.40 | 60 | 70.40 | 40 | 55.20 | 43 ±16 | 57.33 ±12.00 |
| <nobr>store_laptop_and_headphones_random</nobr> | 44 | 59.20 | 32 | 60.00 | 32 | 51.20 | 36 ±6 | 56.80 ±4.40 |
| <nobr>stack_blocks</nobr> | 68 | 72.20 | 84 | 85.80 | 64 | 67.60 | 72 ±10 | 75.20 ±9.10 |
| <nobr>stack_blocks_random</nobr> | 16 | 23.20 | 32 | 38.60 | 28 | 37.00 | 25 ±8 | 32.93 ±7.70 |
| **Avg Standard** | 38.7 | 45.13 | 48.0 | 53.17 | 43.3 | 48.87 | 43 ±5 | 49.06 ±4.02 |
| **Avg Random** | 31.6 | 37.25 | 33.1 | 40.35 | 28.4 | 34.44 | 31 ±2 | 37.35 ±2.95 |

## Era0_Preview

| Task | Seed 0 SR (%) | Seed 0 Score | Seed 1 SR (%) | Seed 1 Score | Seed 2 SR (%) | Seed 2 Score | Avg SR (%) | Avg Score |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| <nobr>align_blocks</nobr> | 30 | 30.00 | 28 | 28.00 | 32 | 32.00 | 30 ±2 | 30.00 ±2.00 |
| <nobr>arrange_largest_number</nobr> | 12 | 18.50 | 16 | 21.20 | 22 | 28.00 | 17 ±5 | 22.57 ±4.75 |
| <nobr>build_tower</nobr> | 74 | 77.60 | 78 | 81.20 | 84 | 86.80 | 79 ±5 | 81.87 ±4.60 |
| <nobr>classify_objects</nobr> | 78 | 81.20 | 86 | 88.50 | 76 | 80.80 | 80 ±5 | 83.50 ±3.85 |
| <nobr>classify_objects_by_language</nobr> | 0 | 4.60 | 2 | 6.60 | 0 | 3.80 | 1 ±1 | 5.00 ±1.40 |
| <nobr>cover_blocks</nobr> | 98 | 98.30 | 98 | 98.10 | 98 | 98.60 | 98 ±0 | 98.33 ±0.25 |
| <nobr>deposit_coin</nobr> | 6 | 11.60 | 2 | 8.00 | 8 | 12.40 | 5 ±3 | 10.67 ±2.20 |
| <nobr>fasten_screws</nobr> | 8 | 31.00 | 16 | 42.20 | 14 | 34.80 | 13 ±4 | 36.00 ±5.60 |
| <nobr>fill_egg_holder</nobr> | 0 | 9.60 | 0 | 9.50 | 2 | 10.00 | 1 ±1 | 9.70 ±0.25 |
| <nobr>fill_pen_holder</nobr> | 30 | 46.40 | 30 | 50.00 | 26 | 50.60 | 29 ±2 | 49.00 ±2.10 |
| <nobr>fold_clothes</nobr> | 86 | 87.60 | 72 | 76.40 | 82 | 84.00 | 80 ±7 | 82.67 ±5.60 |
| <nobr>general_pickup</nobr> | 16 | 16.00 | 10 | 10.00 | 14 | 14.00 | 13 ±3 | 13.33 ±3.00 |
| <nobr>hang_mugs</nobr> | 10 | 19.80 | 18 | 28.80 | 20 | 30.50 | 16 ±5 | 26.37 ±5.35 |
| <nobr>imitate_sorting_sequence</nobr> | 32 | 39.20 | 34 | 42.00 | 28 | 37.40 | 31 ±3 | 39.53 ±2.30 |
| <nobr>insert_key</nobr> | 0 | 10.80 | 0 | 12.30 | 0 | 12.60 | 0 ±0 | 11.90 ±0.90 |
| <nobr>insert_tubes</nobr> | 70 | 77.60 | 58 | 68.80 | 68 | 74.80 | 65 ±6 | 73.73 ±4.40 |
| <nobr>make_kong</nobr> | 56 | 56.00 | 56 | 56.00 | 72 | 72.00 | 61 ±8 | 61.33 ±8.00 |
| <nobr>make_toast</nobr> |  |  |  |  |  |  |  |  |
| <nobr>match_and_pick_from_conveyor</nobr> | 82 | 82.00 | 84 | 84.00 | 82 | 82.00 | 83 ±1 | 82.67 ±1.00 |
| <nobr>organize_table</nobr> | 22 | 62.50 | 26 | 71.00 | 22 | 71.50 | 23 ±2 | 68.33 ±4.50 |
| <nobr>pack_objects_into_box</nobr> | 8 | 29.50 | 22 | 45.20 | 18 | 35.60 | 16 ±7 | 36.77 ±7.85 |
| <nobr>pick_from_conveyor_by_image</nobr> | 0 | 0.00 | 0 | 0.00 | 0 | 0.00 | 0 ±0 | 0.00 ±0.00 |
| <nobr>play_Xylophone</nobr> | 16 | 16.00 | 18 | 18.00 | 20 | 20.00 | 18 ±2 | 18.00 ±2.00 |
| <nobr>play_stacking_toy</nobr> | 0 | 0.00 | 0 | 0.00 | 0 | 0.00 | 0 ±0 | 0.00 ±0.00 |
| <nobr>play_tic_tac_toe</nobr> | 0 | 10.80 | 0 | 18.30 | 0 | 13.50 | 0 ±0 | 14.20 ±3.75 |
| <nobr>plug_in_charger</nobr> | 10 | 10.00 | 4 | 4.00 | 0 | 0.00 | 5 ±5 | 4.67 ±5.00 |
| <nobr>pour_balls_into_vase</nobr> | 62 | 62.00 | 52 | 52.00 | 60 | 60.00 | 58 ±5 | 58.00 ±5.00 |
| <nobr>pour_by_language</nobr> | 0 | 0.00 | 0 | 0.00 | 0 | 0.00 | 0 ±0 | 0.00 ±0.00 |
| <nobr>pour_liquid_into_cup</nobr> | 74 | 74.00 | 88 | 88.00 | 68 | 68.00 | 77 ±10 | 76.67 ±10.00 |
| <nobr>press_by_number</nobr> | 92 | 92.00 | 90 | 90.00 | 88 | 88.00 | 90 ±2 | 90.00 ±2.00 |
| <nobr>push_T</nobr> | 2 | 2.00 | 0 | 0.00 | 0 | 0.00 | 1 ±1 | 0.67 ±1.00 |
| <nobr>put_bottles_into_dustbin</nobr> | 96 | 97.60 | 98 | 98.80 | 100 | 100.00 | 98 ±2 | 98.80 ±1.20 |
| <nobr>solve_equation</nobr> | 18 | 18.00 | 18 | 18.00 | 18 | 18.00 | 18 ±0 | 18.00 ±0.00 |
| <nobr>sort_nesting_dolls_by_size</nobr> | 44 | 44.00 | 42 | 42.00 | 34 | 34.00 | 40 ±5 | 40.00 ±5.00 |
| <nobr>stack_blocks</nobr> | 42 | 47.70 | 58 | 62.20 | 46 | 52.30 | 49 ±8 | 54.07 ±7.25 |
| <nobr>stack_blocks_by_language</nobr> | 0 | 2.00 | 0 | 3.20 | 0 | 2.00 | 0 ±0 | 2.40 ±0.60 |
| <nobr>stack_bowls</nobr> | 80 | 81.80 | 86 | 86.90 | 78 | 81.00 | 81 ±4 | 83.23 ±2.95 |
| <nobr>store_laptop_and_headphones</nobr> | 36 | 52.80 | 46 | 65.20 | 36 | 53.20 | 39 ±5 | 57.07 ±6.20 |
| <nobr>store_tools_in_toolbox</nobr> | 0 | 0.00 | 0 | 0.00 | 0 | 0.00 | 0 ±0 | 0.00 ±0.00 |
| <nobr>swap_T</nobr> | 0 | 0.00 | 0 | 0.00 | 0 | 0.00 | 0 ±0 | 0.00 ±0.00 |
| <nobr>swap_blocks</nobr> | 18 | 18.00 | 14 | 14.00 | 12 | 12.00 | 15 ±3 | 14.67 ±3.00 |
| <nobr>sweep_blocks</nobr> | 6 | 6.00 | 6 | 6.00 | 2 | 2.00 | 5 ±2 | 4.67 ±2.00 |
| **Avg (per seed)** | 32.0 | 37.18 | 33.1 | 38.89 | 32.4 | 37.96 | 33 ±1 | 38.01 ±0.85 |

