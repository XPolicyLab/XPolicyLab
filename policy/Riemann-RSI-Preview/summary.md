# RoboDojo Evaluation Summary

## Overview (by capability dimension)

Progress † marks incomplete evaluation (126 = 42 tasks × 3 seeds). Cells show `mean ±std` across seeds; scores reflect completed cells only. Policies are sorted by overall Avg SR.

| Policy | Progress | Generalization SR (%) | Generalization Score | Precision SR (%) | Precision Score | Long-Horizon SR (%) | Long-Horizon Score | Memory SR (%) | Memory Score | Open SR (%) | Open Score | Avg SR (%) | Avg Score |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Riemann-RSI-Preview | 100% | 50.89 ±1.32 | 54.97 ±1.50 | 77.58 ±2.05 | 80.40 ±1.88 | 59.17 ±1.43 | 72.04 ±0.68 | 72.00 ±2.62 | 74.36 ±2.35 | 69.67 ±5.42 | 76.08 ±4.48 | 65.86 ±1.05 | 71.57 ±0.95 |

## Generalization: Standard vs Random (25 episodes each)

Each cell shows `mean ±std` across seeds, averaged over the 12 Generalization tasks with completed data for that half. Δ = Random − Standard (mean SR). Sorted by Standard SR.

| Policy | Standard SR (%) | Standard Score | Random SR (%) | Random Score | Δ SR | Δ Score |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Riemann-RSI-Preview | 57.11 ±2.28 | 62.36 ±2.12 | 44.67 ±0.98 | 47.58 ±1.11 | -12.44 | -14.78 |

### Per-task Standard vs Random

Additional view only: each of the 12 Generalization tasks is listed as the 25-episode standard split and its `_random` sibling. Columns match the per-policy all-task tables. Existing merged 50-episode scores are unchanged.

#### Riemann-RSI-Preview

| Task | Seed 0 SR (%) | Seed 0 Score | Seed 1 SR (%) | Seed 1 Score | Seed 2 SR (%) | Seed 2 Score | Avg SR (%) | Avg Score |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| <nobr>stack_bowls</nobr> | 76 | 79.00 | 80 | 82.40 | 68 | 72.80 | 75 ±6 | 78.07 ±4.80 |
| <nobr>stack_bowls_random</nobr> | 84 | 85.80 | 60 | 63.60 | 76 | 78.40 | 73 ±12 | 75.93 ±11.10 |
| <nobr>push_T</nobr> | 96 | 96.00 | 88 | 88.00 | 84 | 84.00 | 89 ±6 | 89.33 ±6.00 |
| <nobr>push_T_random</nobr> | 96 | 96.00 | 92 | 92.00 | 80 | 80.00 | 89 ±8 | 89.33 ±8.00 |
| <nobr>pack_objects_into_box</nobr> | 32 | 42.40 | 28 | 38.80 | 32 | 37.00 | 31 ±2 | 39.40 ±2.70 |
| <nobr>pack_objects_into_box_random</nobr> | 8 | 13.00 | 0 | 5.40 | 0 | 6.00 | 3 ±4 | 8.13 ±3.80 |
| <nobr>fold_clothes</nobr> | 84 | 87.20 | 84 | 87.20 | 88 | 90.40 | 85 ±2 | 88.27 ±1.60 |
| <nobr>fold_clothes_random</nobr> | 52 | 60.80 | 68 | 72.80 | 72 | 76.80 | 64 ±10 | 70.13 ±8.00 |
| <nobr>hang_mugs</nobr> | 36 | 50.80 | 12 | 38.00 | 16 | 38.20 | 21 ±12 | 42.33 ±6.40 |
| <nobr>hang_mugs_random</nobr> | 0 | 0.00 | 0 | 3.00 | 0 | 2.80 | 0 ±0 | 1.93 ±1.50 |
| <nobr>sweep_blocks</nobr> | 0 | 0.00 | 0 | 0.00 | 0 | 0.00 | 0 ±0 | 0.00 ±0.00 |
| <nobr>sweep_blocks_random</nobr> | 0 | 0.00 | 0 | 0.00 | 0 | 0.00 | 0 ±0 | 0.00 ±0.00 |
| <nobr>pour_liquid_into_cup</nobr> | 100 | 100.00 | 96 | 96.00 | 100 | 100.00 | 99 ±2 | 98.67 ±2.00 |
| <nobr>pour_liquid_into_cup_random</nobr> | 92 | 92.00 | 96 | 96.00 | 92 | 92.00 | 93 ±2 | 93.33 ±2.00 |
| <nobr>make_toast</nobr> | 44 | 58.00 | 32 | 48.00 | 36 | 51.00 | 37 ±6 | 52.33 ±5.00 |
| <nobr>make_toast_random</nobr> | 8 | 21.00 | 4 | 10.00 | 4 | 7.00 | 5 ±2 | 12.67 ±7.00 |
| <nobr>arrange_largest_number</nobr> | 88 | 89.00 | 80 | 83.20 | 84 | 87.00 | 84 ±4 | 86.40 ±2.90 |
| <nobr>arrange_largest_number_random</nobr> | 52 | 54.80 | 48 | 51.40 | 60 | 61.40 | 53 ±6 | 55.87 ±5.00 |
| <nobr>sort_nesting_dolls_by_size</nobr> | 56 | 56.00 | 60 | 60.00 | 52 | 52.00 | 56 ±4 | 56.00 ±4.00 |
| <nobr>sort_nesting_dolls_by_size_random</nobr> | 36 | 36.00 | 68 | 68.00 | 32 | 32.00 | 45 ±18 | 45.33 ±18.00 |
| <nobr>store_laptop_and_headphones</nobr> | 16 | 28.80 | 4 | 14.40 | 12 | 16.80 | 11 ±6 | 20.00 ±7.20 |
| <nobr>store_laptop_and_headphones_random</nobr> | 20 | 29.60 | 16 | 20.80 | 12 | 23.20 | 16 ±4 | 24.53 ±4.40 |
| <nobr>stack_blocks</nobr> | 96 | 96.60 | 100 | 100.00 | 96 | 96.00 | 97 ±2 | 97.53 ±2.00 |
| <nobr>stack_blocks_random</nobr> | 92 | 92.60 | 96 | 96.00 | 92 | 92.60 | 93 ±2 | 93.73 ±1.70 |
| **Avg Standard** | 60.3 | 65.32 | 55.3 | 61.33 | 55.7 | 60.43 | 57 ±2 | 62.36 ±2.44 |
| **Avg Random** | 45.0 | 48.47 | 45.7 | 48.25 | 43.3 | 46.02 | 45 ±1 | 47.58 ±1.22 |

## Riemann-RSI-Preview

| Task | Seed 0 SR (%) | Seed 0 Score | Seed 1 SR (%) | Seed 1 Score | Seed 2 SR (%) | Seed 2 Score | Avg SR (%) | Avg Score |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| <nobr>align_blocks</nobr> | 90 | 90.00 | 96 | 96.00 | 96 | 96.00 | 94 ±3 | 94.00 ±3.00 |
| <nobr>arrange_largest_number</nobr> | 70 | 71.90 | 64 | 67.30 | 72 | 74.20 | 69 ±4 | 71.13 ±3.45 |
| <nobr>build_tower</nobr> | 86 | 87.60 | 78 | 81.60 | 78 | 81.00 | 81 ±4 | 83.40 ±3.30 |
| <nobr>classify_objects</nobr> | 84 | 89.90 | 90 | 93.00 | 74 | 83.90 | 83 ±8 | 88.93 ±4.55 |
| <nobr>classify_objects_by_language</nobr> | 84 | 88.60 | 70 | 78.20 | 76 | 81.20 | 77 ±7 | 82.67 ±5.20 |
| <nobr>cover_blocks</nobr> | 94 | 95.20 | 88 | 89.20 | 98 | 98.60 | 93 ±5 | 94.33 ±4.70 |
| <nobr>deposit_coin</nobr> | 74 | 75.60 | 84 | 85.20 | 72 | 73.20 | 77 ±6 | 78.00 ±6.00 |
| <nobr>fasten_screws</nobr> | 88 | 94.00 | 84 | 91.00 | 80 | 90.00 | 84 ±4 | 91.67 ±2.00 |
| <nobr>fill_egg_holder</nobr> | 14 | 39.50 | 14 | 39.00 | 10 | 35.70 | 13 ±2 | 38.07 ±1.90 |
| <nobr>fill_pen_holder</nobr> | 6 | 19.30 | 8 | 20.10 | 10 | 21.20 | 8 ±2 | 20.20 ±0.95 |
| <nobr>fold_clothes</nobr> | 68 | 74.00 | 76 | 80.00 | 80 | 83.60 | 75 ±6 | 79.20 ±4.80 |
| <nobr>general_pickup</nobr> | 80 | 80.00 | 58 | 58.00 | 64 | 64.00 | 67 ±11 | 67.33 ±11.00 |
| <nobr>hang_mugs</nobr> | 18 | 25.40 | 6 | 20.50 | 8 | 20.50 | 11 ±6 | 22.13 ±2.45 |
| <nobr>imitate_sorting_sequence</nobr> | 18 | 33.30 | 14 | 26.10 | 30 | 42.10 | 21 ±8 | 33.83 ±8.00 |
| <nobr>insert_key</nobr> | 46 | 54.10 | 44 | 52.40 | 42 | 50.70 | 44 ±2 | 52.40 ±1.70 |
| <nobr>insert_tubes</nobr> | 84 | 87.20 | 88 | 91.20 | 88 | 88.80 | 87 ±2 | 89.07 ±2.00 |
| <nobr>make_kong</nobr> | 78 | 78.00 | 88 | 88.00 | 90 | 90.00 | 85 ±6 | 85.33 ±6.00 |
| <nobr>make_toast</nobr> | 26 | 39.50 | 18 | 29.00 | 20 | 29.00 | 21 ±4 | 32.50 ±5.25 |
| <nobr>match_and_pick_from_conveyor</nobr> | 72 | 72.00 | 86 | 86.00 | 80 | 80.00 | 79 ±7 | 79.33 ±7.00 |
| <nobr>organize_table</nobr> | 14 | 63.00 | 6 | 52.50 | 8 | 56.50 | 9 ±4 | 57.33 ±5.25 |
| <nobr>pack_objects_into_box</nobr> | 20 | 27.70 | 14 | 22.10 | 16 | 21.50 | 17 ±3 | 23.77 ±3.10 |
| <nobr>pick_from_conveyor_by_image</nobr> | 68 | 68.00 | 34 | 34.00 | 48 | 48.00 | 50 ±17 | 50.00 ±17.00 |
| <nobr>play_Xylophone</nobr> | 94 | 94.00 | 86 | 86.00 | 84 | 84.00 | 88 ±5 | 88.00 ±5.00 |
| <nobr>play_stacking_toy</nobr> | 92 | 96.80 | 88 | 94.00 | 80 | 90.20 | 87 ±6 | 93.67 ±3.30 |
| <nobr>play_tic_tac_toe</nobr> | 98 | 99.00 | 100 | 100.00 | 100 | 100.00 | 99 ±1 | 99.67 ±0.50 |
| <nobr>plug_in_charger</nobr> | 76 | 76.00 | 68 | 68.00 | 76 | 76.00 | 73 ±4 | 73.33 ±4.00 |
| <nobr>pour_balls_into_vase</nobr> | 94 | 94.00 | 86 | 86.00 | 82 | 82.00 | 87 ±6 | 87.33 ±6.00 |
| <nobr>pour_by_language</nobr> | 86 | 86.00 | 82 | 82.00 | 90 | 90.00 | 86 ±4 | 86.00 ±4.00 |
| <nobr>pour_liquid_into_cup</nobr> | 96 | 96.00 | 96 | 96.00 | 96 | 96.00 | 96 ±0 | 96.00 ±0.00 |
| <nobr>press_by_number</nobr> | 96 | 96.00 | 100 | 100.00 | 96 | 96.00 | 97 ±2 | 97.33 ±2.00 |
| <nobr>push_T</nobr> | 96 | 96.00 | 90 | 90.00 | 82 | 82.00 | 89 ±7 | 89.33 ±7.00 |
| <nobr>put_bottles_into_dustbin</nobr> | 86 | 90.80 | 94 | 96.40 | 88 | 92.20 | 89 ±4 | 93.13 ±2.80 |
| <nobr>solve_equation</nobr> | 72 | 72.00 | 70 | 70.00 | 66 | 66.00 | 69 ±3 | 69.33 ±3.00 |
| <nobr>sort_nesting_dolls_by_size</nobr> | 46 | 46.00 | 64 | 64.00 | 42 | 42.00 | 51 ±11 | 50.67 ±11.00 |
| <nobr>stack_blocks</nobr> | 94 | 94.60 | 98 | 98.00 | 94 | 94.30 | 95 ±2 | 95.63 ±1.85 |
| <nobr>stack_blocks_by_language</nobr> | 94 | 94.40 | 78 | 79.20 | 88 | 89.20 | 87 ±8 | 87.60 ±7.60 |
| <nobr>stack_bowls</nobr> | 80 | 82.40 | 70 | 73.00 | 72 | 75.60 | 74 ±5 | 77.00 ±4.70 |
| <nobr>store_laptop_and_headphones</nobr> | 18 | 29.20 | 10 | 17.60 | 12 | 20.00 | 13 ±4 | 22.27 ±5.80 |
| <nobr>store_tools_in_toolbox</nobr> | 38 | 75.00 | 18 | 69.00 | 26 | 71.00 | 27 ±10 | 71.67 ±3.00 |
| <nobr>swap_T</nobr> | 92 | 92.00 | 96 | 96.00 | 94 | 94.00 | 94 ±2 | 94.00 ±2.00 |
| <nobr>swap_blocks</nobr> | 38 | 38.00 | 56 | 56.00 | 48 | 48.00 | 47 ±9 | 47.33 ±9.00 |
| <nobr>sweep_blocks</nobr> | 0 | 0.00 | 0 | 0.00 | 0 | 0.00 | 0 ±0 | 0.00 ±0.00 |
| **Avg (per seed)** | 65.9 | 71.48 | 63.3 | 69.09 | 63.2 | 69.00 | 64 ±1 | 69.86 ±1.24 |
