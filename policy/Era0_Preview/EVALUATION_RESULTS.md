# RoboDojo Evaluation Summary

## Overview (by capability dimension)

Progress † marks incomplete evaluation (126 = 42 tasks × 3 seeds). Cells show `mean ±std` across seeds; scores reflect completed cells only. Policies are sorted by overall Avg SR.

| Policy | Progress | Generalization SR (%) | Generalization Score | Precision SR (%) | Precision Score | Long-Horizon SR (%) | Long-Horizon Score | Memory SR (%) | Memory Score | Open SR (%) | Open Score | Avg SR (%) | Avg Score |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Era0_Preview | 97.6% † (123/126) | 41.82 ±1.07 | 47.73 ±1.01 | 33.75 ±2.47 | 40.97 ±2.35 | 38.17 ±0.82 | 48.95 ±0.91 | 54.67 ±1.19 | 56.27 ±1.06 | 19.67 ±1.30 | 21.07 ±1.19 | 37.61 ±0.88 | 43.00 ±0.68 |

## Generalization: Standard vs Random (25 episodes each)

Each cell shows `mean ±std` across seeds, averaged over the 12 Generalization tasks with completed data for that half. Δ = Random − Standard (mean SR). Sorted by Standard SR.

| Policy | Standard SR (%) | Standard Score | Random SR (%) | Random Score | Δ SR | Δ Score |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Era0_Preview | 46.22 ±1.13 | 52.33 ±0.73 | 35.39 ±2.66 | 41.16 ±2.36 | -10.83 | -11.17 |

### Per-task Standard vs Random

Additional view only: each of the 12 Generalization tasks is listed as the 25-episode standard split and its `_random` sibling. Columns match the per-policy all-task tables. Existing merged 50-episode scores are unchanged.

#### Era0_Preview

| Task | Seed 0 SR (%) | Seed 0 Score | Seed 1 SR (%) | Seed 1 Score | Seed 2 SR (%) | Seed 2 Score | Avg SR (%) | Avg Score |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| <nobr>stack_bowls</nobr> | 96 | 96.60 | 96 | 96.60 | 80 | 81.80 | 91 ±8 | 91.67 ±7.40 |
| <nobr>stack_bowls_random</nobr> | 80 | 82.40 | 76 | 78.40 | 80 | 81.80 | 79 ±2 | 80.87 ±2.00 |
| <nobr>push_T</nobr> | 0 | 0.00 | 0 | 0.00 | 4 | 4.00 | 1 ±2 | 1.33 ±2.00 |
| <nobr>push_T_random</nobr> | 0 | 0.00 | 8 | 8.00 | 0 | 0.00 | 3 ±4 | 2.67 ±4.00 |
| <nobr>pack_objects_into_box</nobr> | 20 | 42.00 | 32 | 57.00 | 24 | 44.60 | 25 ±6 | 47.87 ±7.50 |
| <nobr>pack_objects_into_box_random</nobr> | 20 | 41.40 | 16 | 37.00 | 12 | 35.40 | 16 ±4 | 37.93 ±3.00 |
| <nobr>fold_clothes</nobr> | 80 | 82.40 | 76 | 79.20 | 96 | 96.00 | 84 ±10 | 85.87 ±8.40 |
| <nobr>fold_clothes_random</nobr> | 80 | 82.40 | 84 | 86.40 | 88 | 88.00 | 84 ±4 | 85.60 ±2.80 |
| <nobr>hang_mugs</nobr> | 28 | 37.20 | 24 | 38.40 | 40 | 52.20 | 31 ±8 | 42.60 ±7.50 |
| <nobr>hang_mugs_random</nobr> | 4 | 13.60 | 8 | 17.60 | 4 | 15.80 | 5 ±2 | 15.67 ±2.00 |
| <nobr>sweep_blocks</nobr> | 20 | 20.00 | 8 | 8.00 | 16 | 16.00 | 15 ±6 | 14.67 ±6.00 |
| <nobr>sweep_blocks_random</nobr> | 0 | 0.00 | 0 | 0.00 | 4 | 4.00 | 1 ±2 | 1.33 ±2.00 |
| <nobr>pour_liquid_into_cup</nobr> | 84 | 84.00 | 84 | 84.00 | 88 | 88.00 | 85 ±2 | 85.33 ±2.00 |
| <nobr>pour_liquid_into_cup_random</nobr> | 80 | 80.00 | 84 | 84.00 | 44 | 44.00 | 69 ±20 | 69.33 ±20.00 |
| <nobr>make_toast</nobr> | 32 | 39.00 | 20 | 27.00 | 20 | 26.00 | 24 ±6 | 30.67 ±6.50 |
| <nobr>make_toast_random</nobr> |  |  |  |  |  |  |  |  |
| <nobr>arrange_largest_number</nobr> | 28 | 37.40 | 16 | 27.80 | 24 | 34.60 | 23 ±6 | 33.27 ±4.80 |
| <nobr>arrange_largest_number_random</nobr> | 20 | 28.60 | 24 | 28.80 | 20 | 26.60 | 21 ±2 | 28.00 ±1.10 |
| <nobr>sort_nesting_dolls_by_size</nobr> | 56 | 56.00 | 64 | 64.00 | 48 | 48.00 | 56 ±8 | 56.00 ±8.00 |
| <nobr>sort_nesting_dolls_by_size_random</nobr> | 44 | 44.00 | 40 | 40.00 | 36 | 36.00 | 40 ±4 | 40.00 ±4.00 |
| <nobr>store_laptop_and_headphones</nobr> | 36 | 54.40 | 44 | 59.20 | 44 | 59.20 | 41 ±4 | 57.60 ±2.40 |
| <nobr>store_laptop_and_headphones_random</nobr> | 40 | 50.40 | 28 | 43.20 | 16 | 34.40 | 28 ±12 | 42.67 ±8.00 |
| <nobr>stack_blocks</nobr> | 88 | 89.20 | 72 | 75.60 | 76 | 78.40 | 79 ±8 | 81.07 ±6.80 |
| <nobr>stack_blocks_random</nobr> | 40 | 46.60 | 44 | 49.40 | 44 | 50.00 | 43 ±2 | 48.67 ±1.70 |
| **Avg Standard** | 47.3 | 53.18 | 44.7 | 51.40 | 46.7 | 52.40 | 46 ±1 | 52.33 ±0.89 |
| **Avg Random** | 37.1 | 42.67 | 37.5 | 42.98 | 31.6 | 37.82 | 35 ±3 | 41.16 ±2.58 |

## Era0_Preview

| Task | Seed 0 SR (%) | Seed 0 Score | Seed 1 SR (%) | Seed 1 Score | Seed 2 SR (%) | Seed 2 Score | Avg SR (%) | Avg Score |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| <nobr>align_blocks</nobr> | 62 | 62.00 | 50 | 50.00 | 60 | 60.00 | 57 ±6 | 57.33 ±6.00 |
| <nobr>arrange_largest_number</nobr> | 24 | 33.00 | 20 | 28.30 | 22 | 30.60 | 22 ±2 | 30.63 ±2.35 |
| <nobr>build_tower</nobr> | 78 | 80.80 | 72 | 75.80 | 84 | 85.80 | 78 ±6 | 80.80 ±5.00 |
| <nobr>classify_objects</nobr> | 80 | 86.00 | 78 | 83.50 | 86 | 89.60 | 81 ±4 | 86.37 ±3.05 |
| <nobr>classify_objects_by_language</nobr> | 2 | 8.40 | 4 | 10.40 | 0 | 5.60 | 2 ±2 | 8.13 ±2.40 |
| <nobr>cover_blocks</nobr> | 100 | 100.00 | 100 | 100.00 | 96 | 96.60 | 99 ±2 | 98.87 ±1.70 |
| <nobr>deposit_coin</nobr> | 12 | 18.80 | 16 | 21.20 | 16 | 20.00 | 15 ±2 | 20.00 ±1.20 |
| <nobr>fasten_screws</nobr> | 18 | 44.40 | 16 | 42.40 | 24 | 51.80 | 19 ±4 | 46.20 ±4.70 |
| <nobr>fill_egg_holder</nobr> | 0 | 7.70 | 0 | 6.20 | 0 | 8.30 | 0 ±0 | 7.40 ±1.05 |
| <nobr>fill_pen_holder</nobr> | 26 | 45.10 | 34 | 52.30 | 40 | 59.60 | 33 ±7 | 52.33 ±7.25 |
| <nobr>fold_clothes</nobr> | 80 | 82.40 | 80 | 82.80 | 92 | 92.00 | 84 ±6 | 85.73 ±4.80 |
| <nobr>general_pickup</nobr> | 40 | 40.00 | 28 | 28.00 | 30 | 30.00 | 33 ±6 | 32.67 ±6.00 |
| <nobr>hang_mugs</nobr> | 16 | 25.40 | 16 | 28.00 | 22 | 34.00 | 18 ±3 | 29.13 ±4.30 |
| <nobr>imitate_sorting_sequence</nobr> | 44 | 52.60 | 30 | 40.80 | 42 | 50.80 | 39 ±7 | 48.07 ±5.90 |
| <nobr>insert_key</nobr> | 0 | 15.00 | 0 | 14.40 | 0 | 14.40 | 0 ±0 | 14.60 ±0.30 |
| <nobr>insert_tubes</nobr> | 72 | 81.20 | 62 | 70.40 | 70 | 76.80 | 68 ±5 | 76.13 ±5.40 |
| <nobr>make_kong</nobr> | 68 | 68.00 | 68 | 68.00 | 64 | 64.00 | 67 ±2 | 66.67 ±2.00 |
| <nobr>make_toast</nobr> |  |  |  |  |  |  |  |  |
| <nobr>match_and_pick_from_conveyor</nobr> | 80 | 80.00 | 82 | 82.00 | 86 | 86.00 | 83 ±3 | 82.67 ±3.00 |
| <nobr>organize_table</nobr> | 32 | 64.00 | 22 | 69.50 | 24 | 67.00 | 26 ±5 | 66.83 ±2.75 |
| <nobr>pack_objects_into_box</nobr> | 20 | 41.70 | 24 | 47.00 | 18 | 40.00 | 21 ±3 | 42.90 ±3.50 |
| <nobr>pick_from_conveyor_by_image</nobr> | 0 | 0.00 | 0 | 0.00 | 0 | 0.00 | 0 ±0 | 0.00 ±0.00 |
| <nobr>play_Xylophone</nobr> | 24 | 24.00 | 22 | 22.00 | 26 | 26.00 | 24 ±2 | 24.00 ±2.00 |
| <nobr>play_stacking_toy</nobr> | 0 | 0.00 | 0 | 0.00 | 0 | 0.00 | 0 ±0 | 0.00 ±0.00 |
| <nobr>play_tic_tac_toe</nobr> | 0 | 14.10 | 0 | 12.80 | 2 | 14.00 | 1 ±1 | 13.63 ±0.65 |
| <nobr>plug_in_charger</nobr> | 16 | 16.00 | 2 | 2.00 | 6 | 6.00 | 8 ±7 | 8.00 ±7.00 |
| <nobr>pour_balls_into_vase</nobr> | 54 | 54.00 | 54 | 54.00 | 66 | 66.00 | 58 ±6 | 58.00 ±6.00 |
| <nobr>pour_by_language</nobr> | 14 | 14.00 | 16 | 16.00 | 14 | 14.00 | 15 ±1 | 14.67 ±1.00 |
| <nobr>pour_liquid_into_cup</nobr> | 82 | 82.00 | 84 | 84.00 | 66 | 66.00 | 77 ±9 | 77.33 ±9.00 |
| <nobr>press_by_number</nobr> | 94 | 94.00 | 90 | 90.00 | 88 | 88.00 | 91 ±3 | 90.67 ±3.00 |
| <nobr>push_T</nobr> | 0 | 0.00 | 4 | 4.00 | 2 | 2.00 | 2 ±2 | 2.00 ±2.00 |
| <nobr>put_bottles_into_dustbin</nobr> | 98 | 98.80 | 96 | 97.60 | 98 | 98.80 | 97 ±1 | 98.40 ±0.60 |
| <nobr>solve_equation</nobr> | 40 | 40.00 | 40 | 40.00 | 38 | 38.00 | 39 ±1 | 39.33 ±1.00 |
| <nobr>sort_nesting_dolls_by_size</nobr> | 50 | 50.00 | 52 | 52.00 | 42 | 42.00 | 48 ±5 | 48.00 ±5.00 |
| <nobr>stack_blocks</nobr> | 64 | 67.90 | 58 | 62.50 | 60 | 64.20 | 61 ±3 | 64.87 ±2.70 |
| <nobr>stack_blocks_by_language</nobr> | 14 | 17.60 | 12 | 17.60 | 8 | 14.00 | 11 ±3 | 16.40 ±1.80 |
| <nobr>stack_bowls</nobr> | 88 | 89.50 | 86 | 87.50 | 80 | 81.80 | 85 ±4 | 86.27 ±3.85 |
| <nobr>store_laptop_and_headphones</nobr> | 38 | 52.40 | 36 | 51.20 | 30 | 46.80 | 35 ±4 | 50.13 ±2.80 |
| <nobr>store_tools_in_toolbox</nobr> | 0 | 0.00 | 0 | 0.00 | 0 | 0.00 | 0 ±0 | 0.00 ±0.00 |
| <nobr>swap_T</nobr> | 0 | 0.00 | 0 | 0.00 | 0 | 0.00 | 0 ±0 | 0.00 ±0.00 |
| <nobr>swap_blocks</nobr> | 20 | 20.00 | 20 | 20.00 | 12 | 12.00 | 17 ±4 | 17.33 ±4.00 |
| <nobr>sweep_blocks</nobr> | 10 | 10.00 | 4 | 4.00 | 10 | 10.00 | 8 ±3 | 8.00 ±3.00 |
| **Avg (per seed)** | 38.0 | 43.43 | 36.0 | 41.91 | 37.2 | 42.74 | 37 ±1 | 42.70 ±0.76 |

