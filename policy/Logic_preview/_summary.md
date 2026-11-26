# RoboDojo Evaluation Summary

## Overview (by capability dimension)

Progress † marks incomplete evaluation (126 = 42 tasks × 3 seeds). Cells show `mean ±std` across seeds; scores reflect completed cells only. Policies are sorted by overall Avg SR.

| Policy | Progress | Generalization SR (%) | Generalization Score | Precision SR (%) | Precision Score | Long-Horizon SR (%) | Long-Horizon Score | Memory SR (%) | Memory Score | Open SR (%) | Open Score | Avg SR (%) | Avg Score |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| logic_preview | 100% | 34.56 ±2.44 | 41.02 ±2.36 | 36.00 ±1.27 | 44.12 ±1.25 | 34.00 ±1.54 | 50.63 ±2.02 | 57.33 ±1.96 | 57.66 ±1.93 | 21.83 ±0.92 | 22.55 ±0.87 | 36.74 ±0.57 | 43.19 ±0.71 |

## Generalization: Standard vs Random (25 episodes each)

Each cell shows `mean ±std` across seeds, averaged over the 12 Generalization tasks with completed data for that half. Δ = Random − Standard (mean SR). Sorted by Standard SR.

| Policy | Standard SR (%) | Standard Score | Random SR (%) | Random Score | Δ SR | Δ Score |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| logic_preview | 47.56 ±2.86 | 53.67 ±2.46 | 21.56 ±3.01 | 28.38 ±3.10 | -26.00 | -25.28 |

### Per-task Standard vs Random

Additional view only: each of the 12 Generalization tasks is listed as the 25-episode standard split and its `_random` sibling. Columns match the per-policy all-task tables. Existing merged 50-episode scores are unchanged.

#### logic_preview

| Task | Seed 0 SR (%) | Seed 0 Score | Seed 1 SR (%) | Seed 1 Score | Seed 2 SR (%) | Seed 2 Score | Avg SR (%) | Avg Score |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| <nobr>stack_bowls</nobr> | 100 | 100.00 | 100 | 100.00 | 92 | 93.20 | 97 ±4 | 97.73 ±3.40 |
| <nobr>stack_bowls_random</nobr> | 44 | 52.40 | 84 | 85.80 | 68 | 72.80 | 65 ±20 | 70.33 ±16.70 |
| <nobr>push_T</nobr> | 4 | 4.00 | 0 | 0.00 | 0 | 0.00 | 1 ±2 | 1.33 ±2.00 |
| <nobr>push_T_random</nobr> | 0 | 0.00 | 0 | 0.00 | 0 | 0.00 | 0 ±0 | 0.00 ±0.00 |
| <nobr>pack_objects_into_box</nobr> | 0 | 32.00 | 16 | 38.60 | 16 | 38.40 | 11 ±8 | 36.33 ±3.30 |
| <nobr>pack_objects_into_box_random</nobr> | 4 | 25.80 | 12 | 34.00 | 0 | 23.00 | 5 ±6 | 27.60 ±5.50 |
| <nobr>fold_clothes</nobr> | 64 | 68.00 | 88 | 89.60 | 80 | 82.40 | 77 ±12 | 80.00 ±10.80 |
| <nobr>fold_clothes_random</nobr> | 76 | 76.80 | 92 | 92.00 | 88 | 88.80 | 85 ±8 | 85.87 ±7.60 |
| <nobr>hang_mugs</nobr> | 40 | 59.00 | 64 | 73.40 | 40 | 61.00 | 48 ±12 | 64.47 ±7.20 |
| <nobr>hang_mugs_random</nobr> | 0 | 7.40 | 0 | 12.40 | 4 | 15.40 | 1 ±2 | 11.73 ±4.00 |
| <nobr>sweep_blocks</nobr> | 28 | 28.00 | 32 | 32.00 | 12 | 12.00 | 24 ±10 | 24.00 ±10.00 |
| <nobr>sweep_blocks_random</nobr> | 4 | 4.00 | 0 | 0.00 | 0 | 0.00 | 1 ±2 | 1.33 ±2.00 |
| <nobr>pour_liquid_into_cup</nobr> | 64 | 64.00 | 68 | 68.00 | 72 | 72.00 | 68 ±4 | 68.00 ±4.00 |
| <nobr>pour_liquid_into_cup_random</nobr> | 32 | 32.00 | 52 | 52.00 | 44 | 44.00 | 43 ±10 | 42.67 ±10.00 |
| <nobr>make_toast</nobr> | 64 | 69.00 | 76 | 84.00 | 72 | 78.00 | 71 ±6 | 77.00 ±7.50 |
| <nobr>make_toast_random</nobr> | 36 | 48.00 | 28 | 47.00 | 36 | 42.00 | 33 ±4 | 45.67 ±3.00 |
| <nobr>arrange_largest_number</nobr> | 0 | 8.20 | 4 | 11.80 | 4 | 11.00 | 3 ±2 | 10.33 ±1.80 |
| <nobr>arrange_largest_number_random</nobr> | 0 | 3.00 | 0 | 5.20 | 0 | 4.40 | 0 ±0 | 4.20 ±1.10 |
| <nobr>sort_nesting_dolls_by_size</nobr> | 36 | 36.00 | 28 | 28.00 | 20 | 20.00 | 28 ±8 | 28.00 ±8.00 |
| <nobr>sort_nesting_dolls_by_size_random</nobr> | 8 | 8.00 | 12 | 12.00 | 0 | 0.00 | 7 ±6 | 6.67 ±6.00 |
| <nobr>store_laptop_and_headphones</nobr> | 80 | 84.00 | 48 | 64.80 | 40 | 56.80 | 56 ±20 | 68.53 ±13.60 |
| <nobr>store_laptop_and_headphones_random</nobr> | 8 | 24.80 | 20 | 32.80 | 24 | 40.00 | 17 ±8 | 32.53 ±7.60 |
| <nobr>stack_blocks</nobr> | 92 | 93.20 | 88 | 89.20 | 80 | 82.40 | 87 ±6 | 88.27 ±5.40 |
| <nobr>stack_blocks_random</nobr> | 0 | 12.00 | 0 | 12.00 | 0 | 12.00 | 0 ±0 | 12.00 ±0.00 |
| **Avg Standard** | 47.7 | 53.78 | 51.0 | 56.62 | 44.0 | 50.60 | 48 ±4 | 53.67 ±3.01 |
| **Avg Random** | 17.7 | 24.52 | 25.0 | 32.10 | 22.0 | 28.53 | 22 ±4 | 28.38 ±3.79 |

## logic_preview

| Task | Seed 0 SR (%) | Seed 0 Score | Seed 1 SR (%) | Seed 1 Score | Seed 2 SR (%) | Seed 2 Score | Avg SR (%) | Avg Score |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| <nobr>align_blocks</nobr> | 96 | 96.00 | 86 | 86.00 | 92 | 92.00 | 91 ±5 | 91.33 ±5.00 |
| <nobr>arrange_largest_number</nobr> | 0 | 5.60 | 2 | 8.50 | 2 | 7.70 | 1 ±1 | 7.27 ±1.45 |
| <nobr>build_tower</nobr> | 82 | 86.40 | 78 | 83.00 | 88 | 89.80 | 83 ±5 | 86.40 ±3.40 |
| <nobr>classify_objects</nobr> | 52 | 60.90 | 42 | 53.70 | 50 | 60.10 | 48 ±5 | 58.23 ±3.60 |
| <nobr>classify_objects_by_language</nobr> | 0 | 3.20 | 2 | 6.60 | 2 | 4.80 | 1 ±1 | 4.87 ±1.70 |
| <nobr>cover_blocks</nobr> | 100 | 100.00 | 100 | 100.00 | 100 | 100.00 | 100 ±0 | 100.00 ±0.00 |
| <nobr>deposit_coin</nobr> | 32 | 36.40 | 30 | 36.00 | 32 | 38.80 | 31 ±1 | 37.07 ±1.40 |
| <nobr>fasten_screws</nobr> | 40 | 65.20 | 40 | 66.40 | 50 | 75.00 | 43 ±5 | 68.87 ±4.90 |
| <nobr>fill_egg_holder</nobr> | 0 | 9.00 | 0 | 12.80 | 4 | 15.60 | 1 ±2 | 12.47 ±3.30 |
| <nobr>fill_pen_holder</nobr> | 40 | 58.40 | 26 | 49.70 | 42 | 60.60 | 36 ±8 | 56.23 ±5.45 |
| <nobr>fold_clothes</nobr> | 70 | 72.40 | 90 | 90.80 | 84 | 85.60 | 81 ±10 | 82.93 ±9.20 |
| <nobr>general_pickup</nobr> | 78 | 78.00 | 62 | 62.00 | 68 | 68.00 | 69 ±8 | 69.33 ±8.00 |
| <nobr>hang_mugs</nobr> | 20 | 33.20 | 32 | 42.90 | 22 | 38.20 | 25 ±6 | 38.10 ±4.85 |
| <nobr>imitate_sorting_sequence</nobr> | 6 | 8.30 | 0 | 1.80 | 0 | 1.70 | 2 ±3 | 3.93 ±3.30 |
| <nobr>insert_key</nobr> | 0 | 15.00 | 0 | 15.00 | 0 | 15.00 | 0 ±0 | 15.00 ±0.00 |
| <nobr>insert_tubes</nobr> | 50 | 64.40 | 58 | 71.20 | 46 | 63.20 | 51 ±6 | 66.27 ±4.00 |
| <nobr>make_kong</nobr> | 68 | 68.00 | 64 | 64.00 | 66 | 66.00 | 66 ±2 | 66.00 ±2.00 |
| <nobr>make_toast</nobr> | 50 | 58.50 | 52 | 65.50 | 54 | 60.00 | 52 ±2 | 61.33 ±3.50 |
| <nobr>match_and_pick_from_conveyor</nobr> | 84 | 84.00 | 92 | 92.00 | 90 | 90.00 | 89 ±4 | 88.67 ±4.00 |
| <nobr>organize_table</nobr> | 14 | 63.50 | 16 | 57.00 | 12 | 60.00 | 14 ±2 | 60.17 ±3.25 |
| <nobr>pack_objects_into_box</nobr> | 2 | 28.90 | 14 | 36.30 | 8 | 30.70 | 8 ±6 | 31.97 ±3.70 |
| <nobr>pick_from_conveyor_by_image</nobr> | 0 | 0.00 | 0 | 0.00 | 0 | 0.00 | 0 ±0 | 0.00 ±0.00 |
| <nobr>play_Xylophone</nobr> | 0 | 0.00 | 0 | 0.00 | 0 | 0.00 | 0 ±0 | 0.00 ±0.00 |
| <nobr>play_stacking_toy</nobr> | 4 | 18.80 | 2 | 17.40 | 10 | 24.60 | 5 ±4 | 20.27 ±3.60 |
| <nobr>play_tic_tac_toe</nobr> | 6 | 32.60 | 12 | 35.10 | 6 | 41.10 | 8 ±3 | 36.27 ±4.25 |
| <nobr>plug_in_charger</nobr> | 28 | 28.00 | 12 | 12.00 | 14 | 14.00 | 18 ±8 | 18.00 ±8.00 |
| <nobr>pour_balls_into_vase</nobr> | 60 | 60.00 | 56 | 56.00 | 68 | 68.00 | 61 ±6 | 61.33 ±6.00 |
| <nobr>pour_by_language</nobr> | 0 | 0.00 | 0 | 0.00 | 0 | 0.00 | 0 ±0 | 0.00 ±0.00 |
| <nobr>pour_liquid_into_cup</nobr> | 48 | 48.00 | 60 | 60.00 | 58 | 58.00 | 55 ±6 | 55.33 ±6.00 |
| <nobr>press_by_number</nobr> | 100 | 100.00 | 100 | 100.00 | 100 | 100.00 | 100 ±0 | 100.00 ±0.00 |
| <nobr>push_T</nobr> | 2 | 2.00 | 0 | 0.00 | 0 | 0.00 | 1 ±1 | 0.67 ±1.00 |
| <nobr>put_bottles_into_dustbin</nobr> | 90 | 93.40 | 94 | 95.80 | 96 | 97.00 | 93 ±3 | 95.40 ±1.80 |
| <nobr>solve_equation</nobr> | 0 | 0.00 | 4 | 4.00 | 0 | 0.00 | 1 ±2 | 1.33 ±2.00 |
| <nobr>sort_nesting_dolls_by_size</nobr> | 22 | 22.00 | 20 | 20.00 | 10 | 10.00 | 17 ±6 | 17.33 ±6.00 |
| <nobr>stack_blocks</nobr> | 46 | 52.60 | 44 | 50.60 | 40 | 47.20 | 43 ±3 | 50.13 ±2.70 |
| <nobr>stack_blocks_by_language</nobr> | 10 | 10.80 | 12 | 12.40 | 12 | 14.80 | 11 ±1 | 12.67 ±2.00 |
| <nobr>stack_bowls</nobr> | 72 | 76.20 | 92 | 92.90 | 80 | 83.00 | 81 ±10 | 84.03 ±8.35 |
| <nobr>store_laptop_and_headphones</nobr> | 44 | 54.40 | 34 | 48.80 | 32 | 48.40 | 37 ±6 | 50.53 ±3.00 |
| <nobr>store_tools_in_toolbox</nobr> | 0 | 1.00 | 0 | 1.00 | 0 | 0.50 | 0 ±0 | 0.83 ±0.25 |
| <nobr>swap_T</nobr> | 18 | 18.00 | 18 | 18.00 | 28 | 28.00 | 21 ±5 | 21.33 ±5.00 |
| <nobr>swap_blocks</nobr> | 24 | 24.00 | 30 | 30.00 | 42 | 42.00 | 32 ±9 | 32.00 ±9.00 |
| <nobr>sweep_blocks</nobr> | 16 | 16.00 | 16 | 16.00 | 6 | 6.00 | 13 ±5 | 12.67 ±5.00 |
| **Avg (per seed)** | 35.1 | 41.74 | 35.5 | 42.17 | 36.0 | 42.99 | 36 ±0 | 42.30 ±0.62 |

