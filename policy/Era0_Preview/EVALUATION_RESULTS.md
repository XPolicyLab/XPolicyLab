# RoboDojo Evaluation Summary

## Overview (by capability dimension)

Progress † marks incomplete evaluation (126 = 42 tasks × 3 seeds). Cells show `mean ±std` across seeds; scores reflect completed cells only. Policies are sorted by overall Avg SR.

| Policy | Progress | Generalization SR (%) | Generalization Score | Precision SR (%) | Precision Score | Long-Horizon SR (%) | Long-Horizon Score | Memory SR (%) | Memory Score | Open SR (%) | Open Score | Avg SR (%) | Avg Score |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Era0_Preview | 96.8% † (122/126) | 42.78 ±0.60 | 48.84 ±0.17 | 36.00 ±2.07 | 42.91 ±1.72 | 39.75 ±0.35 | 51.25 ±0.43 | 58.78 ±1.59 | 60.10 ±1.37 | 18.58 ±2.66 | 19.58 ±2.65 | 39.18 ±1.13 | 44.54 ±0.91 |

## Generalization: Standard vs Random (25 episodes each)

Each cell shows `mean ±std` across seeds, averaged over the 12 Generalization tasks with completed data for that half. Δ = Random − Standard (mean SR). Sorted by Standard SR.

| Policy | Standard SR (%) | Standard Score | Random SR (%) | Random Score | Δ SR | Δ Score |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Era0_Preview | 46.21 ±1.59 | 52.78 ±0.51 | 37.82 ±1.07 | 44.15 ±1.01 | -8.39 | -8.64 |

### Per-task Standard vs Random

Additional view only: each of the 12 Generalization tasks is listed as the 25-episode standard split and its `_random` sibling. Columns match the per-policy all-task tables. Existing merged 50-episode scores are unchanged.

#### Era0_Preview

| Task | Seed 0 SR (%) | Seed 0 Score | Seed 1 SR (%) | Seed 1 Score | Seed 2 SR (%) | Seed 2 Score | Avg SR (%) | Avg Score |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| <nobr>stack_bowls</nobr> | 96 | 96.60 | 100 | 100.00 | 76 | 77.80 | 91 ±12 | 91.47 ±11.10 |
| <nobr>stack_bowls_random</nobr> | 64 | 68.20 | 76 | 79.00 | 84 | 85.80 | 75 ±10 | 77.67 ±8.80 |
| <nobr>push_T</nobr> | 4 | 4.00 | 0 | 0.00 | 4 | 4.00 | 3 ±2 | 2.67 ±2.00 |
| <nobr>push_T_random</nobr> | 4 | 4.00 | 4 | 4.00 | 0 | 0.00 | 3 ±2 | 2.67 ±2.00 |
| <nobr>pack_objects_into_box</nobr> | 12 | 39.60 | 12 | 41.80 | 16 | 42.20 | 13 ±2 | 41.20 ±1.30 |
| <nobr>pack_objects_into_box_random</nobr> | 28 | 44.80 | 12 | 32.20 | 12 | 32.80 | 17 ±8 | 36.60 ±6.30 |
| <nobr>fold_clothes</nobr> | 88 | 89.60 | 84 | 85.60 | 88 | 89.60 | 87 ±2 | 88.27 ±2.00 |
| <nobr>fold_clothes_random</nobr> | 80 | 81.60 | 80 | 83.20 | 76 | 78.40 | 79 ±2 | 81.07 ±2.40 |
| <nobr>hang_mugs</nobr> | 28 | 40.00 | 32 | 47.40 | 32 | 46.40 | 31 ±2 | 44.60 ±3.70 |
| <nobr>hang_mugs_random</nobr> | 0 | 9.80 | 12 | 21.00 | 4 | 15.20 | 5 ±6 | 15.33 ±5.60 |
| <nobr>sweep_blocks</nobr> | 0 | 0.00 | 12 | 12.00 | 16 | 16.00 | 9 ±8 | 9.33 ±8.00 |
| <nobr>sweep_blocks_random</nobr> | 8 | 8.00 | 4 | 4.00 | 0 | 0.00 | 4 ±4 | 4.00 ±4.00 |
| <nobr>pour_liquid_into_cup</nobr> | 80 | 80.00 | 80 | 80.00 | 80 | 80.00 | 80 ±0 | 80.00 ±0.00 |
| <nobr>pour_liquid_into_cup_random</nobr> | 80 | 80.00 | 76 | 76.00 | 68 | 68.00 | 75 ±6 | 74.67 ±6.00 |
| <nobr>make_toast</nobr> | 32 | 41.00 | 12 | 25.00 | 36 | 39.00 | 27 ±12 | 35.00 ±8.00 |
| <nobr>make_toast_random</nobr> |  |  |  |  |  |  |  |  |
| <nobr>arrange_largest_number</nobr> | 36 | 48.00 | 8 | 25.00 | 28 | 36.80 | 24 ±14 | 36.60 ±11.50 |
| <nobr>arrange_largest_number_random</nobr> | 12 | 22.60 | 20 | 26.80 | 24 | 29.60 | 19 ±6 | 26.33 ±3.50 |
| <nobr>sort_nesting_dolls_by_size</nobr> | 56 | 56.00 | 68 | 68.00 | 52 | 52.00 | 59 ±8 | 58.67 ±8.00 |
| <nobr>sort_nesting_dolls_by_size_random</nobr> | 56 | 56.00 | 36 | 36.00 | 52 | 52.00 | 48 ±10 | 48.00 ±10.00 |
| <nobr>store_laptop_and_headphones</nobr> |  |  | 44 | 61.60 | 52 | 65.60 |  |  |
| <nobr>store_laptop_and_headphones_random</nobr> | 48 | 72.80 | 48 | 60.80 | 36 | 64.00 | 44 ±6 | 65.87 ±6.00 |
| <nobr>stack_blocks</nobr> | 92 | 92.00 | 76 | 79.00 | 84 | 85.20 | 84 ±8 | 85.40 ±6.50 |
| <nobr>stack_blocks_random</nobr> | 48 | 53.40 | 52 | 56.20 | 44 | 50.60 | 48 ±4 | 53.40 ±2.80 |
| **Avg Standard** | 47.6 | 53.35 | 44.0 | 52.12 | 47.0 | 52.88 | 46 ±2 | 52.78 ±0.61 |
| **Avg Random** | 38.9 | 45.56 | 38.2 | 43.56 | 36.4 | 43.31 | 38 ±1 | 44.15 ±1.13 |

## Era0_Preview

| Task | Seed 0 SR (%) | Seed 0 Score | Seed 1 SR (%) | Seed 1 Score | Seed 2 SR (%) | Seed 2 Score | Avg SR (%) | Avg Score |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| <nobr>align_blocks</nobr> | 56 | 56.00 | 36 | 36.00 | 58 | 58.00 | 50 ±11 | 50.00 ±11.00 |
| <nobr>arrange_largest_number</nobr> | 24 | 35.30 | 14 | 25.90 | 26 | 33.20 | 21 ±6 | 31.47 ±4.70 |
| <nobr>build_tower</nobr> | 84 | 87.80 | 84 | 85.80 | 82 | 84.80 | 83 ±1 | 86.13 ±1.50 |
| <nobr>classify_objects</nobr> | 78 | 84.80 | 80 | 86.50 | 88 | 90.50 | 82 ±5 | 87.27 ±2.85 |
| <nobr>classify_objects_by_language</nobr> | 8 | 10.20 | 6 | 8.40 | 0 | 2.60 | 5 ±4 | 7.07 ±3.80 |
| <nobr>cover_blocks</nobr> | 100 | 100.00 | 100 | 100.00 | 100 | 100.00 | 100 ±0 | 100.00 ±0.00 |
| <nobr>deposit_coin</nobr> | 16 | 21.60 | 6 | 13.20 | 8 | 14.40 | 10 ±5 | 16.40 ±4.20 |
| <nobr>fasten_screws</nobr> | 28 | 51.40 | 28 | 55.20 | 36 | 56.00 | 31 ±4 | 54.20 ±2.30 |
| <nobr>fill_egg_holder</nobr> | 2 | 10.20 | 0 | 9.50 | 0 | 11.70 | 1 ±1 | 10.47 ±1.10 |
| <nobr>fill_pen_holder</nobr> | 24 | 43.50 | 36 | 55.60 | 38 | 58.60 | 33 ±7 | 52.57 ±7.55 |
| <nobr>fold_clothes</nobr> | 84 | 85.60 | 82 | 84.40 | 82 | 84.00 | 83 ±1 | 84.67 ±0.80 |
| <nobr>general_pickup</nobr> | 34 | 34.00 | 26 | 26.00 | 40 | 40.00 | 33 ±7 | 33.33 ±7.00 |
| <nobr>hang_mugs</nobr> | 14 | 24.90 | 22 | 34.20 | 18 | 30.80 | 18 ±4 | 29.97 ±4.65 |
| <nobr>imitate_sorting_sequence</nobr> | 42 | 51.00 | 36 | 44.80 | 54 | 60.00 | 44 ±9 | 51.93 ±7.60 |
| <nobr>insert_key</nobr> | 0 | 14.40 | 0 | 14.10 | 0 | 14.40 | 0 ±0 | 14.30 ±0.15 |
| <nobr>insert_tubes</nobr> | 80 | 87.60 | 62 | 72.80 | 74 | 80.40 | 72 ±9 | 80.27 ±7.40 |
| <nobr>make_kong</nobr> | 84 | 84.00 | 68 | 68.00 | 72 | 72.00 | 75 ±8 | 74.67 ±8.00 |
| <nobr>make_toast</nobr> |  |  |  |  |  |  |  |  |
| <nobr>match_and_pick_from_conveyor</nobr> | 82 | 82.00 | 86 | 86.00 | 98 | 98.00 | 89 ±8 | 88.67 ±8.00 |
| <nobr>organize_table</nobr> | 30 | 69.00 | 30 | 73.00 | 26 | 67.00 | 29 ±2 | 69.67 ±3.00 |
| <nobr>pack_objects_into_box</nobr> | 20 | 42.20 | 12 | 37.00 | 14 | 37.50 | 15 ±4 | 38.90 ±2.60 |
| <nobr>pick_from_conveyor_by_image</nobr> | 0 | 0.00 | 0 | 0.00 | 0 | 0.00 | 0 ±0 | 0.00 ±0.00 |
| <nobr>play_Xylophone</nobr> | 22 | 22.00 | 18 | 18.00 | 18 | 18.00 | 19 ±2 | 19.33 ±2.00 |
| <nobr>play_stacking_toy</nobr> | 0 | 0.00 | 0 | 0.00 | 0 | 0.00 | 0 ±0 | 0.00 ±0.00 |
| <nobr>play_tic_tac_toe</nobr> | 0 | 16.20 | 2 | 16.70 | 2 | 17.50 | 1 ±1 | 16.80 ±0.65 |
| <nobr>plug_in_charger</nobr> | 8 | 8.00 | 6 | 6.00 | 8 | 8.00 | 7 ±1 | 7.33 ±1.00 |
| <nobr>pour_balls_into_vase</nobr> | 68 | 68.00 | 62 | 62.00 | 66 | 66.00 | 65 ±3 | 65.33 ±3.00 |
| <nobr>pour_by_language</nobr> | 30 | 30.00 | 20 | 20.00 | 10 | 10.00 | 20 ±10 | 20.00 ±10.00 |
| <nobr>pour_liquid_into_cup</nobr> | 80 | 80.00 | 78 | 78.00 | 74 | 74.00 | 77 ±3 | 77.33 ±3.00 |
| <nobr>press_by_number</nobr> | 94 | 94.00 | 98 | 98.00 | 94 | 94.00 | 95 ±2 | 95.33 ±2.00 |
| <nobr>push_T</nobr> | 4 | 4.00 | 2 | 2.00 | 2 | 2.00 | 3 ±1 | 2.67 ±1.00 |
| <nobr>put_bottles_into_dustbin</nobr> | 98 | 98.50 | 100 | 100.00 | 96 | 97.30 | 98 ±2 | 98.60 ±1.35 |
| <nobr>solve_equation</nobr> | 36 | 36.00 | 30 | 30.00 | 30 | 30.00 | 32 ±3 | 32.00 ±3.00 |
| <nobr>sort_nesting_dolls_by_size</nobr> | 56 | 56.00 | 52 | 52.00 | 52 | 52.00 | 53 ±2 | 53.33 ±2.00 |
| <nobr>stack_blocks</nobr> | 70 | 72.70 | 64 | 67.60 | 64 | 67.90 | 66 ±3 | 69.40 ±2.55 |
| <nobr>stack_blocks_by_language</nobr> | 12 | 17.20 | 6 | 11.20 | 8 | 14.40 | 9 ±3 | 14.27 ±3.00 |
| <nobr>stack_bowls</nobr> | 80 | 82.40 | 88 | 89.50 | 80 | 81.80 | 83 ±4 | 84.57 ±3.85 |
| <nobr>store_laptop_and_headphones</nobr> |  |  | 46 | 61.20 | 44 | 64.80 |  |  |
| <nobr>store_tools_in_toolbox</nobr> | 0 | 0.00 | 0 | 0.00 | 0 | 0.00 | 0 ±0 | 0.00 ±0.00 |
| <nobr>swap_T</nobr> | 0 | 0.00 | 0 | 0.00 | 0 | 0.00 | 0 ±0 | 0.00 ±0.00 |
| <nobr>swap_blocks</nobr> | 30 | 30.00 | 24 | 24.00 | 20 | 20.00 | 25 ±5 | 24.67 ±5.00 |
| <nobr>sweep_blocks</nobr> | 4 | 4.00 | 8 | 8.00 | 8 | 8.00 | 7 ±2 | 6.67 ±2.00 |
| **Avg (per seed)** | 39.5 | 44.86 | 37.0 | 42.94 | 38.8 | 44.38 | 38 ±1 | 44.06 ±0.96 |

