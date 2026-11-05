# RoboDojo Evaluation Summary

## Overview (by capability dimension)

Progress † marks incomplete evaluation (126 = 42 tasks × 3 seeds). Cells show `mean ±std` across seeds; scores reflect completed cells only. Policies are sorted by overall Avg SR.

| Policy | Progress | Generalization SR (%) | Generalization Score | Precision SR (%) | Precision Score | Long-Horizon SR (%) | Long-Horizon Score | Memory SR (%) | Memory Score | Open SR (%) | Open Score | Avg SR (%) | Avg Score |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Hy_XVLA_RoboDojo | 96.8% † (122/126) | 30.88 ±1.19 | 36.41 ±1.61 | 19.67 ±0.42 | 26.50 ±0.31 | 37.17 ±0.96 | 48.93 ±0.35 | 42.22 ±1.23 | 42.66 ±1.25 | 11.83 ±0.31 | 12.70 ±0.20 | 28.35 ±0.45 | 33.44 ±0.42 |

## Generalization: Standard vs Random (25 episodes each)

Each cell shows `mean ±std` across seeds, averaged over the 12 Generalization tasks with completed data for that half. Δ = Random − Standard (mean SR). Sorted by Standard SR.

| Policy | Standard SR (%) | Standard Score | Random SR (%) | Random Score | Δ SR | Δ Score |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Hy_XVLA_RoboDojo | 36.44 ±2.74 | 42.70 ±2.57 | 22.75 ±1.34 | 28.06 ±1.96 | -13.69 | -14.64 |

### Per-task Standard vs Random

Additional view only: each of the 12 Generalization tasks is listed as the 25-episode standard split and its `_random` sibling. Columns match the per-policy all-task tables. Existing merged 50-episode scores are unchanged.

#### Hy_XVLA_RoboDojo

| Task | Seed 0 SR (%) | Seed 0 Score | Seed 1 SR (%) | Seed 1 Score | Seed 2 SR (%) | Seed 2 Score | Avg SR (%) | Avg Score |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| <nobr>stack_bowls</nobr> | 84 | 85.20 | 76 | 77.20 | 72 | 75.00 | 77 ±6 | 79.13 ±5.10 |
| <nobr>stack_bowls_random</nobr> | 52 | 56.80 | 72 | 74.40 | 72 | 75.00 | 65 ±10 | 68.73 ±9.10 |
| <nobr>push_T</nobr> | 0 | 0.00 | 0 | 0.00 | 0 | 0.00 | 0 ±0 | 0.00 ±0.00 |
| <nobr>push_T_random</nobr> | 0 | 0.00 | 0 | 0.00 | 0 | 0.00 | 0 ±0 | 0.00 ±0.00 |
| <nobr>pack_objects_into_box</nobr> | 16 | 42.00 | 28 | 48.80 | 32 | 51.60 | 25 ±8 | 47.47 ±4.80 |
| <nobr>pack_objects_into_box_random</nobr> | 12 | 31.80 | 8 | 32.40 | 8 | 27.40 | 9 ±2 | 30.53 ±2.50 |
| <nobr>fold_clothes</nobr> | 84 | 86.40 | 72 | 75.20 | 68 | 71.20 | 75 ±8 | 77.60 ±7.60 |
| <nobr>fold_clothes_random</nobr> | 76 | 79.20 | 76 | 78.40 | 76 | 79.20 | 76 ±0 | 78.93 ±0.40 |
| <nobr>hang_mugs</nobr> | 12 | 22.60 | 16 | 32.00 | 20 | 35.40 | 16 ±4 | 30.00 ±6.40 |
| <nobr>hang_mugs_random</nobr> | 0 | 7.60 | 4 | 15.00 | 0 | 9.60 | 1 ±2 | 10.73 ±3.70 |
| <nobr>sweep_blocks</nobr> | 20 | 20.00 | 16 | 16.00 | 8 | 8.00 | 15 ±6 | 14.67 ±6.00 |
| <nobr>sweep_blocks_random</nobr> | 8 | 8.00 | 0 | 0.00 | 4 | 4.00 | 4 ±4 | 4.00 ±4.00 |
| <nobr>pour_liquid_into_cup</nobr> | 80 | 80.00 | 76 | 76.00 | 68 | 68.00 | 75 ±6 | 74.67 ±6.00 |
| <nobr>pour_liquid_into_cup_random</nobr> | 64 | 64.00 | 64 | 64.00 | 48 | 48.00 | 59 ±8 | 58.67 ±8.00 |
| <nobr>make_toast</nobr> | 12 | 23.00 | 4 | 15.00 | 8 | 16.00 | 8 ±4 | 18.00 ±4.00 |
| <nobr>make_toast_random</nobr> |  |  |  |  |  |  |  |  |
| <nobr>arrange_largest_number</nobr> | 36 | 39.80 | 4 | 9.80 | 12 | 20.80 | 17 ±16 | 23.47 ±15.00 |
| <nobr>arrange_largest_number_random</nobr> | 0 | 2.00 | 0 | 1.00 | 0 | 0.60 | 0 ±0 | 1.20 ±0.70 |
| <nobr>sort_nesting_dolls_by_size</nobr> | 28 | 28.00 | 28 | 28.00 | 20 | 20.00 | 25 ±4 | 25.33 ±4.00 |
| <nobr>sort_nesting_dolls_by_size_random</nobr> | 0 | 0.00 | 0 | 0.00 | 0 | 0.00 | 0 ±0 | 0.00 ±0.00 |
| <nobr>store_laptop_and_headphones</nobr> | 28 | 43.20 | 24 | 44.00 | 36 | 47.20 | 29 ±6 | 44.80 ±2.00 |
| <nobr>store_laptop_and_headphones_random</nobr> | 0 | 12.00 | 16 | 40.00 |  |  |  |  |
| <nobr>stack_blocks</nobr> | 80 | 80.60 | 88 | 89.20 | 56 | 62.00 | 75 ±16 | 77.27 ±13.60 |
| <nobr>stack_blocks_random</nobr> | 20 | 27.20 | 28 | 33.40 | 20 | 27.80 | 23 ±4 | 29.47 ±3.10 |
| **Avg Standard** | 40.0 | 45.90 | 36.0 | 42.60 | 33.3 | 39.60 | 36 ±3 | 42.70 ±3.15 |
| **Avg Random** | 21.1 | 26.24 | 24.4 | 30.78 | 22.8 | 27.16 | 23 ±2 | 28.06 ±2.27 |

## Hy_XVLA_RoboDojo

| Task | Seed 0 SR (%) | Seed 0 Score | Seed 1 SR (%) | Seed 1 Score | Seed 2 SR (%) | Seed 2 Score | Avg SR (%) | Avg Score |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| <nobr>align_blocks</nobr> | 58 | 58.00 | 56 | 56.00 | 62 | 62.00 | 59 ±3 | 58.67 ±3.00 |
| <nobr>arrange_largest_number</nobr> | 18 | 20.90 | 2 | 5.40 | 6 | 10.70 | 9 ±8 | 12.33 ±7.75 |
| <nobr>build_tower</nobr> | 60 | 64.00 | 54 | 59.20 | 54 | 60.60 | 56 ±3 | 61.27 ±2.40 |
| <nobr>classify_objects</nobr> | 66 | 73.00 | 74 | 80.80 | 56 | 65.30 | 65 ±9 | 73.03 ±7.75 |
| <nobr>classify_objects_by_language</nobr> | 0 | 4.40 | 0 | 2.80 | 0 | 3.00 | 0 ±0 | 3.40 ±0.80 |
| <nobr>cover_blocks</nobr> | 94 | 95.20 | 96 | 96.90 | 96 | 96.60 | 95 ±1 | 96.23 ±0.85 |
| <nobr>deposit_coin</nobr> | 4 | 9.20 | 4 | 8.80 | 10 | 14.80 | 6 ±3 | 10.93 ±3.00 |
| <nobr>fasten_screws</nobr> | 4 | 24.80 | 12 | 32.00 | 12 | 22.60 | 9 ±4 | 26.47 ±4.70 |
| <nobr>fill_egg_holder</nobr> | 0 | 7.70 | 0 | 5.00 | 0 | 8.80 | 0 ±0 | 7.17 ±1.90 |
| <nobr>fill_pen_holder</nobr> | 28 | 47.90 | 20 | 43.50 | 42 | 60.60 | 30 ±11 | 50.67 ±8.55 |
| <nobr>fold_clothes</nobr> | 80 | 82.80 | 74 | 76.80 | 72 | 75.20 | 75 ±4 | 78.27 ±3.80 |
| <nobr>general_pickup</nobr> | 14 | 14.00 | 22 | 22.00 | 16 | 16.00 | 17 ±4 | 17.33 ±4.00 |
| <nobr>hang_mugs</nobr> | 6 | 15.10 | 10 | 23.50 | 10 | 22.50 | 9 ±2 | 20.37 ±4.20 |
| <nobr>imitate_sorting_sequence</nobr> | 4 | 6.40 | 2 | 3.30 | 0 | 1.40 | 2 ±2 | 3.70 ±2.50 |
| <nobr>insert_key</nobr> | 0 | 9.00 | 0 | 7.80 | 0 | 10.80 | 0 ±0 | 9.20 ±1.50 |
| <nobr>insert_tubes</nobr> | 20 | 38.00 | 16 | 32.80 | 22 | 41.60 | 19 ±3 | 37.47 ±4.40 |
| <nobr>make_kong</nobr> | 82 | 82.00 | 70 | 70.00 | 72 | 72.00 | 75 ±6 | 74.67 ±6.00 |
| <nobr>make_toast</nobr> |  |  |  |  |  |  |  |  |
| <nobr>match_and_pick_from_conveyor</nobr> | 74 | 74.00 | 78 | 78.00 | 66 | 66.00 | 73 ±6 | 72.67 ±6.00 |
| <nobr>organize_table</nobr> | 32 | 68.00 | 28 | 74.50 | 20 | 62.50 | 27 ±6 | 68.33 ±6.00 |
| <nobr>pack_objects_into_box</nobr> | 14 | 36.90 | 18 | 40.60 | 20 | 39.50 | 17 ±3 | 39.00 ±1.85 |
| <nobr>pick_from_conveyor_by_image</nobr> | 0 | 0.00 | 0 | 0.00 | 0 | 0.00 | 0 ±0 | 0.00 ±0.00 |
| <nobr>play_Xylophone</nobr> | 6 | 6.00 | 10 | 10.00 | 10 | 10.00 | 9 ±2 | 8.67 ±2.00 |
| <nobr>play_stacking_toy</nobr> | 0 | 0.00 | 0 | 0.00 | 0 | 0.00 | 0 ±0 | 0.00 ±0.00 |
| <nobr>play_tic_tac_toe</nobr> | 2 | 14.50 | 4 | 15.40 | 4 | 28.00 | 3 ±1 | 19.30 ±6.75 |
| <nobr>plug_in_charger</nobr> | 10 | 10.00 | 4 | 4.00 | 6 | 6.00 | 7 ±3 | 6.67 ±3.00 |
| <nobr>pour_balls_into_vase</nobr> | 52 | 52.00 | 54 | 54.00 | 48 | 48.00 | 51 ±3 | 51.33 ±3.00 |
| <nobr>pour_by_language</nobr> | 0 | 0.00 | 2 | 2.00 | 2 | 2.00 | 1 ±1 | 1.33 ±1.00 |
| <nobr>pour_liquid_into_cup</nobr> | 72 | 72.00 | 70 | 70.00 | 58 | 58.00 | 67 ±7 | 66.67 ±7.00 |
| <nobr>press_by_number</nobr> | 74 | 74.00 | 80 | 80.00 | 80 | 80.00 | 78 ±3 | 78.00 ±3.00 |
| <nobr>push_T</nobr> | 0 | 0.00 | 0 | 0.00 | 0 | 0.00 | 0 ±0 | 0.00 ±0.00 |
| <nobr>put_bottles_into_dustbin</nobr> | 98 | 98.50 | 98 | 98.80 | 96 | 97.60 | 97 ±1 | 98.30 ±0.60 |
| <nobr>solve_equation</nobr> | 16 | 16.00 | 14 | 14.00 | 18 | 18.00 | 16 ±2 | 16.00 ±2.00 |
| <nobr>sort_nesting_dolls_by_size</nobr> | 14 | 14.00 | 14 | 14.00 | 10 | 10.00 | 13 ±2 | 12.67 ±2.00 |
| <nobr>stack_blocks</nobr> | 50 | 53.90 | 58 | 61.30 | 38 | 44.90 | 49 ±10 | 53.37 ±8.20 |
| <nobr>stack_blocks_by_language</nobr> | 4 | 7.60 | 0 | 3.60 | 0 | 2.80 | 1 ±2 | 4.67 ±2.40 |
| <nobr>stack_bowls</nobr> | 68 | 71.00 | 74 | 75.80 | 72 | 75.00 | 71 ±3 | 73.93 ±2.40 |
| <nobr>store_laptop_and_headphones</nobr> | 14 | 27.60 | 20 | 42.00 |  |  |  |  |
| <nobr>store_tools_in_toolbox</nobr> | 0 | 0.50 | 0 | 0.00 | 0 | 0.00 | 0 ±0 | 0.17 ±0.25 |
| <nobr>swap_T</nobr> | 0 | 0.00 | 0 | 0.00 | 0 | 0.00 | 0 ±0 | 0.00 ±0.00 |
| <nobr>swap_blocks</nobr> | 8 | 8.00 | 6 | 6.00 | 2 | 2.00 | 5 ±3 | 5.33 ±3.00 |
| <nobr>sweep_blocks</nobr> | 14 | 14.00 | 8 | 8.00 | 6 | 6.00 | 9 ±4 | 9.33 ±4.00 |
| **Avg (per seed)** | 28.3 | 33.44 | 28.1 | 33.62 | 27.1 | 32.52 | 28 ±1 | 33.19 ±0.55 |

