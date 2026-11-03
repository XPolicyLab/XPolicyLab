# RoboDojo Evaluation Summary

## Overview (by capability dimension)

Progress † marks incomplete evaluation (126 = 42 tasks × 3 seeds). Cells show `mean ±std` across seeds; scores reflect completed cells only. Policies are sorted by overall Avg SR.

| Policy | Progress | Generalization SR (%) | Generalization Score | Precision SR (%) | Precision Score | Long-Horizon SR (%) | Long-Horizon Score | Memory SR (%) | Memory Score | Open SR (%) | Open Score | Avg SR (%) | Avg Score |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| go3_agent_preview | 100% | 38.50 ±1.21 | 42.11 ±1.15 | 76.83 ±1.12 | 79.92 ±1.03 | 55.58 ±1.45 | 70.52 ±0.20 | 67.44 ±2.18 | 68.10 ±2.14 | 55.92 ±0.77 | 63.61 ±0.83 | 58.86 ±0.19 | 64.85 ±0.39 |

## Generalization: Standard vs Random (25 episodes each)

Each cell shows `mean ±std` across seeds, averaged over the 12 Generalization tasks with completed data for that half. Δ = Random − Standard (mean SR). Sorted by Standard SR.

| Policy | Standard SR (%) | Standard Score | Random SR (%) | Random Score | Δ SR | Δ Score |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| go3_agent_preview | 47.89 ±0.42 | 51.66 ±0.30 | 29.11 ±2.28 | 32.56 ±2.41 | -18.78 | -19.10 |

### Per-task Standard vs Random

Additional view only: each of the 12 Generalization tasks is listed as the 25-episode standard split and its `_random` sibling. Columns match the per-policy all-task tables. Existing merged 50-episode scores are unchanged.

#### go3_agent_preview

| Task | Seed 0 SR (%) | Seed 0 Score | Seed 1 SR (%) | Seed 1 Score | Seed 2 SR (%) | Seed 2 Score | Avg SR (%) | Avg Score |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| <nobr>stack_bowls</nobr> | 92 | 93.20 | 96 | 96.60 | 92 | 93.20 | 93 ±2 | 94.33 ±1.70 |
| <nobr>stack_bowls_random</nobr> | 44 | 47.00 | 56 | 60.20 | 72 | 74.40 | 57 ±14 | 60.53 ±13.70 |
| <nobr>push_T</nobr> | 84 | 84.00 | 88 | 88.00 | 88 | 88.00 | 87 ±2 | 86.67 ±2.00 |
| <nobr>push_T_random</nobr> | 84 | 84.00 | 64 | 64.00 | 80 | 80.00 | 76 ±10 | 76.00 ±10.00 |
| <nobr>pack_objects_into_box</nobr> | 4 | 11.60 | 0 | 9.80 | 4 | 11.20 | 3 ±2 | 10.87 ±0.90 |
| <nobr>pack_objects_into_box_random</nobr> | 0 | 3.80 | 0 | 3.60 | 0 | 5.80 | 0 ±0 | 4.40 ±1.10 |
| <nobr>fold_clothes</nobr> | 68 | 74.40 | 64 | 71.20 | 72 | 76.00 | 68 ±4 | 73.87 ±2.40 |
| <nobr>fold_clothes_random</nobr> | 32 | 44.00 | 16 | 27.20 | 32 | 44.80 | 27 ±8 | 38.67 ±8.80 |
| <nobr>hang_mugs</nobr> | 4 | 5.20 | 0 | 7.00 | 0 | 3.00 | 1 ±2 | 5.07 ±2.00 |
| <nobr>hang_mugs_random</nobr> | 0 | 2.40 | 0 | 1.20 | 0 | 2.40 | 0 ±0 | 2.00 ±0.60 |
| <nobr>sweep_blocks</nobr> | 0 | 0.00 | 0 | 0.00 | 0 | 0.00 | 0 ±0 | 0.00 ±0.00 |
| <nobr>sweep_blocks_random</nobr> | 0 | 0.00 | 0 | 0.00 | 0 | 0.00 | 0 ±0 | 0.00 ±0.00 |
| <nobr>pour_liquid_into_cup</nobr> | 72 | 72.00 | 76 | 76.00 | 80 | 80.00 | 76 ±4 | 76.00 ±4.00 |
| <nobr>pour_liquid_into_cup_random</nobr> | 48 | 48.00 | 40 | 40.00 | 60 | 60.00 | 49 ±10 | 49.33 ±10.00 |
| <nobr>make_toast</nobr> | 0 | 7.00 | 0 | 8.00 | 4 | 9.00 | 1 ±2 | 8.00 ±1.00 |
| <nobr>make_toast_random</nobr> | 4 | 8.00 | 0 | 6.00 | 0 | 4.00 | 1 ±2 | 6.00 ±2.00 |
| <nobr>arrange_largest_number</nobr> | 64 | 75.00 | 80 | 85.20 | 52 | 64.00 | 65 ±14 | 74.73 ±10.60 |
| <nobr>arrange_largest_number_random</nobr> | 32 | 41.40 | 48 | 55.20 | 52 | 60.00 | 44 ±10 | 52.20 ±9.30 |
| <nobr>sort_nesting_dolls_by_size</nobr> | 76 | 76.00 | 76 | 76.00 | 84 | 84.00 | 79 ±4 | 78.67 ±4.00 |
| <nobr>sort_nesting_dolls_by_size_random</nobr> | 0 | 0.00 | 8 | 8.00 | 0 | 0.00 | 3 ±4 | 2.67 ±4.00 |
| <nobr>store_laptop_and_headphones</nobr> | 4 | 18.40 | 0 | 7.20 | 0 | 9.60 | 1 ±2 | 11.73 ±5.60 |
| <nobr>store_laptop_and_headphones_random</nobr> | 4 | 9.60 | 4 | 8.00 | 4 | 10.40 | 4 ±0 | 9.33 ±1.20 |
| <nobr>stack_blocks</nobr> | 100 | 100.00 | 100 | 100.00 | 100 | 100.00 | 100 ±0 | 100.00 ±0.00 |
| <nobr>stack_blocks_random</nobr> | 80 | 82.40 | 96 | 96.60 | 88 | 89.80 | 88 ±8 | 89.60 ±7.10 |
| **Avg Standard** | 47.3 | 51.40 | 48.3 | 52.08 | 48.0 | 51.50 | 48 ±0 | 51.66 ±0.34 |
| **Avg Random** | 27.3 | 30.88 | 27.7 | 30.83 | 32.3 | 35.97 | 29 ±3 | 32.56 ±2.57 |

## go3_agent_preview

| Task | Seed 0 SR (%) | Seed 0 Score | Seed 1 SR (%) | Seed 1 Score | Seed 2 SR (%) | Seed 2 Score | Avg SR (%) | Avg Score |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| <nobr>align_blocks</nobr> | 96 | 96.00 | 88 | 88.00 | 92 | 92.00 | 92 ±4 | 92.00 ±4.00 |
| <nobr>arrange_largest_number</nobr> | 48 | 58.20 | 64 | 70.20 | 52 | 62.00 | 55 ±8 | 63.47 ±6.00 |
| <nobr>build_tower</nobr> | 98 | 98.20 | 100 | 100.00 | 98 | 98.20 | 99 ±1 | 98.80 ±0.90 |
| <nobr>classify_objects</nobr> | 56 | 68.80 | 70 | 78.50 | 70 | 79.20 | 65 ±7 | 75.50 ±5.20 |
| <nobr>classify_objects_by_language</nobr> | 46 | 60.80 | 66 | 75.40 | 54 | 67.20 | 55 ±10 | 67.80 ±7.30 |
| <nobr>cover_blocks</nobr> | 100 | 100.00 | 100 | 100.00 | 100 | 100.00 | 100 ±0 | 100.00 ±0.00 |
| <nobr>deposit_coin</nobr> | 48 | 54.00 | 62 | 67.20 | 54 | 60.00 | 55 ±7 | 60.40 ±6.60 |
| <nobr>fasten_screws</nobr> | 100 | 100.00 | 98 | 98.40 | 98 | 99.00 | 99 ±1 | 99.13 ±0.80 |
| <nobr>fill_egg_holder</nobr> | 0 | 1.00 | 0 | 0.20 | 0 | 0.80 | 0 ±0 | 0.67 ±0.40 |
| <nobr>fill_pen_holder</nobr> | 22 | 43.80 | 24 | 41.30 | 34 | 51.90 | 27 ±6 | 45.67 ±5.30 |
| <nobr>fold_clothes</nobr> | 50 | 59.20 | 40 | 49.20 | 52 | 60.40 | 47 ±6 | 56.27 ±5.60 |
| <nobr>general_pickup</nobr> | 24 | 24.00 | 18 | 18.00 | 28 | 28.00 | 23 ±5 | 23.33 ±5.00 |
| <nobr>hang_mugs</nobr> | 2 | 3.80 | 0 | 4.10 | 0 | 2.70 | 1 ±1 | 3.53 ±0.70 |
| <nobr>imitate_sorting_sequence</nobr> | 26 | 29.60 | 28 | 31.80 | 32 | 36.40 | 29 ±3 | 32.60 ±3.40 |
| <nobr>insert_key</nobr> | 20 | 32.00 | 18 | 30.00 | 36 | 45.60 | 25 ±9 | 35.87 ±7.80 |
| <nobr>insert_tubes</nobr> | 82 | 88.80 | 78 | 86.40 | 82 | 88.40 | 81 ±2 | 87.87 ±1.20 |
| <nobr>make_kong</nobr> | 100 | 100.00 | 100 | 100.00 | 100 | 100.00 | 100 ±0 | 100.00 ±0.00 |
| <nobr>make_toast</nobr> | 2 | 7.50 | 0 | 7.00 | 2 | 6.50 | 1 ±1 | 7.00 ±0.50 |
| <nobr>match_and_pick_from_conveyor</nobr> | 38 | 38.00 | 42 | 42.00 | 28 | 28.00 | 36 ±7 | 36.00 ±7.00 |
| <nobr>organize_table</nobr> | 28 | 76.00 | 18 | 74.50 | 8 | 72.50 | 18 ±10 | 74.33 ±1.75 |
| <nobr>pack_objects_into_box</nobr> | 2 | 7.70 | 0 | 6.70 | 2 | 8.50 | 1 ±1 | 7.63 ±0.90 |
| <nobr>pick_from_conveyor_by_image</nobr> | 0 | 0.00 | 0 | 0.00 | 0 | 0.00 | 0 ±0 | 0.00 ±0.00 |
| <nobr>play_Xylophone</nobr> | 98 | 98.00 | 98 | 98.00 | 100 | 100.00 | 99 ±1 | 98.67 ±1.00 |
| <nobr>play_stacking_toy</nobr> | 66 | 83.40 | 48 | 78.60 | 46 | 76.60 | 53 ±10 | 79.53 ±3.40 |
| <nobr>play_tic_tac_toe</nobr> | 100 | 100.00 | 100 | 100.00 | 100 | 100.00 | 100 ±0 | 100.00 ±0.00 |
| <nobr>plug_in_charger</nobr> | 70 | 70.00 | 74 | 74.00 | 72 | 72.00 | 72 ±2 | 72.00 ±2.00 |
| <nobr>pour_balls_into_vase</nobr> | 88 | 88.00 | 86 | 86.00 | 86 | 86.00 | 87 ±1 | 86.67 ±1.00 |
| <nobr>pour_by_language</nobr> | 100 | 100.00 | 100 | 100.00 | 100 | 100.00 | 100 ±0 | 100.00 ±0.00 |
| <nobr>pour_liquid_into_cup</nobr> | 60 | 60.00 | 58 | 58.00 | 70 | 70.00 | 63 ±6 | 62.67 ±6.00 |
| <nobr>press_by_number</nobr> | 100 | 100.00 | 100 | 100.00 | 100 | 100.00 | 100 ±0 | 100.00 ±0.00 |
| <nobr>push_T</nobr> | 84 | 84.00 | 76 | 76.00 | 84 | 84.00 | 81 ±4 | 81.33 ±4.00 |
| <nobr>put_bottles_into_dustbin</nobr> | 88 | 92.80 | 82 | 88.90 | 74 | 83.80 | 81 ±7 | 88.50 ±4.50 |
| <nobr>solve_equation</nobr> | 80 | 80.00 | 80 | 80.00 | 86 | 86.00 | 82 ±3 | 82.00 ±3.00 |
| <nobr>sort_nesting_dolls_by_size</nobr> | 38 | 38.00 | 42 | 42.00 | 42 | 42.00 | 41 ±2 | 40.67 ±2.00 |
| <nobr>stack_blocks</nobr> | 90 | 91.20 | 98 | 98.30 | 94 | 94.90 | 94 ±4 | 94.80 ±3.55 |
| <nobr>stack_blocks_by_language</nobr> | 84 | 86.40 | 86 | 87.20 | 90 | 91.20 | 87 ±3 | 88.27 ±2.40 |
| <nobr>stack_bowls</nobr> | 68 | 70.10 | 76 | 78.40 | 82 | 83.80 | 75 ±7 | 77.43 ±6.85 |
| <nobr>store_laptop_and_headphones</nobr> | 4 | 14.00 | 2 | 7.60 | 2 | 10.00 | 3 ±1 | 10.53 ±3.20 |
| <nobr>store_tools_in_toolbox</nobr> | 12 | 55.00 | 6 | 58.00 | 6 | 53.50 | 8 ±3 | 55.50 ±2.25 |
| <nobr>swap_T</nobr> | 98 | 98.00 | 98 | 98.00 | 98 | 98.00 | 98 ±0 | 98.00 ±0.00 |
| <nobr>swap_blocks</nobr> | 44 | 44.00 | 52 | 52.00 | 30 | 30.00 | 42 ±11 | 42.00 ±11.00 |
| <nobr>sweep_blocks</nobr> | 0 | 0.00 | 0 | 0.00 | 0 | 0.00 | 0 ±0 | 0.00 ±0.00 |
| **Avg (per seed)** | 56.2 | 61.91 | 56.6 | 62.62 | 56.8 | 63.07 | 57 ±0 | 62.53 ±0.58 |

