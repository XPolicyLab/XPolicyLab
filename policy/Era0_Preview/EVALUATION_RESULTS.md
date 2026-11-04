# Era0-Preview — Local Evaluation Results

Local reference results for seeds **0, 1 and 2**: **6,225/6,300** planned episodes completed, with **1,716** successes. These are not official leaderboard results. **Overview / Avg Score** is the primary aggregate metric.

## Overview (by capability dimension)

Cells show `mean ±std` across seeds and use every available episode (no minimum threshold). `Episodes` shows total successes/observed episodes; coverage is observed/planned episodes.

| Policy | Episode coverage | Episodes | Generalization SR (%) | Generalization Score | Precision SR (%) | Precision Score | Long-Horizon SR (%) | Long-Horizon Score | Memory SR (%) | Memory Score | Open SR (%) | Open Score | Avg SR (%) | Avg Score |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Era0-Preview | 98.81% | 1716/6225 | 28.89 ±1.29 | 34.99 ±1.31 | 19.67 ±0.42 | 26.50 ±0.31 | 37.17 ±0.96 | 48.93 ±0.35 | 42.22 ±1.23 | 42.66 ±1.25 | 11.83 ±0.31 | 12.70 ±0.20 | 27.96 ±0.47 | 33.16 ±0.35 |

## Generalization: Standard vs Random

Each cell shows `mean ±std` across seeds, averaged over the 12 Generalization tasks with completed data for that half. Δ = Random − Standard (mean SR). Sorted by Standard SR. Statistics use every available episode (no minimum threshold).

| Policy | Standard SR (%) | Standard Score | Random SR (%) | Random Score | Δ SR | Δ Score |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Era0-Preview | 36.44 ±2.74 | 42.70 ±2.57 | 21.85 ±1.78 | 27.68 ±2.24 | -14.60 | -15.02 |

## Era0-Preview

Per-seed columns: `SR (%)`, `Score`, `n` (successes/episodes actually observed for that seed).

| Task | Seed 0 SR (%) | Seed 0 Score | Seed 0 n | Seed 1 SR (%) | Seed 1 Score | Seed 1 n | Seed 2 SR (%) | Seed 2 Score | Seed 2 n | Avg SR (%) | Avg Score |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **Generalization** |  |  |  |  |  |  |  |  |  |  |  |
| <nobr>stack_bowls</nobr> | 68 | 71.00 | 34/50 | 74 | 75.80 | 37/50 | 72 | 75.00 | 36/50 | 71 ±2 | 73.93 ±2.10 |
| <nobr>stack_bowls (standard)</nobr> | 84 | 85.20 | 21/25 | 76 | 77.20 | 19/25 | 72 | 75.00 | 18/25 | 77 ±5 | 79.13 ±4.38 |
| <nobr>stack_bowls_random</nobr> | 52 | 56.80 | 13/25 | 72 | 74.40 | 18/25 | 72 | 75.00 | 18/25 | 65 ±9 | 68.73 ±8.44 |
| <nobr>push_T</nobr> | 0 | 0.00 | 0/50 | 0 | 0.00 | 0/50 | 0 | 0.00 | 0/50 | 0 ±0 | 0.00 ±0.00 |
| <nobr>push_T (standard)</nobr> | 0 | 0.00 | 0/25 | 0 | 0.00 | 0/25 | 0 | 0.00 | 0/25 | 0 ±0 | 0.00 ±0.00 |
| <nobr>push_T_random</nobr> | 0 | 0.00 | 0/25 | 0 | 0.00 | 0/25 | 0 | 0.00 | 0/25 | 0 ±0 | 0.00 ±0.00 |
| <nobr>pack_objects_into_box</nobr> | 14 | 36.90 | 7/50 | 18 | 40.60 | 9/50 | 20 | 39.50 | 10/50 | 17 ±2 | 39.00 ±1.55 |
| <nobr>pack_objects_into_box (standard)</nobr> | 16 | 42.00 | 4/25 | 28 | 48.80 | 7/25 | 32 | 51.60 | 8/25 | 25 ±7 | 47.47 ±4.03 |
| <nobr>pack_objects_into_box_random</nobr> | 12 | 31.80 | 3/25 | 8 | 32.40 | 2/25 | 8 | 27.40 | 2/25 | 9 ±2 | 30.53 ±2.23 |
| <nobr>fold_clothes</nobr> | 80 | 82.80 | 40/50 | 74 | 76.80 | 37/50 | 72 | 75.20 | 36/50 | 75 ±3 | 78.27 ±3.27 |
| <nobr>fold_clothes (standard)</nobr> | 84 | 86.40 | 21/25 | 72 | 75.20 | 18/25 | 68 | 71.20 | 17/25 | 75 ±7 | 77.60 ±6.43 |
| <nobr>fold_clothes_random</nobr> | 76 | 79.20 | 19/25 | 76 | 78.40 | 19/25 | 76 | 79.20 | 19/25 | 76 ±0 | 78.93 ±0.38 |
| <nobr>hang_mugs</nobr> | 6 | 15.10 | 3/50 | 10 | 23.50 | 5/50 | 10 | 22.50 | 5/50 | 9 ±2 | 20.37 ±3.75 |
| <nobr>hang_mugs (standard)</nobr> | 12 | 22.60 | 3/25 | 16 | 32.00 | 4/25 | 20 | 35.40 | 5/25 | 16 ±3 | 30.00 ±5.41 |
| <nobr>hang_mugs_random</nobr> | 0 | 7.60 | 0/25 | 4 | 15.00 | 1/25 | 0 | 9.60 | 0/25 | 1 ±2 | 10.73 ±3.13 |
| <nobr>sweep_blocks</nobr> | 14 | 14.00 | 7/50 | 8 | 8.00 | 4/50 | 6 | 6.00 | 3/50 | 9 ±3 | 9.33 ±3.40 |
| <nobr>sweep_blocks (standard)</nobr> | 20 | 20.00 | 5/25 | 16 | 16.00 | 4/25 | 8 | 8.00 | 2/25 | 15 ±5 | 14.67 ±4.99 |
| <nobr>sweep_blocks_random</nobr> | 8 | 8.00 | 2/25 | 0 | 0.00 | 0/25 | 4 | 4.00 | 1/25 | 4 ±3 | 4.00 ±3.27 |
| <nobr>pour_liquid_into_cup</nobr> | 72 | 72.00 | 36/50 | 70 | 70.00 | 35/50 | 58 | 58.00 | 29/50 | 67 ±6 | 66.67 ±6.18 |
| <nobr>pour_liquid_into_cup (standard)</nobr> | 80 | 80.00 | 20/25 | 76 | 76.00 | 19/25 | 68 | 68.00 | 17/25 | 75 ±5 | 74.67 ±4.99 |
| <nobr>pour_liquid_into_cup_random</nobr> | 64 | 64.00 | 16/25 | 64 | 64.00 | 16/25 | 48 | 48.00 | 12/25 | 59 ±8 | 58.67 ±7.54 |
| <nobr>make_toast</nobr> | 12 | 21.25 | 5/40 | 4 | 15.00 | 1/25 | 8 | 16.00 | 2/25 | 8 ±3 | 17.42 ±2.74 |
| <nobr>make_toast (standard)</nobr> | 12 | 23.00 | 3/25 | 4 | 15.00 | 1/25 | 8 | 16.00 | 2/25 | 8 ±3 | 18.00 ±3.56 |
| <nobr>make_toast_random</nobr> | 13 | 18.33 | 2/15 |  |  |  |  |  |  |  |  |
| <nobr>arrange_largest_number</nobr> | 18 | 20.90 | 9/50 | 2 | 5.40 | 1/50 | 6 | 10.70 | 3/50 | 9 ±7 | 12.33 ±6.43 |
| <nobr>arrange_largest_number (standard)</nobr> | 36 | 39.80 | 9/25 | 4 | 9.80 | 1/25 | 12 | 20.80 | 3/25 | 17 ±14 | 23.47 ±12.39 |
| <nobr>arrange_largest_number_random</nobr> | 0 | 2.00 | 0/25 | 0 | 1.00 | 0/25 | 0 | 0.60 | 0/25 | 0 ±0 | 1.20 ±0.59 |
| <nobr>sort_nesting_dolls_by_size</nobr> | 14 | 14.00 | 7/50 | 14 | 14.00 | 7/50 | 10 | 10.00 | 5/50 | 13 ±2 | 12.67 ±1.89 |
| <nobr>sort_nesting_dolls_by_size (standard)</nobr> | 28 | 28.00 | 7/25 | 28 | 28.00 | 7/25 | 20 | 20.00 | 5/25 | 25 ±4 | 25.33 ±3.77 |
| <nobr>sort_nesting_dolls_by_size_random</nobr> | 0 | 0.00 | 0/25 | 0 | 0.00 | 0/25 | 0 | 0.00 | 0/25 | 0 ±0 | 0.00 ±0.00 |
| <nobr>store_laptop_and_headphones</nobr> | 14 | 27.60 | 7/50 | 20 | 42.00 | 10/50 | 26 | 40.00 | 9/35 | 20 ±5 | 36.53 ±6.37 |
| <nobr>store_laptop_and_headphones (standard)</nobr> | 28 | 43.20 | 7/25 | 24 | 44.00 | 6/25 | 36 | 47.20 | 9/25 | 29 ±5 | 44.80 ±1.73 |
| <nobr>store_laptop_and_headphones_random</nobr> | 0 | 12.00 | 0/25 | 16 | 40.00 | 4/25 | 0 | 22.00 | 0/10 | 5 ±8 | 24.67 ±11.59 |
| <nobr>stack_blocks</nobr> | 50 | 53.90 | 25/50 | 58 | 61.30 | 29/50 | 38 | 44.90 | 19/50 | 49 ±8 | 53.37 ±6.71 |
| <nobr>stack_blocks (standard)</nobr> | 80 | 80.60 | 20/25 | 88 | 89.20 | 22/25 | 56 | 62.00 | 14/25 | 75 ±14 | 77.27 ±11.35 |
| <nobr>stack_blocks_random</nobr> | 20 | 27.20 | 5/25 | 28 | 33.40 | 7/25 | 20 | 27.80 | 5/25 | 23 ±4 | 29.47 ±2.79 |
| **Generalization Avg (per seed)** | 30.2 | 35.79 | 180/590 | 29.3 | 36.03 | 175/575 | 27.1 | 33.15 | 157/560 | 29 ±1 | 34.99 ±1.31 |
| **Precision** |  |  |  |  |  |  |  |  |  |  |  |
| <nobr>fasten_screws</nobr> | 4 | 24.80 | 2/50 | 12 | 32.00 | 6/50 | 12 | 22.60 | 6/50 | 9 ±4 | 26.47 ±4.01 |
| <nobr>plug_in_charger</nobr> | 10 | 10.00 | 5/50 | 4 | 4.00 | 2/50 | 6 | 6.00 | 3/50 | 7 ±2 | 6.67 ±2.49 |
| <nobr>insert_tubes</nobr> | 20 | 38.00 | 10/50 | 16 | 32.80 | 8/50 | 22 | 41.60 | 11/50 | 19 ±2 | 37.47 ±3.61 |
| <nobr>pour_balls_into_vase</nobr> | 52 | 52.00 | 26/50 | 54 | 54.00 | 27/50 | 48 | 48.00 | 24/50 | 51 ±2 | 51.33 ±2.49 |
| <nobr>play_Xylophone</nobr> | 6 | 6.00 | 3/50 | 10 | 10.00 | 5/50 | 10 | 10.00 | 5/50 | 9 ±2 | 8.67 ±1.89 |
| <nobr>deposit_coin</nobr> | 4 | 9.20 | 2/50 | 4 | 8.80 | 2/50 | 10 | 14.80 | 5/50 | 6 ±3 | 10.93 ±2.74 |
| <nobr>insert_key</nobr> | 0 | 9.00 | 0/50 | 0 | 7.80 | 0/50 | 0 | 10.80 | 0/50 | 0 ±0 | 9.20 ±1.23 |
| <nobr>build_tower</nobr> | 60 | 64.00 | 30/50 | 54 | 59.20 | 27/50 | 54 | 60.60 | 27/50 | 56 ±3 | 61.27 ±2.02 |
| **Precision Avg (per seed)** | 19.5 | 26.62 | 78/400 | 19.2 | 26.07 | 77/400 | 20.2 | 26.80 | 81/400 | 20 ±0 | 26.50 ±0.31 |
| **Long-Horizon** |  |  |  |  |  |  |  |  |  |  |  |
| <nobr>put_bottles_into_dustbin</nobr> | 98 | 98.50 | 49/50 | 98 | 98.80 | 49/50 | 96 | 97.60 | 48/50 | 97 ±1 | 98.30 ±0.51 |
| <nobr>fill_pen_holder</nobr> | 28 | 47.90 | 14/50 | 20 | 43.50 | 10/50 | 42 | 60.60 | 21/50 | 30 ±9 | 50.67 ±7.25 |
| <nobr>classify_objects</nobr> | 66 | 73.00 | 33/50 | 74 | 80.80 | 37/50 | 56 | 65.30 | 28/50 | 65 ±7 | 73.03 ±6.33 |
| <nobr>play_tic_tac_toe</nobr> | 2 | 14.50 | 1/50 | 4 | 15.40 | 2/50 | 4 | 28.00 | 2/50 | 3 ±1 | 19.30 ±6.16 |
| <nobr>fill_egg_holder</nobr> | 0 | 7.70 | 0/50 | 0 | 5.00 | 0/50 | 0 | 8.80 | 0/50 | 0 ±0 | 7.17 ±1.60 |
| <nobr>organize_table</nobr> | 32 | 68.00 | 16/50 | 28 | 74.50 | 14/50 | 20 | 62.50 | 10/50 | 27 ±5 | 68.33 ±4.90 |
| <nobr>make_kong</nobr> | 82 | 82.00 | 41/50 | 70 | 70.00 | 35/50 | 72 | 72.00 | 36/50 | 75 ±5 | 74.67 ±5.25 |
| <nobr>play_stacking_toy</nobr> | 0 | 0.00 | 0/50 | 0 | 0.00 | 0/50 | 0 | 0.00 | 0/50 | 0 ±0 | 0.00 ±0.00 |
| **Long-Horizon Avg (per seed)** | 38.5 | 48.95 | 154/400 | 36.8 | 48.50 | 147/400 | 36.2 | 49.35 | 145/400 | 37 ±1 | 48.93 ±0.35 |
| **Memory** |  |  |  |  |  |  |  |  |  |  |  |
| <nobr>cover_blocks</nobr> | 94 | 95.20 | 47/50 | 96 | 96.90 | 48/50 | 96 | 96.60 | 48/50 | 95 ±1 | 96.23 ±0.74 |
| <nobr>match_and_pick_from_conveyor</nobr> | 74 | 74.00 | 37/50 | 78 | 78.00 | 39/50 | 66 | 66.00 | 33/50 | 73 ±5 | 72.67 ±4.99 |
| <nobr>swap_blocks</nobr> | 8 | 8.00 | 4/50 | 6 | 6.00 | 3/50 | 2 | 2.00 | 1/50 | 5 ±2 | 5.33 ±2.49 |
| <nobr>swap_T</nobr> | 0 | 0.00 | 0/50 | 0 | 0.00 | 0/50 | 0 | 0.00 | 0/50 | 0 ±0 | 0.00 ±0.00 |
| <nobr>press_by_number</nobr> | 74 | 74.00 | 37/50 | 80 | 80.00 | 40/50 | 80 | 80.00 | 40/50 | 78 ±3 | 78.00 ±2.83 |
| <nobr>imitate_sorting_sequence</nobr> | 4 | 6.40 | 2/50 | 2 | 3.30 | 1/50 | 0 | 1.40 | 0/50 | 2 ±2 | 3.70 ±2.06 |
| **Memory Avg (per seed)** | 42.3 | 42.93 | 127/300 | 43.7 | 44.03 | 131/300 | 40.7 | 41.00 | 122/300 | 42 ±1 | 42.66 ±1.25 |
| **Open** |  |  |  |  |  |  |  |  |  |  |  |
| <nobr>align_blocks</nobr> | 58 | 58.00 | 29/50 | 56 | 56.00 | 28/50 | 62 | 62.00 | 31/50 | 59 ±2 | 58.67 ±2.49 |
| <nobr>general_pickup</nobr> | 14 | 14.00 | 7/50 | 22 | 22.00 | 11/50 | 16 | 16.00 | 8/50 | 17 ±3 | 17.33 ±3.40 |
| <nobr>stack_blocks_by_language</nobr> | 4 | 7.60 | 2/50 | 0 | 3.60 | 0/50 | 0 | 2.80 | 0/50 | 1 ±2 | 4.67 ±2.10 |
| <nobr>solve_equation</nobr> | 16 | 16.00 | 8/50 | 14 | 14.00 | 7/50 | 18 | 18.00 | 9/50 | 16 ±2 | 16.00 ±1.63 |
| <nobr>classify_objects_by_language</nobr> | 0 | 4.40 | 0/50 | 0 | 2.80 | 0/50 | 0 | 3.00 | 0/50 | 0 ±0 | 3.40 ±0.71 |
| <nobr>pick_from_conveyor_by_image</nobr> | 0 | 0.00 | 0/50 | 0 | 0.00 | 0/50 | 0 | 0.00 | 0/50 | 0 ±0 | 0.00 ±0.00 |
| <nobr>store_tools_in_toolbox</nobr> | 0 | 0.50 | 0/50 | 0 | 0.00 | 0/50 | 0 | 0.00 | 0/50 | 0 ±0 | 0.17 ±0.24 |
| <nobr>pour_by_language</nobr> | 0 | 0.00 | 0/50 | 2 | 2.00 | 1/50 | 2 | 2.00 | 1/50 | 1 ±1 | 1.33 ±0.94 |
| **Open Avg (per seed)** | 11.5 | 12.56 | 46/400 | 11.8 | 12.55 | 47/400 | 12.2 | 12.97 | 49/400 | 12 ±0 | 12.70 ±0.20 |
| **Avg (per seed)** | 27.9 | 33.15 | 585/2090 | 27.5 | 33.18 | 577/2075 | 26.7 | 32.30 | 554/2060 | 27 ±1 | 32.88 ±0.41 |

## Incomplete trials

| Task variant | Seed | Missing episodes |
| --- | ---: | ---: |
| make_toast_random | 0 | 10 |
| make_toast_random | 1 | 25 |
| make_toast_random | 2 | 25 |
| store_laptop_and_headphones_random | 2 | 15 |
