# PyRUA-RSI on RoboDojo: eval seed 0

RoboDojo's own summary (`bash scripts/robodojo.sh summarize`) of the `PyRUA_RSI` policy (PyRUA-RSI): one fixed Python
program per RoboDojo task name, no language model called at evaluation time. Evaluated like the submitted agent: 54 task
configurations at their native episode counts, eval seed 0 (2,100 episodes), one official `robodojo.sh eval` process per
task name, RoboDojo `main` `cb2a37d`, default RGB-only observation. Seed 0 was also the seed the programs were developed
on; the same programs on eval seed 2, never used in development, give Avg SR 80.73 / Avg Score 85.36.

## Overview (by capability dimension)

Progress † marks incomplete evaluation (126 = 42 tasks × 3 seeds). Cells show `mean ±std` across seeds; scores reflect completed cells only. Policies are sorted by overall Avg SR.

| Policy | Progress | Generalization SR (%) | Generalization Score | Precision SR (%) | Precision Score | Long-Horizon SR (%) | Long-Horizon Score | Memory SR (%) | Memory Score | Open SR (%) | Open Score | Avg SR (%) | Avg Score |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| PyRUA_RSI | 33.3% † (42/126) | 60.83 ±0.00 | 65.54 ±0.00 | 92.75 ±0.00 | 93.76 ±0.00 | 82.50 ±0.00 | 89.76 ±0.00 | 93.33 ±0.00 | 93.58 ±0.00 | 81.00 ±0.00 | 88.51 ±0.00 | 82.08 ±0.00 | 86.23 ±0.00 |

## Generalization: Standard vs Random (25 episodes each)

Each cell shows `mean ±std` across seeds, averaged over the 12 Generalization tasks with completed data for that half. Δ = Random − Standard (mean SR). Sorted by Standard SR.

| Policy | Standard SR (%) | Standard Score | Random SR (%) | Random Score | Δ SR | Δ Score |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| PyRUA_RSI | 62.33 ±0.00 | 68.37 ±0.00 | 59.33 ±0.00 | 62.72 ±0.00 | -3.00 | -5.65 |

### Per-task Standard vs Random

Additional view only: each of the 12 Generalization tasks is listed as the 25-episode standard split and its `_random` sibling. Columns match the per-policy all-task tables. Existing merged 50-episode scores are unchanged.

#### PyRUA_RSI

| Task | Seed 0 SR (%) | Seed 0 Score | Seed 1 SR (%) | Seed 1 Score | Seed 2 SR (%) | Seed 2 Score | Avg SR (%) | Avg Score |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| <nobr>stack_bowls</nobr> | 96 | 96.60 |  |  |  |  |  |  |
| <nobr>stack_bowls_random</nobr> | 88 | 89.80 |  |  |  |  |  |  |
| <nobr>push_T</nobr> | 100 | 100.00 |  |  |  |  |  |  |
| <nobr>push_T_random</nobr> | 100 | 100.00 |  |  |  |  |  |  |
| <nobr>pack_objects_into_box</nobr> | 72 | 86.00 |  |  |  |  |  |  |
| <nobr>pack_objects_into_box_random</nobr> | 68 | 76.40 |  |  |  |  |  |  |
| <nobr>fold_clothes</nobr> | 84 | 87.20 |  |  |  |  |  |  |
| <nobr>fold_clothes_random</nobr> | 88 | 90.40 |  |  |  |  |  |  |
| <nobr>hang_mugs</nobr> | 20 | 40.40 |  |  |  |  |  |  |
| <nobr>hang_mugs_random</nobr> | 0 | 15.20 |  |  |  |  |  |  |
| <nobr>sweep_blocks</nobr> | 8 | 8.00 |  |  |  |  |  |  |
| <nobr>sweep_blocks_random</nobr> | 0 | 0.00 |  |  |  |  |  |  |
| <nobr>pour_liquid_into_cup</nobr> | 76 | 76.00 |  |  |  |  |  |  |
| <nobr>pour_liquid_into_cup_random</nobr> | 84 | 84.00 |  |  |  |  |  |  |
| <nobr>make_toast</nobr> | 32 | 42.00 |  |  |  |  |  |  |
| <nobr>make_toast_random</nobr> | 32 | 34.00 |  |  |  |  |  |  |
| <nobr>arrange_largest_number</nobr> | 88 | 92.40 |  |  |  |  |  |  |
| <nobr>arrange_largest_number_random</nobr> | 88 | 90.80 |  |  |  |  |  |  |
| <nobr>sort_nesting_dolls_by_size</nobr> | 76 | 76.00 |  |  |  |  |  |  |
| <nobr>sort_nesting_dolls_by_size_random</nobr> | 64 | 64.00 |  |  |  |  |  |  |
| <nobr>store_laptop_and_headphones</nobr> | 0 | 19.20 |  |  |  |  |  |  |
| <nobr>store_laptop_and_headphones_random</nobr> | 0 | 8.00 |  |  |  |  |  |  |
| <nobr>stack_blocks</nobr> | 96 | 96.60 |  |  |  |  |  |  |
| <nobr>stack_blocks_random</nobr> | 100 | 100.00 |  |  |  |  |  |  |
| **Avg Standard** | 62.3 | 68.37 |  |  |  |  | 62 ±0 | 68.37 ±0.00 |
| **Avg Random** | 59.3 | 62.72 |  |  |  |  | 59 ±0 | 62.72 ±0.00 |

## PyRUA_RSI

| Task | Seed 0 SR (%) | Seed 0 Score | Seed 1 SR (%) | Seed 1 Score | Seed 2 SR (%) | Seed 2 Score | Avg SR (%) | Avg Score |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| <nobr>align_blocks</nobr> | 98 | 98.00 |  |  |  |  |  |  |
| <nobr>arrange_largest_number</nobr> | 88 | 91.60 |  |  |  |  |  |  |
| <nobr>build_tower</nobr> | 100 | 100.00 |  |  |  |  |  |  |
| <nobr>classify_objects</nobr> | 98 | 98.80 |  |  |  |  |  |  |
| <nobr>classify_objects_by_language</nobr> | 86 | 89.60 |  |  |  |  |  |  |
| <nobr>cover_blocks</nobr> | 100 | 100.00 |  |  |  |  |  |  |
| <nobr>deposit_coin</nobr> | 72 | 74.80 |  |  |  |  |  |  |
| <nobr>fasten_screws</nobr> | 96 | 98.00 |  |  |  |  |  |  |
| <nobr>fill_egg_holder</nobr> | 54 | 68.40 |  |  |  |  |  |  |
| <nobr>fill_pen_holder</nobr> | 50 | 68.70 |  |  |  |  |  |  |
| <nobr>fold_clothes</nobr> | 86 | 88.80 |  |  |  |  |  |  |
| <nobr>general_pickup</nobr> | 96 | 96.00 |  |  |  |  |  |  |
| <nobr>hang_mugs</nobr> | 10 | 27.80 |  |  |  |  |  |  |
| <nobr>imitate_sorting_sequence</nobr> | 82 | 83.50 |  |  |  |  |  |  |
| <nobr>insert_key</nobr> | 78 | 81.30 |  |  |  |  |  |  |
| <nobr>insert_tubes</nobr> | 100 | 100.00 |  |  |  |  |  |  |
| <nobr>make_kong</nobr> | 96 | 96.00 |  |  |  |  |  |  |
| <nobr>make_toast</nobr> | 32 | 38.00 |  |  |  |  |  |  |
| <nobr>match_and_pick_from_conveyor</nobr> | 78 | 78.00 |  |  |  |  |  |  |
| <nobr>organize_table</nobr> | 72 | 92.00 |  |  |  |  |  |  |
| <nobr>pack_objects_into_box</nobr> | 70 | 81.20 |  |  |  |  |  |  |
| <nobr>pick_from_conveyor_by_image</nobr> | 68 | 68.00 |  |  |  |  |  |  |
| <nobr>play_Xylophone</nobr> | 100 | 100.00 |  |  |  |  |  |  |
| <nobr>play_stacking_toy</nobr> | 96 | 97.80 |  |  |  |  |  |  |
| <nobr>play_tic_tac_toe</nobr> | 100 | 100.00 |  |  |  |  |  |  |
| <nobr>plug_in_charger</nobr> | 100 | 100.00 |  |  |  |  |  |  |
| <nobr>pour_balls_into_vase</nobr> | 96 | 96.00 |  |  |  |  |  |  |
| <nobr>pour_by_language</nobr> | 94 | 94.00 |  |  |  |  |  |  |
| <nobr>pour_liquid_into_cup</nobr> | 80 | 80.00 |  |  |  |  |  |  |
| <nobr>press_by_number</nobr> | 100 | 100.00 |  |  |  |  |  |  |
| <nobr>push_T</nobr> | 100 | 100.00 |  |  |  |  |  |  |
| <nobr>put_bottles_into_dustbin</nobr> | 94 | 96.40 |  |  |  |  |  |  |
| <nobr>solve_equation</nobr> | 98 | 98.00 |  |  |  |  |  |  |
| <nobr>sort_nesting_dolls_by_size</nobr> | 70 | 70.00 |  |  |  |  |  |  |
| <nobr>stack_blocks</nobr> | 98 | 98.30 |  |  |  |  |  |  |
| <nobr>stack_blocks_by_language</nobr> | 100 | 100.00 |  |  |  |  |  |  |
| <nobr>stack_bowls</nobr> | 92 | 93.20 |  |  |  |  |  |  |
| <nobr>store_laptop_and_headphones</nobr> | 0 | 13.60 |  |  |  |  |  |  |
| <nobr>store_tools_in_toolbox</nobr> | 8 | 64.50 |  |  |  |  |  |  |
| <nobr>swap_T</nobr> | 100 | 100.00 |  |  |  |  |  |  |
| <nobr>swap_blocks</nobr> | 100 | 100.00 |  |  |  |  |  |  |
| <nobr>sweep_blocks</nobr> | 4 | 4.00 |  |  |  |  |  |  |
| **Avg (per seed)** | 79.5 | 83.91 |  |  |  |  | 80 ±0 | 83.91 ±0.00 |