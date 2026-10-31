# PyRUA-RSI on RoboDojo: eval seed 0

RoboDojo's own summary (`bash scripts/robodojo.sh summarize`) of the `PyRUA_RSI` policy (PyRUA-RSI): one fixed Python
program per RoboDojo task name, no language model called at evaluation time. Evaluated like the submitted agent: 54 task
configurations at their native episode counts, eval seed 0 (2,100 episodes), one official `robodojo.sh eval` process per
task name, RoboDojo `main` `cb2a37d`, default RGB-only observation. Seed 0 was also the seed the programs were developed
on; the same programs on eval seed 2, never used in development, give Avg SR 80.73 / Avg Score 85.36. The submitted
agent's summary on the same seed is [`robodojo_summary_rgb.md`](robodojo_summary_rgb.md).

## Overview (by capability dimension)

Progress † marks incomplete evaluation (126 = 42 tasks × 3 seeds). Cells show `mean ±std` across seeds; scores reflect completed cells only. Policies are sorted by overall Avg SR.

| Policy | Progress | Generalization SR (%) | Generalization Score | Precision SR (%) | Precision Score | Long-Horizon SR (%) | Long-Horizon Score | Memory SR (%) | Memory Score | Open SR (%) | Open Score | Avg SR (%) | Avg Score |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| PyRUA_RSI | 33.3% † (42/126) | 72.33 ±0.00 | 78.08 ±0.00 | 83.50 ±0.00 | 85.71 ±0.00 | 80.00 ±0.00 | 88.22 ±0.00 | 95.00 ±0.00 | 95.55 ±0.00 | 83.25 ±0.00 | 88.24 ±0.00 | 82.82 ±0.00 | 87.16 ±0.00 |

## Generalization: Standard vs Random (25 episodes each)

Each cell shows `mean ±std` across seeds, averaged over the 12 Generalization tasks with completed data for that half. Δ = Random − Standard (mean SR). Sorted by Standard SR.

| Policy | Standard SR (%) | Standard Score | Random SR (%) | Random Score | Δ SR | Δ Score |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| PyRUA_RSI | 76.33 ±0.00 | 82.43 ±0.00 | 68.33 ±0.00 | 73.73 ±0.00 | -8.00 | -8.70 |

### Per-task Standard vs Random

Additional view only: each of the 12 Generalization tasks is listed as the 25-episode standard split and its `_random` sibling. Columns match the per-policy all-task tables. Existing merged 50-episode scores are unchanged.

#### PyRUA_RSI

| Task | Seed 0 SR (%) | Seed 0 Score | Seed 1 SR (%) | Seed 1 Score | Seed 2 SR (%) | Seed 2 Score | Avg SR (%) | Avg Score |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| <nobr>stack_bowls</nobr> | 88 | 89.80 |  |  |  |  |  |  |
| <nobr>stack_bowls_random</nobr> | 92 | 92.60 |  |  |  |  |  |  |
| <nobr>push_T</nobr> | 100 | 100.00 |  |  |  |  |  |  |
| <nobr>push_T_random</nobr> | 100 | 100.00 |  |  |  |  |  |  |
| <nobr>pack_objects_into_box</nobr> | 52 | 71.40 |  |  |  |  |  |  |
| <nobr>pack_objects_into_box_random</nobr> | 40 | 63.40 |  |  |  |  |  |  |
| <nobr>fold_clothes</nobr> | 92 | 93.60 |  |  |  |  |  |  |
| <nobr>fold_clothes_random</nobr> | 88 | 90.40 |  |  |  |  |  |  |
| <nobr>hang_mugs</nobr> | 32 | 56.20 |  |  |  |  |  |  |
| <nobr>hang_mugs_random</nobr> | 28 | 46.20 |  |  |  |  |  |  |
| <nobr>sweep_blocks</nobr> | 100 | 100.00 |  |  |  |  |  |  |
| <nobr>sweep_blocks_random</nobr> | 92 | 92.00 |  |  |  |  |  |  |
| <nobr>pour_liquid_into_cup</nobr> | 96 | 96.00 |  |  |  |  |  |  |
| <nobr>pour_liquid_into_cup_random</nobr> | 96 | 96.00 |  |  |  |  |  |  |
| <nobr>make_toast</nobr> | 56 | 63.00 |  |  |  |  |  |  |
| <nobr>make_toast_random</nobr> | 24 | 28.00 |  |  |  |  |  |  |
| <nobr>arrange_largest_number</nobr> | 100 | 100.00 |  |  |  |  |  |  |
| <nobr>arrange_largest_number_random</nobr> | 88 | 91.60 |  |  |  |  |  |  |
| <nobr>sort_nesting_dolls_by_size</nobr> | 96 | 96.00 |  |  |  |  |  |  |
| <nobr>sort_nesting_dolls_by_size_random</nobr> | 76 | 76.00 |  |  |  |  |  |  |
| <nobr>store_laptop_and_headphones</nobr> | 4 | 23.20 |  |  |  |  |  |  |
| <nobr>store_laptop_and_headphones_random</nobr> | 0 | 12.00 |  |  |  |  |  |  |
| <nobr>stack_blocks</nobr> | 100 | 100.00 |  |  |  |  |  |  |
| <nobr>stack_blocks_random</nobr> | 96 | 96.60 |  |  |  |  |  |  |
| **Avg Standard** | 76.3 | 82.43 |  |  |  |  | 76 ±0 | 82.43 ±0.00 |
| **Avg Random** | 68.3 | 73.73 |  |  |  |  | 68 ±0 | 73.73 ±0.00 |

## PyRUA_RSI

| Task | Seed 0 SR (%) | Seed 0 Score | Seed 1 SR (%) | Seed 1 Score | Seed 2 SR (%) | Seed 2 Score | Avg SR (%) | Avg Score |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| <nobr>align_blocks</nobr> | 98 | 98.00 |  |  |  |  |  |  |
| <nobr>arrange_largest_number</nobr> | 94 | 95.80 |  |  |  |  |  |  |
| <nobr>build_tower</nobr> | 100 | 100.00 |  |  |  |  |  |  |
| <nobr>classify_objects</nobr> | 92 | 95.20 |  |  |  |  |  |  |
| <nobr>classify_objects_by_language</nobr> | 78 | 84.40 |  |  |  |  |  |  |
| <nobr>cover_blocks</nobr> | 100 | 100.00 |  |  |  |  |  |  |
| <nobr>deposit_coin</nobr> | 78 | 79.60 |  |  |  |  |  |  |
| <nobr>fasten_screws</nobr> | 96 | 97.40 |  |  |  |  |  |  |
| <nobr>fill_egg_holder</nobr> | 44 | 61.30 |  |  |  |  |  |  |
| <nobr>fill_pen_holder</nobr> | 34 | 59.20 |  |  |  |  |  |  |
| <nobr>fold_clothes</nobr> | 90 | 92.00 |  |  |  |  |  |  |
| <nobr>general_pickup</nobr> | 98 | 98.00 |  |  |  |  |  |  |
| <nobr>hang_mugs</nobr> | 30 | 51.20 |  |  |  |  |  |  |
| <nobr>imitate_sorting_sequence</nobr> | 92 | 95.30 |  |  |  |  |  |  |
| <nobr>insert_key</nobr> | 0 | 14.70 |  |  |  |  |  |  |
| <nobr>insert_tubes</nobr> | 100 | 100.00 |  |  |  |  |  |  |
| <nobr>make_kong</nobr> | 100 | 100.00 |  |  |  |  |  |  |
| <nobr>make_toast</nobr> | 40 | 45.50 |  |  |  |  |  |  |
| <nobr>match_and_pick_from_conveyor</nobr> | 88 | 88.00 |  |  |  |  |  |  |
| <nobr>organize_table</nobr> | 76 | 93.50 |  |  |  |  |  |  |
| <nobr>pack_objects_into_box</nobr> | 46 | 67.40 |  |  |  |  |  |  |
| <nobr>pick_from_conveyor_by_image</nobr> | 44 | 44.00 |  |  |  |  |  |  |
| <nobr>play_Xylophone</nobr> | 98 | 98.00 |  |  |  |  |  |  |
| <nobr>play_stacking_toy</nobr> | 96 | 97.80 |  |  |  |  |  |  |
| <nobr>play_tic_tac_toe</nobr> | 100 | 100.00 |  |  |  |  |  |  |
| <nobr>plug_in_charger</nobr> | 98 | 98.00 |  |  |  |  |  |  |
| <nobr>pour_balls_into_vase</nobr> | 98 | 98.00 |  |  |  |  |  |  |
| <nobr>pour_by_language</nobr> | 98 | 98.00 |  |  |  |  |  |  |
| <nobr>pour_liquid_into_cup</nobr> | 96 | 96.00 |  |  |  |  |  |  |
| <nobr>press_by_number</nobr> | 100 | 100.00 |  |  |  |  |  |  |
| <nobr>push_T</nobr> | 100 | 100.00 |  |  |  |  |  |  |
| <nobr>put_bottles_into_dustbin</nobr> | 98 | 98.80 |  |  |  |  |  |  |
| <nobr>solve_equation</nobr> | 98 | 98.00 |  |  |  |  |  |  |
| <nobr>sort_nesting_dolls_by_size</nobr> | 86 | 86.00 |  |  |  |  |  |  |
| <nobr>stack_blocks</nobr> | 98 | 98.30 |  |  |  |  |  |  |
| <nobr>stack_blocks_by_language</nobr> | 100 | 100.00 |  |  |  |  |  |  |
| <nobr>stack_bowls</nobr> | 90 | 91.20 |  |  |  |  |  |  |
| <nobr>store_laptop_and_headphones</nobr> | 2 | 17.60 |  |  |  |  |  |  |
| <nobr>store_tools_in_toolbox</nobr> | 52 | 85.50 |  |  |  |  |  |  |
| <nobr>swap_T</nobr> | 100 | 100.00 |  |  |  |  |  |  |
| <nobr>swap_blocks</nobr> | 90 | 90.00 |  |  |  |  |  |  |
| <nobr>sweep_blocks</nobr> | 96 | 96.00 |  |  |  |  |  |  |
| **Avg (per seed)** | 81.2 | 85.90 |  |  |  |  | 81 ±0 | 85.90 ±0.00 |
