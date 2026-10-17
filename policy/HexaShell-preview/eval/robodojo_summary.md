# RoboDojo summary: seed 0, layouts 10-19 (interim)

This table comes from RoboDojo's own `scripts/internal/summarize_result.py` (commit `09bd5f5`), run on the episodes in `new_layouts_s0l10.md`. One thing differs from the official run: a cell needs 10 episodes (5 standard + 5 cluttered for Generalization tasks) instead of 50 (25 + 25). The formulas are the official ones.

- Only seed 0 was run. The seed 1 and seed 2 columns are empty, and `±0.00` only means there is one seed. Where the script's text says "25 episodes each", read 5.
- `swap_blocks`: the RoboDojo revision of the first run (`ee67a14`) did not reset its button, so no press was counted (0/10); RoboDojo PR #48 fixes this. `swap_blocks` was developed again in the fixed simulator, its tools were frozen, and it was run again on the same layouts with the fix loaded: 3/10, the value in this table. It has no task notes.
- The official three-seed evaluation will replace this file.


### Overview (by capability dimension)

Progress † marks incomplete evaluation (126 = 42 tasks × 3 seeds). Cells show `mean ±std` across seeds; scores reflect completed cells only. Policies are sorted by overall Avg SR.

| Policy | Progress | Generalization SR (%) | Generalization Score | Precision SR (%) | Precision Score | Long-Horizon SR (%) | Long-Horizon Score | Memory SR (%) | Memory Score | Open SR (%) | Open Score | Avg SR (%) | Avg Score |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| HexaShell-preview | 33.3% † (42/126) | 28.33 ±0.00 | 35.38 ±0.00 | 31.25 ±0.00 | 41.38 ±0.00 | 23.75 ±0.00 | 40.56 ±0.00 | 65.00 ±0.00 | 69.17 ±0.00 | 50.00 ±0.00 | 53.62 ±0.00 | 39.67 ±0.00 | 48.02 ±0.00 |

### Generalization: Standard vs Random (25 episodes each)

Each cell shows `mean ±std` across seeds, averaged over the 12 Generalization tasks with completed data for that half. Δ = Random − Standard (mean SR). Sorted by Standard SR.

| Policy | Standard SR (%) | Standard Score | Random SR (%) | Random Score | Δ SR | Δ Score |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| HexaShell-preview | 33.33 ±0.00 | 38.83 ±0.00 | 23.33 ±0.00 | 31.92 ±0.00 | -10.00 | -6.92 |

#### Per-task Standard vs Random

Additional view only: each of the 12 Generalization tasks is listed as the 25-episode standard split and its `_random` sibling. Columns match the per-policy all-task tables. Existing merged 50-episode scores are unchanged.

#### HexaShell-preview

| Task | Seed 0 SR (%) | Seed 0 Score | Seed 1 SR (%) | Seed 1 Score | Seed 2 SR (%) | Seed 2 Score | Avg SR (%) | Avg Score |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| <nobr>stack_bowls</nobr> | 80 | 83.00 |  |  |  |  |  |  |
| <nobr>stack_bowls_random</nobr> | 100 | 100.00 |  |  |  |  |  |  |
| <nobr>push_T</nobr> | 100 | 100.00 |  |  |  |  |  |  |
| <nobr>push_T_random</nobr> | 100 | 100.00 |  |  |  |  |  |  |
| <nobr>pack_objects_into_box</nobr> | 0 | 17.00 |  |  |  |  |  |  |
| <nobr>pack_objects_into_box_random</nobr> | 0 | 32.00 |  |  |  |  |  |  |
| <nobr>fold_clothes</nobr> | 80 | 84.00 |  |  |  |  |  |  |
| <nobr>fold_clothes_random</nobr> | 20 | 36.00 |  |  |  |  |  |  |
| <nobr>hang_mugs</nobr> | 0 | 17.00 |  |  |  |  |  |  |
| <nobr>hang_mugs_random</nobr> | 0 | 24.00 |  |  |  |  |  |  |
| <nobr>sweep_blocks</nobr> | 0 | 0.00 |  |  |  |  |  |  |
| <nobr>sweep_blocks_random</nobr> | 0 | 0.00 |  |  |  |  |  |  |
| <nobr>pour_liquid_into_cup</nobr> | 0 | 0.00 |  |  |  |  |  |  |
| <nobr>pour_liquid_into_cup_random</nobr> | 0 | 0.00 |  |  |  |  |  |  |
| <nobr>make_toast</nobr> | 0 | 0.00 |  |  |  |  |  |  |
| <nobr>make_toast_random</nobr> | 0 | 0.00 |  |  |  |  |  |  |
| <nobr>arrange_largest_number</nobr> | 20 | 38.00 |  |  |  |  |  |  |
| <nobr>arrange_largest_number_random</nobr> | 20 | 31.00 |  |  |  |  |  |  |
| <nobr>sort_nesting_dolls_by_size</nobr> | 40 | 40.00 |  |  |  |  |  |  |
| <nobr>sort_nesting_dolls_by_size_random</nobr> | 20 | 20.00 |  |  |  |  |  |  |
| <nobr>store_laptop_and_headphones</nobr> | 0 | 4.00 |  |  |  |  |  |  |
| <nobr>store_laptop_and_headphones_random</nobr> | 0 | 8.00 |  |  |  |  |  |  |
| <nobr>stack_blocks</nobr> | 80 | 83.00 |  |  |  |  |  |  |
| <nobr>stack_blocks_random</nobr> | 20 | 32.00 |  |  |  |  |  |  |
| **Avg Standard** | 33.3 | 38.83 |  |  |  |  | 33 ±0 | 38.83 ±0.00 |
| **Avg Random** | 23.3 | 31.92 |  |  |  |  | 23 ±0 | 31.92 ±0.00 |

### HexaShell-preview

| Task | Seed 0 SR (%) | Seed 0 Score | Seed 1 SR (%) | Seed 1 Score | Seed 2 SR (%) | Seed 2 Score | Avg SR (%) | Avg Score |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| <nobr>align_blocks</nobr> | 70 | 70.00 |  |  |  |  |  |  |
| <nobr>arrange_largest_number</nobr> | 20 | 34.50 |  |  |  |  |  |  |
| <nobr>build_tower</nobr> | 20 | 34.00 |  |  |  |  |  |  |
| <nobr>classify_objects</nobr> | 30 | 55.50 |  |  |  |  |  |  |
| <nobr>classify_objects_by_language</nobr> | 30 | 52.00 |  |  |  |  |  |  |
| <nobr>cover_blocks</nobr> | 80 | 82.00 |  |  |  |  |  |  |
| <nobr>deposit_coin</nobr> | 20 | 28.00 |  |  |  |  |  |  |
| <nobr>fasten_screws</nobr> | 20 | 38.00 |  |  |  |  |  |  |
| <nobr>fill_egg_holder</nobr> | 0 | 10.50 |  |  |  |  |  |  |
| <nobr>fill_pen_holder</nobr> | 0 | 3.50 |  |  |  |  |  |  |
| <nobr>fold_clothes</nobr> | 50 | 60.00 |  |  |  |  |  |  |
| <nobr>general_pickup</nobr> | 90 | 90.00 |  |  |  |  |  |  |
| <nobr>hang_mugs</nobr> | 0 | 20.50 |  |  |  |  |  |  |
| <nobr>imitate_sorting_sequence</nobr> | 30 | 53.00 |  |  |  |  |  |  |
| <nobr>insert_key</nobr> | 0 | 15.00 |  |  |  |  |  |  |
| <nobr>insert_tubes</nobr> | 30 | 56.00 |  |  |  |  |  |  |
| <nobr>make_kong</nobr> | 0 | 0.00 |  |  |  |  |  |  |
| <nobr>make_toast</nobr> | 0 | 0.00 |  |  |  |  |  |  |
| <nobr>match_and_pick_from_conveyor</nobr> | 60 | 60.00 |  |  |  |  |  |  |
| <nobr>organize_table</nobr> | 0 | 50.00 |  |  |  |  |  |  |
| <nobr>pack_objects_into_box</nobr> | 0 | 24.50 |  |  |  |  |  |  |
| <nobr>pick_from_conveyor_by_image</nobr> | 20 | 20.00 |  |  |  |  |  |  |
| <nobr>play_Xylophone</nobr> | 60 | 60.00 |  |  |  |  |  |  |
| <nobr>play_stacking_toy</nobr> | 0 | 32.00 |  |  |  |  |  |  |
| <nobr>play_tic_tac_toe</nobr> | 100 | 100.00 |  |  |  |  |  |  |
| <nobr>plug_in_charger</nobr> | 10 | 10.00 |  |  |  |  |  |  |
| <nobr>pour_balls_into_vase</nobr> | 90 | 90.00 |  |  |  |  |  |  |
| <nobr>pour_by_language</nobr> | 10 | 17.00 |  |  |  |  |  |  |
| <nobr>pour_liquid_into_cup</nobr> | 0 | 0.00 |  |  |  |  |  |  |
| <nobr>press_by_number</nobr> | 100 | 100.00 |  |  |  |  |  |  |
| <nobr>push_T</nobr> | 100 | 100.00 |  |  |  |  |  |  |
| <nobr>put_bottles_into_dustbin</nobr> | 60 | 73.00 |  |  |  |  |  |  |
| <nobr>solve_equation</nobr> | 80 | 80.00 |  |  |  |  |  |  |
| <nobr>sort_nesting_dolls_by_size</nobr> | 30 | 30.00 |  |  |  |  |  |  |
| <nobr>stack_blocks</nobr> | 50 | 57.50 |  |  |  |  |  |  |
| <nobr>stack_blocks_by_language</nobr> | 100 | 100.00 |  |  |  |  |  |  |
| <nobr>stack_bowls</nobr> | 90 | 91.50 |  |  |  |  |  |  |
| <nobr>store_laptop_and_headphones</nobr> | 0 | 6.00 |  |  |  |  |  |  |
| <nobr>store_tools_in_toolbox</nobr> | 0 | 0.00 |  |  |  |  |  |  |
| <nobr>swap_T</nobr> | 90 | 90.00 |  |  |  |  |  |  |
| <nobr>swap_blocks</nobr> | 30 | 30.00 |  |  |  |  |  |  |
| <nobr>sweep_blocks</nobr> | 0 | 0.00 |  |  |  |  |  |  |
| **Avg (per seed)** | 37.4 | 45.81 |  |  |  |  | 37 ±0 | 45.81 ±0.00 |
