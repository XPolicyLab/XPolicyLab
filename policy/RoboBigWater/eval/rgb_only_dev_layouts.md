# RoboBigWater, RGB-only: frozen tools on the seed-0 development layouts

Observation: the stock `arx_x5` config (three RGB cameras, joint states, end-effector poses). Depth is estimated inside the policy server by Depth Anything V2 (Metric-Indoor-Base) with its scale fixed from the known table height; the simulator's depth is never read. Tools: the frozen set in `tasks/` (developed earlier with rendered depth, unchanged). Protocol: seed 0, development layouts 0-9 (Generalization tasks: standard 0-4 and cluttered `_random` 0-4), one episode per layout, no optimizer, official success and progress checkers. 420 episodes, 2026-10-11.

## Summary (leaderboard formula: mean of the five task types; Generalization = mean of standard and cluttered)

| | Success rate (%) | Progress score |
|---|---:|---:|
| Generalization | 23.3 | 26.7 |
| Precision | 6.2 | 9.1 |
| Long-Horizon | 5.0 | 21.8 |
| Memory | 36.7 | 40.3 |
| Open | 35.0 | 38.9 |
| **Overall** | **21.2** | **27.4** |

For reference, the same tools with the simulator's rendered depth reached 48.8 / 54.1 on this formula (the previous revision of this adapter); GPT-6 Astra zero-shot through a fixed harness is 22.5 / 29.0 on the leaderboard.

## Per variant (seed 0, 10 episodes per task; 5 + 5 for Generalization tasks)

| Type | Variant | Success | Progress score |
|---|---|---:|---:|
| Generalization | arrange_largest_number | 3/5 | 67.0 |
| Generalization | arrange_largest_number_random | 2/5 | 52.0 |
| Generalization | fold_clothes | 2/5 | 44.0 |
| Generalization | fold_clothes_random | 1/5 | 32.0 |
| Generalization | hang_mugs | 0/5 | 6.0 |
| Generalization | hang_mugs_random | 0/5 | 0.0 |
| Generalization | make_toast | 0/5 | 5.0 |
| Generalization | make_toast_random | 0/5 | 0.0 |
| Generalization | pack_objects_into_box | 0/5 | 14.0 |
| Generalization | pack_objects_into_box_random | 0/5 | 4.0 |
| Generalization | pour_liquid_into_cup | 0/5 | 0.0 |
| Generalization | pour_liquid_into_cup_random | 1/5 | 20.0 |
| Generalization | push_T | 1/5 | 20.0 |
| Generalization | push_T_random | 2/5 | 40.0 |
| Generalization | sort_nesting_dolls_by_size | 1/5 | 20.0 |
| Generalization | sort_nesting_dolls_by_size_random | 0/5 | 0.0 |
| Generalization | stack_blocks | 3/5 | 66.0 |
| Generalization | stack_blocks_random | 2/5 | 46.0 |
| Generalization | stack_bowls | 5/5 | 100.0 |
| Generalization | stack_bowls_random | 5/5 | 100.0 |
| Generalization | store_laptop_and_headphones | 0/5 | 4.0 |
| Generalization | store_laptop_and_headphones_random | 0/5 | 0.0 |
| Generalization | sweep_blocks | 0/5 | 0.0 |
| Generalization | sweep_blocks_random | 0/5 | 0.0 |
| Long-Horizon | classify_objects | 3/10 | 53.0 |
| Long-Horizon | fill_egg_holder | 0/10 | 7.0 |
| Long-Horizon | fill_pen_holder | 0/10 | 6.0 |
| Long-Horizon | make_kong | 0/10 | 0.0 |
| Long-Horizon | organize_table | 0/10 | 22.5 |
| Long-Horizon | play_stacking_toy | 0/10 | 10.0 |
| Long-Horizon | play_tic_tac_toe | 1/10 | 48.0 |
| Long-Horizon | put_bottles_into_dustbin | 0/10 | 28.0 |
| Memory | cover_blocks | 10/10 | 100.0 |
| Memory | imitate_sorting_sequence | 0/10 | 22.0 |
| Memory | match_and_pick_from_conveyor | 5/10 | 50.0 |
| Memory | press_by_number | 0/10 | 0.0 |
| Memory | swap_T | 7/10 | 70.0 |
| Memory | swap_blocks | 0/10 | 0.0 |
| Open | align_blocks | 5/10 | 50.0 |
| Open | classify_objects_by_language | 1/10 | 31.0 |
| Open | general_pickup | 8/10 | 80.0 |
| Open | pick_from_conveyor_by_image | 1/10 | 10.0 |
| Open | pour_by_language | 2/10 | 20.0 |
| Open | solve_equation | 8/10 | 80.0 |
| Open | stack_blocks_by_language | 3/10 | 40.0 |
| Open | store_tools_in_toolbox | 0/10 | 0.0 |
| Precision | build_tower | 0/10 | 1.0 |
| Precision | deposit_coin | 0/10 | 0.0 |
| Precision | fasten_screws | 0/10 | 10.0 |
| Precision | insert_key | 0/10 | 0.0 |
| Precision | insert_tubes | 1/10 | 22.0 |
| Precision | play_Xylophone | 0/10 | 0.0 |
| Precision | plug_in_charger | 0/10 | 0.0 |
| Precision | pour_balls_into_vase | 4/10 | 40.0 |
