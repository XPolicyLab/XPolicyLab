# RoboBigWater, RGB-only: current tools on the seed-0 development layouts

Observation: the stock `arx_x5` config (three RGB cameras, joint states, end-effector poses). Depth is estimated inside the policy server by Depth Anything V2 (Metric-Indoor-Base) with its scale fixed from the known table height; the simulator's depth is never read. Protocol: seed 0, development layouts 0-9 (Generalization tasks: standard 0-4 and cluttered `_random` 0-4), one episode per layout with frozen tools, no optimizer, official success and progress checkers.

Tools: 34 tasks carry the tools developed earlier with rendered depth, unchanged (frozen retest 2026-10-11). 8 tasks (`align_blocks`, `insert_tubes`, `play_tic_tac_toe`, `pour_liquid_into_cup`, `press_by_number`, `push_T`, `stack_blocks`, `stack_blocks_by_language`) carry tools re-evolved with estimated depth on the same development layouts and were retested frozen on 2026-10-12. The other tasks are being re-evolved; this file is updated as they finish.

## Summary (leaderboard formula: mean of the five task types; Generalization = mean of standard and cluttered)

| | Success rate (%) | Progress score |
|---|---:|---:|
| Generalization | 26.7 | 30.0 |
| Precision | 7.5 | 11.9 |
| Long-Horizon | 11.2 | 26.8 |
| Memory | 53.3 | 57.0 |
| Open | 45.0 | 48.1 |
| **Overall (leaderboard formula)** | **28.8** | **34.8** |

Before re-evolution (all 42 tasks with the earlier tools): 21.2% / 27.4.

## Re-evolved tasks

| Task | Earlier tools | Re-evolved (frozen retest) |
|---|---:|---:|
| align_blocks | 5/10 | 9/10 |
| insert_tubes | 1/10 | 2/10 |
| play_tic_tac_toe | 1/10 | 6/10 |
| pour_liquid_into_cup | 1/10 | 3/10 |
| press_by_number | 0/10 | 10/10 |
| push_T | 3/10 | 4/10 |
| stack_blocks | 5/10 | 6/10 |
| stack_blocks_by_language | 3/10 | 7/10 |

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
| Generalization | pour_liquid_into_cup | 1/5 | 20.0 |
| Generalization | pour_liquid_into_cup_random | 2/5 | 40.0 |
| Generalization | push_T | 2/5 | 40.0 |
| Generalization | push_T_random | 2/5 | 40.0 |
| Generalization | sort_nesting_dolls_by_size | 1/5 | 20.0 |
| Generalization | sort_nesting_dolls_by_size_random | 0/5 | 0.0 |
| Generalization | stack_blocks | 4/5 | 83.0 |
| Generalization | stack_blocks_random | 2/5 | 49.0 |
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
| Long-Horizon | play_tic_tac_toe | 6/10 | 87.5 |
| Long-Horizon | put_bottles_into_dustbin | 0/10 | 28.0 |
| Memory | cover_blocks | 10/10 | 100.0 |
| Memory | imitate_sorting_sequence | 0/10 | 22.0 |
| Memory | match_and_pick_from_conveyor | 5/10 | 50.0 |
| Memory | press_by_number | 10/10 | 100.0 |
| Memory | swap_T | 7/10 | 70.0 |
| Memory | swap_blocks | 0/10 | 0.0 |
| Open | align_blocks | 9/10 | 90.0 |
| Open | classify_objects_by_language | 1/10 | 31.0 |
| Open | general_pickup | 8/10 | 80.0 |
| Open | pick_from_conveyor_by_image | 1/10 | 10.0 |
| Open | pour_by_language | 2/10 | 20.0 |
| Open | solve_equation | 8/10 | 80.0 |
| Open | stack_blocks_by_language | 7/10 | 74.0 |
| Open | store_tools_in_toolbox | 0/10 | 0.0 |
| Precision | build_tower | 0/10 | 1.0 |
| Precision | deposit_coin | 0/10 | 0.0 |
| Precision | fasten_screws | 0/10 | 10.0 |
| Precision | insert_key | 0/10 | 0.0 |
| Precision | insert_tubes | 2/10 | 44.0 |
| Precision | play_Xylophone | 0/10 | 0.0 |
| Precision | plug_in_charger | 0/10 | 0.0 |
| Precision | pour_balls_into_vase | 4/10 | 40.0 |
