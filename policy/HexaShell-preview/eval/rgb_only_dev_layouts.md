# HexaShell-preview: frozen tools on the seed-0 development layouts

Observation: the stock `arx_x5` config (three RGB cameras, joint states, end-effector poses) and the fixed camera calibration. No depth is used: neither the simulator's depth nor a depth network. Positions come from calibrated pixel rays intersected with planes whose height is measured in the episode (the table plane is measured from the images once per episode, the first time it is needed) and from two-view triangulation.

Protocol: seed 0, development layouts 0-9 (Generalization tasks: standard layouts 0-4 and cluttered `_random` layouts 0-4), one episode per layout, frozen tools, no optimizer, official success and progress checkers. The tools were evolved on these same layouts, so these numbers are optimistic for unseen layouts.

## Summary (leaderboard formula: mean of the five task types; Generalization = mean of standard and cluttered)

| | Success rate (%) | Progress score |
|---|---:|---:|
| Generalization | 34.2 | 37.7 |
| Precision | 21.3 | 28.4 |
| Long-Horizon | 18.8 | 32.2 |
| Memory | 73.3 | 73.3 |
| Open | 43.8 | 46.6 |
| **Overall (leaderboard formula)** | **38.3** | **43.6** |

Seven tasks (`deposit_coin`, `fill_egg_holder`, `fill_pen_holder`, `hang_mugs`, `insert_key`, `pick_from_conveyor_by_image`, `store_tools_in_toolbox`) originally carried fixed object geometry in their tools (part sizes, templates of fixed scene parts). That geometry was removed and replaced by measurements made in the episode; these seven tasks were then retested with the same protocol, without further development. In 5 of the 10 `fill_egg_holder` episodes a new measurement command returned a numpy boolean that the server could not log, which ended the episode; they count as failures here, and the type conversion is fixed in the shipped tools.

## Per task (10 episodes each; Generalization tasks 5 standard + 5 cluttered)

| Type | Task | Success | Progress score |
|---|---|---:|---:|
| Memory | `cover_blocks` | 10/10 | 100.0 |
| Generalization | `push_T` | 10/10 | 100.0 |
| Memory | `swap_T` | 10/10 | 100.0 |
| Long-Horizon | `play_tic_tac_toe` | 9/10 | 97.5 |
| Memory | `press_by_number` | 9/10 | 90.0 |
| Open | `stack_blocks_by_language` | 9/10 | 92.0 |
| Open | `align_blocks` | 8/10 | 80.0 |
| Memory | `imitate_sorting_sequence` | 8/10 | 80.0 |
| Generalization | `stack_blocks` | 8/10 | 81.5 |
| Generalization | `stack_bowls` | 8/10 | 83.0 |
| Open | `general_pickup` | 7/10 | 70.0 |
| Memory | `match_and_pick_from_conveyor` | 7/10 | 70.0 |
| Precision | `pour_balls_into_vase` | 7/10 | 70.0 |
| Open | `solve_equation` | 7/10 | 70.0 |
| Generalization | `fold_clothes` | 6/10 | 68.0 |
| Precision | `play_Xylophone` | 6/10 | 60.0 |
| Long-Horizon | `classify_objects` | 4/10 | 56.5 |
| Open | `classify_objects_by_language` | 4/10 | 48.0 |
| Generalization | `sort_nesting_dolls_by_size` | 4/10 | 40.0 |
| Generalization | `arrange_largest_number` | 3/10 | 45.5 |
| Precision | `insert_tubes` | 3/10 | 38.0 |
| Long-Horizon | `put_bottles_into_dustbin` | 2/10 | 49.0 |
| Precision | `build_tower` | 1/10 | 25.0 |
| Generalization | `make_toast` | 1/10 | 10.0 |
| Generalization | `pour_liquid_into_cup` | 1/10 | 10.0 |
| Precision | `deposit_coin` | 0/10 | 2.0 |
| Precision | `fasten_screws` | 0/10 | 17.0 |
| Long-Horizon | `fill_egg_holder` | 0/10 | 1.0 |
| Long-Horizon | `fill_pen_holder` | 0/10 | 7.5 |
| Generalization | `hang_mugs` | 0/10 | 0.0 |
| Precision | `insert_key` | 0/10 | 15.0 |
| Long-Horizon | `make_kong` | 0/10 | 0.0 |
| Long-Horizon | `organize_table` | 0/10 | 30.0 |
| Generalization | `pack_objects_into_box` | 0/10 | 8.0 |
| Open | `pick_from_conveyor_by_image` | 0/10 | 0.0 |
| Long-Horizon | `play_stacking_toy` | 0/10 | 16.0 |
| Precision | `plug_in_charger` | 0/10 | 0.0 |
| Open | `pour_by_language` | 0/10 | 13.0 |
| Generalization | `store_laptop_and_headphones` | 0/10 | 6.0 |
| Open | `store_tools_in_toolbox` | 0/10 | 0.0 |
| Memory | `swap_blocks` | 0/10 | 0.0 |
| Generalization | `sweep_blocks` | 0/10 | 0.0 |
