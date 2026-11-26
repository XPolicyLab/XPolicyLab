# RoboDojo's official evaluation

RoboDojo ran the frozen system on its own machines, on a hidden seed: 50 episodes per task (Generalization tasks: 25
standard and 25 cluttered). The 34 tasks outside Open use their tools in `tasks/`; the eight Open tasks use the
generic setup in `profiles/open/`. The overall score is the leaderboard's: the mean of the five task types,
Generalization being the mean of its standard and cluttered halves.

| Task type | Success rate | Progress score |
| --- | ---: | ---: |
| Generalization | 34.50% | 39.22 |
| Precision | 32.00% | 41.25 |
| Long-Horizon | 31.50% | 42.91 |
| Memory | 81.00% | 84.53 |
| Open | 55.25% | 57.97 |
| **Overall** | **46.85%** | **53.18** |

Generalization, standard / cluttered: success rate 38.00% / 31.00%, progress score 43.28 / 35.15.

| Type | Task | Success rate (%) | Progress score | Standard / cluttered success (%) |
| --- | --- | ---: | ---: | ---: |
| Generalization | arrange largest number | 22 | 30.2 | 12 / 32 |
| Generalization | fold clothes | 40 | 47.6 | 56 / 24 |
| Generalization | hang mugs | 4 | 19.0 | 4 / 4 |
| Generalization | make toast | 0 | 1.5 | 0 / 0 |
| Generalization | pack objects into box | 16 | 33.1 | 20 / 12 |
| Generalization | pour liquid into cup | 2 | 2.0 | 4 / 0 |
| Generalization | push T | 96 | 96.0 | 100 / 92 |
| Generalization | sort nesting dolls by size | 52 | 52.0 | 72 / 32 |
| Generalization | stack blocks | 88 | 89.5 | 88 / 88 |
| Generalization | stack bowls | 86 | 86.9 | 92 / 80 |
| Generalization | store laptop and headphones | 0 | 4.8 | 0 / 0 |
| Generalization | sweep blocks | 8 | 8.0 | 8 / 8 |
| Precision | build tower | 30 | 48.0 |  |
| Precision | deposit coin | 16 | 26.4 |  |
| Precision | fasten screws | 18 | 29.8 |  |
| Precision | insert key | 0 | 15.0 |  |
| Precision | insert tubes | 22 | 40.8 |  |
| Precision | play Xylophone | 58 | 58.0 |  |
| Precision | plug in charger | 32 | 32.0 |  |
| Precision | pour balls into vase | 80 | 80.0 |  |
| Long-Horizon | classify objects | 78 | 85.3 |  |
| Long-Horizon | fill egg holder | 0 | 8.8 |  |
| Long-Horizon | fill pen holder | 0 | 6.1 |  |
| Long-Horizon | make kong | 2 | 2.0 |  |
| Long-Horizon | organize table | 0 | 14.0 |  |
| Long-Horizon | play stacking toy | 18 | 57.6 |  |
| Long-Horizon | play tic tac toe | 96 | 99.0 |  |
| Long-Horizon | put bottles into dustbin | 58 | 70.5 |  |
| Memory | cover blocks | 100 | 100.0 |  |
| Memory | imitate sorting sequence | 42 | 63.2 |  |
| Memory | match and pick from conveyor | 74 | 74.0 |  |
| Memory | press by number | 96 | 96.0 |  |
| Memory | swap T | 96 | 96.0 |  |
| Memory | swap blocks | 78 | 78.0 |  |
| Open | align blocks | 78 | 78.0 |  |
| Open | classify objects by language | 72 | 79.4 |  |
| Open | general pickup | 82 | 82.0 |  |
| Open | pick from conveyor by image | 28 | 28.0 |  |
| Open | pour by language | 0 | 0.0 |  |
| Open | solve equation | 86 | 86.0 |  |
| Open | stack blocks by language | 96 | 96.4 |  |
| Open | store tools in toolbox | 0 | 14.0 |  |
