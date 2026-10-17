# Task recipes

The 42 task recipes in this directory are copied byte for byte from
[RoboProbe](https://github.com/RoboProbe/RoboProbe/tree/52300752080325086095854e01977a8cf281b684/policy/RoboDojo_Agent_L3_Inspect/recipes),
commit `52300752080325086095854e01977a8cf281b684`, under the Apache-2.0 license kept in `LICENSE`.
`manifest.json` records the source and the SHA-256 of every file; `roboshell/recipe.py` refuses a file that does not match.

Each recipe is the RoboDojo wiki text of one task: capability dimension, description and score table.
Six of them (`imitate_sorting_sequence`, `make_kong`, `match_and_pick_from_conveyor`, `play_tic_tac_toe`,
`press_by_number`, `swap_blocks`) end with a `## Notes` section written by RoboProbe, not taken from the wiki.
Some notes name commands of RoboProbe's own agent (`give_up`, holding a dimension at its current value, sampled watch frames);
they add no command or observation here, and the text is kept unchanged.

The acting agent's prompt is the official instruction followed by `TASK RECIPE:` and the recipe of its task
(`_random` variants use the base task's recipe), the same text the GPT-6 Astra L3 Inspect EEF baseline and EmbodiedEvo give their agents.
`ROBOSHELL_RECIPE=0` leaves the prompt as the bare instruction. Do not edit these files.
