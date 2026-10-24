# RoboDojo task descriptions

One file per RoboDojo task: the official wiki capability dimension, Description
and process-score ladder, as used for the "with task description" condition of
Zhang et al. (2026b). Copied unchanged from RoboProbe
(https://github.com/RoboProbe/RoboProbe, Apache-2.0), commit
9ffacc54f372beef6479d61711bf4945e31ae4ce,
`policy/RoboDojo_Agent_L3_Inspect/recipes/<task>.md`. A few files end with a
`## Notes` section written by the RoboProbe authors rather than taken from the
wiki (imitate_sorting_sequence, make_kong, match_and_pick_from_conveyor,
play_tic_tac_toe, press_by_number, swap_blocks).

A harness enables them with `runtime.task_descriptions = "<directory>"`; the
driver appends the task's file to the first user message after the instruction.
