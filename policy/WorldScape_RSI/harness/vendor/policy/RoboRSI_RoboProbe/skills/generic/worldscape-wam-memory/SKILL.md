# WorldScape WAM memory Skill

Read this interface before use. Call `call_tool` with `id="worldscape-wam-memory"`,
`arguments={}`, and optionally `stage="predict"`. The official instruction,
RGB and robot state are supplied by the Harness; do not supply task IDs,
coordinates, a checkpoint path or controller overrides.

This Skill runs the memory WSP checkpoint through the adjacent
`WorldScape_Policy_2` adapter. It proposes a complete native joint-action chunk;
the official evaluator executes actions and determines termination. The Agent
receives new RGB/state and execution feedback before choosing another Skill.
Choose this Skill from visible evidence and the current goal, without a fixed
task-to-model rule. Native movement Skills remain available in the same Agent.

Each observed executed action is counted. Complete chunks commit WAM state;
interrupted or uncertain predictions are discarded. Consecutive calls to the
same WAM retain its state. Native movement or switching WAM variants rebases
WAM context at the live observation, so stale prediction history is never
reused. Episode termination releases both model states. Read-only tools do
not break consecutive WAM context. A failed/ambiguous model call ends the trial
and is not automatically replayed. Checkpoint loading is lazy.

Both checkpoint bindings and their file hashes are frozen for the release.
Checkpoint weights are external assets, not text Skill files. A returned chunk
or committed model state does not prove grasp, task completion or native success.
