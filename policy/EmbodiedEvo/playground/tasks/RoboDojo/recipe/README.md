# RoboDojo task recipes

These 42 task recipes are copied verbatim from
[RoboProbe/RoboProbe](https://github.com/RoboProbe/RoboProbe/tree/52300752080325086095854e01977a8cf281b684/policy/RoboDojo_Agent_L3_Inspect/recipes),
commit `52300752080325086095854e01977a8cf281b684`. `LICENSE` preserves the upstream
Apache-2.0 license; `manifest.json` preserves each original checksum.

New runs append the selected recipe to the copied instruction. Set
`EMBODIEDEVO_USE_RECIPE=0` to disable this before creating a run; the default is `1`.
Random variants reuse their base task recipe. Existing frozen inputs are unchanged.
The host-only `recipe.json` records the choice and source.

The upstream text sometimes refers to tools from its original agent. Those
references do not add primitives or observations to EmbodiedEvo. Only the four
functions in the handwritten primitive manual are available. Treat these recipes
as supplied task knowledge, separately from learned Skills/Lessons.
