# Task recipes

The 42 recipe files have their checksums recorded in `manifest.json`.

`knowin_agent.knowledge.task_brief` includes the recipe of a task when the switch `knowledge.recipe`
is on (the default) and verifies its checksum against `manifest.json` first. Random variants
reuse their base task recipe. Where a recipe disagrees with the scoring notes, the scoring
notes win.

A recipe may name unavailable tools; only the API described in the task prompt exists.
Treat these recipes as supplied task knowledge, separately from learned skills and lessons.
