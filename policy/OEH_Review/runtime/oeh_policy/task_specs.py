"""Public task descriptions only; no task-native observation, counter or reducer."""

PRESS_TASK_SPEC = {"id":"press_by_number-public-v1", "source_pin":"ee67a1468510da7624a089164402359f2afc72c8",
    "protocol":"red0 full press/release cycles n0 times -> blue full cycle -> red1 full cycles n1 times -> blue full cycle -> both arms home",
    "global_counts":"Exactly n0/n1/2 respective press edges; excess cannot be repaired by more presses",
    "role_assumption":"Naive red0/red1 left-to-right; cards above the red buttons. Verify visually, never read layout metadata.",
    "release_note":"Full visible spring return before another press; a touch or commanded downstroke is not proof."}
