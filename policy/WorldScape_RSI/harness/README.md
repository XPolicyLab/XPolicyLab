# Harness implementation

See [the policy README](../README.md) for installation, checkpoint bindings,
provider profiles and validation scope.

The execution path is `model.py` → one `AgentBackend` → `SkillEefPolicy`.
The same Agent and tool set are used for every public task. WAM variants are
reviewed code Skills in `vendor/policy/RoboRSI_RoboProbe/skills/generic/`.
`tools.py` dispatches their ids to `AgentBackend._wam_call`; `wam.py` owns
checkpoint/source verification and native explicit commit/discard lifecycle.
`configs/inner_agent.json` freezes the Agent. `configs/wam_skills.json` freezes
the two WSP variants. There is no executor-selection table or learned router.

WAM receives only observations filtered by `contracts.py`. Its joint-action
proposal is checked as a complete chunk before any action is exposed. A final
execution observation commits the chunk before it is passed into WSP's next
history window (memory Skill only). The no-memory Skill samples only the current
observation and resets model context before each prediction; it never commits
causal memory. Partial/uncertain execution discards the prediction. Switching
to native motion or another WAM resets the old context; inactive models do not
accumulate stale observations. Read-only Skill/tools calls preserve continuity.
Trial end closes every loaded model, including after failures.

`release.py` freezes code, documents, configuration and integration entry points.
`development.py` imports offline Skill-asset revisions with evidence hashes
into a separate candidate; it does not conduct recursive experiments or training.
`verify_checkpoints.py` verifies both external weight directories without GPU
inference. It must run again when binding the release on another machine.
