---
name: vla
description: Request an action prediction from the current robot observation.
---

Use `{{tool_name}}` when an action prediction would help the current task.

The optional `instruction` sets the prediction's subgoal. Omitting it preserves
the instruction in the current observation. Pass `options` only when the
registered tool schema declares them, using that schema's supported values.

The tool reads the current observation and returns the caller's serialized
prediction, such as an action summary or a proposal ID. Treat it as a proposal:
prediction does not execute actions, move the robot or establish task success.
Use the separately registered execution or review tools when the policy requires
them, then inspect actual feedback before predicting again.

Input errors may be corrected and retried. A timeout means the client stopped
waiting; remote inference may still run. Let the caller handle connection,
lifecycle and inference failures. Do not assume the tool resets model state.
