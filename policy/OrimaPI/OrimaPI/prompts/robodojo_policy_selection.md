Select the execution policy for a robot episode using the original task,
current camera images, robot proprioception and available_policies.

Each candidate ID names a registered policy with a declared capability and action
contract. Compare these capabilities with the observed task requirements. Treat
missing visual evidence as uncertainty; do not invent capabilities or assume a
checkpoint succeeds on every task. Candidates have already been filtered for
compatibility with the configured subgoal-supervision mode.

Return exactly one JSON object with two nonempty string fields:
- "policy_id": one exact key from available_policies.
- "reason": a brief explanation grounded in the task, observations and capabilities.

Only the selected policy will be loaded. Its identity is fixed until the episode
is reset. Do not return subgoals, actions, code, checkpoint paths, or new policy
definitions. After loading, a separate supervision request will provide the bound
policy's capabilities and ask for the initial subgoal.
