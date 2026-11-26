The top-level code-policy backend is already selected. Choose one of its
registered sub-skills for the original task using current camera images,
observed robot state and available_skills. Do not select pi05 or a backend.
Compare its stated task, geometry and sensing requirements against the evidence;
do not infer broad capabilities from one successful development example.

Shared visual and geometry capabilities are listed in available_tools. Decide
whether the current task needs any of them: tracked points, monocular depth,
public-plane calibration, multi-view localization, geometric constraints or
action verification. Request only tools marked available. No user enable flag
is required. Do not request extra tools solely because they exist.

Return only JSON:
{"skill_id": "one exact available_skills key", "reason": "evidence-based choice", "tools": []}

If the selected skill's existing generated-program controller is sufficient,
leave tools empty. Otherwise list the needed tool names, for example
["estimate_depth", "mark_points", "plan_pose"]. The shared tool loop will then
receive that skill's frozen experience/program and current observations; you
choose each tool call as needed. Tool results are estimates/evidence, never an
official task verdict. Omitting tools also keeps the existing controller path.

If no candidate is suitable, use skill_id "__no_matching_skill__" and explain
why. The caller will reject this sentinel and stop without actions or fallback.
Do not invent another skill, a file path, code, actions or a success verdict.
The chosen skill is bound for this episode. A separate generation call will
receive its frozen experience and successful source plus the current scene.
