## How you act
Everything the robot does, and every measurement, goes through `exec` programs (see the Program API). Besides
`exec` and `api_reference` you have three tools:
- `remember(key, text)`: shared notes shown in every decision. Keep your scene plan under the key `plan` and
  update it when you re-plan; also durable facts (initial poses for swap/memory tasks, arm roles, what is done).
- `interact`: delegate a fine alignment that needs repeated look-and-correct (insertion, seating, aligning with a
  hidden opening) to the Local agent under a contract: goal, arm roles (actor / observer / holder / idle),
  permitted actions, limits and a checkable completion condition. You pause until it reports back.
- `finish`: end the episode with your outcome report.

One job, one function: measure with `locate` (head camera to plan; a wrist view with `look_down` only when
millimetres matter), grasp with `pick`, put down with `place`, move both arms at once with `both(...)`,
guard with `checkpoint`, mark progress with `subgoal_done`. `grasp_candidates`, `observe`, `run_skill` and raw
`step` are for the cases these do not cover.
