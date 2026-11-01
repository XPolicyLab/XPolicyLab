# The interface ladder

The agent and the ablations in `configs/ablations/` form a ladder (Tab. 3): the same
model, runtime, shield and call budget, with more of the interface on each rung. Each
ablation includes `configs/default.toml` and switches parts off; run one with
`--agent configs/ablations/<name>.toml`.

| Configuration | Mode | Interface | Services |
|---|---|---|---|
| `r0` | direct | one action per model call: move a fingertip, set a gripper, observe, wait | none |
| `r1` | direct | R0 plus fixed skills (locate, pick, place) and the learned policy | sam3, simate |
| `r2` | program | programs over the analytic skills; handlers | sam3 (+ the model's vlm) |
| `r3` | program | R2 plus learned skills (`run_policy` and the strategy step) | + simate |
| `default` (R4) | program | R3 plus the reactive layer (guards, controllers) | as R3 |

Ablations of the agent (Tab. 5): `no-handlers` (no `on()`), `no-verifier` (no success
verifier). `with-task-description` adds the task descriptions of Zhang et al. (2026b)
(Tab. 4's second row). `analytic-manipulation` replaces the default tools by object id
(`skills/eef`: `locate` by SAM 3 with object ids, `pick`, `release`, `transfer`, `push`)
with the manipulation skill's `pick` and `place` by name and its straight-line `push`;
R1's fixed skills are those too. On RoboTwin 2.0 the agent runs without learned skills
and with the manipulation skill (`configs/experiments/robotwin-dev.toml`).

Every rung inherits the agent's model, prompt pack (R0 and R1 have their own action
prompts), pointing by the VLM, SAM 3 segmenter, observation contract (XPolicyLab's:
what RoboDojo's official client hands a policy), 20 ms reactive budget and 30 model
calls. The program rungs with
learned skills (R3, the agent) start with the strategy step (`start`); R1 hands the
policy sub-tasks from its action vocabulary instead, wherever the arms are.

## Direct control (R0, R1)

In direct mode the model is called every step on purpose: these rungs are what the
program design is compared against. Each reply is one `<action>{JSON}</action>` or
`<finish>`:

| Action | Arguments | Rungs |
|---|---|---|
| `move` | `arm`, fingertip `position` (x, y, z), optional `orientation` (link6 quaternion), optional `gripper` | R0, R1 |
| `gripper` | `arm`, `value` (0 closed .. 1 open) | R0, R1 |
| `observe` | | R0, R1 |
| `wait` | `seconds` (0-10) | R0, R1 |
| `policy` | `name`, optional `instruction`, `arms`, `seconds` (0-30) | R1 |
| `locate` | `object` | R1 |
| `pick` | `object`, optional `arm` | R1 |
| `place` | `arm`, `target` or `position`, `clearance` | R1 |

Each action runs as a short fixed program over the same skills R2-R4 use
(`src/harness_policy/runtime/direct.py`), so the rungs differ in interface, not in
motion code.
