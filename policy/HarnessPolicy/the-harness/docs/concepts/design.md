# Design: programs and decision points

A one-page mental model for collaborators reading the code or the run traces.

## The idea

Most LLM robot agents keep the model in the loop: look, emit an action (or one skill
call), look again. Here the model is a **programmer, not a controller**: it writes
programs, and a program is a policy for one step of the task. It looks (locate, the
cameras), moves through free space, makes at most one contact with an object (a pick,
push, release, transfer or press), checks and prints the result, and ends. What runs
below the model inside a program (bounded skills, guards and controllers every tick,
handlers, the shield, learned policies through `run_policy`) never waits for it.

The model is asked again at decision points (the paper's §3.2): after each such step,
when something escapes the program, or when what comes next depends on what the robot
has not seen yet (a demonstration, a hidden object, the scene a learned policy left).
Each time it gets the program's output, the scene measured again (every located object
labelled, outlined in the camera images, with each holding arm's grip), the env steps
used and fresh camera images, and writes the next program from where the robot is.

This is how Z2_v01, our best step-by-step harness, worked (one tool call per turn,
looking at what each call did). Version 1.2.0 of the agent took its tools, its
every-turn scene, its start step and its working method into the programs; 1.0.0 had
told the model to write a program for as much of the task as it could foresee, and in
the development comparison those programs looped over several objects without looking
(classify_objects: four programs killed at the 30-minute cell limit with nothing
checked).

## Division of labor: the program and the learned policy

A learned visuomotor policy (Simate in the default agent) was trained on whole RoboDojo
tasks, each demonstration starting at the start pose with the task's own instruction.
It does best in exactly that setting, and it does what the programs' tools do badly:
skilled contact, many times a second, with a memory of past frames. What it cannot do is
read: pick the one object out of several that a name, a number or label written on it,
or a picture shown as the target identifies. So the model's first motion is a choice,
`start(strategy, kind)`, made from the policy's description (`list_policies()`) and the
first images:

| Strategy | When (the prompt's rule) | What happens |
|---|---|---|
| A | a task the policy learned, of a kind its description says it usually finishes | the policy runs the whole task at once, from the untouched start pose; the model supervises |
| C | a task the policy learned, of any other kind | the policy runs first, then the programs finish what it leaves |
| B | picking one object out of several by what identifies it (a name, a number or label, a picture), among objects that stand still | the programs' own tools only |

Everything else goes to the policy, also when the instruction names colours, an order
or categories, or when objects are matched to each other by how they look: those
conditions are part of what it learned. Moving objects go to the policy whatever
identifies the target (the tools are too slow for a conveyor). After B, the model may
still hand the task over (`start("A" or "C", kind)` from the start pose) when its own
tools fail early at a task the policy learned.

The handed-over task runs uninterrupted until the episode ends or the policy stops (its
arms still for 5 s once they have moved). Then the model has 4 calls
(`post_policy_calls`) for what is clearly left, after which the policy runs on by itself
to the end of the episode. Watching the policy and taking over early bought nothing:
1.3.0, which ran the handed-over task in 10 s slices with a look between them, averaged
31.9 on those tasks against Simate alone's 42.4, and the model's own tools after a run
added cost, not score. The [changelog](../changelog.md) has the evidence for each rule.

The result is few model calls: on RoboDojo's evaluation pack 0 (agent 1.4.1, 42 tasks,
three layouts each) 113 of 125 episodes were handed over, at 2.3 calls and $0.003 an
episode; the 12 the programs kept took 16 calls and $0.08. The agent scored 26.2
(RoboDojo average) against 22.2 for Simate alone on the same layouts.

## Who handles a problem, in order

1. **The program itself**, with the analytic skills (perception, the tools by object
   id, `move_eef`, ...), its own checks and retries, the reactive layer (guards that
   run every control tick) and the handlers it registered with `on(event, handler)`.
2. **Learned skills, when the program calls them**: the task handed over with `start`,
   a VLA through `run_policy`, a world-action model through `predict_future`. The
   runtime does not fall back to them by itself.
3. **The model**, only for what the program could not handle. It then writes the next
   program, continuing from where the robot is.

The model is called on these reasons only: `start`, an unhandled event (`escalation`),
an `error`, a program that ended without `finish()` (`program_end`), and replies that
need a follow-up (`verify`, `format`).

The exception is the direct-control rungs of the interface ladder (R0/R1,
`src/harness_policy/runtime/direct.py`): there the model is called every step
(`action`) on purpose; they are the interfaces this design is compared against.

## Where it lives in the code

| Part | Code |
|---|---|
| High level: when the model is called, and why | `src/harness_policy/runtime/driver.py` (`run_program_episode`) |
| Runs the model's programs; variables persist across calls | `src/harness_policy/runtime/kernel.py` |
| The API programs call (`locate`, `pick`, `start`, `run_policy`, `on`, `predict_future`, `run_controller`, `finish`, ...) | `skills/<name>/`, composed by `src/harness_policy/runtime/session.py` |
| The strategy step and the learned policies | `skills/vla/` (`start`, `run_policy`, `list_policies`), `services/<name>.toml` (`client.description`) |
| Reactive layer (guards and controllers every tick) | `src/harness_policy/runtime/reactive.py`, `skills/reactive/` |
| Safety shield, outside the model's reach | `src/harness_policy/runtime/shield.py` (runs in the simulator worker) |
| What the model is told | `prompts/robodojo/`: the API (`core.md`, `learned.md`), the way of working with the tools (`method.md`: plan, one contact then look), the division of labor (`rules_learned.md`) |

More detail: [The episode runtime](runtime.md).

## Reading a run with this in mind

- `summary.json`: `strategy` (the choice and the kind it matched), `model_calls` and
  `decisions` (calls by reason). A handed-over task usually takes two calls, `start`
  and one `program_end`, and ends with the episode; a task the programs keep takes about
  one call per contact. `events_handled` counts events the program dealt with itself;
  `events_escalated` counts those that went to the model.
- `events.jsonl`: every `model_request` carries its `reason`; every `cell` is one
  program run, with the skill events it produced and the scene report after it.
- `kernel.log`: one JSON line per tool call (`locate`, `pick`, ...) with its result.

Many `error` or `escalation` calls in an episode are a signal in themselves: the
program is not absorbing its own failures, and that is usually the thing to fix (a
tool, a check, a handler), not the prompt.

For reference: the step-by-step loop this replaced (grok-4.7, development layouts) used
34 calls and 823k input tokens per episode; agent 1.4.1 used 3.6 calls and 57k input
tokens on average over the 125 evaluation episodes above.
