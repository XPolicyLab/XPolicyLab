# The Harness

The Harness (formerly `harness-policy`) is the code of the paper *Harness Policy*.

A language model writes **programs**, and the program is the policy, on top of a learned
visuomotor policy (a VLA). At the start of a task the model reads the learned policy's
description and decides who does the task:

- **the learned policy**, handed the whole task from the start pose, the setting it was
  trained in, and left to run (strategies A and C): every task it learned, also when the
  instruction names colours, an order or categories;
- **the model's own programs** (strategy B), where the task needs what the policy cannot
  do: picking the one object out of several that a name, a number or label on it, or a
  picture identifies, among objects that stand still.

A program perceives, moves, makes at most one contact with an object, checks the result
and ends; the model looks at the scene measured again and writes the next. Guards and
controllers run every control tick, and a fixed shield checks every command before it
reaches the robot. Only what escapes the program goes back to the model.

```python
# What DeepSeek V4.1 Flash wrote for "Stack the three bowls together." (RoboDojo,
# evaluation pack 0), after one program that looked: the policy learned this task.
# PLAN: three white bowls (o1 left, o2 center, o3 right) — task is nesting/stacking bowls.
# Policy lists "nesting bowls or cups" under "Often does part and leaves the rest" -> strategy C.
start("C", "stacking/nesting three bowls together")
```

Simate stacked the bowls in 230 control steps; the episode took two model calls and
$0.0025. The [Quick tour](getting-started/quicktour.md) follows this episode and one the
programs did themselves, from the first model request to the outcome.

## What is in the box

| | |
|---|---|
| **One agent** | `configs/default.toml` (prompt pack, skills, services, budget) with a model of `configs/models/`, resolved per machine into a lock whose hash is the identity; the paper's ablations include it. Evaluation runs resolve strictly. |
| **Division of labor** | `start(strategy, kind)` is the first motion: A or C hands the task to the learned policy (Simate by default) from the untouched start pose; B keeps it for the programs. After the policy's run the model has 4 calls, then the policy runs on to the end of the episode. |
| **Skills** | perception (SAM 3 segmentation, pointing by the VLM or Molmo2, questions about images), geometry, kinematics and planning, motion, the tools by object id (`locate`, `pick`, `release`, `transfer`, `push`, `move_tip`, with a scene measured again after every program), learned policies (`run_policy`: any XPolicyLab policy), a world-action model (`predict_future`, OpenWAM, for experiments), guards and controllers every tick, event handlers. The agent turns them on by interface layer or by name. |
| **Services** | each model behind a skill is a service with a manifest: how to start it, its port, its VRAM and where its weights come from. `harness up` starts them; on a small machine they load on demand and share a GPU. |
| **Benches** | RoboDojo (dual-arm ARX X5; 42 scored tasks in five dimensions; development layouts disjoint from the evaluation packs). |
| **Ladder** | R0 (one action per model call) to R4 (programs, learned skills, reactive layer, handlers). |
| **Evaluation by others** | develop under XPolicyLab's observation contract and export a policy folder RoboDojo's leaderboard runs. |
| **Traces** | every model request, program, event and action, an HTML viewer, and RoboDojo scores. |

## Where to start

<div class="grid cards" markdown>

- **Get started**

    [Installation](getting-started/installation.md) ·
    [Quickstart](getting-started/quickstart.md) ·
    [Quick tour](getting-started/quicktour.md)

- **Learn by doing**

    [Run the agent on RoboDojo](tutorials/run-the-agent.md) ·
    [Write a skill](tutorials/write-a-skill.md) ·
    [Submit to the leaderboard](tutorials/submit-xpolicylab.md)

- **Understand it**

    [Programs and decision points](concepts/design.md) ·
    [The episode runtime](concepts/runtime.md) ·
    [Configuration and identity](concepts/configuration.md)

- **Look things up**

    [CLI](reference/cli.md) ·
    [Configuration keys](reference/configuration.md) ·
    [Skills](reference/skills/index.md) ·
    [Services](reference/services/index.md)

</div>
