# Skills and prompt packs

A **skill** is what a model's programs can call. A **prompt pack** is what the model is
told about them. Both follow the same switch: the agent turns skills on, a skill whose
service is missing drops out, and the prompt drops its documentation with it, so the
model never reads about an API it cannot call.

## A skill

```text
skills/<name>/
├── skill.toml     name, summary, layer, exports, what it needs
└── __init__.py    a mixin class with the methods
```

```toml
--8<-- "skills/pointing/skill.toml"
```

A skill runs on the embodiment kinds its `embodiments` lists (default `["robot"]`; every
skill so far is a robot's). The kernel's session class is the embodiment's session plus
**every** skill's mixin for that kind, so one skill's methods may use another's (`pick`
uses `segment_text`, `move_eef`, `holding`). The enabled skills decide only which names a
program sees. Skills for different kinds may export the same names (each kind has its
own `finish`, say): they are never enabled together. The keys of `skill.toml` are
in the [configuration reference](../reference/configuration.md#skills); the skills
themselves, with their functions, in the [skills reference](../reference/skills/index.md).

## Which skills a run gets

1. **Requested**: the agent's `[skills] names = [...]`, or the skills of its interface
   layers for the run's embodiment kind: every `base` skill, plus `learned` (vla, wam),
   `reactive`, `handlers` when those layers are on (defaults: learned off, reactive and
   handlers on). A named skill that does not run on the kind drops ("does not run on the
   desktop embodiment"). An `optional`
   skill is in no layer's set: `enable = [...]` adds it. Minus `disable = [...]`. Two
   skills that list each other in `conflicts` (they export some of the same names, such
   as `eef` and `manipulation`) are never enabled together: requesting both is an error.
2. **Available services**: segmenter, pointer, vlm and verifier when the configuration has
   those roles or services and the machine runs them; `policy` and `world_model` when it
   has learned policies or world models.
3. **Resolution**: a skill drops when a service kind in its `needs` is missing, when
   none of its `needs_any` is available, or when a skill it `requires` dropped;
   repeated until nothing changes. A `required` skill (core, geometry, kinematics,
   grasping, memory, verify, motion) that cannot run is an error.

`harness config` shows the result and the reason for each dropped skill:

```text
skill         on   why
pointing      no   needs one of the services pointer, vlm
segmentation  no   no segmenter service
eef           no   no segmenter service
reactive      no   not in this configuration
```

The enabled skills and the dropped ones with their reasons are part of the configuration's
lock, so a run on a machine that lacks a service has another identity
([Configuration and identity](configuration.md)).

## Prompt packs

A pack is a directory, e.g. `prompts/robodojo/`, whose `pack.toml` lists sections in
order; a section with `skills` is included when one of them is enabled:

```toml
--8<-- "prompts/robodojo/pack.toml"
```

Inside a section, markers keep text only when every named skill or feature is on:

```markdown
<!-- if: segmentation -->
  segment_text(observation, prompt, camera="cam_head") -> [Mask]   instance masks, best first, or []
  ...
<!-- end -->
```

(`prompts/robodojo/core.md`; an `<!-- else -->` branch gives text for when the skill is off.)

Markers nest; marker lines never reach the model. Besides skills, two **features** of
the robot connection select text: `measured_gripper` (the gripper readings are measured
openings) and `step_limit` (the episode's step limit is reported). Both hold under the
native contract and neither under XPolicyLab's
([Evaluation through XPolicyLab](xpolicylab.md)).

The tests check every pack against every combination of optional skills: each public
function of an enabled skill must appear as a call in the prompt (except names listed
as `undocumented`), no function of a disabled skill may, and no marker may remain.

| Pack | Used by |
|---|---|
| `prompts/robodojo/` | the agent and the program ablations (R2, R3) on RoboDojo |
| `prompts/robodojo_direct*.md` | R0 and R1 (files, not a pack) |
| `prompts/robodojo_tasks/` | task descriptions (`runtime.task_descriptions`) |

## Why the API documentation lives in the pack

A pack describes one embodiment: its frames, reach, timings and numbers (*"a 15 cm move
takes about 33 ticks"*). The skills' code is the same for every embodiment. The
skills reference shows, for each skill, the text the RoboDojo pack gives the model.
