# Write a skill

This tutorial adds a skill, `wiggle_wrist(arm, amplitude, seconds)`, that rocks a
wrist joint around where it is: a bounded execution like every motion, so guards,
regions, the shield and handlers apply to it as to `move_eef`.

## 1. The manifest

`skills/wiggle/skill.toml`:

```toml
name = "wiggle"
summary = "Rock a wrist joint around where it is."
layer = "base"
mixin = "skills.wiggle:Wiggle"
exports = ["wiggle_wrist"]
order = 95
```

`layer = "base"` puts it in the agent's skills (and every ablation's); `needs = [...]` would name service
kinds it cannot run without (it needs none). `order` places it after `motion` (90) in
the session's method resolution and in the prompt. All keys:
[configuration reference](../reference/configuration.md#skills).

## 2. The mixin

`skills/wiggle/__init__.py`:

```python
"""Rock a wrist joint: a demonstration of a bounded execution."""

import math


class Wiggle:
    def wiggle_wrist(self, arm, amplitude=0.2, seconds=1.0):
        """Rock the last joint of `arm` by ±amplitude rad around where it is, one cycle
        per second, for `seconds`. Returns the execution trace; it ends COMPLETED, or
        with the event that stopped it (a guard, the shield, the region)."""
        joint = self._arm(arm)["joints"][-1]
        start = self._base_command(self.state)

        def command(state, tick):
            target = list(start)
            target[joint] += amplitude * math.sin(2 * math.pi * tick / self.hz)
            return target

        return self._execute("wiggle_wrist", {"arm": arm, "amplitude": amplitude}, command,
                             max_seconds=seconds, region=None, until=None, timeout_is_nominal=True)
```

What the episode core (`runtime/episode.py`) and the robot's session (`runtime/session.py`)
give a robot skill:

| | |
|---|---|
| `self._execute(skill, parameters, command_fn, max_seconds, region, until, timeout_is_nominal=False)` | runs `command_fn(state, tick)` every control tick until it returns `None` (completed) or an `Event`, a guard fires, `until` holds, the arm leaves `region`, or `max_seconds` passes; each command goes through the shield. Returns the trace; a non-nominal event goes to the program's handler or escalates. |
| `self._base_command(state)` | the measured arm joints with each gripper at its last command (holding still) |
| `self._arm(arm)` | the arm's joint group: `{"joints": [...], "gripper": index}` |
| `self.state`, `self.hz`, `self.profile` | the robot state, the control rate, the embodiment profile |
| `self.env.call(name, **arguments)` | a worker call: `observe`, `fk`, `ik`, `plan`, ... |
| `self._failed(skill, parameters, reason)` | a `SKILL_ERROR` trace, for a request the skill cannot carry out |
| other skills' methods | `self.move_eef(...)`, `self.segment_text(...)`, ... |

A skill module may also define `turn_report(session)`, which the session calls after every
program: it returns `{"text": ..., "images": {camera: data URL}}` (or `None`), the text
goes into the model's next turn and the images replace those cameras' plain views. The
tools by object id (`skills/eef`) use it to measure the located objects again every turn.

## 3. Tell the model

A base skill is enabled in every configuration, so every prompt pack must describe it (the
tests check that each public function of an enabled skill appears as a call). In
`prompts/robodojo/core.md`, next to the other motions:

```markdown
<!-- if: wiggle -->
  wiggle_wrist(arm, amplitude=0.2, seconds=1.0)   rock the wrist joint around where it is
<!-- end -->
```

and the same in `prompts/robotwin/core.md`. A configuration that should not have the skill
lists it in `[skills] disable = ["wiggle"]`; one that names its skills explicitly
(`names = [...]`) gets it only when named.

## 4. Check

```bash
python -m unittest discover -s tests -p test_skills.py
harness config configs/experiments/dev.toml   # wiggle  yes
```

Try it against a running worker without a model:

```bash
python scripts/dev_console.py --reset stack_bowls 0
python scripts/dev_console.py --code "print(wiggle_wrist('left_arm', 0.3, 2.0))"
```

`dev_console.py` builds a session with every layer against the worker at `--endpoint`
(default `http://127.0.0.1:8771`), runs the code and prints the executions.

## Skills that need a model

A skill that calls a model names the service kind in `needs` (all of them) or
`needs_any` (one of them) and reaches it through the session's clients
(`self._vision()` for the VLM, the segmenter, the policies). The resolution drops it on
machines without that service, and the prompt drops its text with it:

```toml
--8<-- "skills/vlm/skill.toml"
```

```python
--8<-- "skills/vlm/__init__.py"
```
