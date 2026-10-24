<h1 align="center">The Harness</h1>

<p align="center">
  <b>A model writes programs. The program is the policy.</b>
</p>

<p align="center">
  <a href="https://github.com/EvolvingLMMs-Lab/the-harness/actions/workflows/docs.yml"><img alt="docs" src="https://github.com/EvolvingLMMs-Lab/the-harness/actions/workflows/docs.yml/badge.svg"></a>
  <img alt="python" src="https://img.shields.io/badge/python-3.11-blue">
</p>

> **The Harness** (formerly `harness-policy`) is the code of the paper *Harness Policy*.
> The Python package (`harness_policy`) and the CLI (`harness`) keep their names.

The Harness lets a language model control a robot by writing programs, on top of a
learned visuomotor policy (a VLA). At the start of a task the model reads the policy's
description and decides who does the task: the policy, handed the whole task from the
start pose as it was trained, or the model's own programs, where the task needs what the
policy cannot do (picking the one object a name, a label or a picture identifies). A
program perceives, moves, checks its own results and retries through bounded skills.
Guards and controllers run every control tick, and a fixed shield checks every command
before it reaches the robot. Only what still escapes the program goes back to the model.

```python
# Programs DeepSeek V4.1 Flash wrote on RoboDojo (agent 1.4.1, evaluation pack 0);
# the comments after "->" are ours.

# "Stack the three bowls together." A task the policy learned: hand it over whole.
# Policy lists "nesting bowls or cups" under "Often does part and leaves the rest" -> strategy C.
start("C", "stacking/nesting three bowls together")   # -> Simate stacks them: success, 2 model calls

# "Pick up the blue race toy car by 10 cm." A name picks the object: the program does it.
start("B", "pick up the blue race toy car by 10 cm")
res = locate("blue toy car; blue race car")           # -> o1, seen in two cameras, 10 x 6 cm
# ... next program, after looking at the outlined images:
r = pick("right_arm", "o1", lift=0.10)                # -> "rose with the jaws (measured with SAM 3): held"
```

## Why The Harness

- **The model writes code, not actions.** It looks, then hands the task over or writes
  short programs, one contact with an object each. On RoboDojo's evaluation pack 0 an
  episode took 3.6 model calls on average (median 2) and $0.010 with DeepSeek V4.1 Flash.
- **It adds to the learned policy.** The policy runs what it was trained on, whole and
  uninterrupted; programs do what it cannot. With Simate, agent 1.4.1 scored 26.2
  (RoboDojo average) against 22.2 for Simate alone on the same 42 tasks and layouts.
- **Skills are packages, models are services.** The agent turns skills on by name or by
  interface layer; a skill whose model is missing drops out, and the prompt drops its
  text with it. Any XPolicyLab policy becomes a learned skill with one manifest file.
- **It runs on the machine you have.** A deploy file places the models: one GPU each,
  several on demand sharing one GPU, or none (hosted models). Placement never changes
  what a run computes.
- **Results name what produced them.** Every run records a lock and an identity: the
  prompt, skills, models and settings. Evaluation runs refuse to run a smaller
  configuration.
- **Others can evaluate it.** Develop under XPolicyLab's observation contract and export
  a policy folder RoboDojo's leaderboard runs as is.

## Install

```bash
bash environment/setup.sh            # the harness-policy conda environment (Linux, CUDA 12.8)
source environment/activate.sh       # also reads ~/.harness_env (OPENROUTER_API_KEY, ...)
```

The simulator checkouts and their revisions are in `third_party/sources.toml`
(`python scripts/install_plan.py robodojo` prints the steps). The configuration, the CLI's
`config`, `lock` and `report`, and the tests need only Python 3.11.

## Quick tour

```bash
cp configs/deploy/example-1gpu.toml configs/deploy/mine.toml   # set GPUs and paths
harness doctor --deploy mine                                   # can this machine run it?
harness up --deploy mine                                       # start the models it runs
harness config configs/experiments/dev.toml --deploy mine      # skills on and off, identity
harness run configs/experiments/dev.toml --deploy mine --output runs/dev
harness report runs/dev                                        # RoboDojo Score and success rate
python scripts/view_runs.py runs                               # HTML pages next to the traces
```

## The agent and its ablations

`configs/default.toml` and `configs/ablations/` (the paper's interface ladder and ablations):

<!-- generated: agent -->
| Configuration | What it is |
|---|---|
| `default` | Harness Policy: programs over analytic and learned skills, reactive layer, handlers, verifier |
| `ablations/analytic-manipulation` | with the manipulation skill's pick, place and push in place of the tools by object id |
| `ablations/no-handlers` | without handlers |
| `ablations/no-reactive` | Reactive layer A/B: programs without the reactive layer, no learned policy |
| `ablations/no-verifier` | without the verifier |
| `ablations/r0` | R0: direct control, one action per call |
| `ablations/r1` | R1: fixed skills, one action per call |
| `ablations/r2` | R2: programs over analytic skills, with handlers |
| `ablations/r3` | R3: R2 plus learned skills (the learned policy) |
| `ablations/reflexes` | Reactive layer A/B: programs with the reactive layer and reflexes (intercept), no learned policy |
| `ablations/with-task-description` | with the task description of Zhang et al. (2026b) |
<!-- /generated: agent -->

## Skills

<!-- generated: skills -->
| Skill | Layer | Exports | Needs |
|---|---|---|---|
| core | base | `finish`, `get_state`, `get_observation`, `show`, `ARMS`, `CONTROL_HZ` | – |
| geometry | base | `transform`, `mask_to_plane`, `down_with_yaw`, `project`, `pixel_ray`, `pixel_to_plane`, `triangulate`, `TABLE_Z`, `FORWARD`, `DOWN`, `link6_for_tip`, `fingertip_of` | – |
| kinematics | base | `fk`, `ik`, `plan_path` | – |
| pointing | base | `point_to`, `localize`, `localize_3d` | one of pointer, vlm |
| vlm | base | `query_vlm` | vlm |
| segmentation | base | `segment_text`, `segment_point` | segmenter |
| grasping | base | `plan_grasp` | – |
| memory | base | `recall` | – |
| verify | base | `verify` | – |
| motion | base | `move_joints`, `move_eef`, `open_gripper`, `close_gripper`, `holding`, `hold`, `home` | – |
| manipulation | base | `pick`, `place`, `push` | segmenter; one of pointer, vlm |
| eef | base | `locate`, `pick`, `release`, `transfer`, `push`, `move_tip` | segmenter |
| reactive | reactive | `when`, `watch`, `clear_guards`, `run_controller` | – |
| scene | base | `watch_scene`, `clear_view`, `press` | one of pointer, vlm |
| reflexes | reactive | `intercept` | segmenter |
| handlers | handlers | `on` | – |
| vla | learned | `start`, `run_policy`, `list_policies` | policy |
| wam | learned | `predict_future`, `list_world_models` | world_model |
| mcp | base | `mcp_tools`, `mcp_call` | mcp |
<!-- /generated: skills -->

## Services

<!-- generated: services -->
| Service | Kind | What it is | VRAM |
|---|---|---|---|
| dm05 | policy | DM0.5 VLA with a memory of past frames (Dexmal DM05-MEM-Robodojo-Sim), bf16, through XPolicyLab's OpenDM adapter | 13 GB |
| g05 | policy | G0.5 VLA, RoboDojo's own checkpoint (joint actions, both arms), through XPolicyLab's G05 adapter | 16.4 GB |
| kinematics | kinematics | ARX X5 kinematics (fk, ik, plan_path) through RoboDojo's cuRobo planner, without the simulator | 2 GB |
| mcp-units | mcp | An example MCP tool server (unit conversion) for the mcp skill | – |
| molmo2 | pointer | Molmo2-8B open-vocabulary pointing (point_to, localize, localize_3d), int8 with the vision tower in bf16 | 12.7 GB |
| openwam-policy | policy | OpenWAM-alpha world-action model run as a policy (end-effector actions), RoboDojo's own checkpoint, through XPolicyLab's OpenWAM adapter | 15.3 GB |
| openwam | world_model | OpenWAM world-action model trained on RoboDojo (predict_future): predicted camera views and end-effector motion | 15.3 GB |
| pi05 | policy | pi0.5 VLA, RoboDojo's own checkpoint, through XPolicyLab's Pi_05 adapter (openpi, JAX) | 16 GB |
| qwen3-vl-32b | llm | Qwen3-VL-32B-Instruct in 4 bit, self-hosted: the development model that writes programs (and answers VLM questions) | 21 GB |
| qwen3-vl-8b | vlm | Qwen3-VL-8B-Instruct, self-hosted: questions about images and pointing for development runs without API spend | 18 GB |
| sam3 | segmenter | SAM 3 instance segmentation from text or point prompts (segment_text, segment_point; inside pick and place) | 7.2 GB |
| simate | policy | Simate-beta VLA with a memory of past frames, RoboDojo's own checkpoint, through XPolicyLab's Simate_beta adapter | 14 GB |
<!-- /generated: services -->

## Documentation

The documentation is a MkDocs site built from `docs/` (`mkdocs serve` after
`source environment/activate.sh`; CI builds it on every push).

| | |
|---|---|
| Get started | [Installation](docs/getting-started/installation.md) · [Quickstart](docs/getting-started/quickstart.md) · [Quick tour](docs/getting-started/quicktour.md) |
| Tutorials | [Run the agent on RoboDojo](docs/tutorials/run-the-agent.md) · [Read a run](docs/tutorials/read-a-run.md) · [Run on a small machine](docs/tutorials/small-machine.md) · [Write a skill](docs/tutorials/write-a-skill.md) · [Add a VLA](docs/tutorials/add-a-vla.md) · [Submit to RoboDojo's leaderboard](docs/tutorials/submit-xpolicylab.md) |
| How-to guides | [Configure runs](docs/how-to/configure-runs.md) · [Manage services](docs/how-to/services.md) · [Self-hosted models](docs/how-to/self-hosted-models.md) · [Checks without a model](docs/how-to/pipeline-checks.md) · [Self-improvement](docs/how-to/self-improvement.md) · [Isolation](docs/how-to/isolation.md) · [RoboTwin](docs/how-to/robotwin.md) · [OpenRouter relay](docs/how-to/openrouter-relay.md) · [Extend](docs/how-to/extend.md) |
| Concepts | [Programs and decision points](docs/concepts/design.md) · [The episode runtime](docs/concepts/runtime.md) · [Skills and prompt packs](docs/concepts/skills.md) · [Services and deploys](docs/concepts/services.md) · [Configuration and identity](docs/concepts/configuration.md) · [The ladder](docs/concepts/ladder.md) · [XPolicyLab](docs/concepts/xpolicylab.md) · [Method conformance](docs/concepts/method-conformance.md) |
| Reference | [Benches](docs/reference/benches.md) · [Worker protocol](docs/reference/worker-protocol.md) · [Traces](docs/reference/traces.md) · and, in the built site, the CLI, configuration keys, skills, services, the agent and Python API, generated from the code |

[Contributing](docs/contributing.md): conventions, tests, and how the documentation is
built and checked.
