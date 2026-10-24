# Method conformance checklist

Each claim of the paper's method (Notion "Harness Policy": §3, §4.1, App. A–H) with
where it is implemented, how it is checked, and its status. Re-read the method
sections and update this file before running experiments; every run summary should
report the metrics named here. Status: ✓ conforms, ✗ deviates (fix listed), – not
applicable to the simulated benchmarks.

## High level (§3.2, Tab. 1, Fig. 3, App. F)

| Claim | Implementation | Check / metric | Status |
|---|---|---|---|
| Decision points only at the episode start and when an event reaches the high level; the model never selects individual commands | `runtime/driver.py` episode loop; `finish()` in programs | `decisions` per episode by reason (start / escalation / error / program_end / verify / format), `harness report`; Fig. 3 uses 3 queries | ✓ (F1). Before F1: 27 calls per grok R4 episode, 429 of 546 programs ended normally. Agent 1.4.1 on evaluation pack 0: 3.6 calls per episode (median 2), budget 30 |
| The model judges success (verifier, then finish); the verifier spans layers (Tab. 5) | `<verify>`, `<finish>`; `verify()`, `finish()` in programs | `verifier_calls`, verifier cost in `cost_usd` | ✓ (F1) |
| History h_k: instruction, recent observations, previous programs, their traces, unresolved events | driver messages | tokens per call | ✓ with F1 each message pair is one decision point: the program, its output, the unhandled event and trace, state and images |
| System prompt (App. E): write programs; guards/controllers run without asking; rules | `prompts/robodojo/*.md`, `prompts/robotwin/core.md` | prompt review | ✓, changed (agent 1.2.0): short programs, one contact with an object each, then a look at the scene measured again (`prompts/robodojo/method.md`); programs "as long as the model can foresee" looped over objects without looking (classify_objects: four programs killed at the 30-minute cell limit). **The paper's App. E should say so** |

## Low level (§3.2, App. B)

| Claim | Implementation | Check | Status |
|---|---|---|---|
| Programs compose bounded executions e = (σ, θ, T, Ω, φ); every execution ends with an event and a trace | `session._execute`, `events.py` | unit tests | ✓ |
| Handlers respond to events without querying the model; unresolved events escalate | `on()`, `_finish`, `Escalation` | `events_handled`, `events_escalated` per episode | ✓ |
| Segmentation SAM 3 | service `sam3` (`services/servers/serve_sam3.py`), skill `segmentation` | – | ✓ |
| Open-vocabulary pointing with Molmo2 (Clark et al., 2026) | `point_to` via `[runtime.pointer]`: Molmo2-8B, service `molmo2` (`services/servers/serve_vlm.py`) (LLM.int8, vision backbone in bf16, to fit next to the other servers); without a pointer role `point_to` asks the vlm role; `query_vlm` keeps `[runtime.vlm]` | unit tests; pointing on simulator images | ✓ implemented (F3); changed: not in the agent since 1.5.0, which points with its VLM (the main model): every 1.4 evaluation ran without Molmo2, and programs pointed in 3 of 129 episodes of 1.4.1's run. **The paper's App. B should say so** |
| Calibrated projection, IK, collision-aware motion planning | skills `geometry`, `kinematics`; worker `ik` / `plan` (cuRobo), or the `kinematics` service outside the simulator | – | ✓ projection, IK; planning avoids self collision and the table only. Scene collisions need the depth occupancy map, which the benchmarks do not provide (–) |
| `plan_grasp(obs, mask, top_down=False)` | skill `grasping` (`plan_grasp`): top-down, or a side grasp (horizontal approach from the arm's base) when top-down is not graspable or reachable | unit tests | ✓ (F5). Without depth both use the mask footprint on a plane; no full 6-DoF grasp search |
| `move_eef(..., speed=0.2)`, `close_gripper(arm, force=None)` | speed (default: planner timing), no force | – | minor: default speed differs; no force control in either simulator (–) |
| **Addition (not in App. B):** tools by object id, compositions of the analytic skills: `locate` (SAM 3 in every camera, triangulated, one object per set of pixels, ids for the scene), `pick(arm, id)` (grasp across, by a thin part or an edge; re-measured from above with the wrist camera; slow descent; the check whether the object rose with the jaws), `release`, `transfer`, `push`, `move_tip`, and a scene measured again after every program (`turn_report`) | skill `eef` (the default); skill `scene` (`watch_scene`, `clear_view`, `press`) | `tests/test_eef.py`; development comparisons (changelog 1.2.0, 1.2.1) | ✓ implemented; **the paper's App. B should list them** (programs otherwise rebuild grasping and verification each episode: long generations cut by the provider, logic bugs, steps wasted) |
| Closed-loop `pick(target)` / `place(target, arm)` by name (head camera finds, wrist camera looks closer, guarded descent, retries, intercepting a moving object) | skill `manipulation`: the `analytic-manipulation` ablation, R1's fixed skills and RoboTwin | unit tests; model-free pick/place bench in the simulator | ✓, not the default since agent 1.2.0: its pick held in 2 of 46 calls on the development comparison, the tools by object id scored 21.8 against 16.8 and 19.9 |
| **Benchmark convention (RoboDojo):** most tasks count success only with both arms back at their start pose (`all_robot_back_to_origin`, 0.15 m / 20°) | after `finish()` or `<finish>` the driver runs `__wrap_up__()` in the kernel, which moves each arm to its start joints when the embodiment profile has `return_to_start` (RoboDojo yes, RoboTwin no); same for R0–R4 | unit test; bench episodes end with `wrap_up` events | ✓ implemented; **the paper should state it with the evaluation protocol** |
| `run_policy` bounded and monitored; `predict_future` | Simate (the default since agent 1.3.0; G0.5 before); OpenWAM as a service, not in the agent since 1.5.0 (no 1.4 evaluation had it; `--set 'agent.services.world_models=["openwam"]'` adds it); since agent 1.4.0 the handed-over task runs uninterrupted (no model look between slices, no progress watch: stopping it early bought nothing, and measuring the scene while it ran may harm it, pour_liquid_into_cup 1 of 4 watched against 4 of 4 alone); the stall stop ends a run whose arms moved and stopped, and the model has `post_policy_calls` calls after it, then the policy runs on to the end | unit tests (`tests/test_run_policy.py`); `progress_checks` in each trace | ✓ (RoboTwin: no learned skill served) |
| Division of labor: the program keeps what the learned skill cannot do, the learned skill whole tasks it was trained on | `start(strategy, kind)` (`skills/vla`), the rule in `prompts/robodojo/rules_learned.md`. Agent 1.4.2: strategy B (own tools) only when the task is to pick one object out of several identified by its name, a number or label on it, or a picture, among objects that stand still; every other trained task handed over whole from the start pose (A/C), also with named colours, orders or categories, and conveyors whatever identifies the target; `start` after B allowed; 4 model calls after the policy's run (`post_policy_calls`), then it runs on to the end | evaluation pack 0, paired with Simate alone on the same layouts (agent 1.4.1: 26.2 against 22.2, RoboDojo average); strategy and calls per episode in `summary.json` | ✓, changed: remembering stays with the learned skill (Simate keeps a memory of past frames: officially 56.0 on match_and_pick_from_conveyor, 100 on cover_blocks, 38.7 on swap_blocks; the agent's tools 0 on the first). **The paper's text should say so** |
| `recall` read-only during evaluation | library `memory/` | unit tests | ✓ |
| `depth_to_points` | – | – | – no depth in either benchmark's evaluation config |

## Reactive layer and shield (§3.2, App. C, D, H)

| Claim | Implementation | Check | Status |
|---|---|---|---|
| `when`, `watch`, `run_controller(fn, hz, until, max_seconds, region)`, `clear_guards`; pure, no I/O, per-tick budget, violation → REACTIVE_FAULT | `runtime/reactive.py`, `session.run_controller` | unit tests | ✓ (budget 20 ms) |
| Shield: rejects commands that are bugs: malformed, joint limits, workspace (no further out than the robot already is), joint and Cartesian jump from both the previous accepted command and the measured state | `runtime/shield.py`; worker `[shield]` options | property tests (accepted commands are forwarded unchanged and within every bound; a noisy smooth policy far ahead of a lagging arm passes) | ✓, changed: the shield only rejects, never shortens or clamps, so a program learns its command was wrong; no velocity or acceleration bounds (a robot's controller enforces its own rates; the skills time their motions at the planner's 3.14 rad/s and 10 rad/s²). The earlier velocity, acceleration and step truncation cut G0.5 alone on the official eval cases from 7/14 to 0/6, and the step check measured from the lagging arm rejected our own moves (free-space speed had been lowered to 0.4 m/s for it; back to 0.6). Paper text updated to match. |
| Shield: collisions with a depth occupancy map; force/torque limits | – | – | – no depth, no force sensing in simulation |
| App. H tests: property, fault injection, reactive, replay | `tests/`, `scripts/replay_script.py` | pytest | ✓ |

## Self-improvement (§3.3, §4.6)

| Claim | Implementation | Check | Status |
|---|---|---|---|
| Reflection after each dev episode proposes additions at each layer; kept only if it improves on disjoint dev seeds; content hash; frozen | `reflect.py`, `self_improve.py`, `library.py` | pipeline run | ✓ (first run: candidates rejected, baseline at ceiling) |

## Protocol and ladder (§4.1, Tab. 3, App. A)

| Claim | Implementation | Check | Status |
|---|---|---|---|
| Inputs: instruction, RGB, robot state; no task files, checkers, object annotations or simulator state | worker calls; isolated kernel | code review, `tests/test_isolation.py` | ✓ (F6) |
| Isolated container reaching only the model endpoint and policy servers; workspace and memory deleted after every episode | `runtime/isolation.py`: user, mount and PID namespaces (the user's directories show only the Python installation and, read-only, the harness source; fresh tmpfs workspace); audit hook allows connections only to the services in the kernel options and no process creation or ctypes | `tests/test_isolation.py`; replay in the simulator | ✓ (F6). Not a container image: the hook guards against reaching data and services, not deliberate interpreter escape |
| Frozen configuration recorded by content hash (model, effort, budget, prompts, libraries, embodiment profile) | `manifest.json` `config_sha256`, `embodiment_profile`, `embodiment_sha256` | – | ✓ (F1) |
| Ladder R0–R4 share model, runtime, shield and call budget | `configs/ablations/r*.toml` include `configs/default.toml`; one model file | `tests/test_configs.py` | ✓ |
| R0: one end-effector target / gripper / observation / wait / give-up per call; instruction, ee state, arrival error, shield decision, calls left, two image sets | `prompts/robodojo_direct.md`, `runtime/direct.py` | – | ✓ |
| R1: selects skills from a fixed vocabulary with learned skills (Zhang et al. 2026a; Galanti et al. 2026) | `runtime/direct.py` with `skills = true`: one action per call from move, gripper, observe, wait, locate, pick, place, policy; the skills are fixed programs over the same kernel API | unit tests; scripted pick in the simulator | ✓ (F2) |
| R2: programs, no learned skills; R3: + learned skills; R4: + programmable reactive layer | `runtime.layers` | – | ✓ |
| Tab. 5/6 report model calls and cost per episode | `harness report` | – | ✓ calls by reason, events handled; cost includes pointing and verifier calls made from programs |

## Evaluation by XPolicyLab (RoboDojo leaderboard)

| Claim | Implementation | Check | Status |
|---|---|---|---|
| The submitted policy sees only what the official client gives: instruction, three RGB cameras, measured arm joints, link6 poses, last commanded grippers; no calibration, depth, task name, step limit or outcome | `core/contract.py` (`runtime.contract = "xpolicylab"`): worker filter for development, the same contract in the connector; calibration constants and kinematics from `embodiments/arx_x5` and the `kinematics` service | `tests/test_xpolicylab.py`; kinematics service identical to the worker (FK, IK, plans) | ✓ |
| Grasp checks without a measured gripper | `pick` measures whether the object rose with the jaws (SAM 3); `holding(arm)`, and a pick whose rise cannot be measured, ask the VLM (the main model) about the wrist camera's view | unit tests (`tests/test_eef.py`) | ✓; `vision_calls` and `vision_cost_usd` per episode measure its cost |
| One fixed agent for every task, no per-task router | the agent reads no task descriptions (export refuses a configuration that does); `task_name` only labels traces | `tests/test_xpolicylab.py` | ✓ |
| The evaluated agent is the submitted one | `agent.json` identity checked at start on the evaluator's machine; code fingerprint recorded | `tests/test_xpolicylab.py` | ✓ |

## One harness, any embodiment (§3.1, Tab. 2 of the CVPR draft)

| Claim | Implementation | Check | Status |
|---|---|---|---|
| Bounded executions, events, escalation, the reactive layer and the shield are defined once and instantiated per embodiment; a skill, a prompt and a session belong to an embodiment kind | `runtime/episode.py` (`EpisodeCore`: executions, guards, handlers, model clients, policies; `session_class(kind)`), `runtime/session.py` (`RobotSession`: joint commands, link6 regions, `open_gripper`), `embodiments.KINDS` (robot, desktop) and a file's `kind`, a skill's `embodiments`, an agent's `prompt` table by kind, the bench's kind (`BENCH_EMBODIMENT`), the driver's texts by kind (`EMBODIMENT_TEXT`), the worker's `kind` checked at reset; `runtime/shield.py` `CommandShield` | `tests/test_episode.py`: a session that is not a robot's runs executions, guards (hold; `open_gripper` is an event there), shield rejections and scope exits; skills, embodiment files, agent resolution and the driver keyed by kind. Robot unchanged: every agent and experiment configuration resolves to the same identity as before (17 of 17; default `a15dec1ee3940447`), the suite passes unchanged but for two mock targets | ✓ kernel; the computer embodiment (desktop worker, its skills, prompt and shield rules) is not implemented yet |

## Not needed for the simulated experiments

Web interface for monitoring and human takeover, and the real-robot embodiment
profiles (Tab. 7) are part of the release but not of the simulated experiments; they
are not implemented yet. The XPolicyLab connector (`connectors/xpolicylab.py`) is the
bridge a physical robot behind XPolicyLab would use too.

## Fixes

- F1 (done) Event-driven high level: decision points at start, unresolved escalation,
  program error, and program end without `finish()`; `finish()` and `verify()` inside
  programs; prompts per App. E; calls counted by reason; embodiment profile in the
  manifest.
- F2 (done) R1 fixed skill vocabulary.
- F3 (done) Molmo2 pointing, self-hosted.
- F4 (done) Shield: reject bugs only (limits, workspace, jumps); no rate limits.
- F5 (done) `plan_grasp(..., top_down=False)`.
- F6 (done) Kernel isolation: programs reach only the worker, model and policy endpoints
  and a per-episode workspace.
