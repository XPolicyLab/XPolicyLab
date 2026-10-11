# WorldScape RSI

### Experience-Driven Embodied Intelligence through Policy–Harness Co-Evolution

**WorldScape RSI brings World Action Models and agentic reasoning into a unified framework for long-horizon robotic manipulation.** It connects visual perception, task memory, action generation, execution feedback, and recovery—and makes the resulting experience reusable across subsequent policy and harness versions.

Our central idea is **Policy–Harness Co-Evolution**: improve both the model that proposes actions and the execution framework that turns those proposals into reliable behavior.

**A stronger harness produces better evidence for the next policy update, while an advanced policy contributes to a next-generation intelligent harness — forming a bidirectional recursive self-improvement loop between policy and harness.**

**World Action Models · Agentic Robotics · Multimodal In-Context Learning · Skill Memory · Adaptive Policy Routing · Closed-Loop Control · Execution-Aware Recovery · Code as Policies · Policy–Harness Co-Evolution · Recursive Self-Improvement**

**Contributor:** WorldScape RSI team<br>
**XPolicyLab policy name:** `WorldScape_RSI`<br>
**Target:** RoboDojo · ARX X5 · dual-arm joint control

## Why WorldScape RSI?

Robotic execution requires more than a single action prediction. The system must interpret a goal, recognize the current scene, choose an execution strategy, manage motion constraints, check what actually happened, and recover when an action does not achieve its intended effect.

WorldScape RSI organizes these responsibilities around two complementary capabilities:

- **WorldScape Policy** generates temporally coordinated joint-action chunks from observations and model-supported history.
- **WorldScape Agent** combines visual reasoning with advantage-conditioned task skills and multimodal demonstrations, then acts through structured motion commands and execution feedback.

An episode-level **JEV binary router** provides a common selection interface. The selected executor remains fixed for that episode; each pathway maintains its own observation–action feedback loop. This keeps route selection explicit and execution behavior traceable.

## Architecture

```text
       Task instruction + three RGB views + measured robot state
                                  |
                          JEV episode router
                     /                              \
          WorldScape Policy                    WorldScape Agent
          Visual history                Advantage-conditioned task skills
          Action chunks                         + multimodal ICL
          Commit / discard              Reasoning + motion commands
                     |                  Verification + recovery
                     \                              /
                       Simulation-side execution
                          RoboDojo / Isaac Sim
                                  |
                  New observations and execution feedback
                                  |
                     Versioned execution experience
                                  |
                  Policy–Harness Co-Evolution / RSI
                 Policy experience → harness refinement
                 Harness evidence  → policy improvement
                                  |
                       Validation → next release
```

WorldScape Policy runs through the adjacent `WorldScape_Policy_2` adapter **inside the policy server**. Its checkpoint transforms, RoboDojo camera configuration, action conversion, and temporal memory remain owned by that adapter. WorldScape Agent, powered by Astra, uses the retained motion harness and simulation-side planning stack. Its ICL is not passed to WorldScape Policy.

### A harness that connects reasoning to execution

| Capability | Role in the system |
| --- | --- |
| Advantage-conditioned task skill memory | Supplies reusable procedures, constraints, and recovery guidance for the current task. |
| Multimodal in-context learning | Grounds decisions in task demonstrations and visual references. |
| Structured motion commands | Expresses translation, orientation, grasping, synchronized motion, and return-to-home. |
| Execution-aware feedback | Makes motion failures, observed state, and remaining execution budget available to the control loop. |
| Stateful WorldScape Policy execution | Commits executed action steps, discards interrupted predictions, and resets memory between episodes. |
| Versioned experience | Connects outcomes to the demonstrations, skills, code, and model configuration used to produce them. |

Demonstrations guide decisions in the current scene. Their coordinates are not a fixed trajectory replay. Similarly, a successful motion command does not by itself establish a successful grasp or completed task: subsequent observations and native task evaluation determine the outcome.

## Policy–Harness Co-Evolution

**One stream of execution evidence supports two improvement tracks.**

| Policy track | Harness track |
| --- | --- |
| Curate trajectories for model adaptation. | Convert experience into reusable advantage-conditioned task skills. |
| Learn executor selection from comparative development outcomes. | Build observation–decision–action ICL demonstrations. |
| Evaluate candidate policy and routing checkpoints. | Refine motion procedures, verification, and recovery logic. |

The intended RSI cycle is **execute → analyze → propose updates → validate → version → execute again**. Policy and harness updates can advance at different cadences while sharing evidence and evaluation criteria. Release boundaries keep those changes out of an ongoing episode.

The inference adapter implements the execution and experience-consumption side of this design. Policy training and learned-router training are separate release activities; this package does not automatically retrain WorldScape Policy or rewrite skills after each episode.

## Task memory and demonstrations

The current v2 asset manifest covers **42 tasks**, with **42 task Skill files plus one shared rule file**. Its ICL bank contains:

| Demonstration source | Tasks | Content |
| --- | ---: | --- |
| WorldScape Agent success records (Astra) | 13 | Recorded decisions, actions, and observations. |
| Retained approved collection demonstrations | 2 | Decision demonstrations in the supported loader format. |
| Official task-video references | 27 | Visual task references without a complete agent decision/action trace. |

These are **15 decision demonstrations and 27 video references**, not 42 complete successful trajectories. Per-task sources and hashes are recorded in [`icl/manifest.json`](icl/manifest.json).

The v2 release incorporates the updated `pour_balls_into_vase` demonstration, its task skill, and the associated rotation guard. This demonstrates how new execution experience can enter a versioned harness release without replacing the other task memories.

## Installation

Use the company `dev` integration containing both `policy/WorldScape_RSI` and the adjacent RoboDojo-capable `policy/WorldScape_Policy_2`. The latter must include `deploy_robodojo.yml` and `xpolicylab_adapter/`; a different upstream snapshot may not provide these dependencies.

Use Linux and the supported Python 3.11 policy-server environment. Keep the RoboDojo / Isaac Sim / cuRobo simulation environment separate from the GPU policy environment. Follow the adjacent [WorldScape installation guide](../WorldScape_Policy_2/README.md#installation) for its GPU stack.

```bash
# Run in the prepared policy-server environment.
cd XPolicyLab/policy/WorldScape_RSI
bash install.sh
```

The installer installs this adapter's requirements, registers XPolicyLab, and invokes the adjacent WorldScape installer. Do not run it in the simulation environment as a substitute for that environment's setup.

Shared workspace layout and launch conventions are documented in the [XPolicyLab README](../../README.md). The `env_cfg/` directory belongs to the parent workspace, outside the XPolicyLab checkout.

## Model assets

| Resource | Setup |
| --- | --- |
| Task skills and ICL manifests | Included in the adapter. |
| ICL images and media | Restore from the matching v2 asset archive; media is excluded from Git. |
| WorldScape Policy (WAM) | Supply a compatible **RoboDojo** checkpoint and the adjacent adapter's dependencies. |
| WorldScape Agent (Astra) | Supply a supported provider/model configuration through the external secrets file. |
| Learned JEV | Supply a trained checkpoint matching the feature and class contract. |

Restore the frozen ICL assets from the policy directory:

```bash
python import_icl_assets.py /path/to/Worldscape_RSI-ICL-assets-20261011-v2.zip
```

The importer verifies the archive checksum and individual asset hashes. An archive from another release is not interchangeable. The current submission does not provide a public download URL for this asset archive or a trained JEV checkpoint; obtain the matching release assets from the maintainers before running evaluation. WorldScape Policy weights and agent API credentials are not bundled.

## Configuration

[`deploy.yml`](deploy.yml) defines the adapter defaults. Set the following on the appropriate host:

| Variable | Host | Purpose |
| --- | --- | --- |
| `DEXBOTIC_SECRETS` | Policy server | Absolute path to the external provider configuration. |
| `VLM_AGENT_PROFILE` | Policy server | Profile selecting the actual supported Astra endpoint and model. |
| `WRSI_WAM_CHECKPOINT` | Policy server | Absolute RoboDojo WAM checkpoint directory; overrides the checkpoint-name lookup. |
| `WRSI_JEV_CHECKPOINT` | Policy server | Trained JEV checkpoint path. |
| `WRSI_ROUTE_MODE` | Policy server | `learned` by default; `astra` or `wam` for explicit single-executor runs. |
| `WRSI_OUTPUT_DIR` | Simulation client | Writable directory for episode records. |

Example configuration for a learned-router run:

```bash
export DEXBOTIC_SECRETS=/secure/path/secrets.json
export VLM_AGENT_PROFILE=astra
export WRSI_WAM_CHECKPOINT=/models/robodojo-wam
export WRSI_JEV_CHECKPOINT=/models/jev.pt
export WRSI_ROUTE_MODE=learned
export WRSI_OUTPUT_DIR=/results/worldscape-rsi
```

Replace these example paths and the profile name with the deployed resources. Both hosts must use matching adapter code and configuration. The active integration loads WorldScape Policy in-process; it does **not** use the historical `WRSI_WAM_URL` or `WRSI_WAM_LOCK` settings.

### JEV routing contract

JEV consumes the **initial task instruction, rendered RGB, and measured robot state**. It produces two class probabilities in the order `[wam, astra]`, chooses their maximum, and caches that choice for the episode. These are routing-class probabilities, not calibrated task-success probabilities.

The checkpoint must carry the matching `feature_version`, `classes`, `state_dict`, `trained` marker, and `training_manifest_sha256`. Missing, untrained, or incompatible weights fail explicitly. Setting `WRSI_ROUTE_MODE=astra` or `wam` bypasses learned selection for a named ablation; it does not demonstrate learned routing performance.

## Evaluation

The supported configuration is `RoboDojo`, `arx_x5`, `joint`, and **one environment per policy-server process**. Batch evaluation is disabled. Each episode resets router state and executor state before execution.

### 1. Check assets and local configuration

From the parent workspace in which `XPolicyLab` is importable:

```bash
python -m XPolicyLab.policy.WorldScape_RSI.preflight \
  --task general_pickup --assets-only

python -m XPolicyLab.policy.WorldScape_RSI.preflight \
  --task general_pickup

python -m unittest discover \
  -s XPolicyLab/policy/WorldScape_RSI/tests -t . -v
```

Asset-only preflight validates loading of the task's ICL. Full preflight additionally checks the selected route's local configuration. Neither command performs a paid API request, performs real WorldScape Policy inference, or launches the simulator.

### 2. Launch a task

From `XPolicyLab/policy/WorldScape_RSI`, after supplying the required assets, services, credentials, and environments:

```bash
bash eval.sh \
  RoboDojo general_pickup wrsi_v2 arx_x5 joint \
  0 0 0 wrsi_policy RoboDojo
```

The positional arguments are:

```text
bench_name task_name ckpt_name env_cfg_type action_type seed
policy_gpu_id env_gpu_id policy_conda_env eval_env_conda_env
```

The example uses seed `0`, GPU `0` for each environment, and illustrative Conda environment names. Choose GPU assignments suitable for the deployed resources. `eval.sh` starts the server, waits for readiness, starts the simulation client, and cleans up its server process on exit.

For a fixed WorldScape Agent or WorldScape Policy comparison, set the route explicitly before the same launch command:

```bash
export WRSI_ROUTE_MODE=astra  # WorldScape Agent route in the hybrid adapter.
# Or: export WRSI_ROUTE_MODE=wam
```

These modes preserve the hybrid adapter's installation layout; they are not a separate package with the WAM and JEV code removed.

### Execution interface

WorldScape Agent's `get_action()` returns a structured command envelope consumed by this adapter's **custom `deploy.py`**. The simulation-side harness resolves it through the measured state and motion planner before applying robot actions. A generic joint-action debug client cannot replace this loop.

WorldScape Policy uses `wam_update_obs`, `wam_get_action`, `wam_commit_action`, and `wam_discard_action` through the policy server. The client reports actually executed steps so temporal state remains consistent after interrupted action chunks. RGB decoding follows XPolicyLab's shared image helpers; callers must not introduce a second color-channel conversion.

## Data processing and training

This adapter currently ships the inference integration, asset importer, and JEV architecture. It does not ship a standalone LeRobot conversion pipeline, a JEV training script, or trained routing weights. See [`jev/training_contract.json`](jev/training_contract.json) for the training contract.

For router training, define the utility and tie policy before constructing labels from paired development outcomes. Retain failures and ties. Keep replicas of the same episode/layout within the same data split, and exclude reward, score, success labels, task IDs, seed/layout IDs, and future observations from inference features. One selected successful trajectory does not establish that its executor is better than the alternative.

Policy adaptation and harness refinement belong to the offline RSI workflow. Freeze their outputs before an evaluation release; do not update them from held-out evaluation outcomes during the run.

## Release validation

The checked v2 integration records the following in [`VALIDATION.json`](VALIDATION.json):

| Check | Recorded status |
| --- | --- |
| Offline tests | 18 passed |
| Rotation-guard cases | 8 passed |
| Python compilation and shell syntax | Passed |
| Adjacent WorldScape adapter | Unchanged |
| Other 41 task skills and ICL | Unchanged by the balls update |
| Native simulation of the integrated package | Not yet verified |
| Real WorldScape Policy inference through this integration | Not yet verified |
| Trained JEV checkpoint included | No |

The balls update is grounded in a source run that achieved native success at seed 1 / layout 1 in **374/600 steps**. That source-run result is distinct from regression testing of the integrated package. No official leaderboard performance is claimed here.

Before an evaluation release, supply and freeze the required checkpoints and assets, validate both executor paths through the custom deployment loop, verify consecutive episode resets, and record release hashes. The shared rotation guard also requires cross-task native regression. See [`SUBMISSION_AUDIT.md`](SUBMISSION_AUDIT.md), [`BALLS_UPDATE.json`](BALLS_UPDATE.json), and [`INTEGRATION_NOTES.md`](INTEGRATION_NOTES.md).

## Code map

```text
WorldScape_RSI/
├── model.py, deploy.py       # Policy entry and custom simulation loop
├── deploy.yml               # Runtime configuration
├── wam_backend.py           # Adjacent WorldScape adapter integration
├── wam_loop.py              # Stateful action execution lifecycle
├── jev/                     # Binary routing network and training contract
├── harness/                 # Reasoning, motion execution, and recovery
├── skills/                  # Task memory and shared rules
├── icl/                     # Demonstrations, loader, and asset manifests
├── import_icl_assets.py     # Hash-verified asset restoration
├── preflight.py             # Offline asset/configuration checks
├── tests/                   # Offline integration tests
└── eval.sh                  # Policy-server and environment-client launch
```

Historical utilities retained under the harness are not alternate launch instructions. Use the entry points documented above.

## Acknowledgements

WorldScape RSI builds on [WorldScape-Policy](https://github.com/manifoldai-research/WorldScape-Policy), developed by manifoldai-research, the XPolicyLab integration framework, and RoboDojo's simulation and evaluation infrastructure. Shared setup and submission conventions are documented in the [XPolicyLab README](../../README.md) and [contribution guide](../../CONTRIBUTING.md). Demonstration sources remain
attributed in the ICL manifest.
