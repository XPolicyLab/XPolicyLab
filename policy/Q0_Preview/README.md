# Q0-Preview

**Contributor:** Qoo Tech | **Paper:** | **arXiv:** | **Original code:** [Q0-Preview](https://github.com/XPolicyLab/XPolicyLab/tree/main/policy/Q0_Preview)

Q0-Preview is a physical agent built on frontier multimodal models for robotic manipulation. Its core idea is to fully leverage the spatial computation capabilities of multimodal agent models to calculate the coordinates of spatial points in the world frame. Gripper orientation and contact information are derived entirely from multiple spatial points. This spatial computation framework can also be extended to dexterous hands and humanoid robots, and we will release real-robot results in future updates.

Shared conventions — argument meanings, checkpoint naming, split-machine deployment, `EVAL_ENV_TYPE` — are documented in the [XPolicyLab README](../../README.md). Official results: [RoboDojo LeaderBoard](https://robodojo-benchmark.com/LeaderBoard).

## Installation

Q0-Preview uses a remote model API, local CPU Python tools and a GPU motion controller. Policy execution requires Linux x86_64, Python 3.11 and an NVIDIA GPU. Model weights are hosted by the API provider; RoboDojo and its evaluator environment are supplied separately.

Follow [INSTALLATION.md](INSTALLATION.md) for the complete setup sequence:

1. [API configuration](INSTALLATION.md#api-configuration)
2. [Tool use environment](INSTALLATION.md#tool-use-environment)
3. [Controller environment](INSTALLATION.md#controller-environment)
4. [Evaluator settings](INSTALLATION.md#evaluator-settings) and [verification](INSTALLATION.md#verification)

The guide uses `install.sh` and `requirements.txt` to install Python dependencies. Their roles and the GPU version constraints are explained under [Installation files](INSTALLATION.md#installation-files). Once setup is complete, continue to [Evaluation](#evaluation).

## Data Processing

Not applicable. This policy consumes evaluator observations directly and does not use a demonstration dataset or a data-conversion stage. There is no `process_data.sh`. Task definitions in [assets/tasks/](assets/tasks/), including source provenance and control-step limits, are runtime inputs rather than a training dataset.

## Training

Not applicable. Q0-Preview has no local policy-training stage, learned policy checkpoint or `train.sh`. Evaluation requires external model API access and the runtime assets distributed with this adapter.

## Evaluation

Supported configuration: `RoboDojo` / `arx_x5` / `joint`, at 25 Hz, with one environment per policy server (`eval_batch: false`). Observations must contain decoded RGB for `cam_head`, `cam_left_wrist` and `cam_right_wrist`, camera intrinsics and current camera-to-world extrinsics, arm/gripper state, world-frame end-effector poses and the instruction. Enable `intrinsic_matrix` and `extrinsic_matrix` in the evaluator configuration.

The [observation interface](docs/OBSERVATION_INTERFACE.md) defines the payload, USD/OpenGL camera axes, frame synchronization and missing-field errors. The policy does not query private simulator camera APIs or substitute guessed matrices.

The standard ten-argument entry is:

```bash
bash policy/Q0_Preview/eval.sh \
  <bench_name> <task_name> <ckpt_name> <env_cfg_type> <action_type> <seed> \
  <policy_gpu_id> <env_gpu_id> <policy_env> <eval_env_conda_env>

# Example: policy GPU 0, evaluator GPU 1.
bash policy/Q0_Preview/eval.sh \
  RoboDojo stack_bowls api arx_x5 joint 0 \
  0 1 q0-preview-policy RoboDojo
```

Replace GPU IDs and environment names for your machine. `policy_env` may be a Conda environment name or an absolute environment prefix; the evaluator environment is a Conda name. `api` is a label, not a checkpoint file. For split-machine deployment, follow the shared [Deployment Flow](../../README.md#-deployment-flow) with this policy's setup scripts.

Check the policy environment, package imports and tool sandbox using [INSTALLATION.md](INSTALLATION.md#verification) before starting evaluation.

## Model Assets

No learned checkpoint download is required. Keep the bundled assets with the adapter:


| Location                                           | Purpose                                                                      |
| -------------------------------------------------- | ---------------------------------------------------------------------------- |
| [assets/prompts/](assets/prompts/)                 | Planning, checkpoint and geometry-context prompts                            |
| [assets/geometry/](assets/geometry/)               | Reference documents copied into the Python-tool workspace                    |
| [assets/tasks/](assets/tasks/)                     | Task guidance, completion/process rules, step limits and source fingerprints |
| [controller/calibration/](controller/calibration/) | X5 robot model, CuRobo configuration, collision spheres and provenance       |


The sandbox exposes the geometry references and current task reference under their existing `/work/` paths. These documents and task definitions affect the model's inputs and should be treated as part of the policy.

## Configuration

[deploy.yml](deploy.yml) contains the standard XPolicyLab fields and policy budgets. Select a model configuration from [config/](config/) using `Q0_PREVIEW_AGENT_CONFIG`. Endpoint setup, credentials and model-specific parameters are documented under [API configuration](INSTALLATION.md#api-configuration).

See [Runtime configuration](INSTALLATION.md#runtime-configuration) for environment variables, output paths and configuration precedence.

Default budgets remain:


| Setting                    | Default                                              |
| -------------------------- | ---------------------------------------------------- |
| `max_plans`                | 100                                                  |
| `wall_seconds`             | 10,800 seconds per episode                           |
| `turn_seconds`             | 3,600 seconds per planning window                    |
| `checkpoint_seconds`       | 60 seconds per checkpoint                            |


The episode clock starts with the first observation. Planning and API waits are capped by the remaining episode budget. After expiry, the policy stops planning and holds position until the evaluator closes the episode or exhausts its control-step budget. Initialization and finalization can add to total process time. Control-step limits come from the bundled task definitions.

## Notes

```text
Evaluator observations → XPolicyLab adapter → episode harness ⇄ API model
                                                   ↓
                                            validated action groups
                                                   ↓
Evaluator joint actions ← motion planning and execution
```

A plan contains one action group, or two with an intermediate visual checkpoint. The checkpoint decides whether to continue or replan. Accepted plans carry a short memory into subsequent planning windows. The harness retains the first and recent planning context within a bounded window; full history and raw API attempts remain archived. Python-tool state persists within an episode and is recreated on reset.


| Location                                                  | Responsibility                                                                       |
| --------------------------------------------------------- | ------------------------------------------------------------------------------------ |
| `model.py`, `deploy.py`, `deploy.yml`, evaluation scripts | Official observation/action interface and service lifecycle                          |
| `runtime/`                                                | Episode lifecycle, plan validation, task context, observation validation and budgets |
| `controller/`                                             | Motion planning, CuRobo, collision checks, execution and robot calibration           |
| `harness/`                                                | API tool loop, isolated Python tools, history, caching and archives                  |
| `harness/helpers/`, `assets/`                             | Model-facing Python helpers, prompts, task definitions and reference material        |
| `config/`, `tools/`, `docs/`                              | Configuration, launch utilities and reference documentation                          |


The root keeps the README, installation guide, dependency/license files and standard XPolicyLab entry points. Implementation modules live in the packages above:

```text
Q0_Preview/
├── README.md, INSTALLATION.md, LICENSE
├── model.py, deploy.py, deploy.yml, __init__.py
├── requirements.txt, install.sh, eval.sh, setup_eval_*.sh
├── controller/          # Motion planning and execution
│   └── calibration/    # Robot model and planner assets
├── runtime/             # Episode and observation lifecycle
├── harness/             # Model API and tool orchestration
│   └── helpers/        # Scripts copied into the model's workspace
├── assets/              # Prompts, geometry references and task definitions
├── config/              # Runtime configuration examples and constraints
├── tools/               # Launch and diagnostic utilities
└── docs/                # Evaluator observation interface
```

Each service creates `runs/Q0_Preview/<run_uuid>/`, with per-episode `sessions/<episode_uuid>/` directories for observations, history, API requests/responses/SSE and tool artifacts. Partial or missing responses retain that status. Images are loaded locally and embedded in API requests. Relative output paths resolve against the checkout; explicit run directories must be new and writable. Evaluator scores and videos use the evaluator's own output settings.

The optional `tools/evaluate_tasks.py` scheduler starts one `eval.sh` service per local GPU slot, orders tasks by control-step limit and launches each task once. Use `--help` and `--dry-run` to inspect a schedule. Layouts and scoring belong to the evaluator; multi-host orchestration and simulator asset/shader-cache allocation belong to the deployment layer.
