# Era0-Preview

**Contributor:** RobotEra Team | **Paper:** Technical report planned by 2026-11-18 | **arXiv:** Not available | **Original code:** Inference runtime included in `runtime/`.

Evaluation-only VLA policy for RoboDojo simulation: `arx_x5`, Cartesian `ee` actions, three RGB camera views and batched simulation.

**Release scope:** evaluation code and model assets only. A technical report is planned for public release within two weeks (by 2026-11-18).

**Weights and results:** [roboterax/Era0-5B-Preview-RoboDojo](https://huggingface.co/roboterax/Era0-5B-Preview-RoboDojo).

## Installation

Install [RoboDojo](https://robodojo-benchmark.com/doc/usage/installation/) and place this XPolicyLab checkout inside its workspace. The policy environment requires Python 3.11, a CUDA 12.x toolkit with `nvcc`, and a C++ compiler; the installer creates `.venv` with the dependencies. Install the simulator in its separate RoboDojo environment.

```bash
cd RoboDojo/XPolicyLab/policy/Era0_Preview
bash install.sh
# This revision uses locally prepared model assets; publication is pending.
```

The prepared assets are isolated under the relative `local_directory` in `checkpoint_release.json`. Set `ERA0_CHECKPOINT_PATH` to that directory before evaluation. Keep all supplied files, including `norm_stats.pkl`. The existing public model is an earlier revision and must use its matching adapter.

## Input format

Observations use the standard XPolicyLab schema: `cam_head`, `cam_left_wrist` and `cam_right_wrist` RGB images, a language instruction, and robot state. Actions use Cartesian end-effector poses and gripper values. Use the supplied normalization file unchanged. The prepared configuration uses image history 9/1/1 at interval 40, ten denoising steps and ten executed actions per chunk.

## Evaluation

Use the prepared local assets, then run one task and seed:

```bash
export ERA0_CHECKPOINT_PATH="$(python -c 'import json; from pathlib import Path; print(Path(json.load(open("checkpoint_release.json"))["local_directory"]).resolve())')"
```

```bash
ERA0_EXECUTION_PROFILE=official bash eval.sh RoboDojo stack_bowls \
  era0-preview-sdk arx_x5 ee 0 0 0 .venv RoboDojo
```

The final two arguments select the policy and simulator environments. Replace the final `RoboDojo` with your simulator environment name or Python executable. Argument conventions: [XPolicyLab README](../../README.md).

Run all available tasks with native episode counts for seeds 0, 1 and 2:

```bash
POLICY_ENV=.venv bash eval_three_seeds.sh \
  --checkpoint "$ERA0_CHECKPOINT_PATH" --sim-env RoboDojo \
  --gpu-ids 0,1,2,3,4,5,6,7 --execution-profile official \
  --output "/path/to/eval_$(date +%F_%H%M%S)" --run
```

Use a fresh output directory and replace the simulator environment and GPU IDs as needed. Omit `--run` to preview the plan.

Historical results from the previous release (not results of these prepared assets): **33.44 ± 0.42**, with **122/126** task–seed cells complete and **6,225/6,300** episodes available. Incomplete cells are excluded. See [evaluation results](EVALUATION_RESULTS.md) for the unmodified SDK summary (historical policy label: `Hy_XVLA_RoboDojo`). Evaluation remains incomplete; these are not official leaderboard results. Official scores are published on the [RoboDojo leaderboard](https://robodojo-benchmark.com/LeaderBoard).

Earlier-release debug evidence: [verification results](DEBUG_VERIFICATION.md). Validation of the prepared revision is recorded in [PREPARATION_CHECKS.md](PREPARATION_CHECKS.md).

For a diagnostic seed-0 sweep with ten episodes per task, add `--seeds 0 --eval-num 10`; optionally provide `--tasks-file /path/to/tasks.txt`. Omit `--run` to inspect the plan.
