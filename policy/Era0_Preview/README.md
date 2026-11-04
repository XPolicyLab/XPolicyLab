# Era0-Preview

**Contributor:** RobotEra Team | **Paper:** Not available | **arXiv:** Not available | **Original code:** Inference runtime included in `runtime/`.

Evaluation-only policy for RoboDojo `arx_x5`, Cartesian `ee` actions, with three RGB camera views and batched simulation.

**Weights and results:** [roboterax/Era0-5B-Preview-RoboDojo](https://huggingface.co/roboterax/Era0-5B-Preview-RoboDojo).

## Installation

Install [RoboDojo](https://robodojo-benchmark.com/doc/usage/installation/) and place this XPolicyLab checkout inside its workspace. The policy environment requires Python 3.11; the installer creates `.venv` with the dependencies.

```bash
cd RoboDojo/XPolicyLab/policy/Era0_Preview
bash install.sh
bash download_checkpoint.sh
```

Model assets are downloaded to `checkpoints/era0-preview/`. Keep all supplied files, including `norm_stats.pkl`.

## Input format

Observations use the standard XPolicyLab schema: `cam_head`, `cam_left_wrist` and `cam_right_wrist` RGB images, a language instruction, and robot state. Actions use Cartesian end-effector poses and gripper values. Use the supplied normalization file unchanged.

## Evaluation

Run one task and seed:

```bash
ERA0_EXECUTION_PROFILE=official bash eval.sh RoboDojo stack_bowls \
  era0-preview arx_x5 ee 0 0 0 .venv RoboDojo
```

The final two arguments select the policy and simulator environments. Replace the final `RoboDojo` with your simulator environment name or Python executable. Argument conventions: [XPolicyLab README](../../README.md).

Run all available tasks with native episode counts for seeds 0, 1 and 2:

```bash
POLICY_ENV=.venv bash eval_three_seeds.sh \
  --checkpoint "$(pwd)/checkpoints/era0-preview" --sim-env RoboDojo \
  --gpu-ids 0,1,2,3,4,5,6,7 --execution-profile official \
  --output /path/to/fresh/eval --run
```

Use a fresh output directory and replace the simulator environment and GPU IDs as needed. Omit `--run` to preview the plan.

Local reference Score: **33.16 ± 0.35**, across three seeds with **6,225/6,300** episodes completed. See [evaluation results](EVALUATION_RESULTS.md) for the full tables; official scores are published on the [RoboDojo leaderboard](https://robodojo-benchmark.com/LeaderBoard).

Debug transport and observation-delay checks: [verification results](DEBUG_VERIFICATION.md).
