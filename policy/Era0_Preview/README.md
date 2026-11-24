# Era0-Preview

**Contributor:** RobotEra Team | **Paper:** Technical report planned by 2026-11-18 | **arXiv:** Not available | **Original code:** Inference runtime included in `runtime/`.

Evaluation-only VLA policy for RoboDojo simulation, supporting `arx_x5`, Cartesian `ee` actions and batched environments.

## Installation

Install [RoboDojo](https://robodojo-benchmark.com/doc/usage/installation/) and place this XPolicyLab checkout inside its workspace. The policy environment requires Python 3.11, a CUDA 12.x toolkit with `nvcc`, and a C++ compiler. Install the simulator in its separate RoboDojo environment.

```bash
cd RoboDojo/XPolicyLab/policy/Era0_Preview
bash install.sh
bash download_checkpoint.sh
```

## Model assets

Model repository: [roboterax/Era0-5B-RoboDojo](https://huggingface.co/roboterax/Era0-5B-RoboDojo).

`bash download_checkpoint.sh` downloads the matching model assets to `checkpoints/era0-preview/`. The exact model revision and file checksums are pinned in `checkpoint_release.json`.

Keep `model.safetensors`, `config.json`, the adapter metadata file, processor/tokenizer files and `norm_stats.pkl` together. Set the directory before evaluation:

```bash
export ERA0_CHECKPOINT_PATH="$(pwd)/checkpoints/era0-preview"
```

Observations use the standard XPolicyLab schema: three RGB camera views (`cam_head`, `cam_left_wrist`, `cam_right_wrist`), a language instruction and robot state. Preserve the supplied configuration and normalization file.

## Evaluation

Run one task and seed from this policy directory:

```bash
ERA0_EXECUTION_PROFILE=official bash eval.sh RoboDojo stack_bowls \
  era0-preview arx_x5 ee 0 0 0 .venv RoboDojo
```

The final two arguments select the policy and simulator environments. Replace the final `RoboDojo` with your simulator environment name or Python executable. Argument conventions: [XPolicyLab README](../../README.md).

Run all SDK default tasks with native episode counts for seeds 0, 1 and 2:

```bash
POLICY_ENV=.venv bash eval_three_seeds.sh \
  --checkpoint "$ERA0_CHECKPOINT_PATH" --sim-env RoboDojo \
  --gpu-ids 0,1,2,3,4,5,6,7 --execution-profile official \
  --output "/path/to/eval_$(date +%F_%H%M%S)" --run
```

Omit `--run` to preview the plan. Fewer GPUs are supported, for example `--gpu-ids 0,1`.

To split evaluation across three jobs, run the same command with `--seeds 0`, `--seeds 1` or `--seeds 2`, respectively. Use a separate output directory for each job, the same matching model and checkpoint name, and add `--skip-summary` to avoid concurrent summary writes. Once all jobs finish, run from the RoboDojo root in its simulator environment:

```bash
bash scripts/robodojo.sh summarize
```

For a small diagnostic run, add `--seeds 0 --eval-num 2 --num-envs 2 --tasks-file /path/to/tasks.txt`. Diagnostic runs are partial evaluations.

## Local results

The [official SDK summary](EVALUATION_RESULTS.md) reports **Score 44.54 ± 0.91**, **SR 39.18 ± 1.13%**, with **122/126** task–seed cells complete and **6,215/6,300** episodes saved. Incomplete cells are excluded.

These are partial local results, not official leaderboard scores. See the [RoboDojo leaderboard](https://robodojo-benchmark.com/LeaderBoard) for official results. Current validation evidence: [checks](PREPARATION_CHECKS.md).

Third-party attribution: [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
