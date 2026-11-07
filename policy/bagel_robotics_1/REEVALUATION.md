# Reevaluate Bagel-Robotics-1 checkpoint 1000

Use the [README quickstart](README.md#installation) to install Python 3.10 policy
dependencies, download and verify the pinned files, and pass both
debug checks. The standalone inference runtime is included; the private research
training repository is not needed. Install RoboDojo and its assets separately.

The only policy checkpoint is `cp1000`, from
[bageldotcom/Bagel-Robotics-1](https://huggingface.co/bageldotcom/Bagel-Robotics-1),
revision `2651631c5d08819088475e8778676f5cf6d9afcf`. The downloader and loader reject
files that do not match `artifact-manifest.json`. The VAE, image/text encoders and
tokenizer are downloaded automatically at their separately pinned public revision.
The checkpoint is currently private. Obtain the dedicated repository-only read
credential through the agreed private channel and set `HF_TOKEN` securely as
shown in the README before running the downloader. Public release is pending an
organization administrator visibility change; anonymous checkpoint access will
work only after that change. The required encoder assets are public.

## What a reviewer can reproduce

The adapter loads the pinned checkpoint and implements the recorded inference
settings. Its completed qualification consists of both real-checkpoint debug
transports, four native simulator episodes and eight exact comparison checks on
two recorded observation sequences. The full five-suite research score table has
not been rerun end to end through this packaged adapter.

The commands below start a fresh official evaluation using supplied original
layouts. They do not recreate the research run's three replacement scenes or one
repaired Memory variant. Compare episode selection and seed coverage before
comparing score totals; retain startup failures for committee review. The public
comparison receipts describe the tested scope, but the original research runner
and raw comparison fixtures are not included in this PR. They are not required to
run a fresh evaluation with the packaged model.

## Set the local paths

Replace the example paths and simulator conda environment. Reuse the policy
venv created in the README. GPU commands on a cluster belong inside an allocation;
set `BAGEL_GPU_ID` to its assigned single GPU. On a dedicated workstation, the
example selects GPU 0. Do not overwrite a scheduler's GPU visibility assignment.

```bash
BAGEL_XPL_ROOT=/absolute/path/to/XPolicyLab
BAGEL_POLICY_DIR="$BAGEL_XPL_ROOT/policy/bagel_robotics_1"
BAGEL_POLICY_ENV="$BAGEL_POLICY_DIR/.venv"
BAGEL_SIM_ROOT=/absolute/path/to/RoboDojo
BAGEL_SIM_ENV=your_simulator_conda_env
BAGEL_GPU_ID=0
cd "$BAGEL_SIM_ROOT"
```

Qualification used XPolicyLab `6baccd57ff958570d0db1588370289ee36c8bb70` and
RoboDojo `f57e1633ffed59761bc8e568cfe7530c4b33381c`. Record the revisions used for
reevaluation. A different simulator or numerical backend requires fresh checking.
The [official installation guide](https://robodojo-benchmark.com/doc/usage/install-and-download/)
provides the simulator workspace layout, assets and conda setup.

## Inspect the command, then check two episodes

The dry run does not load the model or simulator:

```bash
EVAL_ENV_TYPE=sim bash scripts/robodojo.sh eval \
  --policy-dir "$BAGEL_POLICY_DIR" --task insert_tubes --ckpt cp1000 \
  --env-cfg arx_x5 --action-type ee --seed 1 --eval-num 2 \
  --policy-env "$BAGEL_POLICY_ENV" --eval-env "$BAGEL_SIM_ENV" \
  --policy-gpu "$BAGEL_GPU_ID" --env-gpu "$BAGEL_GPU_ID" --dry-run
```

Remove `--dry-run` to execute the check. The recorded qualification used original
layouts 0/1 and produced 2/2 successes for `insert_tubes`. Repeating with
`--task store_laptop_and_headphones` produced 1/2 successes and 60/100 mean partial
score. These are four integration episodes, not full-suite performance guarantees.
[Validation evidence](VALIDATION.md) records their scope and backend.

## Evaluate complete tasks or suites

Use the official `benchmark` entry point with `--eval-num native`: the benchmark
then selects its 25/50 episode counts. The following runs the entire Memory suite
sequentially with seed 1. Append `--dry-run` first to inspect all resolved commands.

```bash
EVAL_ENV_TYPE=sim bash scripts/robodojo.sh benchmark \
  --policy-dir "$BAGEL_POLICY_DIR" --ckpt cp1000 \
  --env-cfg arx_x5 --action-type ee --seed 1 --eval-num native \
  --policy-env "$BAGEL_POLICY_ENV" --eval-env "$BAGEL_SIM_ENV" \
  --policy-gpu "$BAGEL_GPU_ID" --env-gpu "$BAGEL_GPU_ID" \
  --dimension memory
```

Use `--dimension generalization`, `precision`, `long-horizon`, `open`, or `all`.
For one complete task, use `--only insert_tubes` with `--dimension precision`.
Generalization includes each standard task and its `_random` sibling; each has
25 episodes per seed. The five suites contain 54 runnable configurations, merged
into 42 task rows. Preserve canonical names such as `play_Xylophone`, `swap_T`,
`push_T` and `push_T_random`.

## Run the three-seed verification scope

Use a dedicated result workspace for a complete run, separate from the smoke
checks. This command is a full benchmark, not a quick installation test. Run it
only with the intended evaluation allocation and disk space. It runs tasks and
seeds sequentially; it never changes the checkpoint or inference settings.

```bash
set -euo pipefail
for BAGEL_EVAL_SEED in 0 1 2; do
  EVAL_ENV_TYPE=sim bash scripts/robodojo.sh benchmark \
    --policy-dir "$BAGEL_POLICY_DIR" --ckpt cp1000 \
    --env-cfg arx_x5 --action-type ee --seed "$BAGEL_EVAL_SEED" \
    --eval-num native --dimension all \
    --policy-env "$BAGEL_POLICY_ENV" --eval-env "$BAGEL_SIM_ENV" \
    --policy-gpu "$BAGEL_GPU_ID" --env-gpu "$BAGEL_GPU_ID"
done
bash scripts/robodojo.sh summarize
```

The official [Quick Evaluation guide](https://robodojo-benchmark.com/doc/usage/quick-evaluation/)
also documents multi-GPU task sharding. Use one unbatched policy process per
worker; `eval_batch: false` is required. Shared state across independent episodes
or changing inference settings is not part of this submission.

## Preserve and report the results

Keep each run's native `_result.json`, logs, source revisions and videos. Run
`bash scripts/robodojo.sh summarize` from the matching RoboDojo checkout and retain
`eval_result/RoboDojo/_summary.md`. The native summary distinguishes incomplete
coverage; its across-seed statistics must not be replaced with statistics over
episodes or combined with the two-episode smoke results.

The default reevaluation uses original supplied layouts. If a scene fails before
any robot action, preserve its startup error and retry history; do not silently
replace the scene or score an infrastructure failure as a policy failure. Seek
the committee's handling decision. The historical research run used three
labeled replacements and one repaired Memory variant, described in
[EVALUATION_RESULTS.md](EVALUATION_RESULTS.md); those exceptions are not activated
by these commands. Official verification and acceptance remain with RoboDojo.

## Common setup problems

| Symptom | Check |
|---|---|
| HTTP 401/403 or repository not found | The checkpoint repository is currently private. Set the supplied repository-only read token as `HF_TOKEN` and use the canonical repository and pinned revision. Removing authentication will not fix private access. After public release, anonymous access can be checked with the token unset. |
| Hash/size mismatch | Check the pinned revision and local file completeness; do not disable verification. |
| Missing encoder or tokenizer | Run the full downloader; `--metadata-only` does not fetch tensors or encoders. |
| Missing `env_cfg` or simulator | Complete the official workspace setup; `env_cfg/` belongs beside XPolicyLab. |
| Shared Isaac/Kit cache permissions | Use a writable installation/cache; the recorded local workaround is documented in VALIDATION.md. No task or scoring patches are supplied. |
| Insufficient GPU memory | Qualification used a 96 GB RTX PRO 6000 Blackwell; a smaller minimum has not been qualified. |

The policy logs the checkpoint identity after file verification. Keep that log
with the run so the evaluated checkpoint is unambiguous.
