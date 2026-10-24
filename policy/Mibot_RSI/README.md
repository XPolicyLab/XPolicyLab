# Mibot_RSI

**Contributor:** Mibot-RSI team | **Original code:** Mibot-RSI (released within three days of the leaderboard listing)

Mibot-RSI is an eval-only agent entry: GPT-6-Astra (Responses API) sees the three RGB cameras and the robot state, writes Python against a small robot SDK and moves the ARX X5 arms (`bench_name=RoboDojo`, `env_cfg_type=arx_x5`, `action_type=joint`, `eval_batch=false`). What it learned while practising on the public seed-0 layouts is a frozen, read-only knowledge snapshot per task, downloaded as the checkpoint `eval`. `runtime/` holds the agent code; `runtime/EXPORT.json` pins every file by sha256, and an evaluation refuses to run if a file differs.

Shared conventions — argument meanings, checkpoint naming, split-machine deployment, `EVAL_ENV_TYPE` — are documented in the [XPolicyLab README](../../README.md). Official results: [RoboDojo LeaderBoard](https://robodojo-benchmark.com/LeaderBoard).

## Installation

The agent runs inside the RoboDojo evaluation client, so it uses two Python environments:

- **Evaluation environment**: RoboDojo's own, for example the `RoboDojo` conda environment. `install.sh` checks numpy, scipy, Pillow, opencv and pyyaml there and verifies `runtime/`. Add `--install-deps` to install missing packages.
- **Policy server environment**: a new, small one. It holds XPolicyLab itself, because the server needs `websockets>=14` and the RoboDojo environment has 12.0.

```bash
cd XPolicyLab/policy/Mibot_RSI
conda create -y -n mibot_rsi_policy python=3.11
bash install.sh --policy-python "$(conda run -n mibot_rsi_policy which python)" "$(conda run -n RoboDojo which python)"
```

Usage: `bash install.sh [--install-deps] [--policy-python <policy env python>] [<evaluation env python>]`. Without the last argument it uses `python3`.

## Data Processing

Unsupported: eval-only entry, so there is no `process_data.sh`.

## Training

Unsupported: eval-only entry, so there is no `train.sh`; no network is trained. The practice code that produced the knowledge snapshots is released with Mibot-RSI within three days of the leaderboard listing.

## Evaluation

First install (above), download the knowledge (Model Assets) and set the model endpoint:

```bash
export MIBOT_RSI_BASE_URL=<Responses API base URL serving gpt-6-astra>
export MIBOT_RSI_API_KEY=<key>
```

Then, from the RoboDojo root:

```bash
bash scripts/robodojo.sh eval --policy-dir XPolicyLab/policy/Mibot_RSI --task stack_bowls --ckpt eval \
  --policy-env mibot_rsi_policy --action-type joint --seed 1 --eval-num native
```

or with the standard XPolicyLab entry, from this directory:

```bash
bash eval.sh <bench_name> <task_name> <ckpt_name> <env_cfg_type> joint <seed> \
  <policy_gpu_id> <env_gpu_id> <policy_conda_env> <eval_env_conda_env>
# Example
bash eval.sh RoboDojo stack_bowls eval arx_x5 joint 1 0 0 mibot_rsi_policy RoboDojo
```

`EVAL_ENV_TYPE=debug` checks the wiring without a simulator or model.

Unlike demo_policy, `deploy.py` runs the agent inside the evaluation client. RoboDojo calls `eval_one_episode(TASK_ENV, model_client)`, and the agent's session runs in a subprocess. Its robot commands, mostly end-effector motions, become joint targets through inverse kinematics and are executed with `TASK_ENV.take_action`, hence `action_type=joint`. The policy server (`model.py`) serves no actions.

**Failures and restarts**

- **Model API errors** (outage, timeout, HTTP 429/5xx, and also HTTP 401–404 and quota errors) are first waited out inside the episode, as set by `model_api_waits_s`. Each wait is shown on the console.
- **Infrastructure failures** are not scored. This covers an API that still fails after the waits, a crashed agent session, a hung simulator (no progress for 10 minutes during a robot command, or a fatal simulator log line), and the 6-hour episode limit. The process exits with code 99, the code RoboDojo's own monitor uses, and `scripts/eval_policy.sh` restarts it with the same `ROBODOJO_RUN_ID` (at most `ROBODOJO_MAX_BASH_RETRIES` − 1 times, 9 by default, shared with RoboDojo's own PhysX restarts). The interrupted layout is then played again from its initial state. Each of these is logged in `runs/submission_reruns.jsonl`.
- **Setup problems** (a missing endpoint or key, invalid settings, a knowledge snapshot that fails its check) exit with code 1 and are printed on the console. Fix the problem, export the same `ROBODOJO_RUN_ID` (printed by `scripts/eval_policy.sh`) and run the same command to continue.

### What the agent sees

- RGB images from the three cameras;
- arm joint positions, gripper state and end-effector poses;
- the task instruction and name;
- the task's `step_lim` and the steps left;
- a plain-language text of the task's scoring rule, written by hand from RoboDojo's public reward code (`task/RoboDojo/tasks/<task>.py`);
- the task's knowledge snapshot.

It does not see the seed, the layout, simulator object states or the reward code.

## Model Assets

The checkpoint `eval` is the knowledge: one snapshot per task for the 42 tasks (a `_random` task uses its standard task's snapshot). It is in the private Hugging Face dataset `ffanyt/Mibot-RSI-knowledge`, and `release.json` pins the dataset commit and every archive's sha256.

```bash
huggingface-cli login                       # or export HF_TOKEN=...; needs access to the dataset
bash download_checkpoint.sh eval 0,1,2      # usage: download_checkpoint.sh <ckpt_name> [seeds, default 1] [python]
```

The knowledge is the same for every evaluation seed: it was written on seed-0 layouts, and seeds 1 and 2 use it unchanged. XPolicyLab looks for a checkpoint folder per seed (`checkpoints/RoboDojo-eval-arx_x5-joint-<seed>/`), so the script downloads the knowledge once to `checkpoints/mibot-rsi-knowledge-eval/` and links each listed seed's folder to it. Before each episode, the task's snapshot is checked against `release.json`.

## Configuration

Keys of `deploy.yml` beyond the standard set:

| Key | Value | Meaning |
| --- | --- | --- |
| `max_model_calls` | `300` | Model calls per episode; at the cap the episode is scored as it stands. |
| `episode_wall_limit_s` | `21600` | 6 h per episode; reaching it is an infrastructure failure (the layout is replayed). |
| `model_api_waits_s` | `[60, 300, 900, 1800, 3600]` | Waits before a model API failure stops the episode. |
| `model_settings` | see file | `gpt-6-astra`, reasoning effort `xhigh`, reasoning summaries `detailed` (only logged), timeouts and retries. No endpoint or key. |
| `runs_root` | `runs` | Per-episode records, relative to this directory; about 0.3–1 GB each. |
| `knowledge_dir` | `null` | Another knowledge folder; its snapshots must still be the ones `release.json` pins. |
| `runner_python` | `null` | Python of the agent session subprocess; by default the evaluation client's. |

Optional environment variables: `MIBOT_RSI_KNOWLEDGE_DIR` and `MIBOT_RSI_RUNS_ROOT` override `knowledge_dir` and `runs_root`; `MIBOT_RSI_HARNESS` names a runtime directory instead of `runtime/` (only a directory with its own `EXPORT.json` is checked file by file); `MIBOT_RSI_ARTIFACTS_DIR` reads a local copy of the dataset instead of Hugging Face.

## Notes

- The model is not deterministic, so a layout can end differently on another run.
- The agent's Python runs in a sandbox: a Python audit hook, plus Landlock on Linux kernels that support it. When Landlock is missing, the console says so and the episode's `run.json` records it.
- The model endpoint is reached directly: its host is added to `NO_PROXY`, so `HTTP(S)_PROXY` does not apply to it.
- `Image.open` in `runtime/` reads only the agent's own images (its recorded camera frames, images it made, demonstration-video frames in its knowledge snapshot), never XPolicyLab observation or trajectory data. Observations reach it as arrays.
