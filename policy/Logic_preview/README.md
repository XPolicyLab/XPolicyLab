# Logic_preview for RoboDojo

Contributed by garlicba.

`Logic_preview` takes a task instruction, three RGB camera views and the absolute dual-arm end-effector state of the ARX X5. It returns 16-dimensional end-effector actions. All tasks use the same paired video and action checkpoint, `joint2b_746_s010000`, and the same inference settings.

## Run the full evaluation

An existing RoboDojo environment can run both the policy and the simulator. The policy requires **Python 3.10 or later**, a CUDA-enabled PyTorch 2.5 or later, and a matching torchvision 0.20 or later. For example, to reuse a compatible Python 3.11 environment and evaluate every runnable task at layout seed 0, run this from the XPolicyLab repository root:

```bash
POLICY_ENV=RoboDojo EVAL_ENV=RoboDojo bash policy/Logic_preview/run_evaluation.sh full --seeds 0
```

The launcher prints the selected environments, Python executable, PyTorch, torchvision and CUDA versions. With no `POLICY_ENV` override, it first tries an existing `logic_preview` environment. It can reuse the simulator environment if its Python and PyTorch stack are compatible and the policy's declared dependencies allow the versions already installed. For example, XPolicyLab requires `websockets>=14`, while some Isaac Sim images pin `websockets==12`. On those images, use a separate policy environment. The launcher creates `logic_preview` when needed, using the simulator's Python major and minor version if it is 3.10 or later. Set `POLICY_ENV` to an existing environment name or prefix to select it explicitly.

To run all three official layout seeds with the default GPU assignments:

```bash
bash policy/Logic_preview/run_evaluation.sh full
```

Use a Linux host with a compatible NVIDIA driver, Conda, and the RoboDojo simulator and assets installed. The simulator environment defaults to `RoboDojo`. The downloader uses curl, zstd and tar with zstd support, and needs access to the model repositories. The installer needs access to Python package indexes. Allow **110 GB of free disk space** for model assets during setup, plus space for the environments and results.

The launcher downloads and verifies the weights and starts a policy server for each evaluation seed. If the selected policy environment needs a compatible PyTorch build, the launcher installs PyTorch 2.11.0 and torchvision 0.26.0 from the CUDA 12.8 wheel index. Set `TORCH_INDEX_URL` to override that index. By default, GPUs 0–3 share inference on one host and GPU 4 runs the simulator. For a two-GPU host where GPU 0 can hold the whole policy, run the same evaluation with GPU 1 assigned to simulation.

```bash
POLICY_GPU=0 ENV_GPU=1 EVAL_ENV=RoboDojo bash policy/Logic_preview/run_evaluation.sh full
```

By default the official client runs layout seeds **0, 1 and 2** in sequence. Pass one or more seeds to `full` (for example, `full --seeds 0` or `full --seeds 0 2`) to select rounds. Each round uses its seed for both the simulator layout and the policy RNG. The default (layout, policy) pairs are **(0, 0), (1, 1) and (2, 2)**. Each round covers the installed runnable inventory at its native episode counts. The reference inventory has 54 runnable task variants, which the official summary groups into 42 entries. The launcher then runs `bash scripts/robodojo.sh summarize`. In a fresh result directory, expect 42 completed official entries for one seed, or **126/126** for the default three. Open `RoboDojo/eval_result/RoboDojo/_summary.md` to inspect the results.

| Shorter run | Command |
|---|---|
| One simulator episode on `stack_bowls` | `bash policy/Logic_preview/run_evaluation.sh smoke` |
| Standard debug loops with ordinary and encoded RGB inputs | `bash policy/Logic_preview/run_evaluation.sh debug` |

## Parallel simulator evaluation

Full runs use the 746 layout scheduling strategy by default. The launcher splits each task evenly into shards of at most 12 layouts. Set `ENV_GPU` to the simulator GPU ids to run shards concurrently. Each client runs its own environment and shares the policy server for the current seed. For example, with GPUs 0–3 for the policy and 4–7 for simulation, run the following command.

```bash
POLICY_GPU=0,1,2,3 ENV_GPU=4,5,6,7 \
  bash policy/Logic_preview/run_evaluation.sh full --seeds 0
```

The scheduler uses the installed RoboDojo inventory and starts expensive shards first using the 746 per-task reference times. Each simulator slot runs the standard client. Repeating a GPU ID in `ENV_GPU` assigns multiple client processes to that GPU when memory allows. Set `LAYOUTS_PER_SHARD` to change the maximum initial shard size, or to `0` for RoboDojo's native task scheduler.

### Multiple machines and recovery

For four eight-GPU machines, start the following command on **each machine** from the XPolicyLab root. Use the same run ID and seed list on all four. Each machine starts its own four-GPU policy server and four simulator clients. All 16 clients draw shards from the current seed before the next seed starts. Omit `--seeds 0` to run the three paired rounds in sequence.

```bash
LOGIC_PREVIEW_RUN_ID=paired-eval-01 POLICY_GPU=0,1,2,3 ENV_GPU=4,5,6,7 \
  bash policy/Logic_preview/run_evaluation.sh full --seeds 0
```

Use identical policy code, checkpoints, simulator assets and task configurations on all nodes. Mount the same RoboDojo `eval_result/` at the same absolute path on each machine, on a filesystem supporting cross-node `flock`, atomic rename and hard links. Use a fresh result root for a new comparison because the official summarizer can also see older runs. For several policy servers on one host, assign separate policy GPU sets and simulator slots to each launcher and prepare their environments before concurrent use. A separately provisioned server can also serve a group via `packed_eval.py --policy-host HOST --port PORT` from the RoboDojo root. Start that server with the group's checkpoint and policy seed.

The run ID identifies the shared plan under `eval_result/.logic-preview/<run-id>/seed-<seed>/`. Relaunch with the same ID and arguments after interruption. Each shard keeps completed episode records and videos, including scored failures. Only unresolved layouts are retried, up to five attempts. When a fatal error appears in the simulator or Kit log, the watchdog stops the affected shard for recovery. Set `SHARD_TIMEOUT_SECONDS` or `SHARD_IDLE_TIMEOUT_SECONDS` to enable a wall-time or no-progress limit. Exit codes 2 and 78 end retries for an incomplete shard. Other tasks can finish. After fixing the cause of a terminal failure, set a new, common `LOGIC_PREVIEW_RETRY_ID` on the launchers to reopen failed shards within the original run.

When the simulator identifies unstable layouts, the scheduler requests the next unused IDs, up to the deficit plus two. It publishes the lowest stable layout IDs up to the native target, with the original scores and three camera videos per episode. A task with too few stable layouts remains incomplete. The launcher runs the official summary after every requested round has finished.

## Download the checkpoint separately

The full evaluation includes this step. To stage the weights ahead of time, run the following command from the XPolicyLab root with Python 3.10 or later.

```bash
PYTHON_BIN=python3 bash policy/Logic_preview/download_checkpoints.sh
```

- **Policy checkpoint.** The downloader fetches five volumes in parallel (28.48 GB total), verifies each SHA-256 digest and reconstructs four files (36.85 GB total) under `joint2b_746_s010000/`. The policy loads its video backbone from this checkpoint's `video.pt`. [ModelScope volumes](https://modelscope.cn/models/garlicba/logic_preview/files)
- **Shared Wan components.** The downloader fetches the VAE, CLIP and UMT5 encoders, tokenizer, configuration and license files (16.66 GB total) from a pinned revision. [Wan release](https://huggingface.co/Wan-AI/Wan2.1-I2V-14B-480P/tree/6b73f84e66371cdfe870c72acd6826e1d61cf279)

The downloader verifies all 13 extracted or downloaded files against the supplied SHA-256 digests and places them under `policy/Logic_preview/weights/`. Rerun the command after an interruption. The ModelScope revision defaults to `master`. Set `LOGIC_PREVIEW_MODEL_REVISION` to select a specific commit or `MODELSCOPE_API_TOKEN` if the repository requires authenticated access.

Set `LOGIC_PREVIEW_WEIGHTS_DIR` to a persistent local directory containing both model directories to reuse it across jobs. The downloader verifies the SHA-256 digests and the launcher reads the same files. A fully staged cache can be mounted read-only.

For offline setup, pass `--parts-dir /path/to/five-volumes` and `--wan-dir /path/to/Wan2.1-I2V-14B-480P` to the downloader. The policy server also accepts `LOGIC_PREVIEW_BUNDLE` and `LOGIC_PREVIEW_WAN_ROOT` for existing verified installations.

## Policy integration

The adapter uses the standard XPolicyLab model, deployment and evaluation entry points, including separate policy server and simulator client scripts. XPolicyLab decodes the camera views. The policy tracks observations separately for each client and resets its history at the start of each episode. It predicts 36 actions, executes up to 24, then replans from the latest observation.

The adapter combines the three RGB views at 240 × 416. It packs each arm's position, quaternion (wxyz) and gripper state in left-to-right arm order and uses the checkpoint's normalization statistics. See the framework documentation for split-machine deployment. [XPolicyLab README](https://github.com/XPolicyLab/XPolicyLab/blob/main/README.md)

The framework also supports its standard ten-argument entry point. For example, with the policy and client environments already installed, run one debug session as follows.

```bash
EVAL_ENV_TYPE=debug LOGIC_PREVIEW_INFERENCE_DEVICE_MAP=balanced \
  bash policy/Logic_preview/eval.sh RoboDojo stack_bowls joint2b_746_s010000 \
  arx_x5 ee 1 0,1,2,3 4 logic_preview RoboDojo
```

Arguments in order are benchmark, task, checkpoint, robot, action type, shared layout and policy seed, policy GPU(s), simulator GPU, policy environment, client environment. The policy uses 10 Euler steps, shift 1, 8 history frames at stride 25 and an episode anchor. Every replanning step predicts 36 actions and executes up to 24. The simulator checks for task termination after each action. Vectorized batches are unsupported.

## Local evaluation results

In a local RoboDojo run predating the paired-seed launcher, `logic_preview` using checkpoint `joint2b_746_s010000` achieved an **Avg Score of 43.19 ± 0.71** and an **Avg SR of 36.74% ± 0.57 percentage points** across the five capability dimensions. The run covered all 42 tasks at each of layout seeds 0, 1 and 2 (126/126 task and seed combinations). Both metrics show the mean and standard deviation across seeds. [Full task and seed breakdown](_summary.md)

## Testing

The original release passed GPU debug runs with the paired checkpoint using both ordinary and encoded RGB inputs. Each mode completed 10 episodes and 200 control steps. Both logs ended with the following lines. [Archived debug results](evidence/standard-debug.json)

```text
[TestEnv] Action Step: 19 / 20 (step_limit)
[TestEnv] Check Episode End: True
[TestEnv] Check Episode End: True
[TestEnv] Episode finished
[MAIN] eval finished
```

Replaying 25 completed Hangzhou scenes (1,200 episodes and 3,600 camera videos) through the layout scheduler reproduced each task's result JSON and the official Markdown summary. This check covered result assembly, including the native fallback. A live GPU comparison of serial and sharded execution is still pending.

Checks on this package covered shell syntax, inference modules, the four loading targets in the checkpoint metadata, launcher arguments, the command sequence for all three modes and reconstruction from five local volumes. A fresh end-to-end run of the packaged launcher, including online weight download, CUDA environment setup and GPU simulation, remains pending.

## Evaluation request

This PR requests inference-only evaluation. Training and data preparation materials are planned for a later submission. The release date and maintainer approval for this scope are pending.
