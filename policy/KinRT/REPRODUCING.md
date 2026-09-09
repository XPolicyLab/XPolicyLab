# Reproducing Full35 / 60k

This guide describes the current Full35 model and the ongoing seed-0 evaluation. The completed simulator subset is `stack_bowls`: **22/25 (88%)**, followed by `stack_bowls_random`: **4/25 (16%)**. Both used one fresh, continuously running policy server, corrected material paths, one simulator environment, joint control, and 50-action chunks. Other tasks remain outside these reported numbers; exact success counts can differ across hardware and simulator/rendering versions.

## Requirements

- Linux with CUDA 12 and a compatible NVIDIA driver. GPU execution passed on Ubuntu 22.04 / RTX 3090 24 GiB / driver `580.95.05`; simulator memory is additional to policy memory. The shared-GPU protocol below uses one simulator environment.
- A complete, working RoboDojo installation with Isaac Sim, robot assets, task configurations, seed-0 layouts, and a simulator Conda environment. The recorded runs used Isaac Sim 5.1. The policy installer does not install the simulator or supply its assets.
- This XPolicyLab revision at `<RoboDojo>/XPolicyLab`, Git, uv, and network access for source/dependency downloads. `install.sh` creates a Python 3.11 policy environment.
- Network access to the public, ungated HF repository `Gleez/kinrt-robodojo-full35-a800-60k`. No account, login, or token is required for this checkpoint.
- Model-download storage of approximately 15.8 GB, plus dependency, cache, simulator, log, and video space. Inference does not require the original training dataset or router-label `.npy`.

The measured simulator used these actual source revisions, followed by the explicit material-path preparation below:

| Component | Revision |
| --- | --- |
| RoboDojo core | `2184bf8844ea9d205382c4aefa3a694311418251` |
| IsaacLab | `afca7b09d60d8beb9c1cb28b43066499940b969b` |
| CuRobo | `895c6517243f8cb091c73c018c8167192d39599a` |

CuRobo was internally clean at the revision above, which differs from the RoboDojo commit's recorded submodule pointer `d17b54ce32cba095c0b000c4c58777075d11de0e`. A recursive checkout of that core commit alone therefore does not recreate the tested simulator. Keep these versions and the actual installed simulator/asset configuration with each result; this guide assumes the RoboDojo environment already works.

Use absolute paths for the following variables; replace their example values with your installed workspace and Conda environment:

```bash
export WORKSPACE=/absolute/path/to/workspace
export ROBODOJO_ROOT="$WORKSPACE/RoboDojo"
export XPL_ROOT="$ROBODOJO_ROOT/XPolicyLab"
export KINRT_OPENPI_ROOT="$WORKSPACE/KinRT_RoboDojo/policy/pi05"
export OPENPI_DATA_HOME="$WORKSPACE/cache/openpi"
export SIM_ENV=RoboDojo
export POLICY_GPU=0
export SIM_GPU=0
cd "$XPL_ROOT/policy/KinRT"
```

## Install And Cache Assets

```bash
bash install.sh "$KINRT_OPENPI_ROOT"
export POLICY_PYTHON="$KINRT_OPENPI_ROOT/.venv/bin/python"
bash download_checkpoint.sh
"$POLICY_PYTHON" full35_assets.py verify \
  --artifact-root "$PWD/checkpoints/KinRT-RoboDojo-Full35-60k"
"$POLICY_PYTHON" prepare_tokenizer.py --check
```

The checkpoint is pinned to HF revision `9460d07a9c7677ef3c72ece08df1f34eba7e45c7`; making the repository public did not change this revision. Anonymous `--assets-only` download passed with a new HF cache and no token, including all 14 delivery entries and checkpoint normalization. A normal download verifies 353 inference-checkpoint files and all 14 delivery-manifest entries; `--assets-only` does not download enough files to execute the model.

Installation pins KinRT to `590d52802cde804cdc2d0ccb672c1a3a90d76f91` and LeRobot to `8fff0fde7c79f23a93d845d1a50e985de01f8b8a`. LeRobot is a separate checkout whose `src/` takes precedence through the policy environment's `.pth` file. The guarded source helper reconstructs the documented task-table compatibility fix and supports v3 Parquet episode reading; the original training-time patch was not supplied.

Dependency installation uses `uv sync --frozen --no-default-groups`. `KINRT_PYPI_MIRROR` selects `pypi` (default), `tencent`, or `original`: only mirror URLs are changed temporarily after TOML validation, while package versions, hashes, and dependency semantics are retained. The original `uv.lock` is restored after synchronization; `pyproject.toml` is not edited. Set `UV_BIN` for a specific uv executable when it is not on PATH.

The installer downloads and verifies the separate PaliGemma tokenizer, which is not contained in the checkpoint repository. Its cache location is `${OPENPI_DATA_HOME}/big_vision/paligemma_tokenizer.model`, and its SHA-256 is `8986bb4f423f07f8c7f70d0dbe3526fb2316056c17bae71b1ea975e77a168fc6`. Keep the same `OPENPI_DATA_HOME` during evaluation. If installation stops because the tokenizer endpoint is unreachable, its policy Python has already been created: use that interpreter to import an identical pre-downloaded tokenizer, then rerun `install.sh`:

```bash
"$POLICY_PYTHON" prepare_tokenizer.py \
  --source-file /absolute/path/to/paligemma_tokenizer.model
"$POLICY_PYTHON" prepare_tokenizer.py --check
```

`POLICY_PYTHON` is `$KINRT_OPENPI_ROOT/.venv/bin/python`; export it explicitly after a stopped installation if needed. This local-copy option supplies the tokenizer only, not the other source and wheel downloads required for installation.

The installer has completed in a newly created Linux policy environment, including dependency synchronization, source preparation, tokenizer download with SHA-256 verification, and original-lockfile restoration. Its strengthened final check passed the actual KinRT adapter import from the new checkout, LeRobot v3.0 source selection, and OpenCV `4.11.0` import. Installation explicitly supplies upstream model import dependency `pytest==9.0.3` and reinstalls the headless OpenCV wheel after removing the GUI wheel's shared files. Full GPU checkpoint restoration, finite synthetic inference, and both standard raw/encoded debug modes also passed in this new environment; its simulator pair replay remains in progress. The [fresh runtime evidence](evidence/fresh_runtime_checks.json) records checks and log digests. The policy installer does not recreate the simulator environment.

## Prepare The Simulator

Stop simulation processes using this checkout before changing its configuration. This explicit helper is separate from `install.sh`:

```bash
"$POLICY_PYTHON" prepare_robodojo.py \
  --robodojo-root "$ROBODOJO_ROOT" --mode check --num-envs 1
"$POLICY_PYTHON" prepare_robodojo.py \
  --robodojo-root "$ROBODOJO_ROOT" --mode apply --num-envs 1
"$POLICY_PYTHON" prepare_robodojo.py \
  --robodojo-root "$ROBODOJO_ROOT" --mode check --num-envs 1
```

The final check must report `would_change=false` for both material sources and the simulator configuration. `check` alone only reports a plan; it does not apply the fixes. The helper changes three known material-path calls in `ground.py` and `table.py` from `Path.resolve()` to `Path.absolute()`, preserving MDL filenames when assets are HF-cache symlinks. It resolves the actual simulator YAML through `env_cfg/arx_x5.yml` and sets `scene.num_envs=1`. When it changes that configuration, original bytes and hashes are recorded in `.kinrt_full35_preparation.json`. `--mode revert` without `--num-envs` restores recorded state only when subsequent edits do not conflict; an originally single-environment YAML is left as it was. Unknown source changes or paths outside the checkout are rejected.

The recorded simulator configuration also uses `dt=0.004`, `decimation=1`, `render_interval=10`, and `scene.env_spacing=7`. Simulation runs with `device=cpu` and `use_fabric=false`, the defaults in the pinned RoboDojo core when those YAML keys are absent. Keep these settings if your installation specifies explicit overrides; the preparation helper changes the environment count, not these other fields. Policy inference and camera rendering still use the selected GPU. Changing the simulator device or Fabric setting changes the protocol and can also affect fluid or garment task support.

Inspect the first recorded frame: the intended table has the `Mahogany` wood material. An earlier run rendered an incorrect coral-colored tabletop; that batch was stopped and all its preliminary results excluded. Material correctness is part of this evaluation setup, not merely a video presentation setting.

```bash
export KINRT_CHECKPOINT_PATH="$XPL_ROOT/policy/KinRT/checkpoints/KinRT-RoboDojo-Full35-60k/checkpoints/60000"
export KINRT_TRAIN_CONFIG_NAME=kinrt_full_robodojo
export KINRT_REPO_ID=RoboDojo_lerobot_v30_video
export KINRT_CHECKPOINT_NUM=60000
export KINRT_ACTION_CHUNK_SIZE=50
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.8
export XLA_PYTHON_CLIENT_ALLOCATOR=platform
```

The `platform` allocator releases unused policy allocations and was used when policy and simulator shared the RTX 3090. One environment is the reported setup; increasing simulator batch size changes the memory requirement and protocol.

## Check The Standard Entry Point

Run both synthetic debug modes before starting the result-reproduction server:

```bash
EVAL_ENV_TYPE=debug DEBUG_OBS_ENCODED=0 \
  bash eval.sh RoboDojo stack_bowls full35_60k arx_x5 joint 0 \
  "$POLICY_GPU" "$SIM_GPU" "$KINRT_OPENPI_ROOT" "$SIM_ENV"
EVAL_ENV_TYPE=debug DEBUG_OBS_ENCODED=1 \
  bash eval.sh RoboDojo stack_bowls full35_60k arx_x5 joint 0 \
  "$POLICY_GPU" "$SIM_GPU" "$KINRT_OPENPI_ROOT" "$SIM_ENV"
```

Both must finish with `[MAIN] eval finished`. The recorded validation completed 10 default synthetic episodes per mode with debug batch size 10; this is distinct from the simulator's one-environment setting. Each `eval.sh` invocation starts and stops its own model server. It is suitable for an independent task check, but two independent invocations do not reproduce the server RNG history used for the result pair below.

Both modes passed again in the fresh policy environment, each reaching `[MAIN] eval finished` at line 461 of its log. The fresh offline smoke also produced finite `50 x 14` actions exactly equal to the original environment's output for the identical synthetic input. These checks verify execution and transport; they do not establish simulator task success.

## Reproduce The Reported Pair

Use **one fresh server**, then run all 25 native `stack_bowls` episodes followed immediately by all 25 native `stack_bowls_random` episodes. Do not call synthetic inference or other tasks on this server first, and do not restart it between the tasks. At the pinned upstream revision, the policy initializes its JAX RNG with key 0 and advances it on every inference; the adapter's `reset()` clears observations without resetting that RNG.

The following uses the adapter's standard server entry point and RoboDojo's official external-policy client. Run it after the independent debug checks have exited. The noninteractive Bash block keeps `setsid` process tracking and exit cleanup consistent when pasted into an interactive terminal:

```bash
bash <<'BASH'
set -euo pipefail
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$SIM_ENV"
cd "$XPL_ROOT/policy/KinRT"
export EVAL_ENV_TYPE=sim
export EVAL_NUM=native
export OMNI_KIT_ACCEPT_EULA=YES
export ISAACLAB_RL_FRAMEWORK=none
export ROBODOJO_MAX_BASH_RETRIES=2
export PYTHONUNBUFFERED=1
RUN_TAG="full35_seed0_$(date -u +%Y%m%dT%H%M%S)"
RUN_DIR="$WORKSPACE/kinrt-results/$RUN_TAG"
mkdir -p "$RUN_DIR"
POLICY_PORT="$(bash "$XPL_ROOT/utils/get_free_port.sh")"
setsid bash setup_eval_policy_server.sh \
  RoboDojo stack_bowls full35_60k arx_x5 joint 0 \
  "$POLICY_GPU" "$KINRT_OPENPI_ROOT" "$POLICY_PORT" 127.0.0.1 \
  >"$RUN_DIR/policy.log" 2>&1 &
POLICY_PID=$!
trap 'kill -TERM -- -"$POLICY_PID" 2>/dev/null || true' EXIT
bash "$XPL_ROOT/utils/wait_for_policy_server.sh" \
  127.0.0.1 "$POLICY_PORT" "$POLICY_PID" "KinRT Full35" 1200
for TASK in stack_bowls stack_bowls_random; do
  ROBODOJO_RUN_ID="${RUN_TAG}_${TASK}" \
    bash "$ROBODOJO_ROOT/scripts/robodojo.sh" client \
    --task "$TASK" --policy-name KinRT \
    --policy-host 127.0.0.1 --policy-port "$POLICY_PORT" \
    --env-cfg arx_x5 --seed 0 --env-gpu "$SIM_GPU" \
    --ckpt full35_60k --action-type joint --eval-num native \
    >"$RUN_DIR/${TASK}.log" 2>&1
done
BASH
```

`EVAL_NUM=native` and `--eval-num native` preserve each task's configured count; `src/eval_client/main.py` reads the environment value after resolving task configuration. Both configurations above require 25 episodes. A shortened smoke run or the debug client's 10 episodes is not this protocol. Inspect each new run's `_result.json` under `RoboDojo/eval_result/RoboDojo/<task>/KinRT/arx_x5/0_ckpt_name=full35_60k,action_type=joint/<run_id>/`: require `eval_time=25`, 25 distinct layout IDs, valid per-episode booleans, and a success rate recomputed from those details. Retain the logs, videos, source revisions, and preparation-helper JSON with their run IDs; do not mix earlier material-error runs or repeated layouts into the count.

The [reference evidence](evidence/seed0_reference_pair.json) records the original 50 episode outcomes, original-result digests, run IDs, and revisions. Its `fresh_install_replay=false` distinguishes that completed compatible-environment run from any subsequent reproduction. The source digests identify the original result files, not the transformed reference record; all episode booleans and scores were checked against those source files.

The currently observed 22/25 and 4/25 are a reference for this setup, not an acceptance condition. Differences in hardware, simulation, rendering, episode execution, or preceding policy calls can change trajectories and RNG consumption. Infrastructure failures should be diagnosed separately from a completed episode's policy failure.

## Full35 Scope

The delivery metadata confirms **35 training tasks, 3,500 episodes, and 1,859,602 frames**; the four router classes are a separate concept. The number 34 describes simulator availability, not a reduced training task count. The available simulator checkout supports 34 canonical tasks, including 12 fixed/random pairs: 46 configurations and 1,700 native episodes in total (`22 x 50 + 12 x (25 + 25)`). This is the planned available scope, not the completed count. The training task that spells `RoboDojo` lacks simulator code, configuration, and layouts; do not substitute another task for it.

The fresh policy environment's 25+25 replay is still running. Planned continuation over the other 44 configurations / 1,650 episodes starts a new policy server, so its records must identify that RNG boundary rather than imply continuity from the pair. Neither that continuation nor the pending replay is included in the completed reference counts.

The two reported configurations constitute one canonical task pair. They do not establish a Full35 average, a multi-seed result, or a matched 60k Pi 0.5 baseline. Continue reporting per-configuration counts and explicitly identify unavailable tasks and incomplete runs.
