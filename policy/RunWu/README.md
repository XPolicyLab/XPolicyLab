# RunWu

**Contributor:** RunWu-RSI authors | **Paper:** Technical report to be released after the leaderboard listing | **arXiv:** TBD | **Original code:** [RunWu-RSI/RunWu-RSI](https://github.com/RunWu-RSI/RunWu-RSI)

RunWu is a model-driven agent policy for RoboDojo: at every decision a multimodal model, reached through a Responses API, reads the camera images and the arm state and issues end-effector targets, which a Cartesian servo executes. Its prompts, tools and parameters are stored in a PolicyBundle, a content-addressed JSON file that is the checkpoint, and one checkpoint name serves all 42 tasks. This adapter supports `bench_name=RoboDojo`, `env_cfg_type=arx_x5` and `action_type=ee`; the runtime (`runwu_rsi`) and the checkpoints stay in the original repository, and `install.sh` and `download_checkpoint.sh` fetch them from there.

Shared conventions — argument meanings, checkpoint naming, split-machine deployment, `EVAL_ENV_TYPE` — are documented in the [XPolicyLab README](../../README.md). Official results: [RoboDojo LeaderBoard](https://robodojo-benchmark.com/LeaderBoard).

## Installation

The policy server runs in a conda environment with Python 3.11 or newer (`setup_eval_policy_server.sh` calls `conda activate`). `install.sh` installs XPolicyLab in editable mode and `runwu_rsi` into the Python you give it:

```bash
cd XPolicyLab/policy/RunWu
conda create -n runwu python=3.11 -y
bash install.sh "$(conda run -n runwu which python)"
```

`runwu_rsi` comes from `RUNWU_RSI_SRC` (a local checkout or a pip requirement) if set, otherwise from the source archive of the original repository at `RUNWU_RSI_REF` (a tag or commit; default `v0.8.1`). It is an archive and not a `git+https` requirement because pip would then also clone the repository's large submodules (RoboDojo, IsaacLab, cuRobo), which the policy server does not need. The simulation side needs only `deploy.py`, which does not import `runwu_rsi`.

## Data Processing

Not required for this eval-only adapter; `process_data.sh` is omitted. The adapter reads no trajectory data. The released checkpoints contain keyframe images and statistics that the bundle generators of the original repository derived from the official LeRobot v2.1 export (`RoboDojo_lerobot_v21_video`). The generators read that export as published (`meta/`, `data/`, `videos/`), with no conversion step, and only when a bundle is built, never during evaluation; see "How the checkpoints were developed".

## Training

Not supported by this integration; `train.sh` is omitted. There is no gradient training: the "weights" are PolicyBundles, produced by the generators in `runwu_rsi/bundles/` and improved offline by the evolution loop in `runwu_rsi/evolution/` of the original repository ([docs/system2.md](https://github.com/RunWu-RSI/RunWu-RSI/blob/v0.8.1/docs/system2.md)).

## Evaluation

```bash
cd XPolicyLab/policy/RunWu
bash download_checkpoint.sh [task|all] [ref]        # fetch the bundles, see "Model Assets"
export ASTRA_ENDPOINT=<Responses API URL ending in /responses> ASTRA_API_KEY=<key>   # see "Notes"
bash eval.sh <bench_name> <task_name> <ckpt_name> <env_cfg_type> <action_type> <seed> \
  <policy_gpu_id> <env_gpu_id> <policy_conda_env> <eval_env_conda_env>

# Example
bash eval.sh RoboDojo stack_bowls RunWu arx_x5 ee 0 0 0 <policy_conda_env> <eval_env_conda_env>

# Example: debug closed loop, no simulator
EVAL_ENV_TYPE=debug bash eval.sh RoboDojo stack_bowls RunWu arx_x5 ee 0 0 0 <policy_conda_env> base
```

Use `ckpt_name=RunWu` and `action_type=ee` for every task. The original repository also contains a runner that evaluates many cases, including random layouts, in parallel and records the code version of each result ([docs/evaluation.md](https://github.com/RunWu-RSI/RunWu-RSI/blob/v0.8.1/docs/evaluation.md) section 4); it uses this same adapter.

## Model Assets

One checkpoint, `ckpt_name=RunWu`, serves all 42 tasks. The policy takes the bundle of the task it is run on (`task_name`); the runtime and the inference procedure are the same for every task, and the bundle carries the task's text and parameters. The generalization variant `<task>_random` of a task (12 of them) runs with the bundle of `<task>`:

```
checkpoints/RunWu/<task_name>/
    <bundle id>.json        # exactly one bundle file; its name is the SHA-256 digest of its content
    demos/                  # optional: images of the one-shot demonstration
    resources/              # optional: files declared by the bundle's function tools
```

The tasks are `align_blocks`, `arrange_largest_number`, `build_tower`, `classify_objects`, `classify_objects_by_language`, `cover_blocks`, `deposit_coin`, `fasten_screws`, `fill_egg_holder`, `fill_pen_holder`, `fold_clothes`, `general_pickup`, `hang_mugs`, `imitate_sorting_sequence`, `insert_key`, `insert_tubes`, `make_kong`, `make_toast`, `match_and_pick_from_conveyor`, `organize_table`, `pack_objects_into_box`, `pick_from_conveyor_by_image`, `play_Xylophone`, `play_stacking_toy`, `play_tic_tac_toe`, `plug_in_charger`, `pour_balls_into_vase`, `pour_by_language`, `pour_liquid_into_cup`, `press_by_number`, `push_T`, `put_bottles_into_dustbin`, `solve_equation`, `sort_nesting_dolls_by_size`, `stack_blocks`, `stack_blocks_by_language`, `stack_bowls`, `store_laptop_and_headphones`, `store_tools_in_toolbox`, `swap_T`, `swap_blocks` and `sweep_blocks`.

`download_checkpoint.sh [task|all] [ref]` fetches only the requested `checkpoints/RunWu/<task>/` directories from the original repository (shallow, sparse git checkout) at `ref` (default `$RUNWU_RSI_REF`, else `v0.8.1`), and skips a directory that already exists. The 42 checkpoints are frozen at the release tag `v0.8.1`; a changed checkpoint has a new bundle id and a new release tag.

`ckpt_name` is resolved with XPolicyLab's shared checkpoint resolver. A checkpoint directory that has no bundle file of its own but a subdirectory named after `task_name` serves all tasks, which is how `RunWu` works. A `ckpt_name` that is a path to a directory or to a bundle file also works, and so does a bundle id (or a unique prefix of at least 8 hex digits) together with `bundle_store`. If `demos/` exists and `RSI_DEMO_STORE` is not set, it is used as the demonstration store; likewise `resources/` becomes `RSI_TOOL_RESOURCE_ROOT`.

### How the checkpoints were developed

- **Layouts.** Development and tuning used the public RoboDojo layouts of seeds 0–2, so scores on those layouts are not held-out results.
- **Scene quantities.** Every object size, placement statistic and other scene quantity in the policy text and tools is measured from evidence and written with its measurement error. The evidence is training episodes, rollout images, and gripper and contact readings. Each value traces to an entry in [docs/physics/](https://github.com/RunWu-RSI/RunWu-RSI/tree/v0.8.1/docs/physics/). Each task document ends with a table of every number its checkpoint uses and that number's source: a measurement entry, a platform constant, the scoring code, or a stated strategy parameter. Object sizes, placement ranges and training-set priors also have per-task parameter tables with their sources and scripts ([docs/physics/params/](https://github.com/RunWu-RSI/RunWu-RSI/tree/v0.8.1/docs/physics/params/)); strategy parameters are not traced one by one. Where a quantity could not be measured, the policy states only the action. The checkpoints contain no layout or configuration information, no asset metadata and no ground-truth depth.
- **Scoring.** The descriptions of how each task is scored were derived from RoboDojo's open-source evaluator code.
- **Task texts.** The task descriptions and scoring conditions in the bundles are taken from RoboDojo's public task documentation. Part of the shared protocol text is adapted from RoboICL (MIT License). See [NOTICE](https://github.com/RunWu-RSI/RunWu-RSI/blob/v0.8.1/NOTICE).
- **Training episodes.** 29 checkpoints carry `demos/`, which are keyframes of RoboDojo training episodes (dataset `RoboDojo_lerobot_v21_video`), and most bundles carry `priors`, which are statistics over the keyframes of the training set. That dataset is licensed under the Apache License 2.0; the attribution is in [NOTICE](https://github.com/RunWu-RSI/RunWu-RSI/blob/v0.8.1/NOTICE).
- **At run time** the policy sees only the camera images and the proprioception that RoboDojo's client sends. It reads no simulator state; the only file it reads from the simulator's assets is the robot URDF, which its inverse-kinematics guard uses.

## Configuration

`deploy.yml` has the standard key set of `policy/demo_policy/deploy.yml`; the launcher fills `task_name`, `seed`, `action_type` and so on, and a null `action_type` is read as `ee`. The keys this policy adds:

| Field | Meaning |
|---|---|
| `bundle_id` | Optional: the content digest of the bundle to load. Wins over `ckpt_name` |
| `bundle_store` | The directory containing `<bundle_id>.json`. Used with `bundle_id`, and to look up a `ckpt_name` that is a bundle id |
| `result_dir` | Output directory of the ExecutionTrace: one run subdirectory per server process, one `<episode>.json` per episode. A null value falls back to `results/` inside this directory |
| `get_action_timeout_s` | Default 100. RoboDojo's client gives every call 120 s and cannot be told otherwise, while one decision (several model turns) can take longer. After this many seconds `get_action` returns a hold of the latest observed pose, and the next call hands out the action the decision produced. 0 disables the hold |
| `request_timeout_s` | Upper bound on the timeout of a single call; it must cover the longest single decision. The public RoboDojo client does not pass it through; the runner of the original repository applies it inside the simulator process |

## Notes

- **Model access.** Every decision calls a Responses API. All 42 bundles name the model `gpt-6-astra`, so evaluation needs a key and an endpoint that serves it, given to the policy server process as `ASTRA_API_KEY` and `ASTRA_ENDPOINT`; there is no field for them in `deploy.yml`. `ASTRA_ENDPOINT` must be an absolute HTTP(S) URL that ends in `/responses`, and the endpoint must accept image input. Responses are streamed by default (`ASTRA_STREAM=0` turns streaming off), and the client connects directly (`ASTRA_NETWORK_MODE=system` uses the system proxy). The key reaches only the policy server process. The model behind the endpoint is outside the checkpoint, so the same bundle can score differently at different times.
- **Supported setup.** Only `bench_name=RoboDojo`, `env_cfg_type=arx_x5` and `action_type=ee`. The runtime builds the end-effector action of the two-arm `arx_x5` layout itself (the model's tool takes six delta values and a gripper value per arm) and does not look the dimensions up with `get_robot_action_dim_info`, so other robots are not supported.
- **Single environment.** `eval_batch` stays `false`; `update_obs_batch` and `get_action_batch` raise `NotImplementedError`.
- **Latency.** If the model endpoint stays unreachable past the decision deadline, `get_action` raises and RoboDojo moves on to the next layout.
- **Scripts.** `eval.sh`, `setup_eval_policy_server.sh` and `setup_eval_env_client.sh` are copied verbatim from `policy/demo_policy`. `deploy.py` is its loop plus one `end_episode` call at the end, because XPolicyLab does not call the policy's `close()`.

## Known Issues

The two issues below occur in the simulator process, not in this adapter or in the checkpoints, so they can affect any policy. They were found in a test of RunWu-RSI `v0.8.1`: 756 episodes (42 tasks × 18 layouts, seeds 0–2) through RoboDojo `09bd5f5`, the version pinned by `v0.8.1`, with Isaac Sim 5.1 on RTX 5090 and RTX 6000 Ada machines. Each hung episode was stopped and its layout rerun; hung episodes are not counted in our results.

### `make_toast` hangs at "Starting the simulation" on some machines

- **Symptom.** No step runs. The log stops after the toaster's collision-cooking and joint warnings, `kit.log` reports `Hang detected` every two minutes, and the process does not recover.
- **Cause.** For tasks whose config has an `Articulation` section (`fill_egg_holder`, `make_toast`, `press_by_number`, `store_laptop_and_headphones`, `swap_blocks` and their `_random` variants), RoboDojo's [`src/eval_client/physx_warning_monitor.py`](https://github.com/RoboDojo-Benchmark/RoboDojo/blob/09bd5f5e762b949f5858ece31347ce62057b3a5e/src/eval_client/physx_warning_monitor.py) does two things before Kit starts:
  - it routes the simulator's stdout and stderr through pipes that Python threads in the same process read;
  - it tries to enlarge those pipes (`_enlarge_pipe`, which ignores a failure).

  While the scene starts, Kit's main thread holds the Python GIL and writes its warnings to stdout. Once the pipe is full, the main thread blocks in `pipe_write`, and the reader thread cannot take the GIL to drain it. At this point `make_toast` writes 13–28 KB of warnings, `store_laptop_and_headphones` about 8.6 KB, and the other monitored tasks 6–7 KB.
- **Affected machines.**
  - Linux counts pipe buffers per user. When one user's processes already hold more than `fs.pipe-user-pages-soft` pages (default 16384) and a process has neither `CAP_SYS_RESOURCE` nor `CAP_SYS_ADMIN`, its new pipes get 8 KB and cannot be enlarged.
  - In containers that run as root, the count covers every root process on the node, including those of other containers.
  - On three such Kubernetes machines, `make_toast` hung in 13 of 13 attempts and `store_laptop_and_headphones` in 1 of 14. On the machines whose pipes could be enlarged to 1 MiB, neither task hung. The GPU driver version was not the cause.
- **Check before an evaluation.** Run the command below as the user that runs the evaluation. It prints `1048576` on a machine that is not affected. `8192`, or a `PermissionError`, means the machine is affected.

  ```bash
  python3 -c "import fcntl,os; r,w=os.pipe(); fcntl.fcntl(r,1031,1<<20); print(fcntl.fcntl(r,1032))"
  ```

- **Workarounds.** Use any one of these:
  - run the evaluation as a non-root user, which has its own pipe budget;
  - give the container `CAP_SYS_RESOURCE`;
  - raise `fs.pipe-user-pages-soft` on the host.

  We verified the first one: on an affected machine, the same `make_toast` case on the same GPU hung as root and ran to completion as a non-root user.

### Startup stalls while Kit downloads materials

- **What happens.** The camera model `Assets/Sensor/Camera/realsense.usd` uses three materials from NVIDIA's online material library: Aluminum_Cast, Aluminum_Anodized and Plastic_ABS. When they are not in Kit's cache, Kit downloads 3 MDL files and 6 textures from `omniverse-content-production.s3.us-west-2.amazonaws.com` while the scene loads, and the main thread waits for them.
- **What we saw.**
  - The downloads took 37–125 s per start (median for each machine) and at most 443 s.
  - In 6 of the 756 counted episodes, Kit reported `Hang detected` for 2–5 minutes and then continued normally.
  - In one start a texture arrived incomplete (`Texture upload failed`), and the camera body was rendered without it; that episode was not affected otherwise.
  - The machine has to reach this server. We did not test a machine without access to it.
- **How to read the logs.**
  - `Hang detected` during this phase is not a reason to stop the process.
  - On machines without `zenity`, Kit's hang dialog writes `sh: zenity: not found` to the log every two minutes. A log file that keeps changing is therefore not a sign of progress; the step counter (`step: N / M`) is.
- **One stop of unknown cause.** One of the 756 episodes (`match_and_pick_from_conveyor`, seed 0, layout 42, RTX 6000 Ada) stopped at step 0, at the first render after reset, and did not recover within two hours. The rerun of that layout succeeded.
  - It was not the pipe issue above: this task does not use the monitor.
  - It was not the material download: all 9 files had already arrived.
