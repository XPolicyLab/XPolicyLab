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

`runwu_rsi` comes from `RUNWU_RSI_SRC` (a local checkout or a pip requirement) if set, otherwise from the source archive of the original repository at `RUNWU_RSI_REF` (a tag or commit; default `v0.1.0`). It is an archive and not a `git+https` requirement because pip would then also clone the repository's large submodules (RoboDojo, IsaacLab, cuRobo), which the policy server does not need. The simulation side needs only `deploy.py`, which does not import `runwu_rsi`.

## Data Processing

Not required for this eval-only adapter; `process_data.sh` is omitted. The adapter reads no trajectory data. The released checkpoints contain keyframe images and statistics that the bundle generators of the original repository derived from the official LeRobot v2.1 export (`RoboDojo_lerobot_v21_video`). The generators read that export as published (`meta/`, `data/`, `videos/`), with no conversion step, and only when a bundle is built, never during evaluation; see "How the checkpoints were developed".

## Training

Not supported by this integration; `train.sh` is omitted. There is no gradient training: the "weights" are PolicyBundles, produced by the generators in `runwu_rsi/bundles/` and improved offline by the evolution loop in `runwu_rsi/evolution/` of the original repository ([docs/system2.md](https://github.com/RunWu-RSI/RunWu-RSI/blob/v0.1.0/docs/system2.md)).

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

Use `ckpt_name=RunWu` and `action_type=ee` for every task. The original repository also contains a runner that evaluates many cases, including random layouts, in parallel and records the code version of each result ([docs/evaluation.md](https://github.com/RunWu-RSI/RunWu-RSI/blob/v0.1.0/docs/evaluation.md) section 4); it uses this same adapter.

## Model Assets

One checkpoint, `ckpt_name=RunWu`, serves all 42 tasks. The policy takes the bundle of the task it is run on (`task_name`); the runtime and the inference procedure are the same for every task, and the bundle carries the task's text and parameters. The generalization variant `<task>_random` of a task (12 of them) runs with the bundle of `<task>`:

```
checkpoints/RunWu/<task_name>/
    <bundle id>.json        # exactly one bundle file; its name is the SHA-256 digest of its content
    demos/                  # optional: images of the one-shot demonstration
    resources/              # optional: files declared by the bundle's function tools
```

The tasks are `align_blocks`, `arrange_largest_number`, `build_tower`, `classify_objects`, `classify_objects_by_language`, `cover_blocks`, `deposit_coin`, `fasten_screws`, `fill_egg_holder`, `fill_pen_holder`, `fold_clothes`, `general_pickup`, `hang_mugs`, `imitate_sorting_sequence`, `insert_key`, `insert_tubes`, `make_kong`, `make_toast`, `match_and_pick_from_conveyor`, `organize_table`, `pack_objects_into_box`, `pick_from_conveyor_by_image`, `play_Xylophone`, `play_stacking_toy`, `play_tic_tac_toe`, `plug_in_charger`, `pour_balls_into_vase`, `pour_by_language`, `pour_liquid_into_cup`, `press_by_number`, `push_T`, `put_bottles_into_dustbin`, `solve_equation`, `sort_nesting_dolls_by_size`, `stack_blocks`, `stack_blocks_by_language`, `stack_bowls`, `store_laptop_and_headphones`, `store_tools_in_toolbox`, `swap_T`, `swap_blocks` and `sweep_blocks`.

`download_checkpoint.sh [task|all] [ref]` fetches only the requested `checkpoints/RunWu/<task>/` directories from the original repository (shallow, sparse git checkout) at `ref` (default `$RUNWU_RSI_REF`, else `v0.1.0`), and skips a directory that already exists. The 42 checkpoints are frozen at the release tag `v0.1.0`; a changed checkpoint has a new bundle id and a new release tag.

`ckpt_name` is resolved with XPolicyLab's shared checkpoint resolver. A checkpoint directory that has no bundle file of its own but a subdirectory named after `task_name` serves all tasks, which is how `RunWu` works. A `ckpt_name` that is a path to a directory or to a bundle file also works, and so does a bundle id (or a unique prefix of at least 8 hex digits) together with `bundle_store`. If `demos/` exists and `RSI_DEMO_STORE` is not set, it is used as the demonstration store; likewise `resources/` becomes `RSI_TOOL_RESOURCE_ROOT`.

### How the checkpoints were developed

- **Layouts.** Development and tuning used the public RoboDojo layouts of seeds 0–2, so scores on those layouts are not held-out results.
- **Object sizes.** Object dimensions in the policy text were taken from the simulator's asset files, with a tolerance of ±1 mm.
- **Scoring.** The descriptions of how each task is scored were derived from RoboDojo's open-source evaluator code.
- **Task texts.** The task descriptions and scoring conditions in the bundles are taken from RoboDojo's public task documentation. Part of the shared protocol text is adapted from RoboICL (MIT License). See [NOTICE](https://github.com/RunWu-RSI/RunWu-RSI/blob/v0.1.0/NOTICE).
- **Training episodes.** 29 checkpoints carry `demos/`, which are keyframes of RoboDojo training episodes (dataset `RoboDojo_lerobot_v21_video`), and most bundles carry `priors`, which are statistics over the keyframes of the training set. That dataset is licensed under the Apache License 2.0; the attribution is in [NOTICE](https://github.com/RunWu-RSI/RunWu-RSI/blob/v0.1.0/NOTICE).
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
