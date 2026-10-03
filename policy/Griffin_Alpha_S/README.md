# Griffin_Alpha_S

**Contributor:** Griffin Labs | **Paper:** [Griffin Alpha-S: an open-weights vision-language-action model](https://griffinlabs.ai/blog/griffin-alpha-s) (blog) | **arXiv:** Not available | **Original code:** https://github.com/griffinlabs-ai/alpha-s

`Griffin_Alpha_S` adapts Griffin Labs' Alpha-S, a Qwen3-VL-4B backbone with a robot-action head pre-trained on a multi-embodiment mixture, to XPolicyLab. The model ships as a [LeRobot](https://github.com/huggingface/lerobot) policy plugin with two heads selected by the checkpoint's `config.json`: `griffin_alpha` (flow-matching action expert, 910M, the default) and `griffin_alpha_fast` (FAST action tokens, autoregressive). Both predict a 50-step action chunk from up to three cameras, the proprioceptive state and a language instruction, and one adapter serves both. There is no vendored source tree: `install.sh` clones the plugin into `alpha-s/` (git-ignored) and installs it editable, `train.sh` fine-tunes through stock `lerobot-train`, and the adapter loads checkpoints from a local directory or straight from the Hugging Face Hub ([collection](https://huggingface.co/collections/griffinlabs/alpha-s)). The RoboTwin 2.0 `aloha-agilex` fine-tune, [`griffinlabs/griffin-alpha-s-robotwin`](https://huggingface.co/griffinlabs/griffin-alpha-s-robotwin), is the checkpoint released for the RoboTwin leaderboard.

Shared conventions — argument meanings, checkpoint naming, split-machine deployment, `EVAL_ENV_TYPE` — are documented in the [XPolicyLab README](../../README.md). Official results: [RoboDojo LeaderBoard](https://robodojo-benchmark.com/LeaderBoard).

## Installation

```bash
cd XPolicyLab/policy/Griffin_Alpha_S
bash install.sh
conda activate <policy_env>  # e.g. griffin_alpha_s (override with GRIFFIN_CONDA_ENV=<name>)
```

Python 3.12 or newer; the installer pins `lerobot[dataset,training]==0.6.1`, `transformers>=5.5.4,<5.6` and `torch>=2.7,<2.12` through the plugin's own `pyproject.toml`. FlashAttention 2 is used when installed (`pip install flash-attn --no-build-isolation`, or a prebuilt wheel — see the `flash` extra in `alpha-s/pyproject.toml`) and the policy falls back to SDPA with a warning otherwise. Installer knobs: `GRIFFIN_CONDA_ENV`, `GRIFFIN_PYTHON_VERSION`, `GRIFFIN_ALPHA_S_REPO`, `GRIFFIN_ALPHA_S_REF`, `GRIFFIN_UPDATE_PLUGIN=1`, `GRIFFIN_TORCH_INDEX` (e.g. a cu128 wheel index).

## Model Assets

| repository | `main` branch | `fast` branch |
|---|---|---|
| [`griffinlabs/Griffin-Alpha-S`](https://huggingface.co/griffinlabs/Griffin-Alpha-S) | pre-trained base, flow head | pre-trained base, FAST head |
| [`griffinlabs/Griffin-Alpha-S-LIBERO`](https://huggingface.co/griffinlabs/Griffin-Alpha-S-LIBERO) | LIBERO fine-tune, flow head | LIBERO fine-tune, FAST head |
| [`griffinlabs/griffin-alpha-s-robotwin`](https://huggingface.co/griffinlabs/griffin-alpha-s-robotwin) | RoboTwin 2.0 `aloha-agilex` fine-tune, flow head (leaderboard release) | — |

Weights are CC BY-NC-SA 4.0; the plugin code is Apache-2.0. The adapter downloads from the Hub on demand when `pretrained_path` (or `ckpt_name`) is a repo id, with `revision: fast` selecting the FAST head. To keep a local copy:

```bash
bash download_checkpoint.sh                                          # -> checkpoints/griffin-alpha-s-robotwin (leaderboard checkpoint)
hf download griffinlabs/Griffin-Alpha-S --local-dir checkpoints/Griffin-Alpha-S                    # flow base
hf download griffinlabs/Griffin-Alpha-S --revision fast --local-dir checkpoints/Griffin-Alpha-S-fast  # FAST base
```

`download_checkpoint.sh [destination] [revision] [repo_id]` fetches the policy files of a Hub checkpoint with `huggingface_hub`; its defaults are the RoboTwin release.

**The RoboTwin checkpoint** was fine-tuned from the base on RoboTwin 2.0 `aloha-agilex`, `demo_clean` only: all 50 tasks, 50 episodes each (2,500 episodes), 6 epochs = 25,776 steps at an effective batch of 128, full fine-tune of backbone and expert, no checkpoint selection (the 6-epoch final). It bakes in the three RoboTwin cameras (`cam_high`, `cam_left_wrist`, `cam_right_wrist`, 480x640) in that prompt order, the 14-D joint state and action (`left arm x6, left gripper, right arm x6, right gripper`, exactly `pack_robot_state`'s order for `aloha_agilex`), relative arm actions with absolute grippers, and 10 Euler steps. Its config records `n_action_steps=50`, but the published numbers were produced executing **25** of the 50 predicted steps per replan, which is what `deploy.yml` sets. Its model card carries the per-task numbers.

The bases are pre-trained with a canonical 32-wide action vector and relative arm actions, so they must be fine-tuned on the target robot before evaluation; the LIBERO fine-tunes use LIBERO's cameras and 7-D OSC deltas and do not transfer to RoboDojo or RoboTwin robots. No RoboDojo checkpoint is published yet — see [Notes](#notes).

## Data Processing

Training consumes **LeRobot v3.0** datasets with the official keys — `observation.state`, `action`, `observation.images.cam_high` / `cam_left_wrist` / `cam_right_wrist` ([official LeRobot conversion](../../README.md#official-lerobot-conversion)). `process_data.sh` runs `XPolicyLab/scripts/transform_lerobot_v30_format.py` unchanged (it needs the parent workspace's `../data/` and `../env_cfg/`, see [Quick Start](../../README.md#-quick-start)), then adds the one step Griffin Alpha-S needs on top of any LeRobot dataset: `alpha-s/scripts/make_finetune_base.py` rebuilds the base checkpoint's saved processors for that dataset — camera keys in prompt order (`cam_high` first, then the wrists), the embodiment prompt, the control-mode string (`joint`), the replan stride, and normalization statistics recomputed in relative-action space — and writes a trainable policy directory to `bases/<bench_name>-<ckpt_name>-<env_cfg_type>-<action_type>/`.

```bash
cd XPolicyLab/policy/Griffin_Alpha_S
bash process_data.sh <bench_name> <ckpt_name> <env_cfg_type> <action_type> [expert_data_num] [task_pattern]

# Example: convert stack_bowls demos for arx_x5 joint control and build the flow-head base
bash process_data.sh RoboDojo stack_bowls arx_x5 joint

# Example: a cotrain set over every arx_x5 task, 50 episodes each, FAST head
GRIFFIN_HEAD=fast bash process_data.sh RoboDojo cotrain arx_x5 joint 50 "*"

# Example: reuse a prepared lerobot_v3.0 export instead of converting
GRIFFIN_DATASET_REPO_ID=RoboDojo_sim_stack_bowls_v30 bash process_data.sh RoboDojo stack_bowls arx_x5 joint
```

The official converter names motors `left_joint_<i>` / `right_joint_<i>` (arm joints first, then the end-effector joints), so the gripper dimensions carry no `gripper` in their names; the script derives them from `utils/robot/_robot_info.json` and passes them as `--relative_exclude_joints`, keeping grippers absolute while arm dimensions are predicted relative to the current state. Overrides: `GRIFFIN_HEAD` (`flow` | `fast`), `GRIFFIN_BASE`, `GRIFFIN_EMBODIMENT_PROMPT`, `GRIFFIN_N_ACTION_STEPS` (default 10 of the 50-step chunk), `GRIFFIN_RELATIVE_ACTIONS=0` (for datasets whose actions are already per-step deltas), `GRIFFIN_RELATIVE_EXCLUDE_JOINTS`, `GRIFFIN_MAKE_BASE_EXTRA_ARGS` (e.g. `--no-include_proprio` or `--freeze_backbone`). A dataset that merges several robots is zero-padded per arm by the converter; set `GRIFFIN_RELATIVE_EXCLUDE_JOINTS` by hand in that case.

## Training

```bash
cd XPolicyLab/policy/Griffin_Alpha_S
bash train.sh <bench_name> <ckpt_name> <env_cfg_type> <action_type> <seed> <gpu_id> [extra lerobot-train flags...]

# Example: fine-tune the base prepared above on GPU 0
bash train.sh RoboDojo stack_bowls arx_x5 joint 0 0

# Example: shorter run with a larger batch
GRIFFIN_STEPS=6000 GRIFFIN_BATCH_SIZE=32 bash train.sh RoboDojo stack_bowls arx_x5 joint 0 0
```

`train.sh` is `lerobot-train --policy.path=bases/<bench_name>-<ckpt_name>-<env_cfg_type>-<action_type> --dataset.repo_id=<same name>` plus the run bookkeeping; trailing arguments are forwarded to `lerobot-train`. Checkpoints land in `checkpoints/<bench_name>-<ckpt_name>-<env_cfg_type>-<action_type>-<seed>/` in lerobot-train's layout (`checkpoints/<step>/pretrained_model/` and a `last` link), which is what evaluation loads. `GRIFFIN_POLICY_PATH` starts from another checkpoint (a directory or `griffinlabs/Griffin-Alpha-S-LIBERO`) when its saved processors already match the dataset; `GRIFFIN_DATASET_REPO_ID`, `GRIFFIN_STEPS`, `GRIFFIN_BATCH_SIZE`, `GRIFFIN_SAVE_FREQ`, `GRIFFIN_NUM_WORKERS` and `VIDEO_BACKEND` are the other knobs. The learning-rate schedule, the two heads' objectives and the relative-action contract are documented in the plugin's [docs/finetuning.md](https://github.com/griffinlabs-ai/alpha-s/blob/main/docs/finetuning.md).

## Evaluation

```bash
cd XPolicyLab/policy/Griffin_Alpha_S
bash eval.sh <bench_name> <task_name> <ckpt_name> <env_cfg_type> <action_type> <seed> \
  <policy_gpu_id> <env_gpu_id> <policy_conda_env> <eval_env_conda_env>

# RoboTwin 2.0, the released checkpoint (put this checkout at RoboTwin/XPolicyLab, see the RoboTwin docs)
bash download_checkpoint.sh
bash eval.sh RoboTwin adjust_bottle checkpoints/griffin-alpha-s-robotwin aloha_agilex joint 0 0 0 griffin_alpha_s RoboTwin
# or straight from the Hub (downloads ~10 GB on first use)
bash eval.sh RoboTwin adjust_bottle griffinlabs/griffin-alpha-s-robotwin aloha_agilex joint 0 0 0 griffin_alpha_s RoboTwin

# RoboDojo, a run trained with train.sh above
bash eval.sh RoboDojo stack_bowls stack_bowls arx_x5 joint 0 0 0 griffin_alpha_s <eval_env_conda_env>

# Offline wiring check, no simulator
EVAL_ENV_TYPE=debug bash eval.sh RoboTwin adjust_bottle griffinlabs/griffin-alpha-s-robotwin aloha_agilex joint 0 0 0 griffin_alpha_s base
```

The RoboTwin checkpoint was trained on `demo_clean` and can be evaluated on either scene setting; RoboTwin's client selects the setting. Results from our own 50 tasks x 100 episodes sweep with the same weights (single seed, 10 Euler steps, 25 of the 50 predicted steps executed per replan, `unseen` instructions):

| setting | success | mean over 50 tasks |
|---|---|---|
| `demo_clean` | 2989 / 5000 | **59.8%** |
| `demo_randomized` | 2310 / 5000 | **46.2%** |

Training used `demo_clean` only, so the randomized number is a domain-randomization generalization figure and is depressed relative to a model trained on randomized data. Per-task numbers are on the [model card](https://huggingface.co/griffinlabs/griffin-alpha-s-robotwin).

`ckpt_name` may be the short run name (combined into `checkpoints/<bench_name>-<ckpt_name>-<env_cfg_type>-<action_type>-<seed>/`), the full run-directory name, a path to a lerobot policy or run directory, or a Hugging Face repo id. `EVAL_ENV_TYPE=debug` runs the offline wiring check (no simulator); leave it unset or set `EVAL_ENV_TYPE=sim` for RoboDojo simulation. For split-machine deployment via `setup_eval_policy_server.sh` / `setup_eval_env_client.sh`, follow the [Deployment Flow](../../README.md#-deployment-flow).

At each replan the adapter packs the observation (`pack_robot_state`: arm then end-effector per arm, the order the official converter writes `observation.state`), feeds the `cameras` to the checkpoint's image keys in order, runs the checkpoint's own pre/post-processors, executes `n_action_steps` of the 50-step chunk, and unpacks them with `unpack_robot_state`. `update_obs_batch` / `get_action_batch` run one batched forward for all active environments; set `eval_batch: true` in `deploy.yml` to use them.

## Configuration

`deploy.yml` keys beyond the standard set:

| Key | Default | Meaning |
|---|---|---|
| `pretrained_path` | `null` | Explicit checkpoint: a lerobot policy directory, a lerobot-train run directory, or a Hub repo id. Overrides `ckpt_name`. |
| `revision` | `null` | Hub branch when the checkpoint is a repo id: `null`/`main` = flow head, `fast` = FAST head. Ignored for local directories. |
| `checkpoint_num` | `null` | lerobot-train step to load from `checkpoints/<run>/checkpoints/<step>/pretrained_model/`; `null` picks `last`, then the highest step. |
| `device` | `cuda` | Torch device (`auto` picks CUDA when available). |
| `prompt` | `null` | Fallback instruction when the observation carries no `instruction`. |
| `cameras` | `[cam_head, cam_left_wrist, cam_right_wrist]` | XPolicyLab cameras in the checkpoint's prompt order; the i-th is fed to the i-th image key the checkpoint was trained with (`cam_head` also matches `cam_high`). |
| `n_action_steps` | `25` | Actions executed per predicted chunk before replanning; `25` reproduces the RoboTwin evaluation. `null` = the checkpoint's value (50 on the RoboTwin checkpoint, 10 on the bases prepared by `process_data.sh`). |
| `num_inference_steps` | `null` | Flow head only: Euler steps; `null` = the checkpoint's (10 on the base). One step matched ten on LIBERO but not on a bimanual robot — measure before lowering. |
| `request_timeout_s` | `300` | Raised from the transport default: loading the 5B model and the first forward take a while. |

## Notes

- **Supported**: `bench_name=RoboTwin` (`env_cfg_type=aloha_agilex`, released checkpoint) and `RoboDojo` (after fine-tuning); `action_type=joint`; any robot registered in both robot-info files; one to three cameras (the checkpoint fixes the camera count and order). `action_type=ee` is refused at load time: the model is control-mode agnostic, but XPolicyLab's shared `pack_robot_state` validates `*_ee_pose` against the arm dimension and cannot pack 7-D poses yet.
- **Checkpoints**: `griffinlabs/griffin-alpha-s-robotwin` is the RoboTwin 2.0 `aloha_agilex` release. No RoboDojo checkpoint is published yet; run `process_data.sh` and `train.sh` to produce one before `eval.sh` on a RoboDojo task. A checkpoint evaluated on a robot it was not fine-tuned for fails with a clear width error (or, for a 32-wide base, is truncated with a warning) rather than acting.
- **Memory**: bf16 backbone (4.4B) + expert (0.9B) is about 11 GB of weights; batched evaluation over ten environments with three cameras fits a 40 GB GPU comfortably.
- **Fine-tuning gotchas** (from the plugin docs): relative actions need `action.names` with the gripper dimensions excluded — `process_data.sh` handles the official converter's names; `lerobot-train` reads `config.json` from `--policy.path` without applying a Hub revision, so a FAST base must come through `process_data.sh` (`GRIFFIN_HEAD=fast`) or a local `hf download --revision fast`; the FAST head's action tokens sit at the end of the sequence, so raise `max_sequence_length` if the input step warns about truncated labels.
