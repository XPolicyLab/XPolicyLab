# SANA_Robot

**Contributor:** SANA-Robot Team | **Paper:** [SANA-Robot: Efficient World Action Model with Hybrid Memory](https://github.com/NVlabs/Sana/blob/robot/asset/SANA-Robot.pdf) | **Project page:** https://nvlabs.github.io/Sana/Robot/ | **Original code:** https://github.com/NVlabs/Sana/tree/robot

`SANA_Robot` serves the two SANA-Robot world-action policies in XPolicyLab/RoboDojo: **SANA-Robot-AR**, which keeps a memory of the episode (the first frame, the last chunks and a recurrent state over all of them), and **SANA-Robot-base**, which maps one observation to one action chunk. Both denoise the future video latents and a chunk of 32 actions together. The adapter is a thin layer over the `sana_robot` package of the SANA repository, which `install.sh` clones into `Sana/` (not vendored): `model.py` feeds the package's own control-loop runner the observations of the environment client, and `robodojo.py` converts RoboDojo ARX-X5 observations and actions the way the training dataset was built.

Shared conventions — argument meanings, checkpoint naming, split-machine deployment, `EVAL_ENV_TYPE` — are documented in the [XPolicyLab README](../../README.md). Official results: [RoboDojo LeaderBoard](https://robodojo-benchmark.com/LeaderBoard).

Supported: `bench_name=RoboDojo`, `env_cfg_type=arx_x5`, `action_type=joint` and `action_type=ee` (one checkpoint serves both: it is trained on joint targets and end-effector poses together).

## Installation

```bash
cd XPolicyLab/policy/SANA_Robot
bash install.sh [env_name]      # default env name: sana-robot
conda activate sana-robot
```

`install.sh` clones the `robot` branch of the SANA repository into `Sana/`, creates the conda environment with the repository's own `environment_setup.sh` (PyTorch 2.9.1+cu128, the CUDA 12.8 toolkit and `flash-attn` compiled from source, about 40 minutes), and installs XPolicyLab into it. To use an existing checkout, set `SANA_ROBOT_ROOT=/path/to/Sana` for every script here (or `sana_robot_root` in `deploy.yml`); `SANA_REPO` and `SANA_REF` choose the repository and the branch to clone.

## Model Assets

The released RoboDojo checkpoint of SANA-Robot-AR, and the two frozen models every SANA-Robot policy needs:

```bash
cd XPolicyLab/policy/SANA_Robot
hf download Efficient-Large-Model/SANA-Robot-AR-RoboDojo --local-dir checkpoints/sana_robot_ar_robodojo
hf download Efficient-Large-Model/LTX-2.3-Diffusers --include "vae/*" --local-dir Sana/checkpoints/LTX-2.3-Diffusers
hf download google/gemma-2-2b-it --local-dir Sana/checkpoints/gemma-2-2b-it
```

| Asset | Source | Default path | Override |
| --- | --- | --- | --- |
| SANA-Robot-AR RoboDojo checkpoint | [SANA-Robot-AR-RoboDojo](https://huggingface.co/Efficient-Large-Model/SANA-Robot-AR-RoboDojo) | the `ckpt_name` argument of `eval.sh` | `checkpoint_path` in `deploy.yml` |
| LTX-2.3 video VAE (diffusers layout; only the `vae/` subfolder is read) | [LTX-2.3-Diffusers](https://huggingface.co/Efficient-Large-Model/LTX-2.3-Diffusers) | `Sana/checkpoints/LTX-2.3-Diffusers` | `SANA_ROBOT_VAE`, or `vae_path` in `deploy.yml` |
| Gemma 2 2B text encoder | [gemma-2-2b-it](https://huggingface.co/google/gemma-2-2b-it) | `Sana/checkpoints/gemma-2-2b-it` | `SANA_ROBOT_TEXT_ENCODER`, or `text_encoder_path` in `deploy.yml` |
| Pretrained policy weights (training only) | [SANA-Robot-AR-Pretrain](https://huggingface.co/Efficient-Large-Model/SANA-Robot-AR-Pretrain) | `Sana/checkpoints/sana_robot_ar_pretrain.pth` | `SANA_ROBOT_PRETRAINED` |

A checkpoint directory is the released checkpoint or a run directory of `train.sh`:

```text
checkpoints/sana_robot_ar_robodojo/      # released
├── sana_robot_ar_robodojo.pth           # the weights
├── meta.json                            # the description of the training dataset: cameras, prompt fields, robot slots
└── normalization.json                   # the normalization of the robot rows

checkpoints/<bench_name>-<ckpt_name>-<env_cfg_type>-<action_type>-<seed>/      # trained here
├── config.yaml                          # the training config: the policy's arguments, the camera sizes, the chunk length
├── meta.json
├── normalization.json
└── checkpoints/latest/model_ema.pth
```

A SANA-Robot-AR directory without a `config.yaml`, such as the released one, reads the policy's arguments and the camera sizes from the RoboDojo recipe of the SANA repository, `Sana/configs/robodojo_ar.yaml`.

## Data Processing

```bash
cd XPolicyLab/policy/SANA_Robot
bash process_data.sh <bench_name> <ckpt_name> <env_cfg_type> <action_type> [expert_data_num]

# Example: convert the official export
bash process_data.sh RoboDojo cotrain arx_x5 ee
```

Training reads the **official RoboDojo LeRobot v3.0 export** (`RoboDojo_lerobot_v30_video`, downloaded by `bash scripts/RoboDojo/download_robodojo_data.sh lerobot_v3.0`; its keys are those of the [official converters](../../README.md#official-lerobot-conversion): `observation.state`, `action` and `observation.images.{cam_high,cam_left_wrist,cam_right_wrist}`). `process_data.sh` runs the repository's `scripts/data/convert_robodojo.sh` on it and writes `data/<bench_name>-<ckpt_name>-<env_cfg_type>-<action_type>/`:

- `sana_robot_zip/`: the episodes as zip archives. The videos are copied without re-encoding; every robot row holds the joint angles, the pose of each flange from the forward kinematics of the ARX-X5 arm, and the gripper closedness.
- `normalization.json`: the normalization of the robot rows.

The dataset is the same for both action types. `SANA_ROBOT_LEROBOT_DIR` points at an export stored elsewhere (default `<workspace>/data/RoboDojo_lerobot_v30_video`); `expert_data_num` converts only the first episodes; `SANA_ROBOT_LATENTS=1` also encodes the video latents once, so that training does not run the video VAE.

## Training

```bash
cd XPolicyLab/policy/SANA_Robot
bash train.sh <bench_name> <ckpt_name> <env_cfg_type> <action_type> <seed> <gpu_id> [extra training args]

# Example: SANA-Robot-AR on eight GPUs
bash train.sh RoboDojo cotrain arx_x5 ee 0 0,1,2,3,4,5,6,7
```

`train.sh` runs the repository's `scripts/train/train_ar.sh` with its RoboDojo recipe (`Sana/configs/robodojo_ar.yaml`) on the dataset of `process_data.sh`, one process per GPU of `gpu_id`. The run directory is `checkpoints/<bench_name>-<ckpt_name>-<env_cfg_type>-<action_type>-<seed>/`; a rerun resumes from its latest checkpoint. Extra arguments go to the trainer as overrides of the recipe (for example `--max_steps 20000`), and `SANA_ROBOT_CONFIG` replaces the recipe.

## Evaluation

```bash
cd XPolicyLab/policy/SANA_Robot
bash eval.sh <bench_name> <task_name> <ckpt_name> <env_cfg_type> <action_type> <seed> \
  <policy_gpu_id> <env_gpu_id> <policy_conda_env> <eval_env_conda_env>

# Example: the released checkpoint on stack_bowls with end-effector actions
bash eval.sh RoboDojo stack_bowls checkpoints/sana_robot_ar_robodojo arx_x5 ee 0 0 0 sana-robot <eval_env_conda_env>

# Example: a run of train.sh
bash eval.sh RoboDojo stack_bowls cotrain arx_x5 ee 0 0 0 sana-robot <eval_env_conda_env>
```

`ckpt_name` may be the short run name, the full run-directory name, or a path to a checkpoint directory. `EVAL_ENV_TYPE=debug` runs the offline wiring check (no simulator); leave it unset or set `EVAL_ENV_TYPE=sim` for RoboDojo simulation. For split-machine deployment via `setup_eval_policy_server.sh` / `setup_eval_env_client.sh`, follow the [Deployment Flow](../../README.md#-deployment-flow).

## Configuration

`deploy.yml` holds the inference settings of the released results (10 denoising steps, video guidance 6, a new chunk every 16 actions, the plain PyTorch path). Its keys beyond the standard set:

| Key | Default | Meaning |
| --- | --- | --- |
| `policy_type` | `ar` | `ar` (SANA-Robot-AR) or `base` (SANA-Robot-base); must be the policy the checkpoint was trained as. |
| `sana_robot_root` | `null` | The checkout of the SANA repository; `null`: `$SANA_ROBOT_ROOT`, else the checkout the `sana_robot` package is installed from (`Sana/` after `install.sh`). |
| `checkpoint_path` | `null` | The checkpoint directory; `null`: resolved from `ckpt_name`. |
| `checkpoint_file`, `checkpoint_step` | `null` | The weights file inside it, or the training step of a run to serve instead of the latest. |
| `normalization_path`, `dataset_meta_path` | `null` | Replace `normalization.json` / `meta.json` of the checkpoint directory. |
| `vae_path`, `text_encoder_path` | `null` | The video VAE and the text encoder (see Model Assets). |
| `policy`, `view_sizes`, `chunk_size`, `video_frame_stride` | `null` | Replace what the checkpoint's `config.yaml` says: the arguments of the policy class, the pixel size of every camera, the actions per chunk and the control steps between two video frames. |
| `steps` | `10` | Denoising steps per chunk. |
| `video_flow_shift`, `action_flow_shift` | `5.0`, `1.0` | Flow shifts of the sampler. |
| `video_cfg_scale` | `6.0` | Classifier-free guidance scale of SANA-Robot-AR's video velocity; the actions are not guided. `1.0` turns it off. |
| `n_action_steps` | `16` | Actions run of every chunk of 32 before the next one is generated; `null`: the whole chunk. A multiple of `video_frame_stride`. |
| `async_delay_steps` | `0` | Actions the robot runs from the running chunk while the next one is generated (asynchronous execution); `0`: the robot waits for every chunk. |
| `optimized` | `false` | `true` runs the policy with the inference kernels of its GPU (H100, Jetson Thor, RTX 5090 / PRO 6000) and as CUDA graphs; `false` is the plain PyTorch path, on which the released results were measured. |
| `precision` | `bf16` | `bf16`, `fp8` or `nvfp4`: the precision of the large matrix products of an optimized policy. |
| `cameras` | `null` | The observation's camera for every view of the dataset, in the dataset's order; `null`: `cam_head`, `cam_left_wrist`, `cam_right_wrist`. |
| `robot_root_poses` | `null` | `{left: [x, y, z, qw, qx, qy, qz], right: [...]}`, the arms' base poses for `action_type=ee`; `null`: RoboDojo's ARX-X5. |
| `request_timeout_s` | `900` | The client's request timeout: with `optimized: true` the first requests of a server compile and capture. |

Environment variables read by the scripts: `SANA_ROBOT_ROOT`, `SANA_REPO`, `SANA_REF` (install); `SANA_ROBOT_LEROBOT_DIR`, `SANA_ROBOT_LATENTS` (data); `SANA_ROBOT_CONFIG`, `SANA_ROBOT_PRETRAINED` (training); `SANA_ROBOT_VAE`, `SANA_ROBOT_TEXT_ENCODER` (data, training and evaluation).

## Notes

- **One environment per server.** `eval_batch` stays `false`: a SANA-Robot-AR episode has its own memory.
- **Every observation counts.** The policy reads every `video_frame_stride`-th frame of the episode, so the client must send the observation after each action it runs, as `deploy.py` does, and run every action of a chunk before asking for the next one.
- **Actions.** `action_type=joint` returns the joint targets of the chunk; `action_type=ee` returns the flange pose `[x, y, z, qw, qx, qy, qz]` in the frame of the environment's `*_ee_pose`, through the arms' base poses. At the start of every episode the server prints how far the reported `*_ee_pose` is from the pose the policy reads; a gap of millimetres means `robot_root_poses` is not the simulator's.
- **Video latents.** The frames are encoded by the `sana_robot` package's stream of SANA's causal LTX-2 VAE in bf16 while they arrive, which gives the latents of one encode of the whole video by that encoder. With `optimized: true` the package compiles the video encoder as well as the policy.
- **Noise.** The sampler's noise is seeded per episode from `seed` and the episode's index on the server, so a rerun of a server repeats its chunks on the same observations.
