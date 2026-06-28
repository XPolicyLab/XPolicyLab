# hy_vla

[Hy-Embodied-0.5-VLA](https://github.com/Tencent-Hunyuan/Hy-Embodied-0.5-VLA)
(Hy-VLA) integrated into XPolicyLab. Hy-VLA is a dual-arm flow-matching
Vision-Language-Action model built on the Hy-Embodied-0.5 MoT backbone, with a
compact memory (MEM) video encoder for multi-frame history and a delta-chunk
action representation.

This policy targets the RoboDojo benchmark with the dual-arm `arx_x5`
embodiment (`-> dual_x5`, `arm_dim [6,6]`, `ee_dim [1,1]`).

## Architecture

The policy **server** runs inside the Hy-Embodied uv venv (torch 2.7 + the
HunYuanVLMoT `transformers` fork + flash_attn) and loads the released
checkpoint. The Isaac Sim env **client** runs in a separate conda env and
talks to the server over a socket, as for every XPolicyLab policy.

`model.py` mirrors Hy-VLA's own `robotwin_eval` adapter:

```
RoboDojo obs (3 cams RGB + dual-arm EEF pose/gripper + instruction)
  -> 16-d dual-arm state (wxyz) + CHW float images
  -> wxyz->xyzw -> UMI coordinate transform
  -> PosRotMat6d -> normalize -> flow-matching forward -> denormalize
  -> RT-relative -> absolute UMI PosQuat -> inverse UMI transform (-> RoboDojo)
  -> xyzw->wxyz -> per-step {left,right}_ee_pose + {left,right}_ee_joint_state
```

## Install

```bash
bash install.sh
```

This clones the Hy-Embodied source tree into `./Hy-Embodied-0.5-VLA` (override
with `HY_VLA_ROOT`), runs `uv sync` to build its venv, and installs XPolicyLab
into that venv. Then download a checkpoint, e.g.:

```bash
# RoboTwin-pretrained release
huggingface-cli download tencent/Hy-Embodied-0.5-VLA-RoboTwin \
  --local-dir Hy-Embodied-0.5-VLA/Hy-Embodied-0.5-VLA-RoboTwin
```

Point `ckpt_path` / `norm_path` in `deploy.yml` at the downloaded checkpoint
(absolute, or relative to `hy_root`).

## Data processing

Compute the normalization statistics the server consumes:

```bash
bash process_data.sh <manifest_csv> <hdf5_dir> <output_pkl> [downsample_rate] [chunk_size]
```

## Training

```bash
CHIEF_IP=127.0.0.1 INDEX=0 NUM_MACHINES=1 NPROC_PER_NODE=8 \
HDF5_DIR=/path/to/robotwin/hdf5 EXP_ROOT=/path/to/experiments \
bash train.sh
```

`train.sh` forwards to the Hy-Embodied RoboTwin/UMI recipe; see that repo for
the full multi-node training documentation.

## Deploy / Evaluate

First run `bash install.sh`. For quick iteration you can launch the server and
client separately (easier to read server errors); on a single machine `eval.sh`
does both:

```bash
bash eval.sh RoboDojo stack_bowls hyvla_dojo_ckpt_v3 arx_x5 50 ee 0 0 0 uv <eval_env_conda_env>
```

Positional args: `<dataset_name> <task_name> <ckpt_name> <env_cfg_type>
<expert_data_num> <action_type> <seed> <policy_gpu_id> <env_gpu_id>
<policy_uv_env> <eval_env_conda_env>`.

Set `eval_env: debug` in `deploy.yml` for offline shape/IO validation before
`sim`.

## Key `deploy.yml` knobs

| Field | Meaning |
|---|---|
| `hy_root` | Hy-Embodied source tree (provides `hy_vla` + `robotwin_eval` + the uv venv). |
| `ckpt_path` / `norm_path` | Checkpoint dir and norm stats (`norm_path: null` -> `<ckpt_path>/norm_stats.pkl`). |
| `with_absolute` | `true` if the model was trained with interleaved rel+abs action supervision. |
| `blend_mode` | `rel_only` / `abs_only` / `rel_abs` action decoding. |
| `exc_action_size` | Env steps executed per network forward. |
| `img_history_size` / `img_history_interval` | MEM video-encoder history cadence (when `use_video_encoder=true`). |
| `policy_uv_env_path` | Hy-Embodied uv venv root for the server. |
