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
with `HY_VLA_ROOT`), runs `uv sync` to build its venv, **overlays RoboDojo
post-training support** onto the clone (see below), and installs XPolicyLab into
that venv. Then download a checkpoint, e.g.:

```bash
# RoboTwin-pretrained release
huggingface-cli download tencent/Hy-Embodied-0.5-VLA-RoboTwin \
  --local-dir Hy-Embodied-0.5-VLA/Hy-Embodied-0.5-VLA-RoboTwin
```

Point `ckpt_path` / `norm_path` in `deploy.yml` at the downloaded checkpoint
(absolute, or relative to `hy_root`).

### RoboDojo overlay

The public Hy-Embodied-0.5-VLA repo does not ship RoboDojo dataset support, and
we do not modify it. XPolicyLab carries the RoboDojo files under
[`robodojo/`](robodojo/) (dataset loader, Hydra config, norm-stats computer,
training launcher) and `install.sh` overlays them onto the clone via
[`apply_robodojo_overlay.py`](apply_robodojo_overlay.py). The overlayer copies
those files in and inserts a `source == "robodojo"` branch into the clone's
`hy_vla/data/vla_dataset.py`. It is idempotent (safe to re-run) and preflights
for the upstream transforms it relies on, so it fails loudly if the public repo
changes. To (re-)apply it manually:

```bash
python apply_robodojo_overlay.py "${HY_VLA_ROOT:-./Hy-Embodied-0.5-VLA}"
```

## Data processing

Compute the normalization statistics that training and the eval-time server
consume. The RoboDojo computer scans the HDF5 tree
(`{hdf5_dir}/{task}/{robot}/data/episode_*.hdf5`) directly — no manifest CSV —
and writes UMI-frame stats (required, matching the model's eval-time frame):

```bash
bash process_data.sh <hdf5_dir> <output_pkl> [downsample_rate] [chunk_size]
```

Defaults mirror `robodojo_hdf5.yaml`: `downsample_rate=1`, `chunk_size=25`.

## Training

Compute `norm_stats.pkl` first (above), then post-train from the UMI pretrain:

```bash
CHIEF_IP=127.0.0.1 INDEX=0 NUM_MACHINES=1 NPROC_PER_NODE=8 \
HDF5_DIR=/path/to/robodojo/hdf5 EXP_ROOT=/path/to/experiments \
NORM_PATH=/path/to/robodojo/norm_stats.pkl \
bash train.sh
```

`train.sh` forwards to the RoboDojo recipe (`scripts/train_robodojo_umi.sh`,
added by the overlay), which launches `dataset=robodojo_hdf5`. Tune via the env
overrides `EXP_ID`, `EXP_ROOT`, `PRETRAIN`, `HDF5_DIR`, `NORM_PATH`,
`NUM_MACHINES`, `NPROC_PER_NODE`, `MAIN_PORT`, `CHIEF_IP`, `INDEX`; see the
Hy-Embodied repo for the full multi-node training documentation.

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
