# RoboDojo SFT

Run commands from the XPolicyLab repository root. Training and inference share
the model bundled in `policy/Awomo05/awomo`; no external Awomo checkout is needed.

## Installation

```bash
bash policy/Awomo05/install.sh training
conda activate awomo05-train
```

The training environment is separate from the default `awomo05` inference environment.

## Data

Set the directory containing the RoboDojo HDF5 task folders, then prepare the index:

```bash
export AWOMO_DATA_ROOT=/datasets/RoboDojo
bash policy/Awomo05/process_data.sh RoboDojo sft arx_x5 joint
```

The supplied `episodes.txt` excludes `swap_T`, `swap_blocks` and `sweep_blocks`
(100 episodes each), leaving 32 tasks and 3,200 episodes. Preparation does not
rewrite images or recompute normalization parameters. Existing indexes are not overwritten.
The dataset decodes legacy and marked RGB JPEGs through XPolicyLab's shared helper;
the reference configuration explicitly selects BGR model inputs afterward.

## Training

Download the [pretrain model](https://huggingface.co/Auwomo/Awomo-0.5I-base) and run [`train.sh`](../train.sh).

### Settings

Use shared model paths and the node-0 address; set `NODE_RANK` to 0–3 across the four nodes.

| Setting | Value |
|---|---|
| GPUs / batch | 32 GPUs; local 8, global 256; no accumulation |
| Updates / LR | 6,755; AdamW `5e-6`; warmup 270; cosine 6,755 |
| Sampling | Uniform stride 1; reference seed 42 |
| Training scope | DiT only; state encoder, VLM/LoRA and VAE frozen |
| EMA | 0.999, including the frozen state encoder |
| Action | Absolute joints, z-score, 14 supervised Canonical80 slots |
| Gripper loss | Slots 16/45 numerator ×2; denominator unchanged |
| Noise / loss | Discrete FP32 timesteps, shift 5; BSMNTW on; Self-Flow off |
| Images / history | BGR mosaic; VAE H320/W288; VLM H448/W448; 20 history slots |
| Precision | FP32 master parameters/checkpoints; BF16 compute; FP32 reduction |
| Pretrain model | `export AWOMO_BASE_CHECKPOINT=/models/awomo-0.5i-pretrain` |
| Qwen3-VL | `export AWOMO_QWEN3_VL=/models/Qwen3-VL-4B-Instruct` |
| FLUX.2 transformer | `export AWOMO_FLUX2_TRANSFORMER=/models/FLUX.2-klein-base-4B/transformer` |
| FLUX.2 VAE | `export AWOMO_FLUX2_VAE=/models/FLUX.2-klein-base-4B/vae` |
| Distributed launch | `export NNODES=4 NODE_RANK=0 MASTER_ADDR=training-node-0 MASTER_PORT=29500` |
| Training command | `bash policy/Awomo05/train.sh RoboDojo sft arx_x5 joint 42 0,1,2,3,4,5,6,7` |

## Evaluation

Use the [published SFT checkpoint](https://huggingface.co/Auwomo/Awomo-0.5I-Robodojo)
and the [adapter evaluation guide](../README.md#evaluation).
