# Post-training Xiaomi-Robotics-0

Xiaomi-Robotics-0 (or XR-0) can be efficiently post-trained from [the pre-trained model](https://huggingface.co/XiaomiRobotics/Xiaomi-Robotics-0-Pretrain).
In the following, we show how we post-train the model to learn a complex manipulation task of packing earphones with 20 hours of data.
For an overview of the method, see [**the Post-training section**](https://robotics.xiaomi.com/xiaomi-robotics-0.html) on the project page.

## Architecture

Xiaomi-Robotics-0 combines a **Qwen3-VL** vision-language model with a **Diffusion Transformer (DiT)**:

1. **VLM** — Qwen3-VL encodes camera images and language instructions into KV-cache.
2. **DiT** — The DiT employs self-attention to jointly process the input tokens and the VLM KV-cache, with AdaLN providing conditioning for the flow-matching timestep. A token sequence -- consisting of a sink token, a state token, and noisy action tokens -- is processed through a series of stacked DiT layers.
3. **Rectified Flow** — The model is trained to predict the velocity field of the rectified flow between a sampled noise and the ground-truth action. During inference, it generates action chunks through a 5-step Euler integration process.

## Installation

1. Download [the pre-trained model](https://huggingface.co/XiaomiRobotics/Xiaomi-Robotics-0-Pretrain) and [demo data](https://huggingface.co/datasets/XiaomiRobotics/xr0_posttrain_demo). Place them under the project root:

    ```
    xr0/
    ├── data/
    │   ├── json/            # Episode annotations (.json)
    │   └── videos/          # Episode videos (.mp4)
    └── pretrained_ckpt/     # Pre-trained model weights
    ```

    We provide a script to convert the downloaded pre-trained checkpoint to PyTorch format:
    ```bash
    python tools/weight_convert.py --model_path path/to/weight
    ```

    The data config ([configs/data/earphone.yaml](configs/data/earphone.yaml)) points to `data/json` by default. Update `data.params.train_datasets.train_path` if you place the data elsewhere.

2. Install dependencies:

    ```bash
    pip install -e .
    ```

    Requirements: Python >= 3.9, CUDA GPU.

    Key dependencies: PyTorch 2.8, Lightning 2.5, DeepSpeed 0.17, Transformers 4.57, Qwen3-VL.

## Training

XR-0 supports two training modes: **synchronous** (default) and **asynchronous**, controlled by the `model.params.model.async_train` flag in [configs/model/XR0.yaml](configs/model/XR0.yaml).

### Synchronous Training (default)

In the synchronous mode (`async_train: false`), the model trains without action prefix conditioning. All action tokens are denoised from scratch, and the loss weight is uniform (1.0) across all tokens.

```
CUDA_VISIBLE_DEVICES=0 RESOURCE_GPU=1 \
bash scripts/train.sh \
data=earphone \
trainer.project="xr0" \
trainer.exp_name="earphone" \
trainer.default_root_dir="test/" \
model=XR0 \
model.params.model.pretrained="pretrained_ckpt/xr0_pretrained.pt"
```

### Asynchronous Training

In the asynchronous mode (`async_train: true`), the model randomly conditions on a prefix of ground-truth actions during training (1–6 steps, with 50% probability). This enables the model to learn prefix-conditioned action generation for asynchronous execution. The loss weight for non-prefix timesteps is computed from the prediction error of a no-grad inference pass, giving higher weight to timesteps where the model's prediction deviates more from the target.

```
CUDA_VISIBLE_DEVICES=0 RESOURCE_GPU=1 \
bash scripts/train.sh \
data=earphone \
trainer.project="xr0" \
trainer.exp_name="earphone" \
trainer.default_root_dir="test/" \
model=XR0 \
model.params.model.async_train=true \
model.params.model.pretrained="pretrained_ckpt/xr0_pretrained.pt"
```

We also support Multi-GPU & Multi-node:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 RESOURCE_GPU=4 \
bash scripts/train.sh \
...
```

In our experiment, we trained with 8 GPUs for 30k steps.

### Configuration

Configs are managed via [Hydra](https://hydra.cc/) + [OmegaConf](https://omegaconf.readthedocs.io/). Key config groups:

| Group | Path | Description |
|-------|------|-------------|
| Data | `configs/data/` | Dataset paths, batch size, normalization stats |
| Model | `configs/model/` | Model type, action/state shapes, `async_train` mode|
| Trainer | `configs/trainer/` | Optimizer, scheduler, DeepSpeed, precision |

Training uses DeepSpeed ZeRO with bf16-mixed precision, AdamW (lr=1e-4, cosine warmup), and gradient clipping (norm=1.0).

## Deployment

Use `mibot/server/deploy.py` to start the inference server and `mibot/server/runtime/client.py` to connect from your robot controller. The server loads the checkpoint and normalization stats, listens on a TCP port, and returns denormalized `(30, 32)` action chunks. The `Client` handles Qwen3-VL tokenization, state composition, and action recovery.

```bash
python mibot/server/deploy.py --model /path/to/checkpoint --port 10086
```

The model produces a 30-step action chunk per inference call. How the robot controller consumes these chunks and schedules the next inference defines two execution modes:

- **Synchronous**: the controller executes all 30 steps of a chunk, then requests the next chunk and waits for the result before continuing. This mode is compatible with both synchronous and asynchronous checkpoints.
- **Asynchronous**: the inference of the next chunk does not wait until the current chunk is fully executed as in the synchronous execution.
The inference starts when the number of steps left for excuting is less than k (we set k=10 in our experiments).
To ensure the next chunk connects smoothly to the current one, the model takes the last `prefix_length` of  the remaining unexecuted actions as the **action prefix** during inference.
The model generates a full 30-step chunk conditioned on this prefix, so the beginning of the new chunk is consistent with the end of the previous one.
The robot controller discards the first N actions that were already executed during the inference latency, and seamlessly continues from the N+1-th action. This mode requires an asynchronously trained checkpoint (`async_train: true`).

## Training Data Format

Each episode contains three synchronized video files (ego, wrist-left, and wrist-right) and a metadata JSON file.
Each metadata JSON file contains (see [here](docs/data_format.md) for more details):

- `num_frames` — total frame count
- `instruction` — task description and multi-view prompt template
- `observations` — paths to the three camera videos
- `proprios` — per-frame proprioceptive state (end-effector pose, joint angles, gripper for both arms)
- `actions` — per-frame action targets (same structure as proprios)

## Building Your Own Dataset

1. **Prepare data** — For each episode, record synchronized camera videos and log per-frame proprioceptive state / action targets. Create a JSON annotation following the [data format specification](docs/data_format.md).

2. **Create a data config** — Copy `configs/data/earphone.yaml` to `configs/data/my_task.yaml`, then update `train_path` to your JSON directory and recompute `mean`/`std` from your dataset.

3. **Train** — Run the following:

    ```bash
    CUDA_VISIBLE_DEVICES=0 RESOURCE_GPU=1 \
    bash scripts/train.sh \
    data=my_task \
    model=XR0 \
    model.params.model.pretrained=/path/to/pretrained_ckpt/
    ```

## XPolicyLab / RoboDojo 接入

若要在 [XPolicyLab](https://github.com/Luminis-Platform/XPolicyLab) 上使用 RoboDojo 仿真数据进行微调，请参考 XPolicyLab 仓库中的接入文档：

```text
XPolicyLab/policy/Xiaomi_Robotics_0/
├── INSTALLATION.md    # 环境安装
├── README.md          # 数据处理 → 训练完整流程
├── process_data.sh    # RoboDojo HDF5 → XR-0 JSON
└── train.sh           # XPolicyLab 统一训练入口
```

典型流程：

```bash
cd XPolicyLab/policy/Xiaomi_Robotics_0
bash install.sh
conda activate mibot

bash process_data.sh RoboDojo cotrain arx_x5 100 ee
bash train.sh RoboDojo cotrain arx_x5 100 ee 0 0,1,2,3,4,5,6,7
```

RoboDojo 原始数据会被转换为 `json/` + `videos/` 目录，并自动生成带 `mean`/`std` 的 Hydra 数据配置。

## Project Structure

```
xr0/
├── configs/                    # Hydra configuration
│   ├── config.yaml             # Top-level config
│   ├── model/XR0.yaml           # XR0 model config
│   ├── data/                   # Dataset configs
│   └── trainer/deepspeed.yaml  # DeepSpeed trainer config
├── mibot/
│   ├── models/
│   │   ├── VLM/qwen3vl.py     # Qwen3-VL backbone
│   │   ├── VLA/XR0.py          # XR0 VLA model (DiT + rectified flow)
│   │   └── runner/base_runner.py  # Lightning training runner
│   ├── data/
│   │   ├── datasets/json_dataset.py
│   │   ├── datamodule/base_datamodule.py
│   │   └── collate/custom_collate.py
│   ├── server/
│   │   ├── deploy.py          # Model loading + server startup
│   │   └── runtime/           # TCP inference server/client
│   └── utils/
│       ├── cfg_utils.py       # Config resolution & DeepSpeed setup
│       ├── cosine_warmup.py   # LR scheduler
│       ├── io.py              # Hard-coded state/action helpers
│       └── model_utils.py     # auto_cast decorator
├── tools/
│   └── train.py               # Training entry point
├── scripts/
│   ├── train.sh               # torchrun launcher
│   └── deploy.sh              # Multi-server tmux launcher
├── assets/
│   └── requirements.txt
├── docs/
│   └── data_format.md         # Data format specification
├── data/
│   ├── json/                  # Episode annotations
│   └── videos/                # Episode videos
└── pretrained_ckpt/
```

## 📚 Citation

If you find this project useful, please consider citing:

```bibtex
@article{cai2026xiaomi,
  title={Xiaomi-Robotics-0: An Open-Sourced Vision-Language-Action Model with Real-Time Execution},
  author={Cai, Rui and Guo, Jun and He, Xinze and Jin, Piaopiao and Li, Jie and Lin, Bingxuan and Liu, Futeng and Liu, Wei and Ma, Fei and Ma, Kun and Qiu, Feng and Qu, Heng and Su, Yifei and Sun, Qiao and Wang, Dong and Wang, Donghao and Wang, Yunhong and Wu, Rujie and Xiang, Diyun and Yang, Yu and Ye, Hangjun and Zhang, Yuan and Zhou, Quanyun},
  journal={arXiv preprint arXiv:2602.12684},
  year={2026}
}
```

## 📄 License

This project is licensed under the [Apache License 2.0](LICENSE).
