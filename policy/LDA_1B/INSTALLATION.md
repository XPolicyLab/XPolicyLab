# LDA_1B Installation

> 上游项目：[LDA-1B: Scaling Latent Dynamics Action Model via Universal Embodied Data Ingestion](https://arxiv.org/abs/2602.12215)
> GitHub: https://github.com/jiangranlv/latent-dynamics-action

## Step 1: 准备 conda 环境

```bash
conda create -n LDA_1B python=3.10
conda activate LDA_1B
```

## Step 2: 安装依赖

`bash install.sh LDA_1B` 。

## Step 3: 下载预训练权重

```bash
pip install -U "huggingface_hub[cli]"
```

### 3.1 Qwen3-VL-4B-Instruct（语言/视觉编码器）

```bash
huggingface-cli download Qwen/Qwen3-VL-4B-Instruct \
    --local-dir checkpoints/Qwen3-VL-4B-Instruct \
    --local-dir-use-symlinks False
```

### 3.2 DINOv3-ViT-S/16（潜在视觉特征编码器）

> 上游 README 明确要求 **DINOv3**。集合页：<https://huggingface.co/collections/facebook/dinov3-68924841bd6b561778e31009>，
> 该集合下属仓库需要先在 HuggingFace 接受许可。最常用的 ViT-S/16 变体：

```bash
huggingface-cli login   # 仅首次需要
huggingface-cli download facebook/dinov3-vits16-pretrain-lvd1689m \
    --local-dir checkpoints/dinov3-vit-s \
    --local-dir-use-symlinks False
```

### 3.3 LDA 策略检查点

```bash
# 通用预训练
huggingface-cli download Wayer2/LDA-pretrain \
    --local-dir checkpoints/LDA-pretrain \
    --local-dir-use-symlinks False

# RoboCasa-GR1 微调
huggingface-cli download Wayer2/LDA-robocasa \
    --local-dir checkpoints/LDA-robocasa \
    --local-dir-use-symlinks False
```

## Step 4: 数据准备（LeRobot v2.1）

XPolicyLab 默认样例数据位于 `data/RoboDojo/test_data/arx_x5`。将其转换为
LDA 上游 `gr00t_lerobot` 训练管线可直接消费的 LeRobot v2.1 数据集：

```bash
conda activate LDA_1B
bash XPolicyLab/policy/LDA_1B/process_data.sh \
    RoboDojo test_data arx_x5 3 joint
```

转换后的数据集会写入
`XPolicyLab/policy/LDA_1B/data/RoboDojo-test_data-arx_x5-3-joint/`
（命名格式：`<dataset_name>-<task_name>-<env_cfg_type>-<expert_data_num>-<action_type>`），
目录结构遵循 LeRobot v2.1：

```
data/
videos/
meta/
  ├─ info.json
  ├─ modality.json
  ├─ episodes.jsonl
  ├─ episodes_stats.jsonl
  ├─ stats.json
  └─ tasks.jsonl
```

并已在上游 `lda/dataloader/gr00t_lerobot/{mixtures.py, data_config.py, embodiment_tags.py}`
注册：

- mixture 名：`xpolicylab`
- robot_type：`arx_x5`
- EmbodimentTag：`ARX_X5`
- DataConfig：`ArxX5DataConfig`

## Step 5: 训练 / 评估

`train.sh` 默认通过脚本自身位置解析以下路径，无需手动 `export`：

| 变量 | 默认值（脚本相对） | 含义 |
|---|---|---|
| `LDA_BASE_VLM` | `<policy>/checkpoints/Qwen3-VL-4B-Instruct` | Qwen3-VL 本地路径 |
| `LDA_VISION_ENCODER` | `<policy>/checkpoints/dinov3-vit-s` | DINOv3 父目录（上游会再拼 `dinov3-vits16-pretrain-lvd1689m`） |
| `LDA_DATA_ROOT` | `<policy>/data` | LeRobot 数据根目录 |
| `LDA_DATA_MIX` | `xpolicylab` | 上游 mixture 名 |
| `LDA_CKPT_ROOT` | `<policy>/checkpoints` | 训练输出根目录（项目约定，与 Step 3 的预训练子目录共存） |
| `LDA_CKPT_SETTING` | `<dataset>-<task>-<env_cfg>-<expert_data_num>-<action_type>-<seed>` | 训练子目录名，对齐 DP 的 6 元组 `ckpt_setting`；`eval.sh` 按相同规则用 `ckpt_name` 索引 |

`<policy>` 指 `XPolicyLab/policy/LDA_1B/` 这个绝对路径，由 `train.sh` 用 `$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)` 在运行时算出，**不依赖 cwd、不写死任何机器路径**。
checkpoints 按 Step 3 的命令下载到对应子目录后，直接跑训练即可；需要换位置再 `export LDA_DATA_ROOT=/your/path bash train.sh ...` 覆盖。
训练产物完整路径为 `<policy>/checkpoints/<ckpt_setting>/checkpoints/steps_*_pytorch_model.pt`；上游 `from_pretrained` 同时需要同目录下的 `config.yaml` 与 `dataset_statistics.json`，`train.sh` 会自动写入。

> 兼容性：早期默认目录为 `<policy>/runs/<dataset>-<ckpt_name>-<env_cfg>-seed<seed>/`（旧的 `LDA_RUN_ROOT` / `LDA_RUN_ID`）。若旧训练产物仍在该位置，`eval.sh` 在新默认路径找不到 checkpoint 时会**自动回退**到旧目录；也可显式 `export LDA_CHECKPOINT_PATH=<...>/steps_<N>_pytorch_model.pt` 直接指定。

```bash
bash XPolicyLab/policy/LDA_1B/train.sh \
    RoboDojo test_data arx_x5 3 joint 0 0

bash XPolicyLab/policy/LDA_1B/eval.sh \
    RoboDojo test_data arx_x5 3 joint 0 0 LDA_1B XPolicyLab
```

## 目录结构参考

```
XPolicyLab/policy/LDA_1B/
├── __init__.py / model.py / deploy.py / deploy.yml
├── install.sh / INSTALLATION.md
├── process_data.sh / train.sh / eval.sh
├── data/                                     # 转换后数据集（process_data 产物）
├── checkpoints/                              # Step 3 预训练子目录 + train.sh 写入的 <ckpt_setting>/
└── LDA-1B/                                   # 上游源码（已注册 arx_x5）
    └── xpolicylab_adapter/                   # XPolicyLab ↔ LDA-1B 适配层（HDF5→LeRobot v2.1、gr00t action_dim 等）
```
