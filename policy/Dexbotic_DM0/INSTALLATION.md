# Dexbotic_DM0 环境配置

本文档记录在 XPolicyLab 中使用 **Dexbotic DM0** 的推荐安装方式。上游 Dexbotic 源码位于 `dexbotic/` 子目录。

## 1. 创建 conda 环境

建议在独立环境中安装，避免 PyTorch / transformers 版本冲突：

```bash
cd policy/Dexbotic_DM0

conda create -n DM0 python=3.10 -y
conda activate DM0

# 先安装与 CUDA 12.x 驱动匹配的 PyTorch（不要用默认 cu130）
pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 \
  --index-url https://download.pytorch.org/whl/cu128

# 训练必需但 install.sh 未包含的依赖
pip install 'deepspeed>=0.18.0' 'numpydantic>=1.6'
```

## 2. 一键安装（推荐）

```bash
cd policy/Dexbotic_DM0
conda activate DM0
bash install.sh
```

`install.sh` 会：

- 在 `dexbotic/` 目录执行 `pip install -e .`
- 在 XPolicyLab 根目录执行 `pip install -e .`，并安装 `h5py pyyaml`（policy server 加载 `process_data` 需要）
- 安装数据转换依赖 `opencv-python-headless`、`tqdm`

## 3. 手动安装

```bash
cd policy/Dexbotic_DM0/dexbotic
pip install -e .

cd ../..   # XPolicyLab 根目录
pip install -e .
pip install opencv-python-headless tqdm
```

## 4. 准备 DM0-base 预训练权重

```bash
pip install -U "huggingface_hub[cli]"
# 如需 gated 模型：hf auth login

mkdir -p dexbotic/checkpoints
cd dexbotic/checkpoints
hf download Dexmal/DM0-base --local-dir DM0-base
```

或使用已有权重：

```bash
export DM0_BASE_MODEL=/path/to/DM0-base
```

## 5. 准备 RoboDojo 原始数据

默认期望目录结构：

```text
<DM0_RAW_DATA_ROOT>/sim_cloud/<task_name>/<env_cfg_type>/*.hdf5
```

可通过环境变量覆盖：

```bash
export DM0_RAW_DATA_ROOT=/path/to/RoboDojo
```

## 6. 安装自检

```bash
python -c "import torch; print('cuda:', torch.cuda.is_available())"
python -c "import deepspeed; print('deepspeed ok')"
python -c "import dexbotic; print('dexbotic ok')"
python -c "import XPolicyLab; print('XPolicyLab ok')"
```

## 7. 常用环境变量

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `DM0_RAW_DATA_ROOT` | （见 `process_data.sh`） | 原始 HDF5 根目录 |
| `DM0_CONVERTED_DATA_ROOT` | `data/<5-tuple>` | 转换输出目录 |
| `DM0_BASE_MODEL` | `dexbotic/checkpoints/DM0-base` | 预训练模型 |
| `DM0_GLOBAL_BATCH_SIZE` | `256` | 全局 batch size |
| `DM0_BATCH_SIZE` | `4` | 每卡 micro batch |
| `DM0_GRAD_ACCUM` | 自动推导 | 梯度累积步数 |
| `DM0_MAX_STEPS` | `60000` | 训练步数 |
| `DM0_SAVE_STEPS` | `2000` | checkpoint 保存间隔 |
| `DM0_CONVERT_WORKERS` | `8` | 数据转换并行数 |
| `DM0_TRAIN_BACKEND` | `deepspeed` | 训练后端（`deepspeed` / `fsdp2` / `ddp`） |

全局 batch 计算公式：

```text
global_batch = DM0_BATCH_SIZE × num_gpus × DM0_GRAD_ACCUM
```

`train.sh` 会校验上述关系；未设置 `DM0_GRAD_ACCUM` 时按 `DM0_GLOBAL_BATCH_SIZE` 自动计算。

环境配置完成后，数据处理与训练入口见 `README.md`。

## XPolicyLab 部署（eval）

已在 GPU 主机完成 debug client 闭环（`setup_eval_policy_server.sh` + `setup_eval_env_client.sh`）。

| 项 | 说明 |
|----|------|
| Server 环境 | `DM0` |
| Client 环境 | `XPolicyLab`（conda） |
| eval 示例 ckpt | `RoboDojo-cotrain-arx_x5-3500-ee-0` |
| expert_data_num | `3500` |
| action_type | `ee` |
| xspark 权重 | `/mnt/xspark-data/final_ckpt/DM_0/RoboDojo-cotrain-arx_x5-3500-ee-0/checkpoint-20000` |
| 备注 | install.sh 含 h5py（XPolicyLab server） |

软链 checkpoint（在 `policy/Dexbotic_DM0/` 下）：

```bash
mkdir -p checkpoints
ln -sfn <xspark_dir> checkpoints/<6-tuple_dir_name>
```

`ckpt_name` 若已是完整 6-tuple（含多个 `-`），eval 脚本直接传入该目录名。

手动评测：

```bash
# terminal 1 — server
bash setup_eval_policy_server.sh RoboDojo stack_bowls RoboDojo-cotrain-arx_x5-3500-ee-0 arx_x5 3500 ee 0 0 DM0 <port> localhost

# terminal 2 — client
bash setup_eval_env_client.sh RoboDojo stack_bowls RoboDojo-cotrain-arx_x5-3500-ee-0 arx_x5 ee 0 0 XPolicyLab "ckpt_name=RoboDojo-cotrain-arx_x5-3500-ee-0,action_type=ee" <port> localhost
```

或使用 `eval.sh`（会等待 server 端口就绪后启动 client）。

