# Xiaomi_Robotics_0 环境配置

本文档记录在 XPolicyLab 中使用 **Xiaomi-Robotics-0 (XR-0)** 的推荐安装方式。训练、数据处理与评测统一使用 **`mibot`** 一个 conda 环境，无需再单独创建 `XPolicyLab` 环境。

## 0. 前置：演示数据

若尚未拉取 RoboDojo 原始数据，请先按 [XPolicyLab README](../../README.md) 准备数据，或使用共享路径：

```text
/vepfs-cnbje63de6fae220/hekun/datasets/RoboDojo/sim_cloud/<task_name>/<env_cfg_type>/*.hdf5
```

## 1. 一键安装（推荐）

```bash
cd policy/Xiaomi_Robotics_0
bash install.sh
conda activate mibot
```

`install.sh` 会：

- 创建 `mibot` conda 环境（Python 3.12）
- 安装 PyTorch 2.8、Flash Attention、XR-0 训练依赖
- 在 `xiaomi_robotics_0/xr0` 与 XPolicyLab 根目录分别执行 `pip install -e .`（评测 client 也走同一环境）

## 2. 手动安装

```bash
cd /vepfs-cnbje63de6fae220/niantian/RoboDojo_env/XPolicyLab/policy/Xiaomi_Robotics_0/xiaomi_robotics_0/xr0

conda create -n mibot python=3.12 -y
conda activate mibot

pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 \
  --index-url https://download.pytorch.org/whl/cu128
pip uninstall -y ninja && pip install ninja
pip install flash-attn==2.8.3 --no-build-isolation

pip install -e .

cd /vepfs-cnbje63de6fae220/niantian/RoboDojo_env/XPolicyLab
pip install -e .
pip install opencv-python-headless tqdm scipy
```

## 3. 准备预训练权重

从 [Xiaomi-Robotics-0-Pretrain](https://huggingface.co/XiaomiRobotics/Xiaomi-Robotics-0-Pretrain) 下载权重，并转换为 XR-0 训练所需的 PyTorch checkpoint：

```bash
cd /vepfs-cnbje63de6fae220/niantian/RoboDojo_env/XPolicyLab/policy/Xiaomi_Robotics_0/xiaomi_robotics_0/xr0

# 方式 A：使用 HuggingFace 缓存目录
python tools/weight_convert.py \
  --model_path hf_pretrain \
  --output_dir pretrained_ckpt \
  --output_filename xr0_pretrained.pt

# 方式 B：使用共享权重（若已存在）
export XR0_PRETRAINED_PATH=/vepfs-cnbje63de6fae220/xspark_shared/xiaomi_checkpoints/xr0_pretrained.pt
```

转换完成后，默认预训练路径为：

```text
policy/Xiaomi_Robotics_0/xiaomi_robotics_0/xr0/pretrained_ckpt/xr0_pretrained.pt
```

## 4. 链接评测权重（可选）

```bash
cd /vepfs-cnbje63de6fae220/niantian/RoboDojo_env/XPolicyLab/policy/Xiaomi_Robotics_0
bash scripts/link_checkpoint.sh RoboDojo cotrain arx_x5 100 ee 0
```

## 5. 安装自检

```bash
conda activate mibot
python -c "import torch; print('cuda:', torch.cuda.is_available())"
python -c "import mibot; print('mibot ok')"
python -c "import XPolicyLab; print('XPolicyLab ok')"
```

环境配置完成后，数据处理、训练与评测入口见 `README.md`。

## XPolicyLab 部署（eval）

已在 GPU 主机完成 debug client 闭环（`setup_eval_policy_server.sh` + `setup_eval_env_client.sh`）。

| 项 | 说明 |
|----|------|
| Server 环境 | `mibot` |
| Client 环境 | `XPolicyLab`（conda） |
| eval 示例 ckpt | `RoboDojo-cotrain-arx_x5-100-ee-0` |
| expert_data_num | `100` |
| action_type | `ee` |
| xspark 权重 | `/mnt/xspark-data/final_ckpt/Xiaomi_Robotics_0/.../last.ckpt/checkpoint` |
| 备注 | 无 flash_attn 时 model 自动 sdpa |

软链 checkpoint（在 `policy/Xiaomi_Robotics_0/` 下）：

```bash
mkdir -p checkpoints
ln -sfn <xspark_dir> checkpoints/<6-tuple_dir_name>
```

`ckpt_name` 若已是完整 6-tuple（含多个 `-`），eval 脚本直接传入该目录名。

手动评测：

```bash
# terminal 1 — server
bash setup_eval_policy_server.sh RoboDojo stack_bowls RoboDojo-cotrain-arx_x5-100-ee-0 arx_x5 100 ee 0 0 mibot <port> localhost

# terminal 2 — client
bash setup_eval_env_client.sh RoboDojo stack_bowls RoboDojo-cotrain-arx_x5-100-ee-0 arx_x5 ee 0 0 XPolicyLab "ckpt_name=RoboDojo-cotrain-arx_x5-100-ee-0,action_type=ee" <port> localhost
```

或使用 `eval.sh`（会等待 server 端口就绪后启动 client）。

