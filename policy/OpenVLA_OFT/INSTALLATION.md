# OpenVLA_OFT 环境配置

## 一键安装

```bash
bash install.sh
```

## 手动安装

### 1. 创建环境

```bash
conda create -n openvla_oft python=3.10.6 -y
conda activate openvla_oft
pip install torch torchvision torchaudio
```

### 2. 安装 OpenVLA-OFT

```bash
cd openvla_oft
pip install -e .
pip install packaging ninja
pip install "flash-attn==2.5.5" --no-build-isolation
```

### 3. 安装 XPolicyLab

```bash
cd ../../..
pip install -e .
```

## 模型与数据路径

| 变量 | 说明 |
|------|------|
| `TFDS_DATA_DIR` | TensorFlow Datasets 根目录 |
| `OPENVLA_TFDS_DATASET_NAME` | 训练用 TFDS 名称 |

基座 VLA 权重通常由 OpenVLA 配置或 HF 指定，见上游文档。

## 训练与评测

详见 [README.md](README.md)。

## XPolicyLab 部署（eval）

已在 GPU 主机完成 debug client 闭环（`setup_eval_policy_server.sh` + `setup_eval_env_client.sh`）。

| 项 | 说明 |
|----|------|
| Server 环境 | `openvla_oft` |
| Client 环境 | `XPolicyLab`（conda） |
| eval 示例 ckpt | `RoboDojo-cotrain-arx_x5-3500-joint-0` |
| expert_data_num | `3500` |
| action_type | `joint` |
| xspark 权重 | `/mnt/xspark-data/final_ckpt/OpenVLA_OFT/seed0/.../100000_chkpt` |
| 备注 | deploy.yml 建议 eval_batch=false（debug） |

软链 checkpoint（在 `policy/OpenVLA_OFT/` 下）：

```bash
mkdir -p checkpoints
ln -sfn <xspark_dir> checkpoints/<6-tuple_dir_name>
```

`ckpt_name` 若已是完整 6-tuple（含多个 `-`），eval 脚本直接传入该目录名。

手动评测：

```bash
# terminal 1 — server
bash setup_eval_policy_server.sh RoboDojo stack_bowls RoboDojo-cotrain-arx_x5-3500-joint-0 arx_x5 3500 joint 0 0 openvla_oft <port> localhost

# terminal 2 — client
bash setup_eval_env_client.sh RoboDojo stack_bowls RoboDojo-cotrain-arx_x5-3500-joint-0 arx_x5 joint 0 0 XPolicyLab "ckpt_name=RoboDojo-cotrain-arx_x5-3500-joint-0,action_type=joint" <port> localhost
```

或使用 `eval.sh`（会等待 server 端口就绪后启动 client）。

