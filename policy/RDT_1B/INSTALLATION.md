# RDT_1B 环境配置

RDT_1B 使用 conda + `rdt/` 上游源码，预训练组件从 HuggingFace 下载。

## 一键安装

```bash
bash install.sh
```

已存在 conda 环境时可跳过创建：

```bash
RDT_SKIP_CONDA_CREATE=1 bash install.sh
```

已有权重目录时软链到 `weights/RDT/`（不下载）：

```bash
RDT_WEIGHTS_SRC=<path_to_weights_root> bash install.sh
```

仅安装依赖、跳过权重：

```bash
RDT_SKIP_WEIGHTS=1 bash install.sh
```

## 手动安装

### 1. 创建 conda 环境

```bash
conda create -n rdt_1b python=3.10 -y
conda activate rdt_1b

pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu121
pip install packaging==24.0 ninja
pip install flash-attn==2.7.2.post1 --no-build-isolation
```

### 2. 安装 RDT 源码

```bash
cd rdt
pip install -r requirements.txt
```

若 PyPI 镜像无法安装 `tfds-nightly` / `tensorflow`，请使用官方源：

```bash
# pip install tfds-nightly==4.9.4.dev202402070044 -i https://pypi.org/simple
# pip install tensorflow==2.15.0.post1 -i https://pypi.org/simple
```

### 3. 安装 XPolicyLab

```bash
cd ../../..
pip install -e .
```

### 4. 下载模型权重（HuggingFace）

```bash
mkdir -p weights/RDT && cd weights/RDT
huggingface-cli download google/t5-v1_1-xxl --local-dir t5-v1_1-xxl
huggingface-cli download google/siglip-so400m-patch14-384 --local-dir siglip-so400m-patch14-384
huggingface-cli download robotics-diffusion-transformer/rdt-1b --local-dir rdt-1b
```

也可在训练时通过环境变量指向本地目录或 HF cache，见 [README.md](README.md)。

## 模型与数据路径

| 变量 / 路径 | 说明 |
|-------------|------|
| `weights/RDT/` | 默认本地下载目录（相对 policy 根） |
| `RDT_HDF5_DIR` | 训练数据目录（覆盖 `data/<4-tuple>/`） |
| `RDT_PRETRAINED_MODEL` | RDT 主模型路径或 HF id |
| `TEXT_ENCODER_NAME` | 文本编码器路径或 HF id |
| `VISION_ENCODER_NAME` | 视觉编码器路径或 HF id |

## 训练与评测

详见 [README.md](README.md)。

## XPolicyLab 部署（eval）

已在 GPU 主机完成 debug client 闭环（`setup_eval_policy_server.sh` + `setup_eval_env_client.sh`）。

| 项 | 说明 |
|----|------|
| Server 环境 | `RDT` |
| Client 环境 | `XPolicyLab`（conda） |
| eval 示例 ckpt | `RoboDojo-cotrain-arx_x5-joint-0` |
| action_type | `joint` |
软链 checkpoint（在 `policy/RDT_1B/` 下）：

```bash
mkdir -p checkpoints
ln -sfn <path_to_trained_ckpt> checkpoints/<5-tuple_dir_name>
```

`ckpt_name` 传入完整 checkpoint 目录名（`<dataset>-<ckpt>-<env_cfg>-<action_type>-<seed>`）。

手动评测：

```bash
# terminal 1 — server
bash setup_eval_policy_server.sh RoboDojo stack_bowls RoboDojo-cotrain-arx_x5-joint-0 arx_x5 joint 0 0 rdt_1b <port> localhost

# terminal 2 — client
bash setup_eval_env_client.sh RoboDojo stack_bowls RoboDojo-cotrain-arx_x5-joint-0 arx_x5 joint 0 0 XPolicyLab "ckpt_name=RoboDojo-cotrain-arx_x5-joint-0,action_type=joint" <port> localhost
```

或使用 `eval.sh`（会等待 server 端口就绪后启动 client）。

