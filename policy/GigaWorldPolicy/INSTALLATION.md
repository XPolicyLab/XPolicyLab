# GigaWorldPolicy 环境配置

## 一键安装

```bash
bash install.sh
```

## 手动安装

### 1. 创建环境

```bash
conda create -n gigaworld-policy python=3.11 -y
conda activate gigaworld-policy
```

### 2. 安装 GigaWorld 依赖

```bash
cd giga_world_policy
pip install -e ./third_party/giga-train
pip install -e ./third_party/giga-models
pip install -e ./third_party/giga-datasets
```

### 3. 安装 XPolicyLab

```bash
cd ../../..
pip install -e .
```

## 模型与数据路径

| 变量 | 说明 |
|------|------|
| `GIGAWORLD_DATA_DIR` | LeRobot 训练数据目录 |
| `GIGAWORLD_NORM_PATH` | `norm_stats_delta.json` 路径 |
| `GIGAWORLD_PRETRAINED_PATH` | Wan2.2 等预训练权重目录或 HF 缓存 |
| `GIGAWORLD_BASE_CONFIG` | 基础训练 JSON 配置 |

## 训练与评测

详见 [README.md](README.md)。内层说明见 [giga_world_policy/Readme.md](giga_world_policy/Readme.md)。
