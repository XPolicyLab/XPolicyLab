# Motus 环境配置

Motus 上游源码位于 [motus/](motus/)，使用 conda 环境 `motus`。

## 一键安装

```bash
bash install.sh
```

## 手动安装

### 1. 创建 conda 环境

```bash
conda create -n motus python=3.10 -y
conda activate motus

pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128
pip install flash-attn --no-build-isolation
```

### 2. 安装 Motus 依赖

```bash
cd motus
pip install -r requirements.txt
pip install --no-deps lerobot==0.3.2
pip install -r requirements/lerobot.txt
pip install -e .
```

### 3. 安装 XPolicyLab

```bash
cd ../..
pip install -e .
```

## 模型与数据路径

| 变量 / 参数 | 说明 |
|-------------|------|
| `WAN_PATH` / `--wan_path` | 含 `Wan2.2-TI2V-5B`、`Qwen3-VL-2B-Instruct`、`Motus/` 的模型根目录 |
| `LEROBOT_DATA_ROOT` | LeRobot 数据集父目录（需指定具体子数据集 `root`） |
| LeRobot 直读 | 配置中填写 `repo_id` + `root=${LEROBOT_DATA_ROOT}/<dataset>` |

预训练组件通常包括：`Motus/`（Stage2）、`Wan2.2-TI2V-5B/`、`Qwen3-VL-2B-Instruct/`。

详细 LeRobot 训练与 T5 缓存流程见 [motus/README.md](motus/README.md)。

## 训练与评测

见 [README.md](README.md)。
