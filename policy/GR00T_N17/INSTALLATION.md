# GR00T_N17 环境配置

本文档记录在 XPolicyLab 中使用 NVIDIA Isaac GR00T N1.7 的推荐安装方式。GR00T N1.7 默认使用 `uv` 管理环境，训练建议使用 Python 3.10 与 CUDA 12.8 的 dGPU 机器。

## 1. 进入项目目录

```bash
cd /vepfs-cnbje63de6fae220/niantian/RoboDojo_env/XPolicyLab/policy/GR00T_N17/gr00t_n17
```

## 2. 安装系统依赖

`ffmpeg` 是 GR00T 读取视频数据的必要依赖；如果需要从 HuggingFace 下载模型或数据，也建议安装 `git-lfs`。

```bash
sudo apt-get update
sudo apt-get install -y ffmpeg git-lfs
git lfs install
```

## 3. 安装 uv

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

如果当前 shell 找不到 `uv`，重新打开终端，或执行：

```bash
source "$HOME/.local/bin/env"
```

## 4. 创建并安装 GR00T 环境

```bash
uv sync --python 3.10
uv run python -c "import gr00t; print('GR00T installed successfully')"
```

如果训练时提示 `CUDA_HOME is unset`，先执行一次：

```bash
uv run bash scripts/deployment/dgpu/install_deps.sh
```

## 5. 安装 XPolicyLab

XPolicyLab 的 policy 需要同时能导入 GR00T 与 XPolicyLab 本体。保持在 `gr00t_n17` 目录下执行：

```bash
uv pip install -e /vepfs-cnbje63de6fae220/niantian/RoboDojo_env/XPolicyLab
uv run python -c "import XPolicyLab; print('XPolicyLab ok')"
```

## 6. 准备 RoboDojo 数据

当前数据路径为：

```text
/vepfs-cnbje63de6fae220/xspark_shared/lerobot/RoboDojo_sim_arx-x5_v30
```

该数据集的 `meta/info.json` 显示 `codebase_version` 为 `v3.0`，而 GR00T N1.7 当前训练入口要求 GR00T-flavored LeRobot v2.1，并额外需要 `meta/modality.json`。建议先复制一份数据再转换，避免改动原始共享数据：

```bash
export DATA_ROOT=/vepfs-cnbje63de6fae220/xspark_shared/lerobot
export SRC_DATASET=RoboDojo_sim_arx-x5_v30
export GR00T_DATASET=RoboDojo_sim_arx-x5_gr00t

cp -a "${DATA_ROOT}/${SRC_DATASET}" "${DATA_ROOT}/${GR00T_DATASET}"

uv run --project scripts/lerobot_conversion \
  python scripts/lerobot_conversion/convert_v3_to_v2.py \
  --root "${DATA_ROOT}" \
  --repo-id "${GR00T_DATASET}"
```

转换完成后，训练数据路径为：

```text
/vepfs-cnbje63de6fae220/xspark_shared/lerobot/RoboDojo_sim_arx-x5_gr00t
```

## 7. 补充 meta/modality.json

GR00T 需要 `meta/modality.json` 描述状态、动作、图像和语言字段。RoboDojo arx-x5 数据中 `observation.state` 与 `action` 都是 14 维，前三路图像为 `cam_high`、`cam_left_wrist`、`cam_right_wrist`。可在转换后的数据目录中创建：

```bash
cat > "${DATA_ROOT}/${GR00T_DATASET}/meta/modality.json" <<'EOF'
{
  "state": {
    "left_arm": { "start": 0, "end": 7 },
    "right_arm": { "start": 7, "end": 14 }
  },
  "action": {
    "left_arm": { "start": 0, "end": 7 },
    "right_arm": { "start": 7, "end": 14 }
  },
  "video": {
    "front": { "original_key": "observation.images.cam_high" },
    "left_wrist": { "original_key": "observation.images.cam_left_wrist" },
    "right_wrist": { "original_key": "observation.images.cam_right_wrist" }
  },
  "annotation": {
    "human.task_description": { "original_key": "task_index" }
  }
}
EOF
```

如果后续改动作预测 horizon 或 modality 切分，需要在准备好 `README.md` 中的 modality config 后重新生成统计信息：

```bash
uv run python gr00t/data/stats.py \
  --dataset-path "${DATA_ROOT}/${GR00T_DATASET}" \
  --embodiment-tag NEW_EMBODIMENT \
  --modality-config-path /tmp/robodojo_arx_x5_config.py
```

## 8. 安装自检

```bash
uv run python -c "import torch; print(torch.cuda.is_available())"
uv run python gr00t/experiment/launch_finetune.py --help
```

环境配置完成后，训练与评测入口见 `README.md`。
