# GR00T_N17

GR00T_N17 是 NVIDIA Isaac GR00T N1.7 在 XPolicyLab 下的源码接入目录。GR00T N1.7 是一个 Vision-Language-Action 模型，使用视觉、语言和机器人本体状态生成连续动作，适合在 LeRobot 格式数据上做后训练与开环评测。

上游源码位于：

```text
policy/GR00T_N17/gr00t_n17
```

安装步骤见 `INSTALLATION.md`。

## 当前数据

RoboDojo arx-x5 数据路径：

```text
${LEROBOT_DATA_ROOT}/RoboDojo_sim_arx-x5_v30
```

该数据是 LeRobot v3.0。GR00T N1.7 当前训练入口要求 GR00T-flavored LeRobot v2.1，并需要额外的 `meta/modality.json`。请先按 `INSTALLATION.md` 转换并补充 metadata。推荐转换后的训练路径为：

```text
${LEROBOT_DATA_ROOT}/RoboDojo_sim_arx-x5_gr00t
```

## 项目结构

```text
GR00T_N17
├── INSTALLATION.md
├── process_data.sh              # 数据转换与 stats 生成
├── train.sh                     # 微调训练
├── README.md
└── gr00t_n17
    ├── gr00t                   # 模型、数据加载、训练和评测代码
    ├── examples                 # 官方微调示例
    ├── getting_started          # 官方数据与部署说明
    ├── scripts                  # 数据转换、部署和 TensorRT 脚本
    └── pyproject.toml
```

## 数据处理

```bash
# 在 policy/GR00T_N17 目录下
bash process_data.sh <bench_name> <ckpt_name> <env_cfg_type> <expert_data_num> <action_type>
```

示例：

```bash
bash process_data.sh RoboDojo cotrain arx_x5 3500 joint
```

默认会从 `RoboDojo_sim_arx-x5_v30` 复制并转换为 GR00T 可用的 LeRobot v2.1，输出到：

```text
${LEROBOT_DATA_ROOT}/RoboDojo-cotrain-arx_x5-3500-joint
```

可通过环境变量覆盖源数据路径：

```bash
GR00T_SRC_DATASET=RoboDojo_sim_arx-x5_v30 bash process_data.sh RoboDojo cotrain arx_x5 3500 joint
```

## 微调训练

```bash
bash train.sh <bench_name> <ckpt_name> <env_cfg_type> <expert_data_num> <action_type> <seed> <gpu_id>
```

示例：

```bash
bash train.sh RoboDojo cotrain arx_x5 3500 joint 0 0,1,2,3,4,5,6,7
```

训练前请确保本地已有：

- `GR00T-N1.7-3B`：`GR00T_BASE_MODEL`（默认 HF id `nvidia/GR00T-N1.7-3B`）
- `Cosmos-Reason2-2B`：`GR00T_COSMOS_MODEL`（默认 HF id `nvidia/Cosmos-Reason2-2B`）

`train.sh` 会自动把 Cosmos 注册到 HF cache，并默认 `HF_HUB_OFFLINE=1` 走本地模型。

训练产物保存在：

```text
policy/GR00T_N17/checkpoints/RoboDojo-cotrain-arx_x5-3500-joint-0
```

## 训练前准备（手动调试时可参考）

进入 GR00T 源码目录：

```bash
# 在 policy/GR00T_N17 目录下/gr00t_n17
```

设置常用路径：

```bash
export POLICY_ROOT="$(pwd)"
export DATASET_PATH=${LEROBOT_DATA_ROOT}/RoboDojo_sim_arx-x5_gr00t
export CKPT_NAME=RoboDojo-cotrain-arx_x5-3500-joint-0
export OUTPUT_DIR="${POLICY_ROOT}/checkpoints/${CKPT_NAME}"
mkdir -p "${OUTPUT_DIR}"
```

准备 GR00T 的 modality config。该配置需要与 `meta/modality.json` 中的 key 保持一致：

```bash
cat > /tmp/robodojo_arx_x5_config.py <<'PY'
from gr00t.configs.data.embodiment_configs import register_modality_config
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.types import (
    ActionConfig,
    ActionFormat,
    ActionRepresentation,
    ActionType,
    ModalityConfig,
)

robodojo_arx_x5_config = {
    "video": ModalityConfig(
        delta_indices=[0],
        modality_keys=["front", "left_wrist", "right_wrist"],
    ),
    "state": ModalityConfig(
        delta_indices=[0],
        modality_keys=["left_arm", "right_arm"],
    ),
    "action": ModalityConfig(
        delta_indices=list(range(0, 16)),
        modality_keys=["left_arm", "right_arm"],
        action_configs=[
            ActionConfig(
                rep=ActionRepresentation.RELATIVE,
                type=ActionType.NON_EEF,
                format=ActionFormat.DEFAULT,
            ),
            ActionConfig(
                rep=ActionRepresentation.RELATIVE,
                type=ActionType.NON_EEF,
                format=ActionFormat.DEFAULT,
            ),
        ],
    ),
    "language": ModalityConfig(
        delta_indices=[0],
        modality_keys=["annotation.human.task_description"],
    ),
}

register_modality_config(
    robodojo_arx_x5_config,
    embodiment_tag=EmbodimentTag.NEW_EMBODIMENT,
)
PY
```

如需在训练前重新生成统计信息：

```bash
uv run python gr00t/data/stats.py \
  --dataset-path "${DATASET_PATH}" \
  --embodiment-tag NEW_EMBODIMENT \
  --modality-config-path /tmp/robodojo_arx_x5_config.py
```

## 微调训练

单卡示例：

```bash
CUDA_VISIBLE_DEVICES=0 \
NUM_GPUS=1 \
MAX_STEPS=10000 \
SAVE_STEPS=1000 \
GLOBAL_BATCH_SIZE=32 \
USE_WANDB=0 \
uv run bash examples/finetune.sh \
  --base-model-path nvidia/GR00T-N1.7-3B \
  --dataset-path "${DATASET_PATH}" \
  --embodiment-tag NEW_EMBODIMENT \
  --modality-config-path /tmp/robodojo_arx_x5_config.py \
  --output-dir "${OUTPUT_DIR}" \
  --experiment-name "${CKPT_NAME}"
```

多卡示例：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
NUM_GPUS=4 \
MAX_STEPS=20000 \
SAVE_STEPS=1000 \
GLOBAL_BATCH_SIZE=128 \
USE_WANDB=0 \
uv run bash examples/finetune.sh \
  --base-model-path nvidia/GR00T-N1.7-3B \
  --dataset-path "${DATASET_PATH}" \
  --embodiment-tag NEW_EMBODIMENT \
  --modality-config-path /tmp/robodojo_arx_x5_config.py \
  --output-dir "${OUTPUT_DIR}" \
  --experiment-name "${CKPT_NAME}"
```

训练产物默认保存在：

```text
policy/GR00T_N17/checkpoints/<bench_name>-<ckpt_name>-<env_cfg_type>-<expert_data_num>-<action_type>-<seed>
```

当前示例使用的 XPolicyLab 命名为：

```text
policy/GR00T_N17/checkpoints/RoboDojo-cotrain-arx_x5-3500-joint-0
```

## 开环评测

训练完成后，可先用 GR00T 官方开环评测检查预测动作是否正常：

```bash
# 在 policy/GR00T_N17 目录下/gr00t_n17
source .venv/bin/activate

uv run python gr00t/eval/open_loop_eval.py \
  --dataset-path "${DATASET_PATH}" \
  --embodiment-tag NEW_EMBODIMENT \
  --model-path "${OUTPUT_DIR}/checkpoint-10000" \
  --traj-ids 0 \
  --action-horizon 16 \
  --steps 200 \
  --modality-keys left_arm right_arm
```

如果保存步数不同，将 `checkpoint-10000` 替换为实际 checkpoint 目录。

## 部署

使用 `eval.sh` 拉起 policy server + env client；也可分别执行 `setup_eval_policy_server.sh` 与 `setup_eval_env_client.sh`：

```bash
# 在 policy/GR00T_N17 目录下
bash install.sh   # 首次：uv sync + 安装 XPolicyLab 到 gr00t .venv

# debug 连通性测试（0 号卡，policy 用 gr00t uv 环境，client 用 mibot）
bash eval.sh RoboDojo sweep_blocks cotrain arx_x5 3500 joint 0 0 0 uv mibot
```

参数说明：

| 参数 | 含义 |
|------|------|
| `3500` | `expert_data_num`，与训练数据规模一致 |
| `joint` | 当前 arx_x5 modality 为 joint 空间相对动作，需与训练 `action_type` 一致 |
| `uv` | policy server 使用 `gr00t_n17/.venv`（见 `deploy.yml` 的 `policy_uv_env_path`） |
| `mibot` | env client 使用的 conda 环境（需已 `pip install -e XPolicyLab`） |

checkpoint 目录约定（6 元组）：

```text
checkpoints/RoboDojo-cotrain-arx_x5-3500-joint-0/
  └── RoboDojo-cotrain-arx_x5-3500-joint-0/
        └── checkpoint-60000/
```

`deploy.yml` 中 `checkpoint_num: last` 会自动选最新 step；也可改为具体步数如 `60000`。

软链接已有权重：

```bash
cd policy/GR00T_N17
bash scripts/link_checkpoint.sh RoboDojo cotrain arx_x5 3500 joint 0 \
  /path/to/upload_ckpts/policy/GR00T_N17/checkpoints/RoboDojo-cotrain-arx_x5-3500-joint-0/RoboDojo-cotrain-arx_x5-3500-joint-0
```

`deploy.yml` 可配置项（部署相关）：

| 字段 | 含义 |
|------|------|
| `model_dir` | 相对 policy 根目录；`null` 时用 `checkpoints/<6元组>/` |
| `checkpoint_num` | `last` 或具体 step（如 `60000`） |
| `cosmos_model_path` | 默认 `nvidia/Cosmos-Reason2-2B`（启动时自动下载），覆盖 checkpoint 内嵌的绝对路径 |
| `embodiment_tag` | 与训练一致，RoboDojo 为 `NEW_EMBODIMENT` |

部署时 checkpoint 与 Cosmos 均通过相对路径或 HuggingFace 仓库 id 配置，无需硬编码机器路径。

## 与 XPolicyLab 的参数约定

XPolicyLab 统一训练入口通常为：

```bash
bash train.sh <bench_name> <ckpt_name> <env_cfg_type> <expert_data_num> <action_type> <seed> <gpu_id>
```

GR00T_N17 当前保留上游原生训练入口，参数可按以下方式映射：

| XPolicyLab 参数 | GR00T 侧含义 |
|---|---|
| `bench_name` | 数据集名称，如 `RoboDojo` |
| `ckpt_name` | 实验名或多任务训练名，如 `cotrain` |
| `env_cfg_type` | 本体配置，如 `arx_x5` |
| `expert_data_num` | 训练轨迹数，当前数据为 3500 |
| `action_type` | RoboDojo 当前建议使用 `joint` |
| `seed` | 随机种子，并参与 checkpoint 目录命名 |
| `gpu_id` | 传给 `CUDA_VISIBLE_DEVICES` |

后续如补充外层 `train.sh`，建议仍按 XPolicyLab 6 元组保存到 `policy/GR00T_N17/checkpoints/`，并在脚本内部调用 `uv run bash examples/finetune.sh`。

## 参考文档

- `gr00t_n17/README.md`: NVIDIA Isaac GR00T N1.7 原项目介绍。
- `gr00t_n17/getting_started/data_preparation.md`: GR00T LeRobot 数据格式说明。
- `gr00t_n17/getting_started/data_config.md`: modality config 写法。
- `gr00t_n17/getting_started/finetune_new_embodiment.md`: 自定义本体微调流程。
