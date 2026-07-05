# RDT_1B

RDT_1B 已按 XPolicyLab policy 方式封装。环境安装见 [INSTALLATION.md](INSTALLATION.md)。

## 约定路径

所有数据与 embedding 通过 **4 元组** 索引，与 `train.sh` / `process_data.sh` 参数一致：

```text
<bench_name>-<ckpt_name>-<env_cfg_type>-<action_type>
```

```text
policy/RDT_1B/
├── data/<4-tuple>/              # HDF5 软链（只读，不污染共享数据集）
├── lang_embeds/<4-tuple>/       # 预编码语言 embedding
│   └── <task>/<env>/lang_embed.pt
├── lang_embeds/empty_lang_embed.pt
├── checkpoints/<5-tuple>/        # 训练输出（4 元组 + seed）
├── process_data.sh
└── train.sh
```

HDF5 目录结构（RoboDojo）：

```text
<task_name>/<env_cfg>/data/episode_*.hdf5
```

## 快速开始

```bash
# 1. 安装环境
cd policy/RDT_1B && bash install.sh && conda activate rdt_1b

# 2. 数据处理：软链数据 + 预编码 embedding（每个数据集跑一次）
bash process_data.sh RoboDojo stack_bowls arx_x5 joint
# 或显式指定源数据目录：
# bash process_data.sh RoboDojo stack_bowls arx_x5 joint /path/to/hdf5_root

# 3. 训练
bash train.sh RoboDojo stack_bowls arx_x5 joint 0 0,1,2,3,4,5,6,7
```

单卡调试：将最后一项改为 `0`。

## process_data.sh

```text
bash process_data.sh <bench_name> <ckpt_name> <env_cfg_type> <action_type> [source_path]
```

| 参数 | 示例 | 说明 |
|------|------|------|
| 前 4 项 | 同 `train.sh`（不含 seed/gpu） | 决定 `data/` 与 `lang_embeds/` 子目录名 |
| `source_path` | 可选 | 源 HDF5 根目录；省略时按顺序查找 `data/<dataset>/<ckpt>` 等 |

源数据查找顺序：`source_path` → `RAW_DATA_ROOT` → `XPolicyLab/data/<dataset>/<ckpt>` → `XPolicyLab/data/<dataset>_<ckpt>`。

常用选项：`--overwrite`（重编码）、`--skip-encode`（仅软链）、`--gpu N`。

## train.sh

```text
bash train.sh <bench_name> <ckpt_name> <env_cfg_type> <action_type> <seed> <gpu_id>
```

训练前需已执行 `process_data.sh`。默认每 1000 step 存 checkpoint，总步数 200000。

权重约定在 `weights/RDT/`：`bash install.sh` 从 HuggingFace 下载，或 `RDT_WEIGHTS_SRC=<dir> bash install.sh` 软链已有权重。DeepSpeed 需设置 `CUTLASS_PATH`（见 [INSTALLATION.md](INSTALLATION.md)）。

## 部署与评测

详见 [INSTALLATION.md](INSTALLATION.md#xpolicylab-部署eval)。

```bash
bash eval.sh RoboDojo stack_bowls RoboDojo-cotrain-arx_x5-joint-0 arx_x5 joint 0 <policy_gpu> <env_gpu> rdt_1b XPolicyLab
```

`ckpt_name` 传入完整 checkpoint 目录名（`<4-tuple>-<seed>`）。
