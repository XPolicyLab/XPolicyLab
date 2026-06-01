# GalaxeaVLA on XPolicyLab

## 1. 安装

```bash
cd XPolicyLab/policy/GalaxeaVLA
bash install.sh
```

`install.sh` 执行 `uv sync` + `uv pip install -e .[dev]`（生成 `GalaxeaVLA/.venv`）。`import XPolicyLab` 通过在脚本里把仓库根加入 `PYTHONPATH` 解决，无需额外安装。

随后**手动**下载（`install.sh` 不会自动下载大文件）：

```bash
export HF_ENDPOINT=https://hf-mirror.com   # 国内镜像，按需

# 1) 系统 ffmpeg（数据 av/mp4 编码用）
sudo apt install -y ffmpeg

# 2) PaliGemma-3B 骨干（g0plus 的 tokenizer + 视觉塔依赖）
hf download google/paligemma-3b-pt-224 \
  --local-dir policy/GalaxeaVLA/weights/paligemma-3b-pt-224
#   g0tiny 改用 HuggingFaceTB/SmolVLM2-500M-Video-Instruct

# 3) G0Plus_3B_base 权重（默认部署权重）
hf download OpenGalaxea/G0-VLA --include "G0Plus_3B_base/*" \
  --local-dir policy/GalaxeaVLA/checkpoints
```

说明：
- 发布的权重文件名是 `model_state_dict.pt`（非 `model.pt`），`model.py` 两种名字都能加载；`dataset_stats.json` 必须与权重位于同一 `checkpoints/` 目录。
- 把 `deploy.yml` 的 `paligemma_path`（或环境变量 `GALAXEA_PALIGEMMA_PATH`）指向第 2 步的骨干目录。

---

## 2. 数据转换

将 XPolicyLab 的 HDF5 轨迹转成 Galaxea LeRobot 格式（用上游写入器，保证与读取器一致）。每帧相机统一为 RGB `(240,320,3)`；state/action 按 `left_arm/left_gripper/right_arm/right_gripper` 写入。

### 2.1 单任务

```bash
cd XPolicyLab/policy/GalaxeaVLA
# dataset_name task_name env_cfg_type expert_data_num action_type
bash process_data.sh RoboDojo pick_place arx_x5 50 joint
```

- 输入：`/mnt/xspark-data/zijian/data/<dataset_name>/<task_name>/<env_cfg_type>/data/episode_*.hdf5`
- 输出：`policy/GalaxeaVLA/data/<dataset_name>-<task_name>-<env_cfg_type>-<expert_data_num>-<action_type>-lerobot/`

### 2.2 批量（多任务合成一个数据集）

把某根目录下所有 `<task>/<env_cfg_type>/data/episode_*.hdf5` 合并为**一个多任务** LeRobot 数据集，每条轨迹的指令（`task`）取其任务目录名。

```bash
cd XPolicyLab/policy/GalaxeaVLA
# dataset_name env_cfg_type action_type batch_root [max_episodes_per_task] [tasks...]
bash process_data_batch.sh RoboDojo_first100 arx_x5 joint \
  /mnt/xspark-data/zijian/final_data/RoboDojo_first100
```

- 输入：`<batch_root>/<task>/<env_cfg_type>/data/episode_*.hdf5`（自动发现所有含 `arx_x5/data` 的任务）。
- 输出：`policy/GalaxeaVLA/data/<dataset_name>-<env_cfg_type>-<action_type>-lerobot/`（单个数据集，`meta/` 下含全部任务的 `tasks.parquet`）。
- `max_episodes_per_task` 缺省 `0` = 每个任务取全部轨迹；可传子集任务名（如 `... build_tower stack_blocks`）。
- 该输出目录即可作为训练数据集：用 `GALAXEA_DATASET_DIR=<该目录>` 传给 `train.sh`，或按默认命名 `data/<dataset_name>-<ckpt_name>-<env_cfg_type>-<expert_data_num>-<action_type>-lerobot` 放置。
- 视频以 h264 编码（本环境 torchcodec 无法解码 AV1）。

---

## 3. 训练（微调）

`train.sh` 遵循 XPolicyLab 标准位置参数格式（与 `process_data.sh` / `eval.sh` 一致）：

```bash
cd XPolicyLab/policy/GalaxeaVLA
# 参数: dataset_name ckpt_name env_cfg_type expert_data_num action_type gpu_id seed [extra hydra...]

# (A) ee（末端位姿）cotrain 微调：复用预转换的 RoboDojo arx-x5 只读数据集
GALAXEA_DATASET_DIR=/mnt/xspark-data/xspark_shared/lerobot/RoboDojo_sim_arx-x5_v21_video \
GALAXEA_PRETRAINED_CKPT=./checkpoints/G0Plus_3B_base/checkpoints \
bash train.sh RoboDojo cotrain arx_x5 100 ee 0,1,2,3 0

# (B) joint 微调：用 process_data.sh 产出的 ./data/<tuple>-lerobot 数据集
bash train.sh RoboDojo robodojo_joint arx_x5 100 joint 0 0
```

- `action_type` 选择任务配置：`ee → real/g0plus_xpolicylab_ee_finetune`，`joint → real/g0plus_xpolicylab_finetune`。
- `ckpt_name` 是 checkpoint 标识；训练/部署按 **6 元组** 定位目录：
  `<dataset_name>-<ckpt_name>-<env_cfg_type>-<expert_data_num>-<action_type>-<seed>`（与 `XPolicyLab/README.md` 一致）。
- `gpu_id` 支持单卡 `0` 或多卡列表 `0,1,2,3`（自动推断 `nproc_per_node`）。
- 数据集目录解析：`GALAXEA_DATASET_DIR` 优先，否则默认 `./data/<dataset_name>-<ckpt_name>-<env_cfg_type>-<expert_data_num>-<action_type>-lerobot`。
- 预训练权重：`GALAXEA_PRETRAINED_CKPT`（默认 `./checkpoints/G0Plus_3B_base/checkpoints`）。
- 输出/缓存目录可用 `GALAXEA_FM_OUTPUT_DIR`、`GALAXEA_FM_DATASET_STATS_CACHE_DIR`、`HF_DATASETS_CACHE` 覆盖（均有默认值）。
- **语言占位符门禁**：若数据集 `meta/tasks.jsonl` 多个 `task_index` 却只有 1 条唯一指令（语言塌缩），`train.sh` 会报错退出；纯视觉运动训练可设 `ALLOW_PLACEHOLDER_LANG=true` 绕过。
- 默认日志关闭（`GALAXEA_LOGGER_MODE=disabled`，避免强制 swanlab 登录）；额外 hydra 覆盖可作为第 8+ 个参数透传。
- 产物写入 `checkpoints/<6-tuple>/<timestamp>/`，其中 `checkpoints/step_*` 由 `model.py` 自动选最新 step 部署。

---

## 4. 部署与评测

`deploy.yml` 的 `eval_env` 决定客户端走 `debug` / `sim` / `real`，切换无需改 `eval.sh`。先用 `debug` 验证维度/通路，再切 `sim` / `real`。

```bash
cd XPolicyLab/policy/GalaxeaVLA
# dataset_name task_name ckpt_name env_cfg_type expert_data_num action_type \
#   seed policy_gpu_id env_gpu_id policy_uv_env_path eval_env_conda_env
bash eval.sh RoboDojo test_data cotrain arx_x5 100 ee \
  0 6 0 ./GalaxeaVLA dp
```

参数要点：
- **6 元组反查权重**：`setup_eval_policy_server.sh` 在 `checkpoints/` 下拼接
  `<dataset_name>-<ckpt_name>-<env_cfg_type>-<expert_data_num>-<action_type>-<seed>`；
  例：`RoboDojo-cotrain-arx_x5-100-ee-0`。目录内若有 `<timestamp>/` 子目录则取最新 run；
  内部 `checkpoints/step_*` 由 `model.py` 选最新 step。预训练基座仍用 `G0Plus_3B_base/checkpoints`。
- 第 10 个 `policy_uv_env_path`：GalaxeaVLA 的 uv 工程目录（含 `.venv`），即 `./GalaxeaVLA`；传 `null` 时脚本默认用 `./GalaxeaVLA`。
- 第 11 个 `eval_env_conda_env`：环境客户端激活的 conda 环境（通常 `XPolicyLab`）。

切换到仿真/真机：把 `deploy.yml` 的 `eval_env` 改成 `sim` 或 `real` 即可，命令不变。

跨机部署：把 `setup_eval_policy_server.sh` 放在 GPU 机后台运行，仿真机用同一 `policy_server_ip:policy_server_port` 连接。

---

## 5. 关键设计

- **配置驱动 `model.py`**：用 Hydra `compose` 加载上游配置（`task=<task_config_name>`），`instantiate(cfg.model.model_arch)` 建模、`instantiate(cfg.data.processor)` 建预处理器，复刻 `eval_libero` 的 `preprocess → predict_action → postprocess` 推理链。
- **obs/action 映射**：`pack_robot_state` 得到 XPolicyLab 规范的 14 维向量，按 `shape_meta` 顺序切分为 Galaxea 的 `left_arm/left_gripper/right_arm/right_gripper`；动作侧逆向用 `unpack_robot_state` 还原为逐时间步动作字典。打包顺序 `[左臂, 左夹爪, 右臂, 右夹爪]` 两侧一致。
- **图像**：相机统一 `(240,320,3)` RGB（上游不做 letterbox，processor 内 `Resize` 硬拉伸到 `224x224`，与上游训练一致）。
- **权重加载**：兼容 `model.pt` / `model_state_dict.pt`，`strict=False` 加载并打印缺失/多余键；`dataset_stats.json` 用上游 `load_dataset_stats_from_json` 解析后注入 normalizer。

---

## 6. 注意事项 / 阻塞项

1. 维度对齐但本体不同，**必须微调**（见顶部说明）。
2. 权重 / 骨干 / uv 环境 / GPU 是运行前提；`install.sh` 与下载步骤已写好但不自动执行。
3. 数据转换器为全新代码，需在真实 HDF5 + ffmpeg + uv 环境中跑通后再用于训练。
4. base ckpt 自带 `config.yaml` 引用内部代码（`GalaxeaZeroProcessor` + bbox 4 相机），部署时忽略，改用公开 `BaseProcessor` + 3 相机（与官方微调路径一致），不影响权重加载。
