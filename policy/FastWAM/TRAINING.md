# FastWAM 训练文档（XPolicyLab 适配版）

> 上游项目：[Fast-WAM: Do World Action Models Need Test-time Future Imagination?](https://arxiv.org/abs/2603.16666)
> GitHub: https://github.com/yuantianyuan01/FastWAM
> 上游训练入口：`FastWAM/scripts/train.py` + `FastWAM/scripts/train_zero1.sh`
> XPolicyLab 入口：`XPolicyLab/policy/FastWAM/train.sh`

本文聚焦“**使用 XPolicyLab 项目数据（HDF5 episodes）训练 FastWAM**”所需的全部步骤、配置含义、踩坑点。
完成本文 Step 1–Step 5 后，单卡或多卡均可直接训练。

---

## 0. 上游与本仓库的关系

```
XPolicyLab/policy/FastWAM/
├── eval.sh / setup_eval_policy_server.sh / setup_eval_env_client.sh
├── train.sh                              # 本仓库的训练入口（封装上游 train_zero1.sh）
├── process_data.sh / process_data_batch.sh
├── deploy.yml / model.py / __init__.py   # XPolicyLab 推理适配
├── INSTALLATION.md / TRAINING.md
├── data/<dataset_id>/                    # process_data.* 产物（LeRobot v2.1 + dataset_stats）
│   ├── lerobot/{data,videos,meta}/
│   └── dataset_stats.json
├── checkpoints/                          # 训练输出根目录
│   ├── ActionDiT_linear_interp_Wan22_alphascale_1024hdim.pt  # Step 2 产物
│   └── <ckpt_setting>/                   # 每次训练运行的 Hydra `output_dir`
└── FastWAM/                              # 上游源码（保持纯净，不要直接改）
    ├── scripts/{train.py,train_zero1.sh,preprocess_action_dit_backbone.py,precompute_text_embeds.py}
    ├── configs/{data,model,task}/        # 上游 Hydra 配置
    ├── src/fastwam/                      # 上游核心代码
    ├── data/text_embeds_cache/xpolicylab/<dataset_id>/  # T5 cache，由 process_data.sh 自动写入
    └── checkpoints/                      # 上游下载的 Wan2.2 等基础模型
```

为什么 `train.sh` 强制走上游 `scripts/train_zero1.sh` 而不是直接调 `train.py`？
1. `train_zero1.sh` 已经包好 `accelerate launch + DeepSpeed ZeRO-1` 配置；
2. 多机训练时 `RUN_ID` 同步逻辑全在那个脚本里；
3. 我们只在外层补 XPolicyLab 需要的 Hydra overrides，不污染上游配置。

---

## 1. FastWAM 用 XPolicyLab 数据需要改什么？（对照原项目）

FastWAM 原本针对的是 RoboTwin / LIBERO 的官方数据集。它们与 XPolicyLab `RoboDojo/arx_x5` 在以下维度有差异，
**`process_data.sh` + `train.sh` 已经把这些差异封装好**，下面列出我们做了什么、以及上游 yaml 默认值在 XPolicyLab 数据上的兼容情况。

| 维度 | 上游 RoboTwin 默认 | XPolicyLab arx_x5 现状 | 需要改吗？ |
|---|---|---|---|
| 数据格式 | LeRobot v2.1（HF 已发布） | HDF5（`data/<ds>/<task>/<env>/data/episode_*.hdf5`） | `process_data.sh` 在转换时已对齐 |
| 相机键 | `cam_high / cam_left_wrist / cam_right_wrist` | `cam_head / cam_left_wrist / cam_right_wrist` | `process_data.py: CAMERA_MAP` 已映射 |
| 视频源分辨率 | 480×640 → resize 到 384×320 | XPolicyLab 渲染分辨率不固定 → 统一 240×320 RGB | `process_data.py: _decode_rgb` 已强制 |
| state/action 维度 | 14（dual-arm 6+1） | joint：14（6+1 双臂），ee：16（7+1 双臂） | `get_action_dim.sh` 自动算 + Hydra override |
| FPS | 由 `meta/info.json` 决定 | 取自 `additional_info.frequency`，回退到 `--fps 10` | `process_data.py` 已读 HDF5 |
| 指令 | 每个 task 一句固定 prompt | 每条 episode 可能不同 | `process_data.py` 按 episode 解析 + 去重写入 `tasks.jsonl` |
| T5 prompt 缓存 | 路径 `./data/text_embeds_cache/robotwin` | 路径 `data/text_embeds_cache/xpolicylab/<dataset_id>` | `process_data.sh` 用 `precompute_text_embeds.py` 自动算 |
| Normalization 统计 | 训练首跑生成 `dataset_stats.json` | 转换时直接预计算 | `process_data.py` 写 `data/<dataset_id>/dataset_stats.json` |
| 归一化 mode | `z-score`（global mean/std） | 默认沿用上游 | 见下文 §6 决策 |
| 模式开关 | `mot_checkpoint_mixed_attn: false` 等 task yaml | 直接复用 `robotwin_uncond_3cam_384_1e-4` | 不动 |
| `action_state_transforms` | `null`（RoboTwin 不做任何额外变换） | 同上 | 不动 |
| `concat_multi_camera` | `"robotwin"`（head + left/right 拼成 384×320） | 同上 | 不动 |
| `num_frames` / `action_video_freq_ratio` | 33 / 4（=32 action + 9 video frame） | 同上 | 不动 |

**结论**：使用 XPolicyLab arx_x5 数据训练 FastWAM 时，**用户层面不需要直接编辑任何上游 yaml**，
所有差异通过 `process_data.sh` 转换产物 + `train.sh` 的 Hydra override 套接好。

---

## 2. Step 1：环境

参见 `INSTALLATION.md`。简单流程：

```bash
cd XPolicyLab/policy/FastWAM
bash install.sh                       # 创建并装填 conda env `fastwam`
conda activate fastwam
```

---

## 3. Step 2：准备 Wan2.2 基础模型 + ActionDiT backbone

这是**只跑一次**的预处理：把 Wan2.2 的视频 DiT 权重按 FastWAM 的形状插值好，保存到本地。

```bash
conda activate fastwam
cd XPolicyLab/policy/FastWAM/FastWAM
mkdir -p checkpoints
export DIFFSYNTH_MODEL_BASE_PATH="$(pwd)/checkpoints"

python scripts/preprocess_action_dit_backbone.py \
  --model-config configs/model/fastwam.yaml \
  --output checkpoints/ActionDiT_linear_interp_Wan22_alphascale_1024hdim.pt \
  --device cuda \
  --dtype bfloat16
```

产物：`XPolicyLab/policy/FastWAM/FastWAM/checkpoints/ActionDiT_linear_interp_Wan22_alphascale_1024hdim.pt`
`train.sh` 启动时会显式检查这个文件，不存在直接报错并打印重跑命令。

> `DIFFSYNTH_MODEL_BASE_PATH` 在我们的 `train.sh` / `setup_eval_policy_server.sh` 里**也会再次 export**，所以
> 训练 / 推理时 Wan2.2 主权重会从 `<policy>/FastWAM/checkpoints/` 下取。

---

## 4. Step 3：数据转换（HDF5 → LeRobot v2.1 + dataset_stats + T5 cache）

### 4.1 单任务

```bash
conda activate fastwam
bash XPolicyLab/policy/FastWAM/process_data.sh \
    RoboDojo test_data arx_x5 3 joint
```

参数顺序与 XPolicyLab 全家桶保持一致：
`dataset_name task_name env_cfg_type expert_data_num action_type`。
单任务默认 `dataset_id = <dataset>-<task>-<env_cfg>-<num>-<action_type>`。

### 4.2 多任务 cotrain（一次合并多个 task 到同一个 LeRobot 数据集）

```bash
# 方式 A：手动传任务列表（逗号分隔），expert_data_num 是“每个 task 各取多少 episode”
bash XPolicyLab/policy/FastWAM/process_data.sh \
    RoboDojo "stack_bowls,press_by_number" arx_x5 3 joint cotrain_demo

# 方式 B：自动发现 data/<dataset>/*/<env>/data/episode_*.hdf5 下的所有 task
bash XPolicyLab/policy/FastWAM/process_data_batch.sh \
    RoboDojo arx_x5 3 joint cotrain_demo
```

可选第 6 个参数 `dataset_id` 控制输出文件夹名（默认 `cotrain_dataset`）。

### 4.3 产物

```
XPolicyLab/policy/FastWAM/data/<dataset_id>/
├── dataset_stats.json                     # FastWAM Normalizer 直接消费
└── lerobot/
    ├── meta/{info.json, tasks.jsonl, episodes.jsonl, episodes_stats.jsonl}
    ├── data/chunk-000/episode_000000.parquet ...
    └── videos/chunk-000/observation.images.{cam_high,cam_left_wrist,cam_right_wrist}/episode_000000.mp4 ...

XPolicyLab/policy/FastWAM/FastWAM/data/text_embeds_cache/xpolicylab/<dataset_id>/
└── <sha256>.t5_len128.wan22ti2v5b.pt ...   # 由 precompute_text_embeds.py 写入
```

T5 缓存键是“**Wan-AI/Wan2.2-TI2V-5B + DEFAULT_PROMPT + context_len=128**”的 sha256，
所以同一 `dataset_id` 不同 seed/batch_size 都可以复用同一份缓存；改任务（改 instruction）才需要重算。

### 4.4 关键 schema 决策（与上游 process_data.py 的差异）

上游 `process_data.py` 在 `tasks.jsonl` 里硬塞 4 条固定槽位：
`[0]=task_name, [1]=instruction, [2]="xpolicylab_quality", [3]="success"`，并把每帧
`task_index=1, coarse_task_index=0, coarse_quality_index=2, quality_index=3`。
这种设计在“**所有 episode 共用一条 instruction**”的单任务情境里能跑，但对 cotrain 多 instruction 直接出错
（`coarse_task_index=0` 指向第一条 instruction，而不是 task 名）。

本仓库的 `process_data.py` 简化为只保留 `task_index` 一个语义：
- `tasks.jsonl` 只包含**去重后的真实 instruction 列表**；
- 每帧 parquet 只写 `task_index`；
- 上游 `LeRobotDataset.__getitem__` 只在 `coarse_task_index` 出现时才会读它，去掉后行为完全等价
  （`augment_instruction` 在 RoboTwin 默认 `drop_high_level_prob=1.0` 下根本不用 `coarse_task`）。

这样 single-task 和 cotrain 走的是同一条代码路径，**多任务训练时每条 episode 用自己的 instruction 编码 T5**。

---

## 5. Step 4：训练

### 5.1 入口

```bash
conda activate fastwam
bash XPolicyLab/policy/FastWAM/train.sh \
    <dataset_name> <task_name> <env_cfg_type> <expert_data_num> <action_type> <seed> <gpu_id> [num_gpus]
```

例：

```bash
# 单卡，单任务
bash XPolicyLab/policy/FastWAM/train.sh \
    RoboDojo test_data arx_x5 3 joint 0 0
```

```bash
# 多卡，单任务（8 卡）
bash XPolicyLab/policy/FastWAM/train.sh \
    RoboDojo test_data arx_x5 3 joint 0 0,1,2,3,4,5,6,7
```

```bash
# 多卡 + cotrain（前提是已 process_data_batch 把多个 task 合到了 cotrain_dataset）
FASTWAM_DATASET_ID=cotrain_dataset \
bash XPolicyLab/policy/FastWAM/train.sh \
    RoboDojo any_placeholder arx_x5 3 joint 0 0,1,2,3,4,5,6,7
```

### 5.2 `train.sh` 内部做了什么

1. `get_action_dim.sh` 算出本次实际 `action_dim`（joint→14 / ee→16）；
2. 根据 `FASTWAM_DATASET_ID`（或默认 data_key）解析出已转换好的 lerobot 目录与 dataset_stats；
3. 校验 ActionDiT backbone 和 T5 cache **必须存在**（缺哪个就提示对应命令），不存在直接退出；
4. cd 到 `FastWAM/`，调上游 `scripts/train_zero1.sh` 并传入下列 Hydra overrides：

```
task=robotwin_uncond_3cam_384_1e-4
seed=${train_seed}
batch_size=${batch_size}                         # 默认 64，可 FASTWAM_BATCH_SIZE 覆盖
gradient_accumulation_steps=${gradient_accumulation_steps}  # 默认 1
num_workers=${num_workers}                       # 默认 8

# 数据集与 normalization 统计指向我们转换好的 LeRobot 数据
data.train.dataset_dirs=[<policy>/data/<dataset_id>/lerobot]
data.val.dataset_dirs=[<policy>/data/<dataset_id>/lerobot]
data.train.text_embedding_cache_dir=<policy>/FastWAM/data/text_embeds_cache/xpolicylab/<dataset_id>
data.val.text_embedding_cache_dir=...（同上）
data.train.pretrained_norm_stats=<policy>/data/<dataset_id>/dataset_stats.json
data.val.pretrained_norm_stats=...（同上）

# action/state 维度对齐
data.train.shape_meta.action.0.raw_shape=${action_dim}
data.train.shape_meta.action.0.shape=${action_dim}
data.train.shape_meta.state.0.raw_shape=${action_dim}
data.train.shape_meta.state.0.shape=${action_dim}
data.val.shape_meta.<...>=${action_dim}
data.train.processor.action_output_dim=${action_dim}
data.train.processor.proprio_output_dim=${action_dim}
data.val.processor.<...>=${action_dim}

# 训练输出根目录
output_dir=<policy>/checkpoints/<ckpt_setting>
```

### 5.3 可调环境变量

| 环境变量 | 默认 | 含义 |
|---|---|---|
| `FASTWAM_DATASET_ID` | `<dataset>-<task>-<env>-<num>-<action>`（单任务）；cotrain 需手动指定 | 选取已转换数据集，便于复用 cotrain 产物 |
| `FASTWAM_CKPT_SETTING` | `<dataset_id>-<seed>` | 训练输出子目录名（写到 `checkpoints/<ckpt_setting>/`） |
| `FASTWAM_BATCH_SIZE` | `64` | 单卡 batch size。OOM 就调小（如 16/8/4） |
| `FASTWAM_GRADIENT_ACCUMULATION_STEPS` | `1` | 等效 batch = batch × accumulation × world_size |
| `FASTWAM_NUM_WORKERS` | `8` | DataLoader workers |
| `PYTORCH_CUDA_ALLOC_CONF` | `expandable_segments:True` | 减少显存碎片；可覆盖 |

可调 Hydra overrides（追加到 `train.sh` 末尾即可，因为我们没有锁死 `train_zero1.sh` 的 `EXTRA_ARGS`，要透传需手动改 `train.sh`；
更常见的做法是直接 `export FASTWAM_BATCH_SIZE=...` 等环境变量，或临时复制一份 `train.sh` 修改 `train_common=()`）。

如果要改 epoch / save_every / eval_every 等，请直接编辑：
`XPolicyLab/policy/FastWAM/FastWAM/configs/task/robotwin_uncond_3cam_384_1e-4.yaml`
（5 个 epoch 是 RoboTwin 完整数据集的设定，对我们 N=3 episodes 的小样本远远不够，**实际训练时务必显著上调** `num_epochs`，
或者改成 `max_steps`，详见 §6）。

### 5.4 训练输出

```
XPolicyLab/policy/FastWAM/checkpoints/<ckpt_setting>/
├── checkpoints/                           # 上游 Trainer 写的所有 step 子目录
│   ├── step_<N>/
│   │   ├── weights/                       # safetensors 分片
│   │   ├── optimizer/
│   │   └── scheduler/ ...
│   └── weights/                           # 软链 / 副本最新 step（视 ZeRO-1 配置）
├── dataset_stats.json                     # 训练首次启动时由 BaseLerobotDataset 二次计算并落盘（与 process_data 写的同名）
└── train_log.txt / accelerate logs ...
```

eval 端的 `setup_eval_policy_server.sh` 会从 `checkpoints/<ckpt_setting>/checkpoints/weights/step_*.pt` 取最大编号那份；
如果 ZeRO-1 写法是分片 safetensors（而不是单文件 `step_*.pt`），请检查上游 `save_every` / Trainer 输出格式
并在 `setup_eval_policy_server.sh` 里把 glob 改成相应的 pattern（当前默认 `step_*.pt`，必要时可设
`FASTWAM_CHECKPOINT_PATH` 直接指定）。

---

## 6. 关键技术细节 / 容易踩坑的点

### 6.1 action/state 归一化模式

`configs/data/robotwin.yaml` 默认：
```yaml
processor:
  use_stepwise_action_norm: False
  norm_default_mode: "z-score"
  action_state_transforms: null
```

- `z-score` 用 `global_mean/global_std`，输出近似 N(0,1) 后 clamp 到 `[-5, 5]`。
- 训练与推理用的是**同一份** `dataset_stats.json`（process_data 写一次，train/eval 都从那里读），所以
  分布不会因为 dataset 重算导致漂移。
- 我们没有改这个 default。如果观察到 gripper（双峰）维度被 z-score 拍扁导致动作不响应，可在 `train.sh`
  里临时追加 `data.train.processor.norm_default_mode=q01/q99 data.val.processor.norm_default_mode=q01/q99`。

### 6.2 `task_name` 与 `ckpt_name` 的区别

- 训练时 `<task_name>` 只参与拼 dataset 文件夹名 + 找 episode 目录。
- 推理时 `eval.sh` 的第 2/3 个参数分别是 `task_name`（仿真任务，决定环境）与 `ckpt_name`（检查点名，决定取哪个权重），
  这样可以用同一个 cotrain checkpoint 评估不同的下游任务。

### 6.3 action_dim 在 eval 与 train 之间的隐患

- `train.sh` 把 `action_output_dim / proprio_output_dim` 都改成了真实 `action_dim`，所以训练的模型形状是对的。
- `deploy_policy.get_model()`（上游写法）**只把 `sim_task` 当作 Hydra override**，没有把 `action_dim` 透进来；
  这意味着 eval 时 processor 的 `action_output_dim/proprio_output_dim` 始终用 yaml 默认值 `14`。
- 因此 **joint 模式（14 维）开箱即用，ee 模式（16 维）目前与 eval 路径不兼容**。
  如果要做 ee 训练 + 评估，需要扩展 `setup_eval_policy_server.sh` 经由 `deploy.yml`
  → `deploy_policy.get_model` → `_compose_sim_cfg(overrides=[...])` 传 `data.train.processor.action_output_dim=16`
  等。这是一个待解决的 TODO，详见末尾“开放问题”。

### 6.4 epoch / steps

- 上游 RoboTwin 用 `num_epochs=5` 但数据集是几十万 episode；
- XPolicyLab arx_x5 小样本（3–几十 episode）建议直接走 `max_steps`：
  在 `configs/task/robotwin_uncond_3cam_384_1e-4.yaml` 把 `num_epochs: 5` 改成 `num_epochs: null` 并设
  `max_steps: 20000` 之类；
- `save_every` 当前 2500、`eval_every` 500，对小数据集偏大，可改 `save_every: 500 eval_every: 100`。

### 6.5 GPU 资源

**Wan2.2-TI2V-5B 单卡训不起来——不是 batch_size 问题，是优化器状态问题。**

AdamW fp32 优化器状态（master + m + v）= 5B × 12 byte ≈ **60 GB**；ZeRO-1 只在多 rank 间分片优化器状态，
单进程 = 单分片 = 60 GB 完整保留。加上 bf16 权重 10 GB、bf16 梯度 10 GB、33 帧×3 路×384×320 视频激活 ~15 GB，
合计 **~95 GB**，单张 A800 80G 必爆 OOM（典型崩在 `deepspeed/runtime/zero/stage_1_and_2.py:step()`）。

把 `FASTWAM_BATCH_SIZE` 调到 1 也救不了：激活只是 ~15 GB 那一项的一个零头，瓶颈在固定的 60 GB 优化器状态。

可行配置（按推荐顺序）：

| 卡数 | 启动命令第 7 个参数 | 每卡优化器状态 | 备注 |
|---|---|---|---|
| 8 卡 80G | `0,1,2,3,4,5,6,7` | 60 / 8 ≈ 7.5 GB | **推荐**，对齐上游 LIBERO 配置 |
| 4 卡 80G | `0,1,2,3` | 15 GB | 紧但能跑，`batch_size=2` 起步 |
| 2 卡 80G | `0,1` | 30 GB | 几乎不行，要切 ZeRO-2 + offload |
| 1 卡 80G | `0` | 60 GB | **不可行**，需改 DS 配置加 CPU offload |

`train.sh` 第 7 个位置参数支持逗号分隔的 GPU 列表，会自动把卡数传给 `train_zero1.sh` → `accelerate launch --num_processes N`。
全局 batch = `batch_size × gradient_accumulation_steps × num_gpus`，
8 卡 + `FASTWAM_BATCH_SIZE=4` 全局就是 32，已经接近 RoboTwin 上游 batch=64 的一半。

**单卡兜底**（确实只有 1 卡可用时）需要改 `FastWAM/scripts/ds_configs/ds_zero1_config.json`：

```json
"zero_optimization": {
    "stage": 2,
    "offload_optimizer": { "device": "cpu", "pin_memory": true },
    ...
}
```

把优化器状态甩到 CPU RAM（需要 ≥ 80 GB 空 RAM），训练会慢 2–4×，但能塞下。**不推荐**作为常态。

### 6.6 显存 / 性能开关

- `model.mot_checkpoint_mixed_attn: true`（来自 `fastwam.yaml`）配 `use_gradient_checkpointing=true`，能省一半 attn 激活；
  上游 RoboTwin task yaml 反而把它设成 `false` 以提速，但需要更多显存。我们暂时沿用 task yaml 的 `false`。
- 加 `bf16` 已经是默认（`mixed_precision: bf16`）。

### 6.7 多任务训练时 T5 prompt 编码

- `process_data.sh` 转换完后会**主动调** `precompute_text_embeds.py`，缓存键基于
  `DEFAULT_PROMPT.format(task=<unique instruction>)` 的 sha256。
- cotrain 多任务时，`tasks.jsonl` 会有多条 instruction，每条都会被独立编码并缓存。
- 训练 / 推理时 `RobotVideoDataset` 按 `task_index → tasks.jsonl[idx]` 取 instruction，再去 cache 取嵌入；
  cache miss 直接 KeyError，请确保 `text_embedding_cache_dir` 与 `dataset_dirs` 一一对应（`train.sh` 已自动对齐）。

### 6.8 视频解码

- LeRobot 内部用 `torchcodec==0.5`，它依赖 conda env 里的 ffmpeg（`libavutil.so.58`），
  我们的 `install.sh` 已经处理；
- DataLoader `num_workers>0` 时上游有 deque-based 视频读取，注意 `_query_videos` 文档里写过：
  **不要再在主进程开第二个 num_workers=0 的 DataLoader**，否则 video reader 句柄共享会段错误。

---

## 7. Step 5：评估

```bash
conda activate fastwam
bash XPolicyLab/policy/FastWAM/eval.sh \
    <dataset_name> <task_name> <ckpt_name> <env_cfg_type> <expert_data_num> \
    <action_type> <seed> <policy_gpu_id> <env_gpu_id> \
    <policy_conda_env> <eval_env_conda_env>
```

例（与 LDA_1B 同款 11 参数）：

```bash
bash XPolicyLab/policy/FastWAM/eval.sh \
    RoboDojo test_data test_data arx_x5 3 \
    joint 0 0 0 \
    fastwam XPolicyLab
```

- `task_name` 与 `ckpt_name` 可不同：用 `cotrain` 的 checkpoint 跑各子任务时只需把 `ckpt_name=cotrain_dataset`。
- `eval.sh` 自己分配空闲端口、起 server 进程、等就绪、再起 client；server 异常退出会立即报错。
- 详见 `INSTALLATION.md` 的 Step 6 与 `model.py`。

---

## 8. 已知开放问题 / 需用户确认

1. **EE 模式（16 维）评估侧不可用**：见 §6.3。要不要补这个？目前只支持 joint 模式 14 维全流程。
2. **`num_epochs / max_steps` 默认值**：上游 RoboTwin 是 `num_epochs=5`，对我们的小样本数据集严重欠拟合，是否需要在 `train.sh` 提供一个 XPolicyLab 默认覆盖（例如 `num_epochs=null max_steps=10000`）？
3. **`norm_default_mode`**：要不要在 train.sh 把默认改成 `q01/q99`？（diffusion policy 实践更常见）
4. **单卡默认 batch size**：当前 `FASTWAM_BATCH_SIZE=64` 是“多卡假设”，单卡跑会 OOM。要不要把默认下调到 4（accumulation=16），多卡再手动覆盖回去？
5. **`process_data.sh` 自动跑 `precompute_text_embeds.py`**：当前只用 1 卡（脚本里没 torchrun），10 条以下 instruction 不到 1 分钟。如果未来上千 instruction，可能要改成 torchrun。
6. **`save_every / eval_every` 与 `num_epochs`**：是否需要按 `expert_data_num` 自动缩放？例如 `save_every = max(100, total_steps // 20)`。

把答案告诉我，我会把 (2)(3)(4) 直接落到 `train.sh` / task yaml 的 XPolicyLab 默认里；(1)(5) 是更大的工作量，
需要你决策优先级。
