# RISE on XPolicyLab

[RISE](https://opendrivelab.com/rise/)（*Self-Improving Robot Policy with Compositional World Model*）在 XPolicyLab 中的集成，**仅包装离线流程**：HDF5/LeRobot 数据准备、value 打标、advantage-conditioned policy 训练，以及经 XPolicyLab policy server 的仿真/真机评测。

产物命名遵循 [XPolicyLab README §4.2](../../README.md)：

| 产物 | 命名 | 默认路径 |
|---|---|---|
| 处理后数据集 | `<bench_name>-<ckpt_name>-<env_cfg_type>-<action_type>-lerobot` | `policy/RISE/data/` |
| 训练 checkpoint | `<bench_name>-<ckpt_name>-<env_cfg_type>-<action_type>-<seed>` | `policy/RISE/checkpoints/` |

路径解析辅助：`xpolicylab_adapter/_artifact_paths.sh`（精确 4 元组 / 4+seed 匹配）。

上游完整三件套（dynamics model、online RL、Piper 真机部署）仍保留在 vendored 源码树中供查阅，但 **XPolicyLab 不提供对应入口脚本**；详见 [§6 与上游 RISE 的关系](#6-与上游-rise-的关系)。

---

环境安装与 Pi0.5 权重准备见 [`INSTALLATION.md`](INSTALLATION.md)。

## 1. 数据准备

RISE 离线训练使用 **LeRobot v2.1**，分辨率 `(240, 320, 3)` RGB，state/action 为 **14 维 joint**。

**注意**：当前 XPolicyLab 推理仅支持 `action_type=joint`。

### 1.1 从 XPolicyLab HDF5 转换

原始 HDF5 位于工作区 `data/<bench_name>/<ckpt_name>/<env_cfg_type>/`（单任务时 `ckpt_name` 通常等于 raw task 目录名）。

```bash
cd policy/RISE
bash process_data.sh RoboDojo test_data arx_x5 joint
# 可选尾参 expert_data_num 限制 episode 数
bash process_data.sh RoboDojo test_data arx_x5 joint 100
```

- **输入**：`data/<bench_name>/<ckpt_name>/<env_cfg_type>/data/episode_*.hdf5`
- **输出**：`policy/RISE/data/<bench_name>-<ckpt_name>-<env_cfg_type>-joint-lerobot/`
- 脚本末尾自动计算 `norm_stats.json`

### 1.2 链接已有 LeRobot 数据集

```bash
cd policy/RISE
bash process_lerobot.sh <path/to/lerobot_dataset> [link_name]
```

推荐 `link_name` 使用标准 tag，例如 `RoboDojo-cotrain-arx_x5-joint-lerobot`。也可用 `RISE_RAW_DATASET` 显式指定 raw 数据集路径。

### 1.3 ckpt_name vs task_name

| 参数 | 用途 |
|---|---|
| `ckpt_name` | 命名实验、定位 processed 数据与 checkpoint；单任务时常等于 raw 数据目录名 |
| `task_name` | **仅评测**：仿真/真机 env client 执行的任务名；可与 `ckpt_name` 不同（如 `ckpt_name=cotrain` 评测 `task_name=stack_bowls`） |

`process_data.sh` / `train.sh` 使用 `ckpt_name`；`eval.sh` 同时需要 `task_name` 与 `ckpt_name`。

---

## 2. 训练（离线 policy）

```bash
# bench_name ckpt_name env_cfg_type action_type seed gpu_id [stage]
bash train.sh RoboDojo stack_bowls arx_x5 joint 0 0 all
```

| Stage | 作用 |
|-------|------|
| `advantage` | norm → value 训练 → 打 advantage 标签，生成 `*_w_adv` |
| `policy` | 在已有 `*_w_adv` 上训练 `Policy_offline_release` |
| `all` | 依次执行 `advantage` → `policy` |

| 资源 | 默认路径 |
|------|----------|
| Pi0.5 预训练 | `weights/pi05_base_pytorch/` |
| Raw 数据集 | `data/<bench>-<ckpt_name>-<env>-<action>-lerobot`（或 `RISE_RAW_DATASET`） |
| Advantage 数据集 | `<raw>_w_adv` |
| 训练 checkpoint | `checkpoints/<bench>-<ckpt_name>-<env>-<action>-<seed>/` |

---

## 3. 部署与评测

Use the `EVAL_ENV_TYPE` environment variable: `debug` / `sim` / `real`；`eval_batch` 控制 batch 推理。切换模式无需改 `eval.sh`。

```bash
bash eval.sh RoboDojo stack_bowls test_data arx_x5 joint 0 0 0 RISE XPolicyLab
```

### 3.1 Checkpoint 解析

`setup_eval_policy_server.sh` 按优先级查找：

1. `RISE_CHECKPOINT_PATH`
2. `checkpoints/<bench>-<ckpt_name>-<env>-<action>-<seed>/`（精确 `ckpt_run_id`）
3. `.../Policy_offline_release/Policy_offline_release/<step>/`（最新 step，或 `RISE_CHECKPOINT_STEP`）

### 3.2 跨机部署

GPU 机运行 `setup_eval_policy_server.sh`，仿真/真机侧 `setup_eval_env_client.sh` 使用同一 `policy_server_ip:policy_server_port`。

---

## 4. 关键设计

### 4.1 `model.py` 推理链

- 相机：`cam_head` → `top_head`，`cam_left_wrist` → `hand_left`，`cam_right_wrist` → `hand_right`
- 图像：resize 到 `(240, 320, 3)` RGB（与训练一致；上游内部再 `resize_with_pad` 到 224×224）
- state：`pack_robot_state` → 14 维；action：`unpack_robot_state`

### 4.2 目录结构

```text
policy/RISE/
├── xpolicylab_adapter/_artifact_paths.sh
├── train.sh, process_data.sh, model.py, eval.sh, ...
├── data/, weights/, checkpoints/
└── RISE/                         # vendored 上游
    ├── process_data.py
    └── deploy/sitecustomize.py   # eval client socket 超时补丁
```

---

## 5. 注意事项

1. **RGB 通道**：`decode_image_bit` 与 live obs 已是 RGB，**禁止**在转换/推理中再加 `cv2.cvtColor(BGR2RGB)`。若曾用错误转换训练，需 **重新跑 `process_data.sh` 并重新训练**。
2. **Advantage 数据集**：直接跑 `policy` stage 前需已有 `<raw>_w_adv`。
3. **action_type**：仅 `joint`。
4. **数据量 ablation**：使用不同 `ckpt_name`（如 `myrun_50ep`）并在 `process_data.sh` 传可选 `expert_data_num`。

---

## 6. 与上游 RISE 的关系

对照 [OpenDriveLab/RISE](https://github.com/OpenDriveLab/RISE)。vendored 树位于 `policy/RISE/RISE/`。

| 位置 | 说明 |
|------|------|
| `policy/RISE/*.sh`、`model.py` 等 | XPolicyLab 包装层 |
| `RISE/process_data.py`、`RISE/deploy/sitecustomize.py` | 相对官方 **新增**，升级上游后需保留 |
| `RISE/policy_and_value/…` | 官方文件，升级时替换对应路径 |

建议工作流：

1. 按 [`INSTALLATION.md`](INSTALLATION.md) 安装环境与 Pi0.5 权重
2. `bash process_data.sh ...` 或 `bash process_lerobot.sh ...`
3. `bash train.sh ... all`
4. `bash eval.sh ...`（`EVAL_ENV_TYPE`: unset/`sim` for simulation, `debug` for offline validation）

---

## 7. 引用

```bibtex
@article{rise2026,
  title={RISE: Self-Improving Robot Policy with Compositional World Model},
  author={Yang, Jiazhi and Lin, Kunyang and Li, Jinwei and Zhang, Wencong and Lin, Tianwei and Wu, Longyan and Su, Zhizhong and Zhao, Hao and Zhang, Ya-Qin and Chen, Li and Luo, Ping and Yue, Xiangyu and Li, Hongyang},
  journal={arXiv preprint arXiv:2602.11075},
  year={2026}
}
```
