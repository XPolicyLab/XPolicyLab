# GalaxeaVLA on XPolicyLab

仅支持 **joint** 控制（14 维双臂关节 + 夹爪）。产物命名遵循 [XPolicyLab README §4.2](../../README.md)：

| 产物 | 命名 | 默认路径 |
|---|---|---|
| 处理后数据集 | `<bench_name>-<ckpt_name>-<env_cfg_type>-joint-lerobot` | `policy/GalaxeaVLA/data/` |
| 训练 checkpoint | `<bench_name>-<ckpt_name>-<env_cfg_type>-joint-<seed>` | `policy/GalaxeaVLA/checkpoints/` |

---

## 1. 安装

```bash
bash install.sh
```

```bash
export HF_ENDPOINT=https://hf-mirror.com   # 国内镜像，按需

sudo apt install -y ffmpeg

hf download google/paligemma-3b-pt-224 \
  --local-dir ./weights/paligemma-3b-pt-224

hf download OpenGalaxea/G0-VLA --include "G0Plus_3B_base/*" \
  --local-dir ./checkpoints
```

---

## 2. 数据转换

HDF5 → Galaxea LeRobot（`dual_arm_joint_robodojo` 格式）；相机 RGB `(480,640,3)`，key 为 `cam_high` / `cam_left_wrist` / `cam_right_wrist`；state/action 为 flat 14 维 `observation.state` / `action`。

### 2.1 单任务

```bash
cd XPolicyLab/policy/GalaxeaVLA
bash process_data.sh RoboDojo stack_bowls arx_x5 100
```

### 2.2 批量（多任务）

```bash
bash process_data_batch.sh RoboDojo cotrain arx_x5 joint \
  /path/to/RoboDojo_data
```

---

## 3. 训练

```bash
# bench_name ckpt_name env_cfg_type expert_data_num action_type seed gpu_id [hydra...]
bash train.sh RoboDojo robodojo_joint arx_x5 100 joint 0 0,1,2,3,4,5,6,7

# 外部 LeRobot 数据集
GALAXEA_DATASET_DIR=/path/to/RoboDojo_sim_arx-x5_v30 \
bash train.sh RoboDojo cotrain arx_x5 100 joint 0 0,1,2,3,4,5,6,7
```

- Hydra task：`real/g0plus_xpolicylab_finetune`
- `action_type` 目前只适配了 `joint`

---

## 4. 部署与评测

```bash
bash eval.sh RoboDojo stack_bowls cotrain arx_x5 joint 0 0 0 GalaxeaVLA XPolicyLab
```
`deploy.yml` 中 `eval_env`：`debug` → `sim` → `real`。`replan_steps`（默认 `5`）控制每次推理后实际执行的动作步数，`null` 表示执行完整 chunk（32 步）。

---

## 5. 策略包结构

| 文件 | 用途 |
|---|---|
| `model.py` | joint 推理（`pack_robot_state` / `unpack_robot_state`） |
| `GalaxeaVLA/configs/data/xpolicylab/dual_arm_joint_robodojo.yaml` | joint 数据 shape_meta（train/deploy 统一） |
| `GalaxeaVLA/configs/task/real/g0plus_xpolicylab_finetune.yaml` | 微调 task 配置 |

---
