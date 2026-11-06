# WSA-Base 接入 WorldArena 2.0 Track 3

该目录只加载原版 `WSA_Base` checkpoint，模型文本输入直接使用官方 observation 的完整 `prompt`。

## 输入与输出

- AgileX：读取 `joint_qpos_left(7)`、`joint_qpos_right(7)`、`cam_high`、`cam_left_wrist`、`cam_right_wrist`，返回 14 维双臂关节动作。
- Franka：读取 `left_end_pose(7)`、`joint_qpos(8)` 中最后一维夹爪，以及 `cam_high`、`cam_left_wrist`；状态和动作顺序为 `[x, y, z, qx, qy, qz, qw, gripper]`。
- 缺字段、维度错误、非 `uint8 HWC RGB` 图像或 NaN/Inf 会直接报错。
- 当前训练配置使用 `image_delta_indices=[0,0,15]`，因此推理端固定输入 `[当前帧, 当前帧]`，不维护跨请求图像历史。状态也只使用当前帧。

## 启动

先安装官方 A 端：

```bash
export WORLDARENA_ROOT=/path/to/WorldArena-2.0
pip install -e "${WORLDARENA_ROOT}/real_world_benchmark"
pip install -r "${WORLDARENA_ROOT}/real_world_benchmark/requirements-a.txt"
```

公共配置：

```bash
export WSA_CHECKPOINT=/path/to/wsa_base/checkpoint
export WSA_STATS_PATH=/path/to/training_stats.json
export WSA_QWEN3_VL_PATH=/path/to/Qwen3-VL-2B-Instruct
export WSA_COSMOS_TOKENIZER_PATH=/path/to/Cosmos-Tokenizer-CI8x8
export HUB_POLICY_URL=https://organizer.example/policy
export POLICY_ID=official_worker_key
```

启动对应本体：

```bash
# AgileX；常规 flat stats 通常会自动匹配 observation.state/action
CUDA_VISIBLE_DEVICES=0 bash evaluation/worldarena/start_agilex.sh

# Franka；按训练 stats 的实际字段设置
export WSA_STATE_STATS_KEYS=observation.state.endpose
export WSA_ACTION_STATS_KEYS=action.endpose
CUDA_VISIBLE_DEVICES=0 bash evaluation/worldarena/start_franka.sh
```

状态归一化和动作反归一化直接调用训练仓库的 `NormalizeTransformFn` / `UnNormalizeTransformFn`。归一化模式优先从 checkpoint 附近的 `train_config.json` 读取；旧 checkpoint 没有该信息时按原版 WSA 默认使用 `mean_std`，无需在启动脚本中手动指定。

`WSA_ACTION_MODE` 同样优先读取训练配置；读取不到时必须显式设置为 `abs` 或 `delta`。delta 模式按 `WSA_ROBOT_TYPE` 从 `transforms/constants.py` 读取 delta mask，因此 `cobot_magic_max` 和 `wr_franka` 的夹爪维会保持绝对值。只有确实使用同一种 8 维 end-pose state/action 表示时，Franka delta 才应设置 `WSA_ALLOW_FRANKA_DELTA=1`。

常用可选项：`WSA_STATS_KEY`、`WSA_STATE_STATS_KEYS`、`WSA_ACTION_STATS_KEYS`、`WSA_PROCESSOR_PATH`、`WSA_EXECUTE_CHUNK_SIZE`、`WSA_DEVICE`、`WSA_DTYPE`。

本地 WebSocket：

```bash
WORLDARENA_TRANSPORT=ws WORLDARENA_HOST=0.0.0.0 WORLDARENA_PORT=8000 \
  bash evaluation/worldarena/start_agilex.sh
```

可用 `WSA_DRY_RUN=1` 做接口冒烟测试：它会严格校验输入并返回 hold action，不加载模型。正式评测必须移除该变量。
