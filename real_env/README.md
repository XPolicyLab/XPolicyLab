# real_env

真机评测 **client 端**模块：连接远端 policy server，在 robot 侧完成 layout 对齐、摆放确认、policy 评测与结果提交。

## 功能

- **Layout 拍摄**（`layout_shot.py`）：保存 `{XONE_ROOT}/layouts/{task}/layout_*.png`
- **评测 GUI**（`workbench.py`）：摆放叠图 → 后台 eval → 成功/失败/终止/重试
- **可选录制**（`recorder.py`）：episode 视频、Xone 格式 HDF5 轨迹
- **环境适配**（`real_env_client.py`）：robot obs/action 与 policy server 通信

GUI **不会**启动 policy server；checkpoint 在 server 侧加载，client 只需 `--host` / `--port`。

## 快速开始

```bash
# 0. 环境：PYTHONPATH 需包含 XPolicyLab 根目录与 {XONE_ROOT}/src（robot 包）
export PYTHONPATH="${XONE_ROOT}/XPolicyLab:${XONE_ROOT}/src"

# 1. 拍摄 layout（评测前，一次性）
python -m real_env.layout_shot \
  --base_cfg x-one-piper-orbbec \
  --task_name stack_bowls \
  --layouts_count 10

# 2. 在 server 机器启动 policy（各 policy 的 eval.sh / setup_policy_server.py）

# 3. 启动评测 GUI（robot client）
bash real_env/run.sh
```

或直接调用（参数见 `run.sh`）：

```bash
python -m real_env.run_real_env_workbench \
  --base_cfg x-one-piper-orbbec \
  --task_name <task_name> \
  --policy_name <policy_name> \
  --ckpt_setting <ckpt_setting> \
  --host <host> \
  --port <port> \
  --eval_episode_num <eval_episode_num>
```

## 常用参数

| 参数 | 说明 |
| --- | --- |
| `--base_cfg` | 本体配置，`{XONE_ROOT}/config/{base_cfg}.yml`(我觉得这个评测需要和数采分开) |
| `--task_name` | `{XONE_ROOT}/task_info/{task_name}.json` |
| `--policy_name` | 加载 `policy.{policy_name}.deploy` |
| `--ckpt_setting` | **必填**，结果路径 `{eval_results}/{policy}/{ckpt_setting}/{task}/` |
| `--host` / `--port` | policy server 地址 |
| `--record_video` / `--record_trajectory` | 启用录制 |
| `--print_config_only` | 只打印路径，不启 GUI |

## 文档

模块设计与状态机见同目录 [AGENTS.md](AGENTS.md)。
