# 测试与部署

本文档记录 real_env 真机 client 的本地测试、layout 拍摄、真机 GUI 启动与排错。

## 目录与 PYTHONPATH

真机目录布局（`XPolicyLab` 为 pipeline 子目录）：

```text
{XONE_ROOT}/
  config/
  task_info/
  layouts/
  src/robot/...
  XPolicyLab/real_env/...
```

运行前需让 Python 找到 `real_env` 与 `robot`：

```bash
export PYTHONPATH="${XONE_ROOT}/XPolicyLab:${XONE_ROOT}/src"
```

`real_env/run.sh` 会自动设置上述路径（假设脚本位于 `XPolicyLab/real_env/run.sh`）。

## 本地单元测试

```bash
conda run -n dev python -m py_compile \
  real_env/real_env_client.py \
  real_env/recorder.py \
  real_env/workbench.py \
  real_env/run_real_env_workbench.py \
  real_env/layout_shot.py

conda run -n dev python -m unittest \
  test.test_real_env_workbench \
  test.test_real_env_recorder
```

虚拟假 robot 全流程（`tmp/virtual_real_env/`，不依赖真机）：

```bash
cd tmp/virtual_real_env && ./run.sh smoke      # 假 robot obs/action
cd tmp/virtual_real_env && ./run.sh headless   # 无 GUI 完整 workflow
```

## Layout 拍摄（评测前）

```bash
python -m real_env.layout_shot \
  --base_cfg x-one-piper-orbbec \
  --task_name stack_bowls \
  --layouts_count 10
```

详见 `docs/layout_shot.md`。

## 真机 GUI 启动

**推荐：** 编辑 `real_env/run.sh` 中的参数后执行：

```bash
bash real_env/run.sh
```

**或直接调用：**

```bash
python -m real_env.run_real_env_workbench \
  --deploy_config policy/ACT/deploy.yml \
  --task_name stack_bowls \
  --base_cfg x-one-piper-orbbec \
  --policy_name ACT \
  --ckpt_setting RoboDojo_real-stack_bowls-piper-200-joint \
  --host <policy_server_host> \
  --port <policy_server_port> \
  --eval_episode_num 10
```

注意：

- 入口**不会**启动 policy server；需先在 server 机器启动 ModelServer。
- 入口**不修改** `sys.path`；依赖 `PYTHONPATH` 或 `run.sh`。
- `--ckpt_setting` **必填**，仅用于 client 结果目录分段。

检查配置：

```bash
python -m real_env.run_real_env_workbench \
  --task_name stack_bowls \
  --base_cfg x-one-piper-orbbec \
  --policy_name ACT \
  --ckpt_setting debug-test \
  --host 127.0.0.1 \
  --port 12345 \
  --print_config_only
```

用户当前真机测试习惯（server 侧）：

```bash
policy/ACT/eval.sh RoboDojo_real stack_bowls piper 200 joint 0 0 ACT_env Xone
```

client GUI 需额外传 `--ckpt_setting`（可与 server 侧 dataset/task 命名对齐）。若参数与真机本地 eval 文件不一致，先向维护者确认。

## 参数说明

### RealEnv 必需

| 参数 | deploy key | 用途 |
| --- | --- | --- |
| `--base_cfg` | `base_cfg` | `{XONE_ROOT}/config/{base_cfg}.yml` |
| `--task_name` | `task_name` | `{XONE_ROOT}/task_info/{task_name}.json` |
| `--policy_name` | `policy_name` | `policy.{policy_name}.deploy` |
| `--host` | `host` | policy server host |
| `--port` | `port` | policy server port |
| `--ckpt_setting` | `ckpt_setting` | 结果目录 `{eval_results}/{policy}/{ckpt_setting}/{task}/` |

### 可选

| 参数 | deploy key | 用途 |
| --- | --- | --- |
| `--seed` | `seed` | policy 侧随机种子 |
| `--eval_batch` | `eval_batch` | batch eval |
| `--force_reach_mode` | `force_reach_mode` | action 后等待 robot 到位 |

### Workbench

| 参数 | 用途 |
| --- | --- |
| `--eval_episode_num` | 目标 episode 数 |
| `--poll_hz` | GUI 刷新频率 |
| `--record_video` | 启用视频 recorder |
| `--record_trajectory` | 启用 HDF5 轨迹 recorder |
| `--record_fps` / `--record_crf` / `--record_camera` | recorder 参数 |
| `--offscreen` | 无显示器环境 |

### 运行前目录检查

```text
{XONE_ROOT}/config/{base_cfg}.yml
{XONE_ROOT}/task_info/{task_name}.json
{XONE_ROOT}/layouts/{task_name}/layout_{episode_idx:06d}.png
```

## 快速排错

```bash
python -c "from robot.robot import get_robot; print('robot ok')"
python -c "import policy.ACT.deploy as d; print('deploy ok')"
python -c "from PyQt5.QtWidgets import QApplication; print('PyQt5 ok')"
```

## 评测 GUI 流程摘要

1. 启动后自动 `prepare_task()` → 校验 layout 目录。
2. `准备任务` / 自动 `start_episode(0)` → 加载 layout，进入摆放。
3. `摆放完成` → 保存 placement，后台 eval；预览切为纯 live 画面。
4. 评测结束 → `成功` / `失败`；或评测中 `提前结束` / `异常终止`。
5. 可多轮直至 `eval_episode_num` 完成。
