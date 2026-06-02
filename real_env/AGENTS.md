# real_env 模块设计

真机评测 **client** 重构线。改 `real_env/` 代码前必读；用法见 [README.md](README.md)。

## 三层职责

| 层 | 文件 | 职责 |
| --- | --- | --- |
| 环境适配 | `real_env_client.py` | 加载 config/task_info、robot、ModelClient；obs/action 转换；调用 `policy.{name}.deploy` |
| 工作流 | `workbench.py` | GUI 状态机、layout、placement、评测线程、成功/失败/终止/重试、`result_events.jsonl` |
| 录制 | `recorder.py` | 可选插件：video、trajectory（HDF5） |
| 辅助 | `helpers.py` | `build_state`、`create_move_data`、`camera_meta`、`load_yaml` |
| Layout | `layout_shot.py` | 评测前拍摄 layout 参考图（独立 GUI，非 workbench 流程） |

**核心边界：**

- `RealEnv` 不判断成功/失败、不推进 episode index、不管理 recorder。
- `WorkbenchController` 是唯一 workflow 状态持有者。
- `result_events.jsonl` 为 append-only；abort/retry 时不得删除。
- Recorder 插件只写 `episode_dir/recorder/{plugin}/`；workbench 不解析插件内部格式。

## 目录约定

```text
{XONE_ROOT}/
  config/{base_cfg}.yml
  task_info/{task_name}.json
  layouts/{task_name}/layout_{episode:06d}.png
  eval_results/{policy_name}/{ckpt_setting}/{task_name}/
    result_events.jsonl
    episode_000000/
      placement.png
      placement_metadata.json
      recorder/video/ ...
      recorder/trajectory/ ...
  src/robot/...
  XPolicyLab/real_env/...
```

`XONE_ROOT` = `XPolicyLab` 上一级（pipeline 根）。`ckpt_setting` **必填**，仅用于 client 结果目录分段；checkpoint 由 policy server 加载，client 不传 `ckpt_dir`。`seed` 可出现在 `deploy_cfg` 给 policy 用，**不**参与结果路径。

## RealEnv（`real_env_client.py`）

**负责：** `get_obs` / `take_action` / `reset` / `reset_robot` / `eval_one_episode`；`request_stop` / `clear_stop`（仅 Python flag，不直接 halt robot）。

**不负责：** GUI 状态、成功率统计、recorder 产物。

`deploy_cfg` 必填：`base_cfg`, `task_name`, `policy_name`, `host`, `port`, `ckpt_setting`。

加载 config 时会 `pop("collect")`，避免评测时创建 `CollectAny`。`reset_robot()` 只 reset 真机；`reset()` 额外向 policy server 发 `reset`。评测线程开始时用 `reset_robot()`，episode 结束 `finish_episode()` 再完整 reset。

Obs/action 转换在 `helpers.py`；policy import 路径为 `policy.{policy_name}.deploy`（非 `XPolicyLab.policy`）。

### 公开接口

```python
class RealEnv:
    def __init__(self, deploy_cfg: dict) -> None: ...
    def get_obs(self, env_idx: int = 0) -> dict: ...
    def eval_one_episode(self) -> None: ...
    def reset_robot(self) -> None: ...
    def reset(self) -> None: ...
    def take_action(self, action: dict) -> None: ...
    def is_episode_end(self) -> bool: ...
    def finish_episode(self) -> None: ...
    def request_stop(self, reason: str = "operator_abort") -> None: ...
    def clear_stop(self) -> None: ...
    def close(self) -> None: ...
```

`request_stop()` 只设置 stop flag；`take_action()` / `is_episode_end()` 检查 stop flag；不调用 robot stop/halt。`m_robot_lock` 保护 `get_obs()`、`take_action()`、`reset()` 的 robot 访问。

`task_info` 要求 `step_lim`；`instructions` 缺失时 fallback 到 task name。

## Workbench 状态机

```text
TASK_INIT -> LAYOUT_READY -> PLACEMENT -> EVALUATING -> AWAIT_RESULT -> (下一 episode PLACEMENT | TASK_FINISHED)
                              ^              |
                              |         EPISODE_ABORTED (abort 后可 retry)
                              +-- retry_episode
```

| 状态 | 含义 |
| --- | --- |
| `PLACEMENT` | 用户摆放；GUI 显示 live + layout 叠图（alpha 默认 0.35） |
| `EVALUATING` | 后台 `reset_robot` → eval → recorder；**GUI 预览 alpha 强制为 0** |
| `AWAIT_RESULT` | 评测结束，等待用户点成功/失败 |
| `EPISODE_ABORTED` | 异常终止，未提交，可重试 |

**停止语义：**

| 操作 | 方法 | 结果状态 | 产物 |
| --- | --- | --- | --- |
| 提前结束 | `finish_eval_early()` | `AWAIT_RESULT` | 保留，recorder `stop()` |
| 异常终止 | `abort_episode()` | `EPISODE_ABORTED` | 清理未提交产物，recorder `abort()` |

`EVALUATING` 中不可直接 retry；须先 abort。`PLACEMENT` 中无 retry（重新摆即可）。

### GUI 按钮

| 按钮 | 启用状态 | 动作 |
| --- | --- | --- |
| `准备任务` | `TASK_INIT` / `ERROR` | prepare task 后自动进入 episode 0 |
| `摆放完成` | `PLACEMENT` | 保存 placement，启动评测线程 |
| `成功` / `失败` | `AWAIT_RESULT` | 提交结果，推进 episode |
| `重试本轮` | `AWAIT_RESULT` / `EPISODE_ABORTED` | 回到当前 `PLACEMENT` |
| `提前结束` | `EVALUATING` | 进入 `AWAIT_RESULT` |
| `异常终止` | `EVALUATING` | 进入 `EPISODE_ABORTED` |
| `退出` | 始终 | 若评测中先 stop 并等待线程，再 `env.close()` |

### 产物

Workbench 直接写 `episode_*/placement.png`、`placement_metadata.json`（含 task、policy、ckpt_setting、episode、layout_path、alpha 等）及 task 级 `result_events.jsonl`（`episode_committed` / `episode_aborted` / `episode_retried`）。

Abort/retry 清理：`placement.png`、`placement_metadata.json`、`recorder/`、episode 根下 `*.mp4`（旧约定兼容）。不删 `result_events.jsonl` 与 layout 图。

## Recorder

`EpisodeRecorder` 按 fps 采样 `env.get_obs()`，分发给插件。

- 视频：`episode_dir/recorder/video/{cam}.mp4` + `manifest.json`
- 轨迹：`episode_dir/recorder/trajectory/trajectory.hdf5`（Xone HDF5：`vision/`、`state/`、`timestamps`、相机内外参）

评测线程：`recorder.start(episode_dir)` → `env.eval_one_episode()` → `recorder.stop()`。abort/retry 走 `recorder.abort()`。

Recorder 不判断成功/失败、不推进 episode、不控制 robot。

## Layout 拍摄（`layout_shot.py`）

输出 `{XONE_ROOT}/layouts/{task_name}/layout_{episode_idx:06d}.png`（及同名 `.json` 元数据）。依赖 `cam_head`、PyQt5、`robot` 包。与 workbench 独立；`prepare_task()` 只校验 layout 目录存在。

GUI：`保存当前` / `保存并下一轮` / `下一轮` / `退出`。

## 入口

| 入口 | 用途 |
| --- | --- |
| `run_real_env_workbench.py` | 评测 GUI 主入口 |
| `real_env/run.sh` | 设置 PYTHONPATH 并启动 workbench |
| `layout_shot.py` | Layout 拍摄 GUI |

禁止在 Python 里改 `sys.path`；路径由 `run.sh` 设置 `PYTHONPATH=${XONE_ROOT}/XPolicyLab:${XONE_ROOT}/src`。

## 测试与真机

```bash
python -m unittest test.test_real_env_workbench test.test_real_env_recorder
```

假 robot 全流程：`tmp/virtual_real_env/`（不提交主线）。

Server 侧习惯：`policy/ACT/eval.sh RoboDojo_real stack_bowls piper 200 joint 0 0 ACT_env Xone`。Client GUI 需额外传 `--ckpt_setting`；若与真机本地 eval 脚本参数不一致，先向用户确认。

排错：

```bash
python -c "from robot.robot import get_robot; print('robot ok')"
python -c "import policy.ACT.deploy as d; print('deploy ok')"
python -c "from PyQt5.QtWidgets import QApplication; print('PyQt5 ok')"
```

## 待办

- 从 `result_events.jsonl` 恢复中断任务
- Error 状态恢复策略（配置 / 评测线程 / robot 错误）
- Recorder 临时文件提交、action log、新数据类型插件
- `ResultStore` 统一封装 result log
- 真机 GUI 入口实际 smoke

**暂不做：** RealEnv 管 GUI workflow；`request_stop()` 调 robot halt；attempt 子目录；已提交 episode 回滚。
