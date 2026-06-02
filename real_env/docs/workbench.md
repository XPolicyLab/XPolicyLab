# Workbench 模块

`real_env/workbench.py` 是当前真机 GUI workflow controller。它负责 GUI 状态机和 episode 流程，不负责 robot 细节。

## 职责

Workbench 负责：

- 校验 task layout 目录。
- 自动加载每个 episode 的 layout。
- 显示 live image、layout overlay（`PLACEMENT` 阶段 alpha 默认 0.35；`EVALUATING` 阶段预览 alpha 临时为 0）。
- 保存 placement 原图和 metadata。
- 在后台线程中运行 RealEnv episode。
- 提交成功/失败结果。
- 终止正在评测的 episode。
- 重试当前 episode。
- 写入 `result_events.jsonl`。
- 清理当前未提交 episode 产物。

Workbench 不负责：

- robot 初始化和 action 下发。
- policy server 连接。
- recorder 内部数据格式。
- 从 result log 恢复任务状态。

## 主要类

### `WorkflowState`

当前状态：

```text
TASK_INIT
LAYOUT_READY
PLACEMENT
EVALUATING
AWAIT_RESULT
EPISODE_ABORTED
TASK_FINISHED
ERROR
```

状态机完整说明见 `real_env/design.md`。

### `WorkbenchState`

保存 GUI workflow 状态：

- `workflow_state`
- `active_episode`
- `committed_episode_num`
- `success_num`
- `target_episode_num`
- `alpha`
- `live_image`
- `layout_image`
- `layout_path`
- `episode_dir`
- `abort_reason`
- `last_error`
- `logs`

### `WorkbenchController`

核心 controller，不依赖 Qt。主要方法：

| 方法 | 功能 |
| --- | --- |
| `prepare_task()` | 校验 layout 目录，进入 `LAYOUT_READY` |
| `start_episode(episode_idx=None)` | 加载当前 layout，创建 episode 目录，进入 `PLACEMENT` |
| `update_live_obs()` | 从 env 读取 obs，更新 live image |
| `save_placement()` / `finish_placement()` | 保存 placement，并启动后台评测 |
| `process_eval_events()` | 处理评测线程结束事件 |
| `mark_success()` | 提交 success，更新统计，推进 episode |
| `mark_fail()` | 提交 fail，更新统计，推进 episode |
| `abort_episode(reason="operator_abort")` | 仅 `EVALUATING` 可用，向 RealEnv 写入英文 reason 并调用 `request_stop()` |
| `finish_eval_early(reason="operator_early_finish")` | 仅 `EVALUATING` 可用，向 RealEnv 写入英文 reason 并调用 `request_stop()` |
| `retry_episode(reason="operator_retry")` | `AWAIT_RESULT` / `EPISODE_ABORTED` 可用，回到当前 `PLACEMENT` |
| `current_display_image()` | 返回 live/layout blend 图；评测中 alpha=0 |

### `RealEnvWorkbench`

Qt GUI wrapper。用于启动窗口、处理 Qt events，并暴露 controller wrapper：

- `prepare_task()`
- `save_placement()`
- `finish_placement()`
- `mark_success()`
- `mark_fail()`
- `abort_episode()`
- `retry_episode()`

## GUI 按钮

当前按钮和启用条件：

| 按钮 | 启用状态 | 动作 |
| --- | --- | --- |
| `准备任务` | `TASK_INIT` / `ERROR` | prepare task 后自动进入 episode 0 |
| `摆放完成` | `PLACEMENT` | 保存 placement，启动评测线程 |
| `成功` | `AWAIT_RESULT` | 提交 success |
| `失败` | `AWAIT_RESULT` | 提交 fail |
| `重试本轮` | `AWAIT_RESULT` / `EPISODE_ABORTED` | 不提交结果，回到当前 `PLACEMENT` |
| `提前结束` | `EVALUATING` 且未请求停止 | 任务完成后提前结束评测，进入 `AWAIT_RESULT` |
| `异常终止` | `EVALUATING` 且未请求停止 | 中断当前评测，进入 `EPISODE_ABORTED` |
| `退出` | 始终可点 | 关闭窗口；如果正在评测，先请求停止并等待评测线程退出，再调用 `env.close()` |

`placement_metadata.json` 中的 `alpha` 记录**摆放阶段**保存时的叠图透明度；评测阶段 GUI 强制 alpha=0 仅影响预览，不回写 metadata。

## 产物

Workbench 直接生成：

```text
episode_000000/
  placement.png
  placement_metadata.json
```

`placement_metadata.json` 包含：

- task
- policy
- ckpt_setting
- episode
- timestamp
- layout_path
- alpha
- output_image_path

当前还会写 task 级事件日志：

```text
result_events.jsonl
```

事件包括：

- `episode_committed`
- `episode_aborted`
- `episode_retried`

## Abort / Retry 清理

当前 abort/retry 清理当前 episode 未提交产物：

```text
placement.png
placement_metadata.json
recorder/
*.mp4
```

这不会删除：

- `result_events.jsonl`
- episode 目录本身
- layout 图片
- recorder 插件产物目录之外的未知文件

`recorder/` 是当前插件目录；`*.mp4` 是旧 recorder 兼容规则。

## 测试入口

单元测试：

```bash
conda run -n dev python -m unittest test.test_real_env_workbench
```
