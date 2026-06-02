# Real Env GUI 统一设计

本文档只描述 `real_env/` 下当前真机 client GUI 重构线的统一设计。模块细节见：

- `real_env/real_env.md`
- `real_env/workbench.md`
- `real_env/recorder.md`
- `real_env/testing_deployment.md`
- `real_env/layout_shot.md`
- `real_env/TODO.md`

## 目标

真机评测由三层组成：

- `real_env`：真机环境适配器。负责 robot、policy server、obs/action schema 转换，以及 policy-facing env API。
- `workbench`：GUI workflow controller。负责状态机、layout、placement、评测线程、成功/失败/终止/重试、结果事件。
- `recorder`：episode 数据录制插件。后续负责接收或采样 obs，并通过插件把不同类型产物写入 episode 目录。

核心边界：

- `RealEnv` 不判断成功/失败，不推进 GUI episode，不管理 recorder。
- `WorkbenchController` 是唯一 workflow 状态持有者。
- `result_events.jsonl` 是 append-only task 级日志，abort/retry 清理时不能删除。
- recorder 长期按插件目录管理产物，workbench 不解析各插件的内部数据格式。

## 目录约定

`XPolicyLab` 作为 x-one-pipeline 的子目录存在：

```text
XONE_ROOT/
  config/
  task_info/
  layouts/
  src/
  XPolicyLab/
```

当前 `real_env/constants.py` 通过 `XPolicyLab` 的上一级目录计算 `XONE_ROOT`。

Layout 固定放在：

```text
{XONE_ROOT}/layouts/{task_name}/layout_{episode_idx:06d}.png
```

结果目录默认放在：

```text
{XONE_ROOT}/eval_results/{policy_name}/{ckpt_setting}/{task_name}/
```

## 状态机

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

状态含义：

- `TASK_INIT`：GUI 已启动，task 级资源尚未校验。
- `LAYOUT_READY`：layout 目录已校验，可开始 episode。
- `PLACEMENT`：用户摆放物品，GUI 显示 live image 与 layout 叠图（默认 alpha=0.35）。
- `EVALUATING`：后台线程执行 `env.reset_robot()`、recorder start、`env.eval_one_episode()`、recorder stop；GUI 预览临时将叠图 alpha 设为 0，仅显示实时画面。
- `AWAIT_RESULT`：评测线程正常结束，等待用户提交成功、失败或重试本轮。
- `EPISODE_ABORTED`：用户中断当前 episode，不计入统计，可以重试本轮。
- `TASK_FINISHED`：目标 episode 数完成。
- `ERROR`：出现不可恢复错误，需要人工处理。

主路径：

```text
TASK_INIT
  -> LAYOUT_READY
  -> PLACEMENT
  -> EVALUATING
  -> AWAIT_RESULT
  -> PLACEMENT / TASK_FINISHED
```

合法迁移：

| 当前状态 | 事件 | 下一个状态 | 动作 |
| --- | --- | --- | --- |
| `TASK_INIT` | `prepare_task` | `LAYOUT_READY` | 校验 `{XONE_ROOT}/layouts/{task_name}/` |
| `LAYOUT_READY` | `start_episode` | `PLACEMENT` | 创建 episode 目录，加载 layout |
| `PLACEMENT` | `finish_placement` | `EVALUATING` | 保存 `placement.png` 和 metadata，后台评测 |
| `EVALUATING` | `eval_finished` | `AWAIT_RESULT` | 停止 recorder，等待判定 |
| `EVALUATING` | `finish_eval_early` | `AWAIT_RESULT` | `env.request_stop()`，等待线程退出，保留 episode 产物，recorder 正常 stop |
| `EVALUATING` | `abort_episode` | `EPISODE_ABORTED` | `env.request_stop()`，等待线程退出，清理未提交产物，写 abort event |
| `AWAIT_RESULT` | `mark_success` | `PLACEMENT` 或 `TASK_FINISHED` | 写 success，更新统计，推进 episode |
| `AWAIT_RESULT` | `mark_fail` | `PLACEMENT` 或 `TASK_FINISHED` | 写 fail，更新统计，推进 episode |
| `AWAIT_RESULT` | `retry_episode` | `PLACEMENT` | 清理未提交产物，写 retry event，不推进 episode |
| `EPISODE_ABORTED` | `retry_episode` | `PLACEMENT` | 清理未提交产物，写 retry event，不推进 episode |
| 任意状态 | `fatal_error` | `ERROR` | 记录错误 |

不支持：

- `PLACEMENT` 中 retry：用户本来就在摆放阶段。
- `EVALUATING` 中 retry：必须先 abort，等进入 `EPISODE_ABORTED` 后重试。
- `TASK_FINISHED` 中 retry：已提交统计，后续如果要修改应做独立结果编辑能力。

## Episode 产物

每个 episode 使用稳定目录：

```text
eval_results/
  {policy_name}/
    {ckpt_setting}/
      {task_name}/
        result_events.jsonl
        episode_000000/
          placement.png
          placement_metadata.json
          recorder/
            video/
              manifest.json
              cam_head.mp4
            trajectory/
              manifest.json
              trajectory.hdf5
          cam_head.mp4                  # 旧 recorder 兼容
          cam_left_wrist.mp4            # 旧 recorder 兼容
          cam_right_wrist.mp4           # 旧 recorder 兼容
```

当前 abort/retry 清理范围：

```text
episode_000000/placement.png
episode_000000/placement_metadata.json
episode_000000/recorder/
episode_000000/*.mp4
```

不清理：

- `result_events.jsonl`
- episode 目录本身
- layout 图片
- 未来 recorder 插件目录之外的未知文件

## 结果事件

`result_events.jsonl` 采用 append-only。当前事件：

```json
{"event":"episode_committed","episode":0,"result":"success","timestamp":"...","success":1,"total":1}
{"event":"episode_aborted","episode":0,"reason":"operator_abort","timestamp":"..."}
{"event":"episode_retried","episode":0,"reason":"operator_retry","from_state":"EPISODE_ABORTED","timestamp":"..."}
```

当前会写事件，但还不会从事件日志恢复状态。恢复能力见 `real_env/TODO.md`。

## Recorder 长期设计

旧 `VideoRecorder` 会把 `{camera_name}.mp4` 直接写到 episode 目录。当前清理 `*.mp4` 只是兼容旧约定，不作为长期 contract。

当前 recorder 已经按插件产物目录保存视频和轨迹：

```text
episode_000000/
  recorder/
    video/
      manifest.json
      cam_head.mp4
      cam_left_wrist.mp4
    trajectory/
      manifest.json
      trajectory.hdf5
```

规则：

- recorder 插件只写入 `episode_dir/recorder/{plugin_name}/`。
- workbench 不理解插件内部文件格式。
- abort/retry 时可以删除整个 `episode_dir/recorder/`。
- `manifest.json` 是插件产物清单，记录插件名、版本、输出文件、数据类型、帧数、shape、状态等元信息。

## 当前能力

已实现：

- 固定 layout 自动加载。
- placement 原图保存和 metadata。
- GUI 主路径。
- 成功/失败提交。
- `EVALUATING` 中断与提前结束。
- 评测过程中 GUI 自动隐藏 layout 叠图（alpha=0）。
- layout 拍摄工具（`layout_shot.py`）。
- GUI 正在评测时的关闭保护。
- `AWAIT_RESULT` / `EPISODE_ABORTED` 重试本轮。
- abort/retry 清理当前未提交产物。
- recorder 中心采样线程：按 fps 采样 obs，分发给插件。
- 视频 recorder：写入 `episode_dir/recorder/video/`。
- 轨迹 recorder：按 Xone HDF5 结构写入 `episode_dir/recorder/trajectory/trajectory.hdf5`。
- 单元测试 + offscreen GUI smoke。

未实现：

- recorder 新数据类型插件扩展点。
- recorder 临时文件提交机制。
- 从 `result_events.jsonl` 恢复任务。
- 更完整的 error recovery。
