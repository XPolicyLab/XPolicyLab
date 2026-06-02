# RealEnv 模块

`real_env/real_env_client.py` 是当前真机环境适配器。它的目标是保持小而稳定，只提供 policy 运行所需的真机 I/O 和 env API。

## 职责

RealEnv 负责：

- 加载 `config/{base_cfg}.yml` 和 `task_info/{task_name}.json`。
- 初始化 `ModelClient`。
- 通过 `robot.robot.get_robot(base_cfg=...)` 初始化 robot。
- 将 robot 原始 obs 转成 policy obs schema。
- 将 policy action 转成 robot `move_data`。
- 动态 import policy deploy，并执行 `eval_one_episode()` / `eval_one_episode_batch()`。
- 提供 GUI 中断所需的 stop flag。
- 提供真机 client 生命周期接口 `close()`。

RealEnv 不负责：

- GUI 状态机。
- 成功/失败统计。
- episode index 推进。
- recorder 产物管理。
- retry/abort 的 workflow 语义。

## 公开接口

当前对齐 debug/sim env 的接口：

```python
class RealEnv:
    def __init__(self, deploy_cfg: dict) -> None: ...

    @property
    def deploy_cfg(self) -> dict: ...
    @property
    def task_info(self) -> dict: ...

    def get_obs(self, env_idx: int = 0) -> dict: ...
    def get_obs_batch(self, env_idx_list: list[int]) -> list[dict]: ...

    def eval_one_episode(self) -> None: ...
    def eval_one_episode_batch(self) -> None: ...

    def reset_robot(self) -> None: ...
    def reset(self) -> None: ...
    def take_action(self, action: dict) -> None: ...
    def take_action_batch(self, action_list: list[dict], env_idx_list: list[int]) -> None: ...

    def is_episode_end(self) -> bool: ...
    def finish_episode(self) -> None: ...
    def get_running_env_idx_list(self) -> list[int]: ...

    def request_stop(self, reason: str = "operator_abort") -> None: ...
    def clear_stop(self) -> None: ...
    def close(self) -> None: ...
```

`request_stop()` / `clear_stop()` 是 GUI workflow 的附加控制接口，不要求 policy deploy 直接调用。

## 配置

`deploy_cfg` 必须包含：

| key | 用途 |
| --- | --- |
| `base_cfg` | 加载 `{XONE_ROOT}/config/{base_cfg}.yml`（评测前会去掉其中的 `collect` 段，避免创建 `CollectAny`） |
| `task_name` | 加载 `{XONE_ROOT}/task_info/{task_name}.json` |
| `policy_name` | import `policy.{policy_name}.deploy` |
| `host` | policy server host |
| `port` | policy server port |

常用可选字段：

| key | 用途 |
| --- | --- |
| `eval_batch` | 外层入口决定调用 single 还是 batch eval |
| `force_reach_mode` | action 后等待 `robot.is_move()` 结束 |
| `ckpt_setting` | 结果目录 `{eval_results}/{policy}/{ckpt_setting}/{task}/` 的分段名（必填） |
| `seed` | policy 侧随机种子；不用于 workbench 结果目录 |

`ckpt_setting` 只影响 client 侧结果保存路径；checkpoint 文件由 policy server 加载，client 不传递 `ckpt_dir`。

`task_info` 当前要求：

| key | 用途 |
| --- | --- |
| `step_lim` | `is_episode_end()` 的步数上限 |
| `instructions` | reset 时随机选择 instruction；缺失时 fallback 到 task name |

## Obs / Action 转换

转换逻辑在 `real_env/helpers.py`：

- `camera_meta(...)`
- `build_state(...)`
- `create_move_data(...)`

当前 RealEnv 输出 obs：

```python
{
    "data_format_version": "v1.0",
    "instruction": "...",
    "env_idx": 0,
    "additional_info": {"frequency": ...},
    "vision": {...},
    "state": {...},
}
```

action 转换支持网络返回的 numpy-like payload，包括当前已修复的 bytes/base64 变体。

## 中断语义

`request_stop(reason)` 只设置 Python stop flag 和 reason：

- 不主动调用 robot 的 stop/halt/emergency-stop。
- `is_episode_end()` 会因为 stop flag 返回 true。
- `take_action()` 在转换和下发 action 前检查 stop flag，设置后直接返回。
- `force_reach_mode` 等待 robot move 完成时也检查 stop flag。

停止评测后的 robot 状态恢复由下一次常规 `reset()` 完成。

## Robot 访问锁

当前用 `m_robot_lock` 保护直接访问 robot 的路径：

- `get_obs()` 持锁调用 `robot.get_obs()`。
- `take_action()` 持锁调用 `robot.move()`，并在 `force_reach_mode` 下查询 `robot.is_move()`。
- `reset()` 持锁调用 `robot.reset()`。

`request_stop()` 不访问 robot，因此不持锁。

`reset()` 还会向 policy server 发送 `reset` 命令，用于清空 server 侧时序状态。这个命令的返回值没有业务含义，RealEnv 只读取并丢弃返回包，避免不同 policy server 对 reset ack 的序列化差异影响 GUI / client 流程。

## 生命周期

`reset_robot()` 只 reset 真机；`reset()` 在 robot reset 之外还向 policy server 发送 `reset`。

评测线程开始时调用 `reset_robot()`，episode 结束后 `finish_episode()` 再次 reset robot 并通知 policy server。

`close()` 关闭 ModelClient 连接。robot scheduler / sensor 的清理由上层入口或 layout 拍摄工具在退出时显式处理。

## 当前限制

- 真机 batch 暂时只支持 batch size 1。
- `finish_episode()` 只保留兼容语义，GUI 不依赖它做成功率统计。
- recorder 不属于 RealEnv 职责。
