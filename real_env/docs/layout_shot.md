# Layout 拍摄

`real_env/layout_shot.py` 用于在真机上拍摄 workbench 所需的 layout 参考图。

## 输出路径

```text
{XONE_ROOT}/layouts/{task_name}/layout_{episode_idx:06d}.png
```

与 workbench 加载约定一致（见 `docs/design.md`）。

每次保存还会写同名 `.json` 元数据（timestamp、shape、episode 等）。

## 依赖

- PyQt5
- `robot` 包（`{XONE_ROOT}/src` 需在 PYTHONPATH）
- `{XONE_ROOT}/config/{base_cfg}.yml`
- 相机需包含 `cam_head`（GUI 预览与保存均使用 head 相机）

## 启动

```bash
PYTHONPATH=${XONE_ROOT}/XPolicyLab:${XONE_ROOT}/src \
python -m real_env.layout_shot \
  --base_cfg x-one-piper-orbbec \
  --task_name stack_bowls \
  --layouts_count 10
```

参数：

| 参数 | 说明 |
| --- | --- |
| `--base_cfg` | robot 配置文件名（不含 `.yml`） |
| `--task_name` | 任务名，决定 `layouts/{task_name}/` |
| `--layouts_count` | 需要拍摄的 layout 数量（episode 0 .. N-1） |
| `--poll_hz` | 预览刷新频率，默认 30 |
| `--offscreen` | 无显示器时使用 |
| `--print_config_only` | 只打印路径，不启动 GUI |

## GUI 操作

| 按钮 | 作用 |
| --- | --- |
| 保存当前 | 将当前画面写入 `layout_{idx:06d}.png` |
| 保存并下一轮 | 保存后 episode index +1 |
| 下一轮 | 不保存，仅切换 index |
| 退出 | 关闭窗口并清理 robot 节点 |

## 与 workbench 的关系

layout 拍摄是评测前的独立步骤；workbench 的 `prepare_task()` 只校验 `layouts/{task_name}/` 目录存在，不会自动拍摄。
