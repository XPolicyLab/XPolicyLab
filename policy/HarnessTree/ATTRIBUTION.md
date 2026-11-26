# Attribution / 借鉴说明

- `colour_toolkit.py`、`colour_tools.py`、`colour_sight.py`、`colour_kinematics.py` 和 `colour_constants.py` 借鉴自 PyRUA-Lean 0.2.0 的 RoboDojo colour host，包括运动学、夹爪和 `locate` 的彩色几何计算。保留 Apache-2.0 版权声明；运行时不导入 `pyrualean`。见 `NOTICE`。
- 策略通过本项目的 SAM3 HTTP 服务进行分割，技能通过 `HarnessRobot`（`robo`）访问机器人。任务程序改编自 `PyRUA_RSI/programs`；共享辅助函数与阶段拆分不代表独立重写算法。
- The kinematics, gripper tooling, and colour-based `locate` geometry are adapted from the PyRUA-Lean 0.2.0 RoboDojo colour host. Copyright notices are retained; the runtime does not import `pyrualean`. See `NOTICE`.
- Segmentation uses Meta's SAM3, served by HarnessTree's external SAM3 HTTP wrapper (pins and artifact identity in [README](README.md#model-assets)). No SAM3 weights or source files are redistributed in this adapter. Skills use the `HarnessRobot` facade. The task programs are adapted from `PyRUA_RSI/programs`; shared helpers and stage refactors do not make the underlying algorithms independent implementations.
- 本适配器按照指令选择预编写的 Python 任务程序，不在测试时生成工作流；这些改动不代表符合 Agent Track 的要求。
- This adapter selects prewritten Python task programs from instructions; it does not generate workflows at test time. The adaptation does not establish Agent Track compliance.
