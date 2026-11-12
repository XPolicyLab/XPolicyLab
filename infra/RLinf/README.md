# XPolicyLab · RLinf

From the XPolicyLab root, run `bash infra/RLinf/scripts/run.sh <config-name>`. For RoboDojo, set `ROBODOJO_ROOT` to the official simulator checkout. Training and evaluation share the launcher. The task config sets `runner.only_eval` and accepts Hydra overrides.

在 XPolicyLab 根目录执行 `bash infra/RLinf/scripts/run.sh <config-name>`，把占位符换成任务配置名。使用 RoboDojo 时，将 `ROBODOJO_ROOT` 指向官方仿真仓库。训练与评估共用启动器，任务配置通过 `runner.only_eval` 选择模式，也支持 Hydra 参数覆盖。

[English user guide](doc/GUIDE.md) · [中文使用手册](doc/GUIDE_zh.md)

From the XPolicyLab root, run `bash infra/RLinf/scripts/test_cpu.sh` to check task configurations, policy updates, checkpoint loading and simulator interfaces on CPU. Test dependencies are listed in [the CPU requirements](tests/requirements-cpu-test.txt) and [the optional OpenPI data requirements](tests/requirements-cpu-sft-data.txt).

从 XPolicyLab 根目录运行 `bash infra/RLinf/scripts/test_cpu.sh`，可在 CPU 上检查任务配置、策略更新、权重加载和仿真接口。依赖见 [CPU 测试依赖](tests/requirements-cpu-test.txt)和[可选的 OpenPI 数据测试依赖](tests/requirements-cpu-sft-data.txt)。

`core/` defines model, observation and configuration interfaces. `workers/` connects them to RLinf. Simulator adapters live in `envs/` and policy implementations and configs under [policy/](../../policy). Use `scripts/` for launchers, `tests/` for regression tests, and `doc/` for documentation. `RLinf/` is the pinned upstream submodule. RLinf writes training outputs under `runner.logger.log_path`.

`core/` 定义模型、观测和配置接口，`workers/` 将它们接入 RLinf。仿真适配位于 `envs/`，策略实现和配置位于 [policy/](../../policy)。`scripts/` 存放启动脚本，`tests/` 存放回归测试，`doc/` 存放文档。`RLinf/` 是固定版本的上游子模块。RLinf 将训练结果写到 `runner.logger.log_path` 指定的目录。
