# Attribution / 借鉴说明

HarnessTree is an independently packaged public XPolicyLab adapter. Its design is inspired by prior open RoboDojo agent approaches, including the PyRUA-Lean colour-geometry host and the use of instruction-selected skill programs in PyRUA-style agents.

The retained colour geometry helpers are licensed under Apache-2.0; see [NOTICE](NOTICE) and the [Apache license](LICENSE-Apache-2.0-colour-host.txt). HarnessTree does not import `pyrualean` at runtime. Segmentation uses an external [SAM3 HTTP service](README.md#model-assets), and skills access the robot through `HarnessRobot`. HarnessTree selects prewritten programs rather than generating workflows at test time; this does not establish Agent Track compliance.

HarnessTree 是独立打包发布的 XPolicyLab 适配器。其设计受到 RoboDojo 早期开源智能体方案的启发，包括 PyRUA-Lean 的颜色几何宿主设计，以及 PyRUA 风格智能体按指令选择技能程序的做法。

保留的颜色几何辅助模块采用 Apache-2.0 许可，详见 [NOTICE](NOTICE) 和 [Apache 许可证](LICENSE-Apache-2.0-colour-host.txt)。HarnessTree 运行时不导入 `pyrualean`。分割通过外部 [SAM3 HTTP 服务](README.md#model-assets) 完成，技能通过 `HarnessRobot` 访问机器人。HarnessTree 选择预编写的程序，不在测试时生成工作流；这并不意味着符合 Agent Track 的要求。
