<div align="center">

<h1>XPolicyLab</h1>

<p><a href="README.md">English</a> | 简体中文</p>

<p><strong>面向机器人策略评测与部署的统一标准与开放生态</strong></p>

<p>
<a href="https://xpolicylab.github.io/">Website</a> |
<a href="https://arxiv.org/abs/2608.09892">arXiv</a> |
<a href="https://github.com/XPolicyLab/XPolicyLab">GitHub</a> |
<a href="https://robodojo-benchmark.com/LeaderBoard">RoboDojo Leaderboard</a> |
<a href="https://robotwin-platform.github.io/leaderboard">RoboTwin Leaderboard</a>
</p>

<img src="assets/teaser.png" alt="XPolicyLab overview" width="100%"/>

<p><em>把 N 个策略接到 M 个评测环境 —— 从 O(N×M) 降到 O(N+M)。</em></p>

</div>



XPolicyLab 是策略代码与评测环境之间的共享层。每个模型的依赖、权重与训练配方放在 `policy/<POLICY>/`；XPolicyLab 负责那些枯燥但容易出错的部分 —— 服务化、观测/动作契约，以及评测接线。截至 2026 年 9 月，生态已接入 **46 个机器人策略**，覆盖 VLA、world-action、模仿学习与记忆增强等家族；同一套 adapter 可服务 RoboTwin、RoboDojo 仿真，以及标准化真机评测。

仓库级概念与接入步骤从本文开始。安装命令、权重布局与训练细节以各策略自己的 README 为准。

## 📚 目录

- [XPolicyLab 能做什么](#-xpolicylab-能做什么)
- [已支持的 Benchmark 与基础设施](#-已支持的-benchmark-与基础设施)
- [已接入策略](#-已接入策略)
- [框架概览](#-框架概览)
- [快速开始](#-快速开始)
- [通用工作流](#-通用工作流)
- [部署流程](#-部署流程)
- [标准数据格式](#-标准数据格式)
  - [图像解码只能走 `decode_image_bit`](#图像解码只能走-decode_image_bit)
  - [官方 LeRobot 转换](#官方-lerobot-转换)
- [数据与 Checkpoint](#-数据与-checkpoint)
- [接入你自己的策略](#-接入你自己的策略)
- [编程 Agent](#-编程-agent)
- [引用](#-引用)
- [联系方式](#-联系方式)



## 🚀 XPolicyLab 能做什么

- **环境隔离**：策略模型跑在自己的 conda/uv 环境里，仿真器、benchmark 或机器人客户端单独运行。
- **远程部署**：策略 server 与环境 client 通过 websocket 连接，可同机也可跨机。
- **统一 adapter 契约**：安装、数据转换、训练、服务化、评测共用同一套高层生命周期。
- **大规模策略库**：复用 VLA/WAM、模仿学习基线与参考模板的 adapter。
- **Benchmark / 基础设施集成**：把 XPolicyLab 挂进 benchmark 或仿真工作区，而不把策略代码绑死在某一个环境上。



## 🌐 已支持的 Benchmark 与基础设施

XPolicyLab 与具体 benchmark 解耦：任意 benchmark、仿真器或真机方案都可以作为环境 client，挂到同一套策略侧接口上 —— 每个策略一个 adapter，每个环境一个 client。下列 benchmark 均已接入；RoboDojo 与 RoboTwin 的官方排行榜由 XPolicyLab 提交驱动。

<div align="center">
<img src="assets/benchmarks.png" alt="Cross-platform evaluation through XPolicyLab" width="70%"/>
<p><em>通过共享代码库与标准化服务接口做跨平台评测。</em></p>
</div>

**Benchmarks**

- **[RoboDojo](https://github.com/RoboDojo-Benchmark/RoboDojo)**：基于仿真器的评测，以及 RoboDojo 格式数据导出。[RoboDojo Leaderboard](https://robodojo-benchmark.com/LeaderBoard) 覆盖五个能力维度（Generalization、Precision、Long-Horizon、Memory、Open）上的 42 个仿真任务，以及三套双臂本体上的 18 个真机任务。
- **[RoboTwin](https://github.com/RoboTwin-Platform/RoboTwin)**：通过策略专用 adapter 与转换脚本作为 benchmark 与数据源。[RoboTwin 2.0 Leaderboard](https://robotwin-platform.github.io/leaderboard) 覆盖干净与随机设置下 50 个双臂操作任务。
- **[RMBench](https://github.com/RoboTwin-Platform/RMBench)**：基于 RoboTwin 2.0 的记忆依赖操作 benchmark，九个双臂任务覆盖不同任务记忆复杂度（[论文](https://arxiv.org/abs/2603.01229)，[网站](https://rmbench.github.io/)）。其参考策略已接入为 [Mem-0](policy/Mem_0/README.md)。

**基础设施**

- **[RLinf](https://github.com/RLinf/RLinf)** *（即将支持）*：策略开发与部署工作流的基础设施目标。
- **StarVLA**：基础设施与策略栈；见 [policy/starVLA](policy/starVLA/README.md)。



## 🧭 已接入策略

当前已接入 46 个策略，覆盖 VLA、world-action、模仿学习与记忆增强等家族，另加 [demo_policy](policy/demo_policy/README.md) 作为最小参考 adapter。顶层 adapter 位于 `policy/`；每个策略 README 记录该模型的论文/仓库链接、环境、数据格式、训练入口与 checkpoint 布局。


| Policy                                   | Policy                                               | Policy                                      | Policy                                                     | Policy                                                         | Policy                                            |
| ---------------------------------------- | ---------------------------------------------------- | ------------------------------------------- | ---------------------------------------------------------- | -------------------------------------------------------------- | ------------------------------------------------- |
| [A1](policy/A1/README.md)                | [AHA-WAM](policy/AHA_WAM/README.md)                  | [ABot-M0](policy/Abot_M0/README.md)         | [Being-H05](policy/Being_H05/README.md)                    | [DM0](policy/Dexbotic_DM0/README.md)                           | [Dexora-1B](policy/Dexora_1B/README.md)           |
| [DreamZero](policy/DreamZero/README.md)  | [EventVLA](policy/EventVLA/README.md)                | [FastWAM](policy/FastWAM/README.md)         | [G0](policy/GalaxeaVLA/README.md)                          | [G0.5](policy/G05/README.md)                                   | [GO-1](policy/GO1/README.md)                      |
| [GR00T-N1.7](policy/GR00T_N17/README.md) | [GigaWorld-Policy](policy/GigaWorldPolicy/README.md) | [H-RDT](policy/H_RDT/README.md)             | [Hy-Embodied-0.5-VLA](policy/Hy_Embodied_05_VLA/README.md) | [InternVLA-A1](policy/InternVLA_A1/README.md)                  | [InternVLA-A1.5](policy/InternVLA_A1_5/README.md) |
| [LDA-1B](policy/LDA_1B/README.md)        | [LingBot-VA](policy/LingBot_VA/README.md)            | [LingBot-VLA](policy/LingBot_VLA/README.md) | [Meituan-Robotics-0](policy/Meituan_Robotics_0/README.md)  | [Mem-0](policy/Mem_0/README.md)                                | [MolmoAct2](policy/MolmoAct2/README.md)           |
| [OLA-SEM](policy/OLA_SEM/README.md)      | [OpenDM](policy/OpenDM/README.md)                    | [OpenVLA-OFT](policy/OpenVLA_OFT/README.md) | [OpenWAM](policy/OpenWAM/README.md)                        | [π0](policy/Pi_0/README.md)                                    | [π0.5](policy/Pi_05/README.md)                    |
| [π0-Fast](policy/Pi_0_Fast/README.md)    | [RDT-1B](policy/RDT_1B/README.md)                    | [RISE](policy/RISE/README.md)               | [SmolVLA](policy/SmolVLA/README.md)                        | [Spatial Forcing](policy/Spatial_Forcing/README.md)            | [Spirit v1.5](policy/Spirit_v15/README.md)        |
| [TinyVLA](policy/TinyVLA/README.md)      | [X-VLA](policy/X_VLA/README.md)                      | [X-WAM](policy/X_WAM/README.md)             | [Xiaomi-Robotics-0](policy/Xiaomi_Robotics_0/README.md)    | [Xiaomi-Robotics-1 (XR-1)](policy/Xiaomi_Robotics_1/README.md) | [StarVLA](policy/starVLA/README.md)               |
| [ACT](policy/ACT/README.md) | [DP](policy/DP/README.md) | [Evo-1](policy/Evo_1/README.md) | [InternW0_delta](policy/InternW0_delta/README.md) | [demo_policy](policy/demo_policy/README.md) | |


接入自有策略，或报名排行榜，都通过 PR —— 见 [接入你自己的策略](#-接入你自己的策略)。

## 🧩 框架概览

XPolicyLab 把模型侧依赖与环境侧依赖拆开，两侧保留各自原生栈，可本地或远程运行。一个 adapter 同时服务 benchmark、仿真器与真机。

<div align="center">
<img src="assets/infra.png" alt="XPolicyLab infrastructure" width="100%"/>
<p><em>XPolicyLab 基础设施。一个 adapter 服务 benchmark、仿真器与真机。</em></p>
</div>

```text
Policy environment                         Evaluation / benchmark environment
------------------                         ----------------------------------
policy/<POLICY>/model.py     <---ws--->    env client / simulator / robot
policy server                              environment client
deploy.yml deployment config               benchmark task and observation API
```

典型 adapter 包含：

```text
policy/<POLICY>/
├── README.md                    # policy-specific guide
├── INSTALLATION.md              # optional detailed setup notes
├── __init__.py                  # keeps XPolicyLab.policy.<POLICY> importable
├── install.sh                   # environment setup
├── process_data.sh              # optional data conversion
├── train.sh                     # optional training
├── eval.sh                      # same-machine evaluation
├── setup_eval_policy_server.sh  # policy-side server
├── setup_eval_env_client.sh     # environment-side client
├── deploy.yml                   # deployment config
├── deploy.py                    # deployment loop
└── model.py                     # model adapter, served by the policy server
```

`model.py` 实现面向模型的 API。`deploy.py` 把环境观测桥接到模型 server 调用。最小参考见 [policy/demo_policy](policy/demo_policy/README.md)。

`model.py` 应定义如下形态的 `Model` 类：


| 方法                                    | 契约                                                                   |
| ------------------------------------- | -------------------------------------------------------------------- |
| `__init__(model_cfg)`                 | 从 `deploy.yml` 加载模型配置、checkpoint、processor 与按次运行的覆盖项。               |
| `update_obs(obs)`                     | 用一个观测字典更新模型状态。                                                       |
| `update_obs_batch(obs_list)`          | 用一组观测字典更新模型状态。                                                       |
| `get_action()`                        | 返回一个动作 chunk（动作字典列表）。                                                |
| `get_action_batch(env_idx_list=None)` | 返回与活跃环境索引对齐的 batch 动作 chunk。                                         |
| `reset()`                             | 在评测 episode 之间清空模型侧状态。无参数 —— 需要首帧观测的策略应先 `reset()`，再正常 `update_obs`。 |


策略 server 在 `update_obs` / `update_obs_batch` 之前已解码相机颜色，因此 `obs["vision"][<camera>]["color"]` 始终是图像数组 —— `model.py` 从不解码。

默认策略 server 协议是 websocket（`deploy.yml` 中 `protocol: ws`）；`legacy_tcp` 仅留给尚未迁移的 adapter。传输层替你处理重连、重试、keepalive 与模型加载冷启动 —— 正常 adapter 不必碰它。

<details>
<summary>传输细节与超时调参（仅在评测卡住或断连时需要）</summary>

- **重试是安全的**：每个请求带 `request_id`，client 在重连后复用；server 用缓存回答重复请求，而不会把非幂等调用跑两遍。`timeout` 错误是例外 —— server 可能仍在执行该调用，对该 trial 应视为致命错误，不要重试。
- **Server 重启会中止本次 run**：若重连落到不同 server 进程，client 抛出 `ServerRestartedError`，因为新 server 已丢失模型状态。
- **冷启动**：server 在打开端口前先加载模型，过早连上的 client 会重试（默认预算 15 分钟）。`eval.sh` 也会通过 `wait_for_policy_server.sh` 卡住 client。
- **错误**：client 只看到 `str(exc)`；模型失败的完整 traceback 记在 *policy server* 一侧，先查那里。
- **序列化** 是带 numpy 支持的 msgpack（`torch.Tensor` 自动转换）。三个注意点：`tuple` 会变成 `list`；解码后的 numpy 数组是只读 view（原地修改前先 copy）；int 型 dict key 会变成字符串。

可选 `deploy.yml` 键 —— 省略则用默认值：


| Key                                        | Default | Purpose                                      |
| ------------------------------------------ | ------- | -------------------------------------------- |
| `request_timeout_s`                        | `120.0` | 单次 `update_obs` / `get_action` 超时 —— 推理慢就调大。 |
| `max_connect_attempts`                     | `180`   | server 仍在加载时的冷启动重试次数。                        |
| `connect_retry_delay_s`                    | `5.0`   | 上述重试之间的间隔。                                   |
| `max_connect_seconds`                      | `900.0` | 整个重试循环的墙钟上限；`0` 表示关闭。                        |
| `connect_timeout_s`                        | `30.0`  | 单次连接尝试超时。                                    |
| `handshake_timeout_s`                      | `60.0`  | HELLO 往返超时。                                  |
| `ws_ping_interval_s` / `ws_ping_timeout_s` | `20.0`  | Keepalive ping/pong；`null` 关闭。               |
| `close_timeout_s`                          | `10.0`  | 关闭握手上限。                                      |

</details>




## ⚡ 快速开始

把 XPolicyLab 当作普通 Python 项目克隆，用于 adapter 开发、离线检查、基于已准备数据的训练，或自建环境 client：

```bash
mkdir demo_env
cd demo_env
git clone https://github.com/XPolicyLab/XPolicyLab.git
cd XPolicyLab
pip install -e .
```

开始模型侧开发不需要仿真器：自带下载脚本可拉取已准备好的 RoboDojo 数据 —— 多种仿真导出版本，以及 HDF5 `RoboDojo_real` 真机数据 —— 用于训练与离线调试。若把 `XPolicyLab/` 作为 RoboDojo 仓库内的子包使用，请改用 RoboDojo 自己的数据下载脚本。

下载一个小的 Hugging Face demo 包，并把数据放在 `XPolicyLab/` 旁：

```bash
# From demo_env/XPolicyLab
bash scripts/RoboDojo/download_robodojo_data.sh demo
```

会得到：

```text
demo_env/
├── data/        # demo data, including a small 10-episode HuggingFace bundle
└── XPolicyLab/
```

同一脚本可拉取完整导出 —— `hdf5`、`lerobot_v3.0`、`lerobot_v2.1` 与 `real`（真机 HDF5）—— 各自进独立的 `../data/` 目录。两套 LeRobot 导出由[官方转换器](#官方-lerobot-转换)生成，你也可以按自己的任务子集或分辨率重新生成。

有了这套布局，就可以在接入仿真 backed benchmark 之前，先测数据转换、模型加载、训练脚本与 debug 模式评测。

```bash
export EVAL_ENV_TYPE=debug
cd policy/demo_policy
bash install.sh
bash eval.sh RoboDojo stack_bowls demo arx_x5 joint 0 0 0 base base
```

任意 adapter 的模板相同 —— 替换 `demo_policy` 与参数即可：

```bash
export EVAL_ENV_TYPE=debug
cd policy/<POLICY>
bash eval.sh <bench_name> <task_name> <ckpt_name> <env_cfg_type> <action_type> \
  <seed> <policy_gpu_id> <env_gpu_id> <policy_env_or_uv_path> <eval_env_conda_env>
```

做 RoboDojo 仿真时，把 `XPolicyLab/` 挂在仿真侧的 `env_cfg/`、`scripts/`、`src/eval_client/`、`task/` 目录旁。

## 🔄 通用工作流

多数 adapter 顶层形态相同。有的策略会增加额外参数、吃上游原生数据集，或不支持训练。与本文模板不一致时，以该策略 README 为准。

```bash
cd policy/<POLICY>

# Install the policy environment.
bash install.sh

# Optional: convert or prepare policy-specific data.
bash process_data.sh <bench_name> <ckpt_name> <env_cfg_type> <action_type> [extra_args...]

# Optional: train.
bash train.sh <bench_name> <ckpt_name> <env_cfg_type> <action_type> <seed> <gpu_id> [extra_args...]

# Evaluate on one machine.
bash eval.sh <bench_name> <task_name> <ckpt_name> <env_cfg_type> <action_type> <seed> \
  <policy_gpu_id> <env_gpu_id> <policy_env_or_uv_path> <eval_env_conda_env>
```



### 参数含义

跑 `eval.sh` 时，你主要在回答：**哪个 benchmark 家族**、**现在跑哪个任务**、**加载哪个 checkpoint**、**哪套机器人配置**、**关节还是末端动作**、**哪个 seed**。同一套名字贯穿 `process_data.sh`、`train.sh` 与 `eval.sh`，不必每步改名。


| 参数                             | 通俗含义                         | 示例                                                                |
| ------------------------------ | ---------------------------- | ----------------------------------------------------------------- |
| `bench_name`                   | 本次 run 所属的 benchmark / 数据集家族 | `RoboDojo`、`RoboTwin`                                             |
| `task_name`                    | 环境 client 现在要跑的任务            | `stack_bowls`、`push_T` —— 可与训练时见过的任务不同                            |
| `ckpt_name`                    | 要加载的权重：短昵称、完整 run 目录名，或路径    | `cotrain`、`RoboDojo-cotrain-arx_x5-joint-0`、`checkpoints/my_run/` |
| `env_cfg_type`                 | 机器人 / 相机 / 场景配置键             | `arx_x5`                                                          |
| `action_type`                  | 策略输出的动作空间                    | 通常为 `joint` 或 `ee`                                                |
| `seed`                         | 训练或评测 seed / 布局 id           | `0`、`1`、`2`                                                       |
| `policy_gpu_id` / `env_gpu_id` | 模型与仿真/client 各用哪张 GPU        | `0`、`1`                                                           |
| `policy_env_or_uv_path`        | 策略 server 的 conda 环境名或 uv 路径 | 你的策略侧环境                                                           |
| `eval_env_conda_env`           | 仿真 / 机器人 client 的 conda 环境   | 你的评测侧环境                                                           |


**`ckpt_name` 如何解析。** 通常传训练时用的短昵称，例如 `cotrain`，XPolicyLab 会与其它参数拼成 `checkpoints/RoboDojo-cotrain-arx_x5-joint-0/`。也可以传完整目录名或路径 —— 相对路径相对策略目录解析，绝对路径亦可。部分 adapter 会认 `deploy.yml` 里的显式键（`checkpoint_path`、`model_path` 等）。不确定时查该策略 README。

**一个具体的评测例子：**

```bash
cd policy/AHA_WAM
bash eval.sh RoboDojo stack_bowls cotrain arx_x5 joint 0 0 0 aha_wam robodojo
# loads checkpoints/RoboDojo-cotrain-arx_x5-joint-0/ and evaluates on stack_bowls
```



## 🔌 部署流程

评测时，策略 server 与环境 client 通过 websocket 通信。这种拆分让你可以把 Isaac Sim / 机器人驱动放在一台机器上，把沉重的 VLA 放在另一台。

同机评测用 `eval.sh` 即可 —— 它会启动 server、跑 client，并在结束后清理。

跨机部署时，在 GPU 机器上启动策略 server，并绑定 `0.0.0.0`，以便其它机器可达。Client 连的是策略机的真实 IP，不是 `0.0.0.0`。

```bash
cd policy/<POLICY>
bash setup_eval_policy_server.sh \
  <bench_name> <task_name> <ckpt_name> <env_cfg_type> <action_type> <seed> \
  <policy_gpu_id> <policy_env_or_uv_path> <policy_server_port> 0.0.0.0
```

然后在仿真或机器人机器上启动环境 client：

```bash
cd policy/<POLICY>
bash setup_eval_env_client.sh \
  <bench_name> <task_name> <ckpt_name> <env_cfg_type> <action_type> <seed> \
  <env_gpu_id> <eval_env_conda_env> <additional_info> \
  <policy_server_port> <policy_server_ip>
```

`<additional_info>` 是转发给环境 client 的逗号分隔 `key=value` 字符串。`eval.sh` 会自动建成 `ckpt_name=<ckpt_name>,action_type=<action_type>`，对多数 adapter 这就是正确默认。

`EVAL_ENV_TYPE` 选择环境侧后端：

- 未设置或 `sim`：真实仿真 backed 评测（集成已安装时）。
- `debug`：离线接线检查 —— 无 Isaac、无真机，只验 shape 与 IO。
- `real`：真机 client 路径（硬件集成存在时）。



## 📐 标准数据格式

XPolicyLab 标准化 adapter、转换器与环境 client 之间传递的观测与轨迹字典。个别策略可将该标准格式再转成上游原生格式。

### 图像解码只能走 `decode_image_bit`

> **始终通过 `decode_image_bit` 解码，并通过 `encode_image_bit` 编码。** 二者都在 `XPolicyLab.utils.process_data`。自行解码图像 bits 不受支持：它们有两种字节格式，手写解码器对一种正确、对另一种会通道颠倒。`decode_image_bit` 会区分二者，并对每一版数据返回 RGB，因此其输出不需要再做通道交换。离线转换与训练必须走它。运行时观测已解码，所以 `model.py` 完全不应解码。

存储的图像 bits 有两种格式，解码后都是 RGB：


| 格式           | 写入方式                                                     | 标准解码器看到的结果                                    |
| ------------ | -------------------------------------------------------- | --------------------------------------------- |
| **legacy**   | 把 RGB 数组直接交给 `cv2.imencode`（它按 BGR 读入）                   | 红蓝对调 —— 字节相对 JPEG 标准通道颠倒，`cv2.imdecode` 再颠倒回来 |
| **standard** | `encode_image_bit`：先转 BGR，并在 JPEG `COM` 段写入载荷 `XPL-RGB1` | 颜色正确                                          |


Legacy 数据永不迁移 —— JPEG 无法无损换通道 —— 因此两种格式会长期共存，甚至出现在同一次训练中。标记写在 buffer 内而非文件属性上，使单个 buffer 自描述；`COM` 是标准段，所有解码器都会跳过，只占 12 字节且不破坏兼容。若必须在无 OpenCV 时读取这些 buffer：PIL 会把标记暴露为 `Image.open(...).info["comment"]`，检查是否为 `b"XPL-RGB1"`，缺失时自行反转通道。不能跳过检查 —— 只在新数据上测过的 PIL loader 看起来完美，却会悄悄毁掉旧 episode。

所有位姿值为 `[x, y, z, qw, qx, qy, qz]`。图像端到端为 RGB —— `decode_image_bit` 返回 RGB，管线其它处不做通道转换。注意一个命名差异：运行时观测里相机外参是 `extrinsics_matrix`，轨迹文件里是 `extrinsic_matrix`。

<details>
<summary>Observation 数据格式</summary>

```text
Observation Data Format
├── data_format_version                        string, optional
├── instruction / instructions                 string or list[str]
├── env_idx                                    int, optional for batched eval
├── additional_info/
│   └── frequency                              int, optional
├── vision/
│   ├── cam_head/
│   │   ├── color                              (H, W, 3) RGB, decoded by the server
│   │   ├── depth                              (H, W) or (H, W, 1), optional
│   │   ├── intrinsic_matrix                   (3, 3), optional
│   │   ├── extrinsics_matrix                  (4, 4), optional
│   │   └── shape                              (2,) or (3,), optional
│   ├── cam_left_wrist/                        optional
│   ├── cam_right_wrist/                       optional
│   ├── cam_wrist/                             optional for single-arm robots
│   └── cam_third_view/                        optional
└── state/
    ├── left_arm_joint_state                   (DOF,), optional
    ├── left_ee_joint_state                    (EEF_DOF,), optional
    ├── left_ee_pose                           (7,), optional
    ├── left_tcp_pose                          (7,), optional
    ├── left_delta_ee_pose                     (7,), optional
    ├── right_arm_joint_state                  (DOF,), optional
    ├── right_ee_joint_state                   (EEF_DOF,), optional
    ├── right_ee_pose                          (7,), optional
    ├── right_tcp_pose                         (7,), optional
    ├── right_delta_ee_pose                    (7,), optional
    ├── arm_joint_state                        (DOF,), optional for single-arm robots
    ├── ee_joint_state                         (EEF_DOF,), optional for single-arm robots
    ├── ee_pose                                (7,), optional for single-arm robots
    ├── tcp_pose                               (7,), optional for single-arm robots
    ├── delta_ee_pose                          (7,), optional for single-arm robots
    └── mobile/                                optional
        ├── base_pose                          (7,)
        └── base_twist                         (6,), [vx, vy, vz, wx, wy, wz]
```

</details>

<details>
<summary>Trajectory 数据格式</summary>

```text
Trajectory Data Format
├── data_format_version                        string, e.g. "v1.0"
├── instruction / instructions                 string, or JSON-serialized list[str]
├── subtasks                                   JSON-serialized annotations, optional
├── additional_info/
│   └── frequency                              int
├── vision/
│   ├── cam_head/
│   │   ├── colors                             (T, H, W, 3), uint8 RGB or encoded stream
│   │   ├── depths                             (T, H, W) or (T, H, W, 1), optional
│   │   ├── intrinsic_matrix                   (3, 3) or (T, 3, 3), optional
│   │   ├── extrinsic_matrix                   (4, 4) or (T, 4, 4), optional
│   │   └── shape                              (2,) or (3,), optional
│   ├── cam_left_wrist/                        optional
│   ├── cam_right_wrist/                       optional
│   ├── cam_wrist/                             optional for single-arm robots
│   └── cam_third_view/                        optional
├── action/                                    action targets, same key naming as state/ below
└── state/
    ├── left_arm_joint_states                  (T, DOF), optional
    ├── left_ee_joint_states                   (T, EEF_DOF), optional
    ├── left_ee_poses                          (T, 7), optional
    ├── left_tcp_poses                         (T, 7), optional
    ├── left_delta_ee_poses                    (T, 7), optional
    ├── right_arm_joint_states                 (T, DOF), optional
    ├── right_ee_joint_states                  (T, EEF_DOF), optional
    ├── right_ee_poses                         (T, 7), optional
    ├── right_tcp_poses                        (T, 7), optional
    ├── right_delta_ee_poses                   (T, 7), optional
    ├── arm_joint_states                       (T, DOF), optional for single-arm robots
    ├── ee_joint_states                        (T, EEF_DOF), optional for single-arm robots
    ├── ee_poses                               (T, 7), optional for single-arm robots
    ├── tcp_poses                              (T, 7), optional for single-arm robots
    ├── delta_ee_poses                         (T, 7), optional for single-arm robots
    └── mobile/                                optional
        ├── base_poses                         (T, 7)
        └── base_twists                        (T, 6), [vx, vy, vz, wx, wy, wz]
```

</details>

常用转换辅助函数：

```python
from XPolicyLab.utils.load_file import load_hdf5
from XPolicyLab.utils.process_data import (
    decode_image_bit,
    encode_image_bit,
    get_robot_action_dim_info,
)
```

`decode_image_bit` 与 `encode_image_bit` 是轨迹图像 bits 唯一受支持的编解码对（见[上文](#图像解码只能走-decode_image_bit)）。二者对输入处理对称 —— 单帧或序列，以及轨迹文件使用的各种容器 —— 已转换的值会原样通过。`get_robot_action_dim_info(env_cfg_type)` 返回机器人特定的 `arm_dim` 与 `ee_dim` 列表，adapter 不必硬编码动作维度。

[CONTRIBUTING.md](CONTRIBUTING.md#modelpy) 说明了 RGB 例外，以及新机器人如何在两份 `_robot_info.json` 中注册。

### 官方 LeRobot 转换

许多策略在 LeRobot 数据集上训练，而不是上面的轨迹格式。`scripts/transform_lerobot_v21_format.py` 与 `scripts/transform_lerobot_v30_format.py` 是官方转换器 —— 各对应一个 LeRobot 数据集版本，输出相同的键。

<details>
<summary>LeRobot 格式</summary>

| Key | Shape | Content |
| --- | --- | --- |
| `observation.state` | `(D,)` float32 | `left_arm_joint_states` + `left_ee_joint_states` + `right_arm_joint_states` + `right_ee_joint_states`，按该顺序拼接 |
| `action` | `(D,)` float32 | 相同布局，来自轨迹的 `action/` 组 |
| `observation.images.cam_high` | `(3, H, W)` video | `vision/cam_head/colors` |
| `observation.images.cam_left_wrist` | `(3, H, W)` video | `vision/cam_left_wrist/colors` |
| `observation.images.cam_right_wrist` | `(3, H, W)` video | `vision/cam_right_wrist/colors` |

</details>

为某个 benchmark 发布的现成 LeRobot 导出 —— 例如[快速开始](#-快速开始)中的 `lerobot_v2.1` / `lerobot_v3.0` —— 即按此方式生成，因此消费它们的策略无需再做自己的转换步骤。

> **在 LeRobot 数据上训练的策略必须在自己的 README 的 `Data Processing` 中声明**：数据集版本，以及键是否与上文一致。若一致，写明转换器名称，并说明 `process_data.sh` 是缺失还是仅做链接与规范化。若不一致 —— 上游原生布局、额外键、latent 树、不同相机名 —— 写明差异及如何产出该布局。[policy/RISE](policy/RISE/README.md) 与 [policy/AHA_WAM](policy/AHA_WAM/README.md) 是第一种的实例，[policy/LingBot_VA](policy/LingBot_VA/README.md) 是第二种。

<details>
<summary>运行转换</summary>

两个脚本都接收 `<bench_name>.<task_name>.<env_cfg_type>` glob，从 `../data/` 读轨迹、从 `../env_cfg/` 读机器人维度，并把所有匹配目标合并为一个数据集，写到 `HF_LEROBOT_HOME/<repo_id>`。在旁置这两个目录的 checkout 仓库根目录运行（[快速开始](#-快速开始)）。

```bash
# One robot, every task under it.
python scripts/transform_lerobot_v30_format.py "<bench_name>.*.<env_cfg_type>" --repo_id my_dataset

# Several robots merged, 50 episodes per task/env, downscaled.
# Without --resolution the target size comes from the first source frame.
python scripts/transform_lerobot_v21_format.py "<bench_name>.*.*" \
  --max_episode 50 --resolution 240x320
```

- **`D` 是按最宽维度 padding，不是按机器人。** 每条臂会零填充到匹配目标中最宽的维度，因此一个数据集可混多种机器人；`robot_type` 为 `unified_robot`，电机名为 `left_joint_<i>` / `right_joint_<i>`。
- **三个相机键始终存在。** 源数据缺失的相机会用黑帧填充，使特征跨机器人稳定。
- **图像为 RGB**，经 `decode_image_bit` 解码后不再交换通道（见[上文](#图像解码只能走-decode_image_bit)）。
- **仅支持关节空间双臂。** 二者读取 `*_arm_joint_states` / `*_ee_joint_states`，对仅含位姿或单臂键的轨迹会失败。
- 两个版本除数据集版本外，只在编码吞吐上不同：v3.0 用 8 个 worker 写图，并以 CRF 18 流式写视频。

</details>



## 💾 数据与 Checkpoint

训练与数据准备通常按可预测方式命名，以便评测无需猜测即可找到：

```text
<bench_name>-<ckpt_name>-<env_cfg_type>-<action_type>
<bench_name>-<ckpt_name>-<env_cfg_type>-<action_type>-<seed>
```

因此若训练时用了 `bench_name=RoboDojo`、`ckpt_name=cotrain`、`env_cfg_type=arx_x5`、`action_type=joint`、`seed=0`，run 会落在 `checkpoints/RoboDojo-cotrain-arx_x5-joint-0/`。评测时 `ckpt_name` 如何映射回这些目录，见[通用工作流](#-通用工作流)。

策略也可能使用上游原生布局，或在 `deploy.yml` 中写显式路径。在假定命名约定前先查策略 README。本地小数据集试用见[快速开始](#-快速开始)。

## 🤝 接入你自己的策略

欢迎社区策略 —— 开 PR 添加 `policy/<POLICY>/`。进入官方 [RoboDojo](https://robodojo-benchmark.com/LeaderBoard) 与 [RoboTwin](https://robotwin-platform.github.io/leaderboard) 排行榜同样**需要** PR，并附上可复现结果的 checkpoint。

如何搭 adapter、提交必须包含什么、PR 前要跑哪些检查、以及 PR 模板，都在 [CONTRIBUTING.md](CONTRIBUTING.md)。从 [policy/demo_policy](policy/demo_policy/README.md) 开始，或运行 `bash scripts/create_policy.sh <POLICY_NAME>`。

## 🤖 编程 Agent

[.agents/skills](.agents/skills) 下有两个 skill，Cursor、Claude Code 和 Codex 会通过 `.cursor/skills` 与 `.claude/skills` 的 symlink 自动加载。[AGENTS.md](AGENTS.md) 是常驻规则。

- `xpolicylab-model-integration` —— 构建 adapter。一句 `Integrate <POLICY_NAME> into XPolicyLab` 即可。
- `xpolicylab-adapter-check` —— PR 前审计（`Check policy/<POLICY_NAME>`）。

不支持这些机制时要粘贴的清单在 [CONTRIBUTING.md](CONTRIBUTING.md#using-a-coding-agent)。

## 📝 引用

若 XPolicyLab 对你的研究有帮助，请引用：

```bibtex
@article{community2026xpolicylab,
  title={{XPolicyLab}: A Unified Standard and Open Ecosystem for Robot Policy Evaluation and Deployment},
  author={Community, XPolicyLab and Chen, Tianxing and Chen, Yue and Nian, Tian and Cai, Zijian and Chen, Guangyu and Lin, Wenwei and Liang, Qiwei and Xiang, Peicheng and Su, Kailun and others},
  journal={arXiv preprint arXiv:2608.09892},
  year={2026}
}
```



## 📬 联系方式

项目负责人 Tianxing Chen：[chentianxing2002@gmail.com](mailto:chentianxing2002@gmail.com)

由 **MMLab@HKU** 与 **THU** 主导的协作开源项目。

**Core Lead Authors**：Tianxing Chen, Yue Chen, Tian Nian, Zijian Cai, Guangyu Chen, Wenwei Lin, Qiwei Liang.

完整贡献者列表 —— 覆盖每个已接入策略 —— 见[项目网站](https://xpolicylab.github.io/)。
