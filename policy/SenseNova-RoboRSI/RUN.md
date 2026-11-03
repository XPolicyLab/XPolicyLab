# SenseNova-RoboRSI 主版本运行说明

RC1 适配器：每卡 1 个共享 VLA、每卡 10 路的真实评测验收尚未完成。本 PR 不声明并发测试通过，也未包含十路评测编排器或并发验收结果。

本 PR 在 XPolicyLab 仓库中提供 `adaptive_2_50` 策略适配器。相对仓库根目录的位置为：

```text
policy/SenseNova-RoboRSI
```

该版本保留 `adaptive_2_50` 源码基线，运行时默认允许批准 5–50 步。策略服务启动前设置 `ROBORSI_ADAPTIVE_MIN_STEPS`（2–50整数，未设置为5）可改变下限，设置为2可恢复之前的2–50范围；上限及VLA预测长度仍为50。省略批准步数时使用 `max(5, 下限)`，显式低于下限的值会被拒绝。实际范围写入配置、提示词和proposal；恢复中断任务需使用原范围。保留原始 agent 执行流程、真实 layout 透传、跨进程动作回放恢复和完整轨迹。本 PR 的修改限于此策略目录。

## 1. 工作目录与运行前提

先将 XPolicyLab 仓库部署到完整官方 RoboDojo 工作区，在实际部署后的策略目录中设置本文共用变量：

```bash
POLICY_DIR="$(pwd -P)"  # 当前目录须为实际部署后的 XPolicyLab/policy/SenseNova-RoboRSI
WORKSPACE="$(cd "$POLICY_DIR/../../.." && pwd)"
SIM_ENV=RoboDojo
cd "$POLICY_DIR"
```

启动器根据策略文件的实际位置确定 RoboDojo 根目录，即 `XPolicyLab/` 的父目录。设置 shell 变量 `WORKSPACE` 只是方便执行命令，不会改变启动器选择的根目录。

XPolicyLab 仓库提供共享服务、工具和策略源码；官方模拟器、Assets 与任务配置由外层 RoboDojo 工作区提供，策略虚拟环境和模型权重由安装流程准备。安装必须在最终物理路径完成，迁移源码后在新位置真实安装并生成安装收据，不能复用旧位置的虚拟环境或收据来宣称安装完成。

目录关系应为：

```text
WORKSPACE/
├── env_cfg/                  # 官方配置
├── env/                      # 官方模拟器环境
├── task/                     # 官方任务
├── scripts/eval_policy.sh    # 官方启动脚本
├── src/eval_client/main.py   # 官方评测入口
├── utils/load_file.py        # 官方工具
├── ...                      # 其余官方资源和 Assets
└── XPolicyLab/policy/SenseNova-RoboRSI/
```

可先检查几个必需入口；这些检查不代替完整环境安装：

```bash
test -f "$WORKSPACE/scripts/eval_policy.sh"
test -f "$WORKSPACE/src/eval_client/main.py"
test -f "$WORKSPACE/env_cfg/arx_x5.yml"
test -f "$WORKSPACE/env/seed_manager/seed_manager.py"
test -f "$WORKSPACE/utils/load_file.py"
```

运行需要完整匹配的官方仿真工作区。当前适配器使用 `RoboDojo`、`arx_x5`、`joint`，一个 policy server 对应一个环境。

## 2. 安装与运行时 API 配置

需要 Linux x86_64、NVIDIA CUDA 兼容驱动、Bash、`setsid`、编译工具、namespace / memfd 支持，以及官方模拟器 Conda 环境。详细依赖与锁定版本见 [INSTALL.md](INSTALL.md)。

```bash
cd "$POLICY_DIR"
bash bootstrap.sh --check
bash install.sh
```

`bootstrap.sh --check` 检查固定构建输入；`install.sh` 构建策略、模型工具和 π0.5 环境，并下载固定 CLI、ICL、tokenizer、checkpoint。源码副本不包含这些安装产物。模拟器继续使用官方 Conda 环境；若环境名不是 `RoboDojo`，修改 `SIM_ENV`。

运行前在当前 shell 设置服务地址、模型和 key。下面地址与模型仅为占位符，请替换为实际的 Responses 兼容服务地址和模型；不要把真实配置或密钥写入源码：

```bash
export ROBORSI_CODEX_BASE_URL='https://example.invalid/v1'
export ROBORSI_CODEX_MODEL='your-model'
read -r -s -p 'CODEX_API_KEY: ' CODEX_API_KEY
printf '\n'
export CODEX_API_KEY
```

`ROBORSI_CODEX_BASE_URL` 与 `ROBORSI_CODEX_MODEL` 均为必填环境变量，没有默认值；缺失或为空时评测会提前报错。推理强度仍为 `max`，API key 仍由 `CODEX_API_KEY` 提供。使用与预期实验一致的服务与模型；切换模型会改变实验条件。安装与离线检查不需要这些运行时配置。

模型请求的 `X-Trajectory-Id` 标识同一个逻辑 episode。新 episode 生成一次 ID；策略每次启动 Codex CLI 时重新注入该 ID，因此 Codex 新建 session、resume，以及通过 `hybrid_resume` 跨进程恢复时都保持相同的 ID。CLI 本身不会持久化这个自定义请求头。服务端可按该请求头归集完整模型交互。

`trajectory.json` 和 `episode-owner.json` 中的 `trajectory_id` 是这个稳定 ID；`episode_id` 与输出目录名仍标识独立的执行尝试，因此一次 episode 恢复后的多个目录可以共享同一个 `trajectory_id`。恢复请求记录并校验原轨迹 ID。旧记录可从原恢复链、runtime 父目录的 `trajectory.json.episode_id` 或归档 `manifest.json.episode_id` 继承；缺少可核实 ID 的恢复会报错，避免将同一轨迹静默拆成新 ID。

## 3. 单任务、单 layout

以下命令明确运行 `align_blocks`、eval seed=0、layout=0，policy/VLA 和 simulator 都使用 GPU 0。新实验使用独立 run ID 与持久输出目录：

```bash
cd "$POLICY_DIR"
RUN_TAG="main-align_blocks-s0-l0-$(date -u +%Y%m%dT%H%M%SZ)"
RUN_ROOT="$POLICY_DIR/outputs/$RUN_TAG"
mkdir -p "$RUN_ROOT/state" "$RUN_ROOT/tmp" "$RUN_ROOT/trajectories"

ROBODOJO_RUN_ID="$RUN_TAG" \
ROBORSI_NATIVE_LAYOUT_ID=0 \
ROBORSI_REQUIRE_NATIVE_IDENTITY=1 \
ROBORSI_RUNTIME_STATE_DIR="$RUN_ROOT/state" \
ROBORSI_TRAJECTORY_DIR="$RUN_ROOT/trajectories" \
TMPDIR="$RUN_ROOT/tmp" \
EVAL_ENV_TYPE=sim EVAL_NUM=1 \
bash eval.sh RoboDojo align_blocks roborsi-codex-v1 arx_x5 joint 0 0 0 uv "$SIM_ENV"
```

`eval.sh` 接收且只接收十个位置参数：

| 顺序 | 参数 | 示例 |
| --- | --- | --- |
| 1 | benchmark | `RoboDojo` |
| 2 | task | `align_blocks` |
| 3 | agent label | `roborsi-codex-v1`，不是 checkpoint 路径 |
| 4 | environment config | `arx_x5` |
| 5 | action type | `joint` |
| 6 | eval seed | `0`，与 layout ID 分开 |
| 7 | policy GPU | `0`，policy/VLA 使用的卡 |
| 8 | simulator GPU | `0`，可与 policy 分开 |
| 9 | policy environment | `uv` 使用本目录 `.venv/bin/python`；也支持环境或解释器路径 |
| 10 | simulator Conda environment | `$SIM_ENV` |

实际启动链是 `eval.sh` → `setup_eval_env_client.sh` → 官方 `scripts/eval_policy.sh` → scoped `native_entry.py` → 官方 `src/eval_client/main.py`。wrapper 增加真实场景选择与收据校验，评分和 reward 仍由官方评测器执行。

`ROBORSI_NATIVE_LAYOUT_ID` 选择真实 pending layout，并验证 reset 的场景 SHA-256。该 run ID 下已完成或 abandoned 的 layout 不会被重写为新运行。新实验使用新 run ID，恢复使用原 run ID。

显式选择 layout 时，wrapper 内部会把 `EVAL_NUM` 设为 `native`，再用只含目标 layout 的队列限定一场，避免原有累计计数阻止恢复；因此仍是单场执行。

## 4. 原生场数、动作步数与预算

`EVAL_NUM=1` 表示该任务最多运行一个有效 episode。保留官方任务配置的原生场数时使用 `EVAL_NUM=native`，同时取消显式 layout：

```bash
cd "$POLICY_DIR"
FULL_TAG="main-native-align_blocks-s0-$(date -u +%Y%m%dT%H%M%SZ)"
FULL_ROOT="$POLICY_DIR/outputs/$FULL_TAG"
mkdir -p "$FULL_ROOT/state" "$FULL_ROOT/tmp" "$FULL_ROOT/trajectories"
unset ROBORSI_NATIVE_LAYOUT_ID ROBORSI_NATIVE_SCENE_SHA256

ROBODOJO_RUN_ID="$FULL_TAG" \
ROBORSI_REQUIRE_NATIVE_IDENTITY=1 \
ROBORSI_RUNTIME_STATE_DIR="$FULL_ROOT/state" \
ROBORSI_TRAJECTORY_DIR="$FULL_ROOT/trajectories" \
TMPDIR="$FULL_ROOT/tmp" \
EVAL_ENV_TYPE=sim EVAL_NUM=native \
bash eval.sh RoboDojo align_blocks roborsi-codex-v1 arx_x5 joint 0 0 0 uv "$SIM_ENV"
```

具体原生场数以目标官方工作区的任务配置为准；整数 `EVAL_NUM` 也受原生场数上限约束。42 个任务各测一个 layout 是 42 个单场测试，不等于完整官方多场评测。本目录的 `eval.sh` 每次启动一个任务；逐任务运行时为各任务分配独立 run ID 和输出目录。

| 限制 | 当前代码的含义 |
| --- | --- |
| 每任务动作步数 | fresh config 的 `review_window_steps`；当前 42 项范围 200–1900，`align_blocks` 为 200；必须与原生收据的 `step_limit` 相等 |
| 实际已完成步数 | 以模拟器 `take_action_cnt` 和 action ACK 为准；VLA 候选不算已执行 |
| 自适应动作 | VLA proposal为50步；批准范围默认5–50，环境变量 `ROBORSI_ADAPTIVE_MIN_STEPS` 可设置下限，省略长度时为 `max(5, 下限)` |
| agent 累计时间 | `episode_seconds=86400`，是原始 adaptive agent 的 wall-time 预算，不是官方通用时间上限 |
| 技术恢复 | 初次运行之外最多 4 次恢复；准备恢复时剩余不足 300 秒会拒绝 |
| agent 轮数/token | `agent.yaml` 的 `max_turns=null`、`max_tokens=null`，未另加固定上限 |
| 单次 RPC | `deploy.yml` 的 `request_timeout_s=3660（清理等待3600秒，预留60秒回执余量）`，不等于整个任务预算 |

没有用于替代每任务原生限制的通用 `MAX_STEPS` 环境变量。修改任务预算、proposal 或 agent 配置会改变实验条件。

## 5. 轨迹和官方结果

未指定 `ROBORSI_TRAJECTORY_DIR` 时，默认轨迹位于策略目录下：

```text
outputs/trajectories/<task>/seed-<eval-seed>/<episode-id>/
```

单场示例使用 `$RUN_ROOT/trajectories/`。每个 episode 保留 controller、观测 RGB 图像、实际 joint actions/ACK/反馈、VLA proposals/reviews、请求/响应和原生身份收据。完整 owned runtime 位于该 episode 的 `runtime-*/` 下，包含 `agent/workspace`、`agent/controller`、模型 sessions 与 segment 日志。

`reset`、`close` 和取消运行不会删除这些轨迹。复制时保留整个 episode 及符号链接关系；只复制 `runtime-*` 会漏掉其同级持久 controller。可选 `ROBORSI_TRACE_ROOT` 为附加归档，不替代完整 episode；该归档排除凭据文件名并保留外链文本。

官方评分、视频和原生续跑进度保存在 `$WORKSPACE/eval_result/` 下。核对原 run ID 和 episode `details` 的 `layout_id`、`success`、`score`：成功 episode 分数为 1，失败也可能有过程分；顶层 `score` 为有效 episode 平均分乘 100，`success_rate` 为成功数除以有效 episode 数。`unstable` 另计并排除该分母；`score=0` 是有效评分，没有评分的技术中断应保留为未评分。

## 6. 断进程后恢复

官方自动 restart 保留原生 layout 队列，不会自动生成本适配器的动作回放恢复请求。跨进程恢复要求：原生 task/seed/run ID/layout/scene SHA 收据、连续已完成 actions、已用时间和原 workspace。缺少真实场景身份的旧轨迹会被拒绝。

先确认旧 case 的写进程已经停止。以下仍以原任务 `align_blocks`、eval seed=0 为例，将 `SOURCE_RUNTIME` 改成真实目录，不能传外层 episode 目录：

```bash
SOURCE_RUNTIME=/absolute/trajectory/align_blocks/seed-0/EPISODE_ID/runtime-XXXX
RESUME_ROOT="$POLICY_DIR/outputs/resume-$(date -u +%Y%m%dT%H%M%SZ)"
REQUEST="$RESUME_ROOT/resume-request.json"
mkdir -p "$RESUME_ROOT/state" "$RESUME_ROOT/tmp" "$RESUME_ROOT/trajectories"

cd "$WORKSPACE"
"$POLICY_DIR/.venv/bin/python" -m XPolicyLab.policy.SenseNova-RoboRSI.hybrid_resume \
  --source "$SOURCE_RUNTIME" \
  --output "$REQUEST"
```

必须使用模块方式准备请求，不能直接执行 `python hybrid_resume.py`。可选 `--native-progress /absolute/eval_result/.../_resume_RUN_ID.json` 用来验证官方进度，不会推断或改写 layout。

先校验再启动，位置参数中的 task 和 seed 必须与原 episode 相同：

```bash
cd "$POLICY_DIR"
unset ROBORSI_RESUME_MANIFEST
"$POLICY_DIR/.venv/bin/python" resume_eval.py --request "$REQUEST" --check-only -- \
  RoboDojo align_blocks roborsi-codex-v1 arx_x5 joint 0 0 0 uv "$SIM_ENV"

ROBORSI_RUNTIME_STATE_DIR="$RESUME_ROOT/state" \
ROBORSI_TRAJECTORY_DIR="$RESUME_ROOT/trajectories" \
TMPDIR="$RESUME_ROOT/tmp" \
EVAL_NUM=1 \
"$POLICY_DIR/.venv/bin/python" resume_eval.py --request "$REQUEST" -- \
  RoboDojo align_blocks roborsi-codex-v1 arx_x5 joint 0 0 0 uv "$SIM_ENV"
```

`resume_eval.py` 自动绑定原 run ID、task、eval seed、layout、scene SHA 并启用原生身份校验。若 shell 残留与请求冲突的这些环境变量，先取消对应旧值，不要修改请求来迎合旧值。

恢复在真实 layout 上从物理 step 0 重放已 ACK 的 joint commands，校验位姿与 head image 后沿用 workspace，用原始 recovery prompt 开新的模型会话。旧 pending proposal 被丢弃；保留 session 不等于恢复模型隐藏状态。已用时间和重放时间都扣除累计预算，任务逻辑进度不会清零。

`--check-only` 不 claim 请求；实际 controller 启动后创建 `.claimed`。每个请求只启动一次，不删除该标记重用。再次技术失败时保留现场，从该次 runtime 准备新的有界恢复请求。layout 和回放校验不保证 VLA RNG、模型输出或最终分数逐条相同。

## 7. 常用变量与并发

| 变量 | 用途 |
| --- | --- |
| `CODEX_API_KEY` | 必需，仅运行时设置 |
| `ROBORSI_CODEX_BASE_URL` | 必需，Responses 兼容服务地址，无默认值 |
| `ROBORSI_CODEX_MODEL` | 必需，服务提供的模型名称，无默认值 |
| `EVAL_ENV_TYPE=sim` | 官方模拟器；debug 不产生真实官方评分 |
| `EVAL_NUM=1/native` | 单场或保留原生场数 |
| `ROBODOJO_RUN_ID` | 新实验唯一 ID；恢复由请求绑定原 ID |
| `ROBORSI_NATIVE_LAYOUT_ID` | 可选，选择真实 pending layout；不设则沿用官方队列 |
| `ROBORSI_REQUIRE_NATIVE_IDENTITY=1` | 强制核验原生场景身份 |
| `ROBORSI_TRAJECTORY_DIR` | 轨迹根；相对路径按策略目录解析，推荐绝对路径 |
| `ROBORSI_RUNTIME_STATE_DIR` | 绝对路径，隔离启动/预检状态，本身不代表 episode 恢复 |
| `TMPDIR` | 该 case 的临时目录，先创建 |
| `ROBORSI_SHARED_VLA_PORT` | 可选，复用操作者确认匹配的本节点 π0.5 服务，默认由策略管理独立服务 |
| `ROBORSI_TRACE_ROOT` | 可选附加归档根 |
| `ROBORSI_EXECUTION_BUDGET_FILE` | 可选，controller 写入预算收据 |

并发 case 使用独立 run ID、state/tmp/轨迹目录和 GPU 分配。默认 VLA 动态选择端口，无需写死端口。当前共享模式检查端口可连接，不会自动核验服务的 checkpoint；需自行确认是匹配的本节点 π0.5 服务。关闭 policy 不会终止共享服务，也不会恢复历史 RNG；不能把 WAM 服务端口当作 π0.5 服务复用。

`eval.sh` 等待 policy server 的启动超时为 1200 秒。这是启动阶段的等待期限，不会延长已建立的 agent 执行预算。

## 8. 部署入口与并发验证边界

本包保留的 `install.sh` 是直接安装依赖的入口；`bash bootstrap.sh --check` 只检查构建输入。安装阶段不读取 API key，会下载锁定的公共依赖、ICL、CLI、tokenizer 和权重。安装不会补齐官方模拟器、完整 Assets 或所需的匹配 Kit 扩展，须先准备这些环境。

此策略目录包含可选的 `environment/run.py` namespace 启动器，但本 PR 不分发部署 plan。使用该启动器时须显式传入针对当前部署验证过的 `--plan /absolute/current-plan.json`。常规 `install.sh` / `eval.sh` 直接使用前文的官方工作区布局。源码提交不代表安装或评测就绪。

共享 VLA 是显式选择的运行方式：先启动匹配的本节点 π0.5 服务，再通过 `ROBORSI_SHARED_VLA_PORT` 将端口传给各路独立 policy；每路使用独立 run ID、state、tmp、轨迹及模型会话。未设置该变量时，每路默认管理独立 VLA。每卡 1 个共享 VLA、每卡 10 路评测的真实验收仍待完成，本 PR 未包含十路编排器或相关验收结果。需要在目标工作区验证服务、模拟器入口和 GPU 分配，才能声明对应的并发能力或任务成功率。
