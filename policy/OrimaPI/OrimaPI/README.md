# OrimaPI

正式 RoboDojo 评测统一使用 `configs/robodojo.yaml`：公共仿真环境、API 配置及两种顶层策略只维护一份。pi0.5 与 code-policy 通过启动参数切换，任务专属参数放在各自的 policy / 子 skill 下。

当前 GPT 默认直连 Copilot，账户凭据独立保存在 `configs/copilot.credentials.json`。
配置方式见 [Copilot API](doc/copilot-api.md)；无需修改当前 Codex 会话或本地 bridge。

## 启动命令

在仓库根目录执行，固定使用 pi05 测试 `stack_bowls` 的一个回合，不接入 GPT，不需要 API Key。

```bash
OMNI_KIT_ACCEPT_EULA=YES \
XLA_PYTHON_CLIENT_PREALLOCATE=false \
.venv/bin/python -m orimapi eval \
  --config configs/robodojo.yaml \
  --task stack_bowls --episodes 1 \
  --mode direct --fixed-policy pi05
```

`--mode direct` 和 `--fixed-policy pi05` 必须同时指定，才能跳过高层模型调用。资源路径从配置文件读取，不通过环境变量指定。

## 自有 Code Skill 冒烟命令

```bash
OMNI_KIT_ACCEPT_EULA=YES LIVESTREAM=0 \
.venv/bin/python -m orimapi eval \
  --config configs/code_smoke.yaml \
  --task deposit_coin --episodes 1 \
  --mode direct --fixed-policy code-policy --exec-horizon 1 \
  --output-root runs/code-smoke
```

`gripper-smoke` 是自有的仿真诊断程序：保持初始机械臂姿态，有限开合夹爪，随后保持直到 RoboDojo 官方终止。不调用 GPT、不加载 pi0.5，也不尝试完成投币；预期任务失败，成功标准是完整记录官方结果，而非进程退出码。仅用于仿真，不用于真实机器人。

| 参数 / 配置 | 说明 |
| --- | --- |
| `--config configs/code_smoke.yaml` | 独立 code skill 配置；仿真解释器和资源路径在 YAML 中指定，不依赖外部项目。 |
| `policies.code-policy.code_path` | 此诊断配置直接加载自有程序，相对于配置文件目录；不启用 GPT 子 skill 选择。 |
| `code_parameters.close_steps` / `open_steps` | 关闭和重新打开夹爪的阶段步数，默认各 20；按实际观测步数推进。 |
| `code_parameters.max_gripper_step` | 每步命令相对当前夹爪观测的最大变化，默认 0.1，取值 `(0, 0.1]`。 |
| `--exec-horizon 1` | 每步重新观察；此 skill 即使收到更大预算也只返回一个动作。 |
| `--output-root runs/code-smoke` | 每次创建独立运行目录，含 `events.jsonl`、`summary.json` 和 `evaluator/` 下的官方结果、仿真日志及视频。 |

只检查加载，不启动仿真：

```bash
.venv/bin/python -m orimapi check-skill --config configs/code_smoke.yaml --skill code-policy --load
```

## 自有硬币抓取实验

在已有推理环境中补充 IK 依赖：

```bash
uv pip install --python .venv/bin/python 'ikpy==3.4.2'
```

执行一个开发回合，仅尝试定位、抓取和抬起，不执行插入：

```bash
OMNI_KIT_ACCEPT_EULA=YES LIVESTREAM=0 \
.venv/bin/python -m orimapi eval \
  --config configs/coin_pickup.yaml \
  --task deposit_coin --episodes 1 \
  --mode direct --fixed-policy code-policy --exec-horizon 1 \
  --output-root runs
```

| 参数 / 配置 | 说明 |
| --- | --- |
| `code_parameters.robot_urdf` / `robot_config` / `arm_config` | 公开机器人结构和标定路径，相对启动工作目录。 |
| `code_parameters.camera_config` / `camera_template` / `table_config` | 公开相机及桌面参数路径；不读取任务布局、物体真值或评分器。 |
| `code_parameters.max_joint_step` | 每步关节命令相对观测的最大变化，默认 0.06 rad。 |
| `code_parameters.hover_height` / `lift_height` | 相对视觉定位点的接近、抬升高度，默认 0.10 / 0.12 m。 |
| `code_parameters.close_steps` | 关闭夹爪阶段的步数，默认 25；不能据此认定抓取成功。 |
| `code_parameters.closed_gripper` | 抓取时夹爪归一化目标，默认 0.15；闭合和抬升使用同一目标，不以完全闭合代替稳定夹持。 |
| `code_parameters.grasp_height_offset` | 抓取点相对可见硬币定位点的向上偏移，默认 0.012 m，用于避开支架；只适用于该开发抓取方案。 |
| `code_parameters.stage_timeout_steps` | 单个运动阶段步数上限，默认 140；超限后保持姿态。 |
| `--output-root runs` | 每轮保存源码 `code_sources/`、标定 `code_assets/`、阶段画面 `skill_frames/`、`code_skill_stage` 事件及官方结果。 |

该程序仅供仿真开发：图像定位或 IK 检查失败时停止推进；没有完整的碰撞规划，也不用于真实机器人。没有插入阶段，所以官方完整任务成功与抓取阶段进展必须分开看。`orimapi/skills/code_skill_rsi/` 通过本仓库的 `.git/info/exclude` 保持忽略，`.gitignore` 未因目录移动而修改；本地忽略规则不随 Git 同步。运行目录中的源码快照用于追溯；当前加载器仍只接受可信代码，不是自动生成代码的安全沙箱。

## Align Blocks 代码生成实验

开发 seed 0，直到首次官方成功；随后冻结文字经验和成功代码，在 seed 1、2 上各生成并执行一个测试程序：

```bash
OMNI_KIT_ACCEPT_EULA=YES LIVESTREAM=0 \
.venv/bin/python -m orimapi.skills.code_skill_rsi.align_blocks.experiment \
  --study-root orimapi/skills/code_skill_rsi/align_blocks/studies/20261019-dev0 \
  --runs-root runs --dev-seed 0 --eval-seeds 1 2 --max-attempts 0 \
  --llm-endpoint https://api.githubcopilot.com/responses \
  --llm-model gpt-6-astra \
  --llm-auth-command /usr/bin/python3 /root/.codex/copilot-auth.py
```

| 参数 / 文件 | 说明 |
| --- | --- |
| `--base-config 文件` | 默认使用独立目录内的 `config.yaml`；指定仿真解释器、资源路径和动作接口。 |
| `--study-root 目录` | 保存候选源码、任务拆解、文字反思、经验库及台账；同一目录重复执行会继续未完成阶段，先确认旧进程已结束。 |
| `--runs-root 目录` | 原生运行记录与视频根目录；经验库中不保存视频或关键帧。 |
| `--dev-seed N` | 开发使用的 seed。 |
| `--eval-seeds A B` | 不参与开发的 seed；每个只测一次，测试结果不更新经验库。 |
| `--max-attempts N` | 开发尝试编号上限，0 表示直到首次官方成功；增大该值可以继续同一实验。 |
| `best.json` / `frozen_experience.json` | 首次官方成功后冻结的代码及文字经验；不是对所有布局成功的保证。 |
| `ledger.json` | 分开记录开发、测试和中断；进程异常不伪造成官方任务失败。 |

代码只允许生成 `build_plan(scene)` 纯函数，在低权限、限时限内存且受 seccomp 限制的进程中产生动作阶段数据，再由可信 IK/运动工具执行。当前为仿真开发实验，不是真实机器人安全控制器。`code_skill_rsi/` 仍被 Git 忽略；首次成功和两次跨 seed 测试不代表完整基准成绩。

## 七任务 Code Skill 研究

```bash
OMNI_KIT_ACCEPT_EULA=YES LIVESTREAM=0 PYTHONDONTWRITEBYTECODE=1 \
.venv/bin/python -m orimapi.skills.code_skill_rsi.batch \
  --config configs/robodojo.yaml \
  --research-root orimapi/skills/code_skill_rsi --runs-root runs \
  --tasks build_tower classify_objects classify_objects_by_language \
    deposit_coin fasten_screws fill_egg_holder fold_clothes \
  --study-name 20261020-dev0 --reference-name 20261019-dev0 \
  --attempts-per-round 1 --rounds 0 \
  --llm-endpoint https://api.githubcopilot.com/responses \
  --llm-model gpt-6-astra \
  --llm-auth-command /usr/bin/python3 /root/.codex/copilot-auth.py
```

| 参数 / 文件 | 说明 |
| --- | --- |
| `--study-name` | 每个任务的研究版本目录；同名启动会继续已登记的试验，不覆盖旧结果。 |
| `--reference-name` | 读取各任务同名旧研究的开发经验与候选程序；不存在则从头开始，不导入验证反馈。 |
| `--reference-root` | 旧研究的根目录，默认与 `--research-root` 相同；独立试验目录可用它引用已有开发经验。 |
| `--runtime baseline` / `reactive` | 原版研究运行时 / 支持有界中途复查、腕部相机定位和双臂同步移动的实验运行时。不同运行时使用独立研究版本；后者尚不代表已通过官方测试。 |
| `--attempts-per-round 1` | 每轮给每个未完成任务增加一次开发尝试，再轮到下一任务。 |
| `--rounds 0` | 持续研究；正整数限制本次研究轮数，不会降低发布门槛。磁盘不足或基础设施异常会退出。 |
| `--research-root` / `--runs-root` | 文字经验、源码和研究台账的根目录 / 官方仿真记录的根目录。 |
| `stop_after_trial` | 在研究根目录放置此文件，当前开发回合记录完后退出；删除后可继续。 |
| `progress.json` | 研究进度；开发中、验证失败与达标分开记录，不是正式 skill 注册表。 |
| `studies/<版本>/publication/` | 发布时生成的验证证据与注册信息；不复制进正式 skill。 |

开发使用官方 seed 集合 0，首次成功后冻结代码与经验，再测试集合 1、2。三个集合必须全部完成官方测试，至少两个集合成功，且至少一个成功验证布局不同于开发布局，才允许发布。验证失败后创建新版本；重复使用这些集合属于工程迭代，不是独立基准成绩。程序异常、只有部分得分、缺少官方结果均不算成功。批处理只做研究与验证，不会自动修改正式配置。修改研究运行时代码前须等当前试验结束；已冻结但未完成验证的版本不能更换运行时。此研究模块仍在 Git 忽略的 RSI 目录内。

## GPT 选择 Code Skill 官方评测

顶层只有 `pi05` 和 `code-policy` 两种策略。指定 `code-policy` 后，由它内部的 GPT 根据任务和当前观测选择子 skill，再读取该子 skill 的冻结经验与成功程序，生成本回合的控制程序。已注册的子 skill 包括 `align_blocks`、`arrange_largest_number`、`cover_blocks`、`build_tower`、`classify_objects`、`play_tic_tac_toe`，均位于 `policies.code-policy.code_skills` 下，不与 pi0.5 并列。数字排序子 skill 仅声明支持普通 `arrange_largest_number`，不支持 `arrange_largest_number_random`；随机版保留在研究目录，尚未正式接入。不加载 pi0.5，不进行顶层 GPT 重规划；子 skill 可在接力时复核当前画面，不把评测结果写回经验库。`build_tower` 和 `classify_objects` 分别完成三个 seed 集合、各一回合的小样本测试，各有两个成功；这不是重复运行或完整基准成绩的保证。普通分类 skill 不覆盖仍在研究中的 `classify_objects_by_language`。棋盘任务使用不同对手种子的验证方式，见下面的三回合命令。

```bash
OMNI_KIT_ACCEPT_EULA=YES LIVESTREAM=0 \
.venv/bin/python -m orimapi eval-suite \
  --config configs/robodojo.yaml \
  --fixed-policy code-policy --mode direct \
  --llm-wire-api responses \
  --llm-endpoint https://api.githubcopilot.com/responses \
  --llm-model gpt-6-astra \
  --llm-auth-command /usr/bin/python3 /root/.codex/copilot-auth.py \
  --llm-header Editor-Version=vscode/1.110.0 \
  --llm-header Copilot-Integration-Id=vscode-chat \
  --llm-timeout-s 120 \
  --tasks align_blocks cover_blocks build_tower classify_objects --profile smoke --episodes 1 --seeds 0 1 2 \
  --exec-horizon 1 --no-record-frames --output-root runs
```

| 参数 / 文件 | 说明 |
| --- | --- |
| `--config configs/robodojo.yaml` | 与 pi0.5 共用唯一的正式评测配置；不再维护独立的 code skill 评测 YAML。 |
| `--fixed-policy code-policy` | 固定顶层 code-policy；内部的 GPT 子 skill 选择仍会执行。 |
| `--policy-backend code-policy` | 按后端筛选并固定唯一的顶层 code-policy，与上面的入口效果一致。 |
| `--mode direct` | 完整任务交给 code-policy；内部仍调用 GPT 选择子 skill、规划和生成代码。 |
| `--exec-horizon 1` | 首版必须为 1，每步读取最新观测后再产生一个动作。 |
| `--llm-*` | skill 选择、规划和代码生成共用端点、模型、鉴权及超时配置。 |
| `policies.code-policy.code_skills.<skill_id>` | 注册子 skill 的能力描述、控制器和冻结技能清单；增加子 skill 不增加顶层 policy。 |
| `code_skills.<skill_id>.code_bundle` | 相对 YAML 的技能清单路径；清单中的程序、经验、提示词和源码路径相对清单文件。 |
| `policies.code-policy.code_selection_prompt` | 子 skill 选择提示词，相对 YAML；GPT 返回 `skill_id`，不返回顶层 `policy_id`。 |
| `code_parameters.geometry` | 机器人和相机标定路径，相对 YAML 解析，不通过环境变量指定。 |
| `--seeds 0 1 2` | 分别使用三个官方布局集合；`layout_id` 是集合内编号，从 0 开始，不等于集合 seed。外部 GPT 返回不受 `--policy-seed` 保证复现。 |
| `--dry-run` | 只检查官方任务、布局、技能文件和资源，不调用 GPT、不启动仿真。 |
| `--resume runs/<运行目录>` | 沿用现有套件恢复规则；完成的任务不重跑，已有评分的部分结果不会自动重试。技能、经验或配置变更会拒绝续跑。 |
| `--no-record-frames` | 将本次运行的 `runtime.record_frames` 设为 false，不保存 GPT 输入图片；不影响官方视频。省略时沿用 YAML，亦可用 `--record-frames` 显式开启。 |

每个回合的生成代码、规划、文字请求和响应保存在 `runs/<运行目录>/code_episodes/<episode_id>/`，子 skill 选择记录是其中的 `selection-request.json` 和 `selection-response.json`；加载时的冻结技能快照在 `code_bundles/`，标定快照在 `code_assets/`。官方结果仍位于该运行目录的 `RoboDojo/`，统计和 monitor 沿用原链路。monitor 区分顶层 `policy_id=code-policy` 和内部选择的 `skill_id`；每回合重新选子 skill，已加载的控制器可以复用。

`program.py` 和 `experience.json` 位于 `orimapi/skills/code_skills/<skill_id>/`，是只读参考资料，不会原样回放历史布局的坐标。生成程序经过隔离校验后执行；GPT、代码校验或执行故障不会伪造成官方任务失败/成功，也不会回退到 pi0.5。接力复核的文字记录位于回合目录下的 `inspections/`。

正式 skill 目录只保留运行所需的代码、`skill.json`、成功程序、实际使用的经验和提示词。开发脚本、完整历史经验、实验台账和发布报告保存在 `orimapi/skills/code_skill_rsi/`；发布报告位于对应研究实验的 `publication/` 下，不参与正式加载或打包。成功程序及实际使用的失败/成功经验不会因目录清理而删除。

普通数字排序使用下面的单任务入口。上方套件示例不包含数字排序，因为其官方任务展开会同时带上尚未正式支持的 `_random` 场景：

```bash
OMNI_KIT_ACCEPT_EULA=YES LIVESTREAM=0 \
.venv/bin/python -m orimapi eval \
  --config configs/robodojo.yaml \
  --task arrange_largest_number --episodes 1 \
  --fixed-policy code-policy --mode direct \
  --llm-wire-api responses \
  --llm-endpoint https://api.githubcopilot.com/responses \
  --llm-model gpt-6-astra \
  --llm-auth-command /usr/bin/python3 /root/.codex/copilot-auth.py \
  --llm-header Editor-Version=vscode/1.110.0 \
  --llm-header Copilot-Integration-Id=vscode-chat \
  --llm-timeout-s 120 \
  --exec-horizon 1 --no-record-frames --output-root runs
```

`eval --task` 只执行指定任务；正式数字排序 skill 仅支持上面的普通任务，不应改成 `--task arrange_largest_number_random`。seed 使用 YAML 中的 `xpolicylab.seed`（默认 0）。可选到 code-policy 时保留无损 RGB，与研究和套件入口一致；固定 pi05 的单任务输入行为保持不变。`eval-suite --tasks arrange_largest_number` 仍按官方规则展开普通和随机两个任务，但由于随机版没有正式 skill 支持，会拒绝启动；官方任务配对规则没有修改。研究目录中的历史验证仅覆盖普通任务：开发 seed 0 和验证 seed 2 为 100/100，验证 seed 1 失败；这些记录不代表 `_random` 已取得成功，也不是全量成功率保证。

## 共享视觉与几何能力

code-policy 在请求 GPT 选择子 skill 时自动提供能力目录，不需要工具开关参数。GPT 返回的 `tools` 为空或省略时，沿用该 skill 原有的控制程序生成链路；选择工具时，读取同一份冻结经验和成功程序作为参考，再根据当前图像逐次调用跟踪、深度估计、三维定位、几何求解和验证工具。工具支路不要求生成一份新的完整 Python 控制程序。

默认深度资源配置在 `configs/robodojo.yaml` 的 `policies.code-policy.code_parameters.depth_estimation` 中。缺少本地权重时，只将深度能力告知为不可用，不阻止其他 skill 执行。模型仅在 GPT 调用 `estimate_depth` 时加载；同一进程复用权重，同一帧复用估计，新帧不复用旧深度图。仅支持度量深度权重，不把相对深度当成米。

所有 LLM 请求（含策略/子 skill 选择、监督和代码生成）默认限制单张编码图片为 300,000 字节、完整 JSON 请求为 2,000,000 字节，后者包含 Base64 膨胀和文字上下文。已符合限制的 PNG/JPEG 保持原编码，其余图片仅压缩发给模型的副本为 JPEG，不缩放、不裁剪，也不静默丢弃相机视角；本地动作模型、跟踪、深度和几何工具仍使用原始观测。压缩后仍超限或总请求过大时，在联网前报不可直接重试的错误，需显式减少视图或文字，或使用带坐标映射的裁剪图。图片处理与请求检查源码纳入 code 工具链身份，新行为不能混入旧冻结研究版本。

另一台机器需要补充依赖和权重时执行：

```bash
uv pip install --python .venv/bin/python -e '.[code-skills,code-depth]'
.venv/bin/huggingface-cli download \
  depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf \
  --revision 8078d68a9c75a972131914f6afd0c1723be0da7f \
  --include config.json preprocessor_config.json model.safetensors README.md \
  --local-dir checkpoint/depth-anything-v2-metric-indoor-small
```

继续使用前面的 code-policy 启动命令即可；只需改变资源位置或设备时追加：

```bash
--code-depth-model-path checkpoint/depth-anything-v2-metric-indoor-small \
--code-depth-device cpu
```

| 参数 / 配置 / 记录 | 说明 |
| --- | --- |
| `--code-depth-model-path` | 本地 Depth Anything 度量深度模型目录；相对启动目录，不通过环境变量指定。不是启用开关，也不强制 GPT 调用。 |
| `--code-depth-device` | `cpu` 或 `cuda:N`；默认从 YAML 读取，目前为 `cpu`。只影响深度模型，不改变 pi0.5 或仿真设备。 |
| `code_parameters.depth_estimation.model_path` | YAML 中的权重路径相对 YAML 文件解析；运行时不下载权重、不执行模型仓库的远程代码。 |
| `code_parameters.tool_limits` | 可选调用预算：`max_calls` 默认 60，`max_tools_per_turn` 默认 8，`history_size` 默认 12；只限制工具支路，不选择是否使用工具。 |
| `selection-request.json` / `selection-response.json` | 保存提供给 GPT 的 `available_tools` 及 GPT 自己选择的 `tools`。 |
| `code_episodes/<episode_id>/tools/` | 保存逐次工具决策的文字请求和响应，不保存图像或深度数组。 |
| `events.jsonl` | `code_skill_tool_*` 事件记录实际工具调用、返回、丢失目标、动作后验证和额外能力请求；Monitor 的事件日志可查看摘要。 |
| `--dry-run` | 只检查任务计划和资源身份，不调用 GPT、不运行深度推理、不启动仿真。 |

已有 Monitor 进程需重启才能加载新增事件类型。模型深度与平面投影均明确标为估计；可用 `calibrate_depth` 根据 GPT 在当前图像选择的公开桌面点修正全局尺度，但不保证局部精度。三角测量要求不同当前视角中的同一物理点。动作完成和几何约束满足只作为局部证据，不能代替抓取成功或官方任务成功。工具支路已完成离线测试、真实深度推理和少量倒水任务开发仿真，尚无该任务的官方成功；原 skill 的历史成绩不代表工具支路的成绩。

## insert_tubes Code Skill

```bash
OMNI_KIT_ACCEPT_EULA=YES LIVESTREAM=0 \
.venv/bin/python -m orimapi eval-suite \
  --config configs/robodojo.yaml --mode direct --fixed-policy code-policy \
  --tasks insert_tubes --profile smoke --episodes 1 --seeds 0 1 \
  --exec-horizon 1 --no-record-frames --output-root runs \
  --llm-endpoint https://api.githubcopilot.com/responses \
  --llm-wire-api responses --llm-model gpt-6-astra \
  --llm-auth-command /usr/bin/python3 /root/.codex/copilot-auth.py \
  --llm-header Editor-Version=vscode/1.110.0 \
  --llm-header Copilot-Integration-Id=vscode-chat \
  --llm-timeout-s 180
```

`--seeds 0 1 --episodes 1` 分别运行两个官方 seed 集合的首回合。相同冻结版本已在 seed 0、1 得到官方 `success=true, score=1.0`，分别执行 465、469 步，未追加第三个 seed；这是小样本准入结果，不代表完整基准成功率。保持原生 500 步上限和双臂归位要求，不加载 pi0.5；`--output-root` 可修改结果目录。

GPT 根据当前相机图像绑定管子、孔位和管帽，固定控制器执行几何测量、抓取、插入及归位。同一只手按远孔到近孔的顺序放置，避免后续搬运经过已插入的管子；完整路径检查、遮挡后的换视角测量和失败保护仍保留。正式目录仅包含 `controller.py`、`vision.py`、`experience.json`、`skill.json` 四个文件，复用的机器人运动与视觉工具也列入清单。研究过程及失败结果保留在 RSI 和 runs 目录，不复制图片或视频到经验文件。

## stack_bowls Code Skill

```bash
OMNI_KIT_ACCEPT_EULA=YES LIVESTREAM=0 \
.venv/bin/python -m orimapi eval \
  --config configs/robodojo.yaml --mode direct --fixed-policy code-policy \
  --task stack_bowls --episodes 1 \
  --exec-horizon 1 --no-record-frames --output-root runs \
  --llm-endpoint https://api.githubcopilot.com/responses \
  --llm-wire-api responses --llm-model gpt-6-astra \
  --llm-auth-command /usr/bin/python3 /root/.codex/copilot-auth.py \
  --llm-header Editor-Version=vscode/1.110.0 \
  --llm-header Copilot-Integration-Id=vscode-chat \
  --llm-timeout-s 180
```

`--task stack_bowls --episodes 1` 只执行该任务的一个回合，seed 由 YAML 中的 `xpolicylab.seed` 指定，默认 0。该冻结版本已在 seed 0、1 的首回合均得到官方 `success=true, score=1.0`，分别执行 481、584 步；未追加 seed 2，不代表全量成功率。保持原生 800 步上限及双臂归位要求，不加载 pi0.5。

GPT 负责当前图像的碗口绑定与抓取复核，控制器执行空手观察、夹碗沿、慢速搬运、叠放和归位。正式目录只有控制器、视觉几何、经验、清单 4 个文件，复用的公开机器人运动辅助代码也纳入清单校验；失败试验保留在研究目录，不随正式 skill 发布。`eval-suite --tasks stack_bowls` 会按官方配对规则带上尚未验证、未注册的 `stack_bowls_random` 并拒绝启动，因此只测试普通叠碗时使用上面的 `eval` 命令。

## make_kong 固定阶段 Code Skill

```bash
uv pip install --python .venv/bin/python -e '.[code-skills]'

OMNI_KIT_ACCEPT_EULA=YES LIVESTREAM=0 \
.venv/bin/python -m orimapi eval-suite \
  --config configs/robodojo.yaml --mode direct --fixed-policy code-policy \
  --tasks make_kong --profile smoke --episodes 1 --seeds 0 1 \
  --exec-horizon 1 --no-record-frames --output-root runs \
  --llm-endpoint https://api.githubcopilot.com/responses \
  --llm-wire-api responses --llm-model gpt-6-astra \
  --llm-auth-command /usr/bin/python3 /root/.codex/copilot-auth.py \
  --llm-header Editor-Version=vscode/1.110.0 \
  --llm-header Copilot-Integration-Id=vscode-chat \
  --llm-timeout-s 180
```

`--seeds 0 1 --episodes 1` 分别执行两个官方布局集合的首回合；该冻结版本两回合均得到官方 `success=true, score=1.0`，达到两个不同 seed 通过的准入条件，没有追加 seed 2。这是小样本结果，不代表全量成功率。

GPT 仅进行当前图像的目标绑定与补牌顶面定位，动作由固定阶段程序执行。原生上限仍为 600 步；官方成功后结束，不为了后续归位延长回合。`code-controller-skill/v1` 保留实际控制器，不添加占位程序或无用工具配方；正式目录只有 7 个运行输入文件。`trimesh` 已加入 `code-skills` 依赖，用于读取公开机器人网格。

## 井字棋 Code Skill 三回合命令

```bash
OMNI_KIT_ACCEPT_EULA=YES LIVESTREAM=0 \
.venv/bin/python -m orimapi eval-suite \
  --config configs/robodojo.yaml \
  --fixed-policy code-policy --mode direct \
  --llm-wire-api responses \
  --llm-endpoint https://api.githubcopilot.com/responses \
  --llm-model gpt-6-astra \
  --llm-auth-command /usr/bin/python3 /root/.codex/copilot-auth.py \
  --llm-header Editor-Version=vscode/1.110.0 \
  --llm-header Copilot-Integration-Id=vscode-chat \
  --llm-timeout-s 120 \
  --tasks play_tic_tac_toe --profile smoke --episodes 3 --seeds 0 \
  --exec-horizon 1 --no-record-frames --output-root runs
```

`--episodes 3 --seeds 0` 在同一个官方布局集合内连续执行三回合；本任务的对手随机数由原生 `layout_id` 驱动。仅用 `--episodes 1 --seeds 0 1 2` 会各从 `layout_id=0` 开始，不能当成三个不同对手种子。以官方实际有效回合编号为准，不筛选布局或伪造恢复状态。

该 skill 从当前 RGB 重新定位棋盘和棋子，复用经过校验的回合函数，一次放一枚并在对手行动后复查；不是回放固定坐标。冻结版本连续三回合的官方结果为 `100 / 100 / 50` 分，成功 `2/3`，仅代表固定棋盘上三个对手种子的小样本验证。失败回合保留在 `runs/`，验证报告保留在研究目录，不回写正式经验；正式目录仍仅保留 13 个运行必需文件。

## 工具程序研究命令

`swap_T` 和 `press_by_number` 已注册为正式子 skill。三 seed 准入样本分别为 3/3 和 2/3；另一次正式入口 seed 0 复测两者均失败，不能据此认为策略已稳定。交换任务仍会耗尽归位步数，按钮任务的视觉计数仍可能与实际触发不一致。资格和额外复测记录保存在各自的 `skill.json`，未删除失败结果。

`put_bottles_into_dustbin` 已加入正式子 skill。开发 seed 0 和独立验证 seed 1 官方成功，seed 2 失败，分数分别为 `100 / 100 / 40`，达到 2/3 小样本准入门槛。随后正式入口 seed 0 复测失败，得分 40/100，因此策略仍不稳定，不能把准入成绩当作全量成功率。该版本复用冻结的函数和文字经验，在同一 Python 会话中根据当前观测逐段执行、修正代码；测试回合的修改不会写回冻结库。失败研究与官方运行记录继续保留。

单独运行瓶子任务，每个 seed 一个回合：

```bash
OMNI_KIT_ACCEPT_EULA=YES LIVESTREAM=0 \
.venv/bin/python -m orimapi eval-suite \
  --config configs/robodojo.yaml --mode direct --fixed-policy code-policy \
  --tasks put_bottles_into_dustbin --profile smoke --episodes 1 --seeds 0 1 2 \
  --exec-horizon 1 --no-record-frames --output-root runs \
  --llm-endpoint https://api.githubcopilot.com/responses \
  --llm-wire-api responses --llm-model gpt-6-astra \
  --llm-auth-command /usr/bin/python3 /root/.codex/copilot-auth.py \
  --llm-header Editor-Version=vscode/1.110.0 \
  --llm-header Copilot-Integration-Id=vscode-chat \
  --llm-timeout-s 180
```

瓶子子 skill 的 `code_parameters.vision_resources_path` 指定视觉模型清单，`vision_model_root` 指定模型根目录，两者都相对 YAML 文件解析；清单中的模型路径相对这个根目录。`vision_device` 指定视觉模型设备，当前配置为 `cuda:0`。迁移机器时调整这些配置，不需要设置资源路径环境变量。正式目录仅保留 14 个运行输入文件，不包含研究日志或视频。`code_workspace/` 是最近回合的工作副本；完整的程序和反馈在 `events.jsonl` 中按 `episode_id` 区分，模型请求和回复分别保存在 `code_episodes/<episode_id>/programs/`。

按正常评测入口运行这两个正式 skill，每个 seed 一个回合：

```bash
OMNI_KIT_ACCEPT_EULA=YES LIVESTREAM=0 \
.venv/bin/python -m orimapi eval-suite \
  --config configs/robodojo.yaml --mode direct --fixed-policy code-policy \
  --tasks swap_T press_by_number --profile smoke --episodes 1 --seeds 0 1 2 \
  --exec-horizon 1 --no-record-frames --output-root runs \
  --llm-endpoint https://api.githubcopilot.com/responses \
  --llm-wire-api responses --llm-model gpt-6-astra \
  --llm-auth-command /usr/bin/python3 /root/.codex/copilot-auth.py \
  --llm-header Editor-Version=vscode/1.110.0 \
  --llm-header Copilot-Integration-Id=vscode-chat \
  --llm-timeout-s 180
```

工具型 bundle 使用 `orimapi.code-tool-skill/v1`，只保留控制器、工具程序、冻结经验和清单，不添加占位的 `program.py` 或无用提示词；仍属于 `code-policy`，不是新增顶层 policy。工具由 GPT 根据当前场景选择。

仅做本地隔离测试，不调用 GPT、不启动仿真：

```bash
.venv/bin/python -m orimapi.research.tool_loop check \
  --recipe configs/tool_recipes/approach_close.json
```

显式启动研究时使用以下命令。示例未注册为正式 skill；开发只用 seed 0 的首个布局，最多修改三次，每次修改通过回归测试后才重跑官方回合：

```bash
OMNI_KIT_ACCEPT_EULA=YES LIVESTREAM=0 \
.venv/bin/python -m orimapi.research.tool_loop develop \
  --config configs/robodojo.yaml \
  --task-spec configs/tool_recipes/pour_research_spec.json \
  --recipe configs/tool_recipes/approach_close.json \
  --study-root orimapi/skills/code_skill_rsi/pour_liquid_into_cup/studies/tool-repair-001 \
  --runs-root runs --max-repairs 3 \
  --llm-endpoint https://api.githubcopilot.com/responses \
  --llm-model gpt-6-astra \
  --llm-auth-command /usr/bin/python3 /root/.codex/copilot-auth.py \
  --llm-header Editor-Version=vscode/1.110.0 \
  --llm-header Copilot-Integration-Id=vscode-chat \
  --llm-timeout-s 120
```

开发成功生成 `frozen.json` 后，用同样的配置、任务规格、模型参数和研究目录，将 `develop` 改为 `validate --seed 1`，移除 `--recipe`、`--max-repairs`；完成后以 `--seed 2` 再运行一次。验证不修改候选、不重试同一个验证 seed，也不自动合入正式 skill。

| 参数 | 说明 |
| --- | --- |
| `--task-spec 文件` | 公开任务名、官方动作预算和任务要求，不包含隐藏物体状态或布局坐标。 |
| `--recipe 文件` | 初始有界工具程序及 JSON 回归用例，供 `check` 或 `develop` 使用。 |
| `--study-root 路径` | 独立研究目录，保存候选、修复、测试与冻结记录；已有研究不能被 `develop` 覆盖。 |
| `--runs-root 路径` | 官方仿真运行目录的根路径，每回合生成独立子目录。 |
| `--max-repairs 0..5` | 首次尝试之后最多修复次数，默认 3；0 表示仅验证初始候选。 |
| `validate --seed 1/2` | 冻结后验证，各 seed 只允许一次，不调用优化器。 |

首次实现只自动修复有界工具组合，不能任意改共享视觉/IK或仿真程序。执行 GPT 自动选择可用工具；没有 `--code-tools` 开关。设计与安全边界见 `doc/code-tools.md`。

## 多任务启动命令

指定主任务做流程测试，每个展开后的任务只跑一个回合，pi05 权重常驻：

```bash
OMNI_KIT_ACCEPT_EULA=YES \
XLA_PYTHON_CLIENT_PREALLOCATE=false \
.venv/bin/python -m orimapi eval-suite \
  --config configs/robodojo.yaml \
  --tasks stack_bowls insert_key --profile smoke --episodes 1 --seeds 0
```

按官方配额运行完整仿真基准：

```bash
OMNI_KIT_ACCEPT_EULA=YES \
XLA_PYTHON_CLIENT_PREALLOCATE=false \
.venv/bin/python -m orimapi eval-suite \
  --config configs/robodojo.yaml --all-tasks --profile official
```

继续执行
```bash
OMNI_KIT_ACCEPT_EULA=YES \
XLA_PYTHON_CLIENT_PREALLOCATE=false \
.venv/bin/python -m orimapi eval-suite \
  --config configs/robodojo.yaml \
  --resume runs/<suite_id>
```

只检查任务计划，不加载权重、不启动仿真：

```bash
.venv/bin/python -m orimapi eval-suite --all-tasks --profile official --dry-run
```

恢复原有套件，跳过已校验的完整作业：

```bash
OMNI_KIT_ACCEPT_EULA=YES XLA_PYTHON_CLIENT_PREALLOCATE=false \
.venv/bin/python -m orimapi eval-suite --resume runs/<suite_id>
```

| 参数 | 说明 |
| --- | --- |
| `--tasks A B` / `--tasks-file 文件` / `--all-tasks` | 三选一；任务文件每行一个名称，可使用 `#` 注释。泛化主任务自动包含随机变体。 |
| `--profile official` | 使用官方 50 或 25+25 回合配额，默认 seed 0、1、2；全量共 6,300 个有效回合。 |
| `--profile smoke --episodes N` | 每个原生任务/seed 跑 N 回合，默认 N=1、seed=0；不是完整基准成绩。 |
| `--seeds 0 1 2` | 明确选择官方布局 seed，不切换 checkpoint。 |
| `--policy-gpu 0` / `--simulator-gpu 0` | 相对于当前可见 GPU 列表的索引；两者独立选择。 |
| `--policy-seed 0` | 每个任务/seed 作业的模型 RNG 起点，回合内及回合间不重新播种。 |
| `--job-timeout-s 秒` | 单作业超时；默认至少使用配置中的评测超时，并按每回合 600 秒扩展预算。 |
| `--min-free-gib 5` | 最低磁盘余量，运行时低于该值停止当前作业。输出空间估算不保证覆盖所有任务。 |
| `--fail-fast` | 单个作业出错后停止队列；正常任务失败不算运行错误。 |
| `--resume 目录` | 仅恢复新版 v2 套件，必须匹配原配置、代码和资源；部分结果不会自动重跑，不支持迁移旧版目录。 |
| `--output-root 目录` | 套件输出根目录，默认为 `runs/`。 |
| `--dry-run` | 输出展开后的计划，不创建评测运行目录。 |
| `--mode closed_loop` | 显式启用 GPT 严格闭环；不指定时，新的 `eval-suite` 仍默认纯 pi05 的 `direct`。 |
| `--exec-horizon N` | 每次策略推理后最多执行 N 个动作，再根据最新观测重新推理；必须为正整数。覆盖 YAML 的 `runtime.exec_horizon`，未指定时沿用配置，当前默认 5。适用于 `eval` 和 `eval-suite`，与 GPT 复查间隔独立。 |
| `--replan-interval-chunks N` | 严格闭环每完整执行 N 个动作块就停止，等待新 GPT 指令；与每块动作数独立。默认从 YAML 读取，当前为 20。旧名 `--replan-interval-steps` 仍可使用。 |
| `--llm-model 模型ID` | 覆盖 GPT 模型标识，必须支持图像输入和现有 JSON 返回协议。 |
| `--llm-reasoning-effort 等级` | 覆盖 `supervisor.reasoning_effort`；默认不指定，沿用接口默认设置。统一用于首次规划、中途复查、子 skill 选择、代码生成和研究复盘。可选 `none`、`minimal`、`low`、`medium`、`high`、`xhigh`、`max`，需接口及模型支持；不支持时直接报错，不自动降级。配置和生成请求日志会记录指定的等级，`null` 表示未指定，不代表服务端没有推理。 |
| `--llm-endpoint URL` | 覆盖完整的 GPT 接口地址；命令行参数优先于环境变量和 YAML。 |

`eval-suite` 固定使用 pi05，并默认选择唯一的 pi05 候选。`direct` 不需要 API Key；`closed_loop` 固定动作模型，但仍调用 GPT 生成子目标。

## GPT 闭环启动命令

先在运行终端配置凭据环境变量，默认名称为 `OPENAI_API_KEY`，不要把密钥写入 YAML。
将 `YOUR_VISION_MODEL` 替换为实际支持图像和 JSON 输出的模型标识。先验证两个任务：

```bash
OMNI_KIT_ACCEPT_EULA=YES \
XLA_PYTHON_CLIENT_PREALLOCATE=false \
.venv/bin/python -m orimapi eval-suite \
  --config configs/robodojo.yaml \
  --mode closed_loop --fixed-policy pi05 \
  --llm-model YOUR_VISION_MODEL --replan-interval-chunks 20 \
  --tasks general_pickup insert_key --profile smoke --episodes 1 --seeds 0
```

按官方配额运行 GPT 闭环全量评测：

```bash
OMNI_KIT_ACCEPT_EULA=YES \
XLA_PYTHON_CLIENT_PREALLOCATE=false \
.venv/bin/python -m orimapi eval-suite \
  --config configs/robodojo.yaml \
  --mode closed_loop --fixed-policy pi05 \
  --llm-model YOUR_VISION_MODEL --replan-interval-chunks 20 \
  --all-tasks --profile official
```

`--resume runs/<suite_id>` 默认读取该运行保存的 `config.json`，包括模式、模型和 N；
仍需提供 API 凭据，并通过源码、配置、提示词和资源身份校验。显式 `--config` 或其他覆盖参数必须与原计划一致。

第 0 步先等待 GPT 指令。上面的 `N=20` 配合 `exec_horizon=5`，每次指导后执行
20 个动作块，共 100 步，再暂停等待 GPT；不会把动作块截短来凑 interval。
如果策略实际返回的动作块不足 5 步，完整执行该块仍计为一次，实际总步数相应减少。
例如在启动命令中添加 `--exec-horizon 10 --replan-interval-chunks 10`，
每块执行 10 步，每 10 块请求 GPT，仍是每 100 步复查一次（策略返回足够动作时）。

| YAML 参数 | 说明 |
| --- | --- |
| `runtime.supervisor_schedule: blocking` | 严格等待，套件闭环只接受此值；旧的单任务异步行为可显式配置为 `async`。 |
| `runtime.exec_horizon: 5` | 每个动作块最多交付 5 个动作，独立于 GPT 指导间隔；可用 `--exec-horizon N` 覆盖。 |
| `runtime.replan_interval_steps: 20` | 保留旧字段名；严格闭环按已完整执行的动作块计数，默认每 20 块请求 GPT，每块 5 步时共 100 步。旧 `async` 模式仍按单个环境动作计数。 |
| `runtime.supervisor_timeout_s: 30` | 单次 GPT 请求等待上限。 |
| `runtime.max_consecutive_failures: 3` | 严格闭环在同一观测点最多连续失败 3 次；不会在等待或重试时继续执行旧指令。 |
| `runtime.supervisor_retry_backoff_s: 1` | 重试初始退避秒数，指数增加，单次不超过请求等待上限。 |
| `runtime.max_supervisor_calls: 100` | 每回合 GPT 调用上限，包含每次重试。 |

GPT 故障耗尽重试后记录执行错误，默认继续下一任务，不生成虚假官方失败回合；
`--fail-fast` 可停止后续任务。pi05 故障或无法回收的在途请求会停止套件，不继续复用。

## Copilot 闭环启动命令

当前机器的 `configs/robodojo.yaml` 已显式配置本地 Copilot bridge：
`http://127.0.0.1:4142/v1/responses`、`gpt-6-astra`、`auth.type: none`。
上游登录凭据由 bridge 管理，不写入项目配置；无鉴权模式只允许回环地址。
`require_model_match: true` 要求响应的 `model` 与请求完全一致，缺失或不一致直接报错，
不会切换其他模型。策略选择、代码生成和监督请求共用该配置。

本机已用真实接口验证单图、三相机图片及 8 路并发 JSON 请求；这些检查不启动仿真。
以下直连 Copilot 的示例适用于另有可用鉴权脚本的机器，本机运行无需该脚本。

复用本机已登录的 Copilot 鉴权脚本，只跑一个任务、一个回合。模型 ID、接口、
请求头和脚本路径均显式传入，不读取 Codex 当前会话配置，也不需要 `OPENAI_API_KEY`：

```bash
OMNI_KIT_ACCEPT_EULA=YES \
XLA_PYTHON_CLIENT_PREALLOCATE=false \
.venv/bin/python -m orimapi eval-suite \
  --config configs/robodojo.yaml \
  --mode closed_loop --fixed-policy pi05 \
  --llm-wire-api responses \
  --llm-endpoint https://api.githubcopilot.com/responses \
  --llm-model gpt-6-astra \
  --llm-auth-command /usr/bin/python3 /root/.codex/copilot-auth.py \
  --llm-header Editor-Version=vscode/1.110.0 \
  --llm-header Copilot-Integration-Id=vscode-chat \
  --llm-timeout-s 90 --replan-interval-chunks 20 \
  --tasks general_pickup --profile smoke --episodes 1 --seeds 0
```

## 少量task测试
```bash
OMNI_KIT_ACCEPT_EULA=YES \
XLA_PYTHON_CLIENT_PREALLOCATE=false \
.venv/bin/python -m orimapi eval-suite \
  --config configs/robodojo.yaml \
  --mode closed_loop --fixed-policy pi05 \
  --llm-wire-api responses \
  --llm-endpoint https://api.githubcopilot.com/responses \
  --llm-model gpt-6-astra \
  --llm-auth-command /usr/bin/python3 /root/.codex/copilot-auth.py \
  --llm-header Editor-Version=vscode/1.110.0 \
  --llm-header Copilot-Integration-Id=vscode-chat \
  --llm-initial-prompt prompts/robodojo_initial.md \
  --llm-review-prompt prompts/robodojo_review.md \
  --llm-timeout-s 90 \
  --replan-interval-chunks 20 \
  --exec-horizon 5 \
  --tasks classify_objects_by_language cover_blocks deposit_coin align_blocks \
  --profile smoke --episodes 3 --seeds 0 --policy-seed 0 \
  --output-root runs
```

| 参数 | 说明 |
| --- | --- |
| `--llm-wire-api` | `chat_completions` 或 `responses`；当前 YAML 使用 `responses`，修改 URL 不会自动切换协议。 |
| `supervisor.reasoning_effort` | 默认 YAML `null`，不发送思考等级字段、交由服务端决定；这不同于字符串 `none`。显式配置时，Responses 使用 `reasoning.effort`，Chat Completions 使用 `reasoning_effort`。新设置对新启动进程生效，不改写已有实验记录。 |
| `--llm-auth-command 命令 参数...` | 执行可信的本地命令，标准输出必须仅为 token。使用 argv，不经过 shell；路径建议使用绝对路径，不要将凭据放进参数。 |
| `--llm-header 名称=值` | 可重复；仅接受非敏感的 `Editor-Version`、`Copilot-Integration-Id`、`User-Agent`，不能传入 Authorization 或 Cookie。 |
| `--llm-timeout-s 秒` | 单次监督请求的时间预算，包括获取凭据和一次 401 刷新重试；同时设置 Runtime 等待上限。 |
| `--llm-stream` / `--no-llm-stream` | Responses 是否使用 SSE，默认关闭；流式模式只在收到完整完成事件后才交付决策。 |

也可在 YAML 的 `supervisor` 下配置 `wire_api`、`headers`、`stream`，以及：

```yaml
auth:
  type: command
  command: [/usr/bin/python3, /root/.codex/copilot-auth.py]
  refresh_interval_s: 300
  timeout_s: 30
```

命令鉴权的 token 只缓存在内存中，默认 300 秒刷新；`timeout_s` 是单次凭据命令上限，
不能超出监督请求剩余预算。一次监督调用内部最多因 401 刷新并重发一次；
第二次 401 或 403 不继续重试。外层重试仍计入 `max_supervisor_calls`。
凭据错误归为 GPT 故障，等待期间不会继续执行动作。

保留 `auth.type: env` 时仍从 `supervisor.api_key_env` 指定的环境变量读取 API Key。
`--resume runs/<suite_id>` 会恢复保存的协议、模型、请求头和鉴权命令，
但需要本机仍具备有效登录；token 不写入运行配置或日志。

## 两阶段 Prompt 三任务冒烟

首次生成一个子目标，中途检查执行情况并决定保持、更换或重试；不生成完整计划列表。
以下命令每个任务执行一个回合，共三个回合，继续使用每块 5 步、每 20 块请求 GPT：

```bash
OMNI_KIT_ACCEPT_EULA=YES \
XLA_PYTHON_CLIENT_PREALLOCATE=false \
LIVESTREAM=0 \
.venv/bin/python -m orimapi eval-suite \
  --config configs/robodojo.yaml \
  --mode closed_loop --fixed-policy pi05 \
  --llm-wire-api responses \
  --llm-endpoint https://api.githubcopilot.com/responses \
  --llm-model gpt-6-astra \
  --llm-auth-command /usr/bin/python3 /root/.codex/copilot-auth.py \
  --llm-header Editor-Version=vscode/1.110.0 \
  --llm-header Copilot-Integration-Id=vscode-chat \
  --llm-initial-prompt prompts/robodojo_initial.md \
  --llm-review-prompt prompts/robodojo_review.md \
  --llm-timeout-s 90 --replan-interval-chunks 20 \
  --tasks classify_objects_by_language cover_blocks deposit_coin \
  --profile smoke --episodes 1 --seeds 0 --policy-seed 0 \
  --output-root runs
```

| 参数 / YAML 配置 | 说明 |
| --- | --- |
| `--llm-initial-prompt 文件` / `supervisor.initial_prompt_path` | 没有生效子目标时使用，包括首次失败后的重试。默认 YAML 指向 `robodojo_initial.md`。 |
| `--llm-review-prompt 文件` / `supervisor.review_prompt_path` | 已有子目标时使用，包括 `keep` 后的后续复查。默认 YAML 指向 `robodojo_review.md`。 |
| `supervisor.prompt_path` | 旧统一 prompt；未配置某阶段路径或设为 `null` 时，该阶段回退到此文件。 |

命令行 prompt 路径相对启动目录，YAML 路径相对 YAML 文件；每回合重置后重新使用首次 prompt。
旧统一 prompt 对照组可将两个命令行路径都指向 `prompts/robodojo_supervisor.md`。
改用 `--episodes 5 --seeds 0 1 2` 可做每组 45 回合的小样本对比，不代表完整基准成绩。

`events.jsonl` 的 `supervisor_configured` 保存两份 prompt 全文与哈希；
每次 `supervisor_prompt` 事件记录 `prompt_phase`（`initial` / `review` / `selection`）、
`prompt_sha256` 和 `request_id`，可关联请求、返回和生效决策。选择策略仍使用独立的 selection prompt。
冻结计划包含阶段 prompt 哈希，`--resume` 不允许跨版本混用。命令创建新 run，不追加到旧实验。

## 监控启动命令

剩余任务的双 GPU 自动研究、预算继承、候选修订、两 seed 合入及服务恢复方式见
[自动 RSI 研究队列](doc/automatic-rsi-campaign.md)。

新启动的 `eval` / `eval-suite` 在运行目录保存 `invocation.json`，Monitor 的“执行命令”
展示实际启动参数，不再根据配置拼接。保留参数顺序、相对路径及工作目录；密钥等敏感参数脱敏。
同时记录已设置的 `OMNI_KIT_ACCEPT_EULA`、`XLA_PYTHON_CLIENT_PREALLOCATE`、`LIVESTREAM`、
`CUDA_VISIBLE_DEVICES` 白名单值，不读取或展示其他环境变量。Shell 原始换行、引号、变量写法及重定向无法自动记录。
`--resume` 不覆盖首次启动记录，恢复命令追加到 `events.jsonl` 的 `invocation_recorded` 事件。
没有启动记录的历史实验显示“未记录”，不会猜测原始输入；更新后需重启 Monitor 生效。

在另一个终端独立启动，只读查看单任务或 v2 套件，不启动评测或加载权重：

```bash
.venv/bin/python -m orimapi.monitor \
  --runs-root ./runs \
  --native-results-root ./third_party/RoboDojo/eval_result \
  --host 127.0.0.1
```

默认监听 80 端口，浏览器访问 `http://127.0.0.1`。相对路径以启动命令的工作目录为准。
缺少 HTTP 依赖时安装可选依赖：`.venv/bin/python -m pip install -e '.[monitor]'`。

需要通过服务器 IP 从外部直接访问时，使用：

```bash
.venv/bin/python -m orimapi.monitor \
  --runs-root ./runs \
  --native-results-root ./third_party/RoboDojo/eval_result \
  --host 0.0.0.0
```

访问 `http://服务器IP`，不要将 `0.0.0.0` 当作浏览器访问地址。云安全组和主机防火墙需允许客户端访问 TCP 80；服务无认证，请限制来源 IP，或使用下方的 SSH 转发。

| 参数 | 说明 |
| --- | --- |
| `--runs-root 目录` | 必填，包含各次运行目录的根目录；不存在时显示空列表，不创建或修改它。 |
| `--native-results-root 目录` | RoboDojo 的 `eval_result/` 根目录，下面应有 `RoboDojo/`。用于读取当前作业的中间成绩；省略时仍可查看进度及已归档结果。 |
| `--host 地址` | 默认 `127.0.0.1`。服务无登录认证；非回环地址只能用于可信网络或受认证代理保护的部署。 |
| `--port 端口` | 默认 `80`；端口占用或绑定权限不足时报错，可显式换其他端口，例如 `--port 8080`。 |
| `--poll-interval 秒` | 活跃运行的读取间隔，默认 `1`，允许 `0.1` 至 `30`。 |

远程服务器建议保留回环监听，并在本机执行 `ssh -L 8080:127.0.0.1:80 用户@服务器`，随后在本机访问 `http://127.0.0.1:8080`。
监控可以在评测前、中、后启动；关闭监控不影响评测。不需要新增环境变量。

## 环境变量说明

| 环境变量 | 作用 |
| --- | --- |
| `OMNI_KIT_ACCEPT_EULA=YES` | 表示接受 NVIDIA Isaac Sim 的许可协议，用于无交互启动。执行前应确认接受该许可。 |
| `XLA_PYTHON_CLIENT_PREALLOCATE=false` | 关闭 JAX 的显存预分配，改为按需申请，减少与 Isaac Sim 争抢显存；不代表设置了显存使用上限。 |
| `OPENAI_API_KEY` | 环境变量鉴权的默认 GPT 凭据变量，仅闭环或策略选择需要；变量名由 `supervisor.api_key_env` 指定。命令鉴权不使用它。 |
| `ORIMAPI_LLM_MODEL` / `ORIMAPI_LLM_ENDPOINT` | 可覆盖 YAML 中的 GPT 模型和接口；也可使用优先级更高的 `--llm-model` / `--llm-endpoint` 参数。 |

命令前缀设置的环境变量只对本次命令及其子进程生效，不会永久修改终端环境。
