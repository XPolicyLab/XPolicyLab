# WALL-WM-2.0：公开依赖安装

项目名称： **WALL-WM-2.0**

本说明只安装 **WALL-WM-2.0 的 VAP policy server**。RoboDojo / Isaac Sim 仿真环境使用官方 RoboDojo
安装流程单独创建；两套 Python/CUDA 环境不能合并。

## 1. 支持范围和公开来源

- Linux x86_64，**CPython 3.11**；需要兼容 PyTorch cu130 的 NVIDIA 驱动。
- 所有 Python 运行依赖来自 [PyPI](https://pypi.org/simple) 或
  [PyTorch 官方 wheel 源](https://download.pytorch.org/whl/)。
- `environment/requirements.lock` 固定完整传递依赖和 wheel 的 SHA256。安装启用
  `--require-hashes --only-binary :all:`，不接受私有 find-links 或源码包替代。
- Torch `2.12.1+cu130`、Torchvision `0.27.1+cu130`、Triton `3.7.1` 使用官方
  `download-r2.pytorch.org` URL；其他包来自 PyPI。具体版本及公开 URL 见
  [PUBLIC_SOURCES.json](environment/PUBLIC_SOURCES.json)。
- `wall_x` 模型代码位于 PR 的 `upstream/`。常量兼容模块采用
  [X-Square-Robot/wall-x](https://github.com/X-Square-Robot/wall-x) 固定提交
  `ce2a1ae31d9ee1028b97ca40e3f66034cf8c0e5f` 的 Apache-2.0 公开实现。
  不复制或安装 proprietary `x2robot_utils`。本 checkpoint 的统计来自 adapter 的显式
  JSON 与 checkpoint 自带 normalizer，不使用旧包的私有默认统计表。

不依赖现有 Conda 环境、私有 wheel、CPFS 包缓存、editable 安装或 `host.env`。
Conda 仅是可选的 Python 3.11 获取方式，不是运行要求。Python 解释器也不打包到 PR；
可用操作系统、python.org 或 conda-forge 安装。

默认安装器不安装 Isaac Sim、FlashAttention、SageAttention、DeepSpeed、xcompute、
x2robot_dataset 或其他训练/数据收集依赖。模型 checkpoint、Qwen、RoboDojo Assets、
CUDA/Triton 编译缓存也不包含在环境或 PR 中。

## 2. 创建独立环境

在 `XPolicyLab/policy/vap/` 下执行：

```bash
# 使用 PATH 中的 python3.11；目标必须不存在
bash install.sh
bash install.sh check
CUDA_VISIBLE_DEVICES=0 bash install.sh check --gpu
```

默认创建 `.venv`。已有目标目录会报错，绝不覆盖。显式选择解释器和目标：

```bash
bash install.sh install /path/to/new-vap-policy-env \
  --python /path/to/python3.11
bash install.sh check /path/to/new-vap-policy-env --gpu
```

环境通过标准库 `venv` 创建。安装器从 PyPI 下载并验证固定版本的 pip wheel，
再从该 wheel 用 `--python <新环境>/bin/python` 引导目标环境，不要求外部 Python 已安装 pip，
不修改用于启动安装器的 Python。安装器用原子 `mkdir` 预留最终目标，以最终路径创建 venv，
再用 pip `--python <最终解释器> --prefix <临时目录>` 暂存运行依赖。因此 `bin/pip` 和 `activate`
不会因发布搬移而指向旧的临时路径。bootstrap 专用的 setuptools/wheel 在并入运行依赖前卸载，
避免与运行锁中的同名包冲突；最终 setuptools 版本由完整运行锁提供。运行依赖通过隔离导入检查后
并入最终 venv；普通安装失败会删除本轮预留的目标以允许重试，并发安装者创建的目标不会被删除。
不需要 root、Conda 或访问旧环境。

安装缓存默认在 adapter 的 `.cache/pip/`。可通过 `VAP_INSTALL_CACHE` 指定一个独立
缓存目录；缓存中的文件也必须满足锁文件哈希。最终运行不读取安装缓存。若需零缓存
网络重建，选择一个新的空缓存目录。默认只访问 `https://pypi.org/simple` 和 PyTorch
官方源；如官方 PyPI 传输过慢，可显式添加
`--index-url https://mirrors.aliyun.com/pypi/simple` 使用公共镜像。镜像只改变传输，
所有 wheel 仍必须匹配 PyPI 的哈希；不要传入私有索引或 `find-links`。

`check` 直接调用目标环境的 `bin/python`，不要求 PATH 中存在另一个 `python3.11`。检查使用
`python -I`、禁用用户 site-packages，并检查关键模型模块来自新环境或
本 adapter。`--gpu` 额外执行一个小型 CUDA 矩阵乘法；它不是模型加载或仿真验收。

## 3. 模型和仿真资源

### 3.1 RoboDojo 代码、仿真环境和 Assets

这三项按 [RoboDojo 官方安装与资源下载说明](https://robodojo-benchmark.com/doc/usage/install-and-download/)
准备。VAP 的 `install.sh` 只安装 policy 环境，不另行安装或打包 RoboDojo、Isaac Sim
及仿真 Assets。新机器按官方流程下载代码、创建仿真环境、下载 Assets，并运行官方的
`utils/update_embodiment_config_path.py` 更新机器人资源路径。已有完整安装可以直接复用。

准备完成后，只需填写自己的路径：

```bash
export ROBODOJO_ROOT=/path/to/RoboDojo
export ROBODOJO_SIM_ENV=/path/to/conda/envs/RoboDojo
# 如果资源单独存放，此处指向包含 Background、Eval_Layout 等子目录的 Assets 本身。
export ROBODOJO_ASSETS=/path/to/Assets
```

| 输入 | 如何使用 |
| --- | --- |
| RoboDojo 代码目录 | 将本 PR 的 adapter 放在 `$ROBODOJO_ROOT/XPolicyLab/policy/vap`，从该 RoboDojo 目录启动 |
| 仿真环境目录 | 传给 VAP 标准 `eval.sh` 的第十个参数，要求完整环境路径 |
| Assets 目录 | 官方默认读取 `$ROBODOJO_ROOT/Assets`；资源可直接放在此处，也可通过软链接指向单独目录 |

`ROBODOJO_ROOT` 和 `ROBODOJO_ASSETS` 是下面命令使用的路径变量，不是新增的
RoboDojo 配置开关。与 WallX 一样，adapter 根据自身所在的
`RoboDojo/XPolicyLab/policy/vap` 目录结构确定 benchmark 根目录。
因此仅设置变量不会把另一份 checkout 内的 adapter 切换到指定的 RoboDojo。

如果官方安装已经生成有效的 `$ROBODOJO_ROOT/Assets`，直接使用即可。
若资源单独存放，且该位置尚不存在，可创建链接；下面命令不会覆盖已有目录或链接：

```bash
if [ ! -e "$ROBODOJO_ROOT/Assets" ] && [ ! -L "$ROBODOJO_ROOT/Assets" ]; then
  ln -sT "$ROBODOJO_ASSETS" "$ROBODOJO_ROOT/Assets"
fi
test -d "$ROBODOJO_ROOT/Assets/Eval_Layout"
cd "$ROBODOJO_ROOT"
# conda 应已按官方说明配置在 PATH 中；用于激活官方仿真环境。
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ROBODOJO_SIM_ENV"
python utils/update_embodiment_config_path.py
```

移动 RoboDojo 或更换 Assets 后，按官方说明重新更新资源路径。
当前 VAP 仿真启动会调用 XPolicyLab 的 Conda client launcher，需确保 `conda` 可用；
policy 环境仍使用前文创建的独立 venv。

### 3.2 模型资源

以下输入必须由用户明确提供，不能通过默认私有路径替代：

```bash
export VAP_QWEN_MODEL=/path/to/Qwen3.5-9B
# 完整 VAPv2 checkpoint 在 serving 时严格恢复所有持久 DiT/VAE 张量，不读取 Wan。
# 训练或 legacy VAP 初始化 Wan 时必须显式提供 WALLX_MODEL_ROOT，但这不是
# 本 serving launcher 的输入。
# 仿真环境遵循 RoboDojo 官方安装；不要把 Isaac Sim 安装到 .venv
export ROBODOJO_SIM_ENV=/path/to/robodojo-sim-env
```

checkpoint 必须包含：

```text
CHECKPOINT/
  model.safetensors
  config.yml
  normalizer_action.pth
```

公开仓库为 [`x-square-robot/wall-wm-2.0-robodojo`](https://huggingface.co/x-square-robot/wall-wm-2.0-robodojo)。
[`environment/CHECKPOINT_SOURCES.json`](environment/CHECKPOINT_SOURCES.json) 固定仓库、`revision: "main"`
和三个必需文件名，不保存预设 size 或 SHA256。`download_checkpoint.py` 下载 `main` 当前指向的三个 payload
文件，只检查文件清单完整且各文件非空；目标目录必须不存在，下载完成后才发布到目标路径。
每个文件使用有界的 64 MiB 分块、默认 6 个并发 worker（`--workers 1` 到 `--workers 16`）和有限次数的瞬时错误重试，
并以 ETag 绑定续传内容：

```bash
python -I download_checkpoint.py --output checkpoints/wall-wm-2.0-robodojo --workers 6
python -I download_checkpoint.py --verify-only --output checkpoints/wall-wm-2.0-robodojo
# 直接 Hub 不可达时可显式选择公共镜像；仍读取 main 当前指向的文件。
python -I download_checkpoint.py --base-url https://hf-mirror.com \
  --output checkpoints/wall-wm-2.0-robodojo --workers 6
```

若传输中断，下载器保留 `<目标名>.partial-*` 未完成目录并输出 `CHECKPOINT_INCOMPLETE <路径>`。
使用相同目标目录和公共镜像，将输出的路径显式传给 `--resume` 继续；不要把 partial 目录直接作为运行 checkpoint：

```bash
python -I download_checkpoint.py --base-url https://hf-mirror.com \
  --output checkpoints/wall-wm-2.0-robodojo \
  --resume checkpoints/wall-wm-2.0-robodojo.partial-EXAMPLE
```

组装前会再次校验完整分块。下载器读取当前 `main` 的公开文件元数据：LFS 文件必须与公开
`lfs.oid`（完整文件的 SHA256）一致。Xet `xetHash` 或 bridge ETag 是另一类对象标识，不是普通文件
SHA256；ETag 用于分块和远端版本绑定。`config.yml` 这类普通 Git 文件没有 LFS `oid`，由分块校验、
ETag 绑定及实际下载文件的 SHA256 provenance 提供检查。HEAD 缺少 `Content-Length`（例如 chunked 响应）时使用该必需文件的仓库 API 大小；
HEAD 如果提供长度则仍须与 API 大小一致。弱 ETag 不用于 `If-Range`。镜像若忽略 Range，仅允许单个完整小文件的普通 GET：严格校验状态、相同 ETag、精确字节数和没有尾部额外字节；
不会把新的整文件响应追加到中断的旧分块。已组装的文件也会保留，复用前检查其绑定的远端
ETag 和实际 SHA256；有 LFS `oid` 的文件还必须与当前公开 `oid` 一致。续传遇到目标名、来源镜像、
远端文件大小或 ETag 变化会拒绝继续，不会将旧分块与变化后的远端文件拼接。成功后 partial 目录会被消费；
失败或未完成目录不会发布为可用 checkpoint。

下载 provenance 写在目标旁的 `<目标名>.provenance.json`，记录实际下载文件的 size 和 SHA256，仅用于追溯，
不是发布时的预设白名单；`--verify-only` 重新计算实际文件属性，检查当前本地 payload 与该记录一致。
替换 Hugging Face `main` 中的文件后，无需修改 PR、下载脚本或 manifest。新的下载命令获取当时的 `main`；
已有目录仍拒绝覆盖，也不会自动更新，因此请为替换后的 checkpoint 指定一个新的输出目录。`main` 是可变
分支，下载器不再验证它对应某个历史 commit 或历史大小/哈希，旧下载与旧成绩不会自动代表新文件。
若结构、config、normalizer、Qwen 或动作定义不兼容，仍须完成相应适配和回归/仿真验证；新的正式成绩必须使用
新 checkpoint 独立重新评测，不能沿用旧 run。这里公开了可获取方式，但尚未声称已在干净机器上完成完整安装和
cold-cache 下载复现。

`download_qwen.py` 默认从官方 Hugging Face 完整 revision 下载 `Qwen/Qwen3.5-9B`。
运行所需 13 个文件的 size、SHA256 和 checkpoint Qwen identity 固定于
`environment/QWEN_SOURCES.json`。目标目录必须不存在，全部文件验证后才发布到目标路径：

```bash
python -I download_qwen.py --output /path/to/new-Qwen3.5-9B
python -I download_qwen.py --verify-only --output /path/to/Qwen3.5-9B
# 直接 Hub 不可达时，可显式选择公共镜像；镜像只是 transport。
python -I download_qwen.py --output /path/to/new-Qwen3.5-9B \
  --base-url https://www.modelscope.cn/models --transport-revision master
```

完整文件清单、Apache-2.0 license 链接与官方 revision 见 manifest。镜像 `master`
仅是传输别名，不能替代官方 commit identity；不会重新上传或替换官方 Qwen。
下载 provenance 写在目标旁的 `<目标名>.provenance.json`，不放入模型目录：运行时会将目录
内所有 JSON 纳入 checkpoint identity，额外文件会导致不一致。`--verify-only` 同时要求
13 个 payload 文件和该旁置 provenance 匹配。不得使用未完成的 `*.partial-*` 目录。

`VAP_OUTPUT_ROOT`、`VAP_CACHE_ROOT` 分别控制运行结果与缓存。默认结果位于 RoboDojo
checkout 同级目录的 `results/RUN_ID/`（不是 adapter 内）；编译与模型缓存默认位于 adapter 的
`.cache/`。README 示例将两者显式指向提交目录之外的可写路径，干净复测也应这样做。
标准 `eval.sh` 和未修改的官方 `scripts/robodojo.sh eval` 都会生成共享运行 ID 并初始化
`$VAP_OUTPUT_ROOT/$ROBODOJO_RUN_ID`，不要求预先设置 `VAP_RUN_DIR`。官方 setup-only
server/client 分别根据相同的 bench、task、checkpoint、robot、action、seed、eval 模式和
eval 次数推导同一 ID，不读取各自的时间戳、PID 或端口。相对 checkpoint 按标准 resolver 的
`policy/vap/` 基准规范化；bare name 同样优先选择 concatenated run-directory，再回退到
`checkpoints/<name>`，server 和 client 启动目录不同也不会改变身份。server 原子创建目录并写入身份与
共享 server 请求签名，client 校验两者后复用；共享签名不含 task 和 eval 次数，使显式身份下的
同一 batch server 可以服务所分配的多个 task，而不要求知道每个 client 的 episode 数，但仍绑定
checkpoint、robot、action、seed 和 eval 模式。默认 combined ID 仍包含 task 和 eval 次数；
已有目录不会被覆盖，相同请求重跑也不是断点续跑。
直接 `eval.sh` 仍使用唯一时间戳 owner ID；显式传入的 run ID 和 smoke 每任务 ID 优先。
VAP `eval.sh` 另外将 server/client 日志分别写入 `$VAP_RUN_DIR/server.log`
和 `client.log`，记录 `request.json` 与 `exit_code.txt`；官方统一入口使用其原有调度日志，
调用方应保存该日志，运行身份与模型准备记录仍位于共享 `VAP_RUN_DIR`。
拆分 server/client 时，双方必须显式提供相同的 `ROBODOJO_RUN_ID` 和
`VAP_RUN_DIR`，先启动 server；通过 XPolicyLab 标准 client launcher 调用本 adapter 的
`setup_eval_env_client.sh`，并设置 `ROBODOJO_SIM_ENV`。不要求官方父仓库自动选择
policy-specific client setup；client 不会自行创建缺失目录或加入身份、请求不符的目录。
`smoke_all_tasks.sh` 的 split 模式按未过滤的任务 assignment 为同组 server/client 复用
`<run-id>_<bootstrap-task>`；split resume 只跳过执行，不改变 assignment/bootstrap；combined
parallel eval resume 保持原有的活跃任务 assignment 和每任务 run ID。server 和 client
默认 summary 文件分离。原生结果使用同一 run ID 写在 RoboDojo `eval_result/`；有效 smoke
必须具有该 run 对应的新结果目录、`eval_time >= 1` 的 `_result.json` 和预期视频。
不能仅依赖 launcher exit 0：未修改的父 launcher 可能掩盖 client 失败，固定版本的原生 client
也不保证 Kit 异常的非零退出状态。VAP 在环境端 adapter 内配置自己的 transport 等待时限，
不要求父 client 转发这些字段；兼容补丁不能作为 XPolicyLab-only 复测的前置条件。
`POLICY_ENV` 可显式覆盖 `.venv`，`SIM_ENV` 由标准 `eval.sh` 第十个参数提供。

## 4. 用真实 checkpoint 验收

标准十参数入口如下；先选择空闲 GPU，不终止其他任务：

```bash
cd XPolicyLab/policy/vap
export VAP_QWEN_MODEL=/path/to/Qwen3.5-9B
export VAP_OUTPUT_ROOT=/path/to/external/vap-results
export VAP_CACHE_ROOT=/path/to/external/vap-cache
export ROBODOJO_SIM_ENV=/path/to/robodojo-sim-env
export VAP_REQUIRE_NO_WAN=1  # 可选审计：清除 Wan 变量，任何 Wan 文件 open 立即失败

EVAL_ENV_TYPE=debug ROBODOJO_RUN_ID=vap_public_debug_raw \
  bash eval.sh RoboDojo stack_bowls /path/to/CHECKPOINT \
  arx_x5 ee 0 0 0 .venv /path/to/lightweight-debug-client-env

EVAL_ENV_TYPE=debug DEBUG_OBS_ENCODED=1 ROBODOJO_RUN_ID=vap_public_debug_encoded \
  bash eval.sh RoboDojo stack_bowls /path/to/CHECKPOINT \
  arx_x5 ee 0 0 0 .venv /path/to/lightweight-debug-client-env
```

两种 debug 均须完成 10 episodes、exit 0、无 traceback。debug client 可以是轻量环境，
需要 NumPy、OpenCV headless、PyYAML、WebSockets，不需要 Isaac Sim 或 Torch。

真实仿真需要官方 RoboDojo 完整环境和有效 Assets，不要求修改 RoboDojo 父仓库或应用
[诊断用 WebSocket timeout 兼容补丁](compat/README.md)。从 RoboDojo 根目录运行标准 combined 入口，
不要提供外部 run ID：

```bash
unset ROBODOJO_RUN_ID VAP_RUN_DIR
EVAL_NUM=1 bash scripts/robodojo.sh eval \
  --policy-dir XPolicyLab/policy/vap \
  --task stack_bowls --ckpt /path/to/CHECKPOINT \
  --policy-env /path/to/vap-policy-env --eval-env "$ROBODOJO_SIM_ENV" \
  --policy-gpu 0 --env-gpu 1
```

固定版本的父 client 在构造 WebSocket client 时不转发 VAP 的 `deploy.yml` timeout 字段。
环境端 VAP adapter 因此自行读取同目录配置，在自己的首次 policy RPC 前配置现有官方 client
的公开 `PolicyEvalClientConfig`：request timeout 为 600 秒，后续重连循环的总预算为 1800 秒。
初始连接已由父 client 建立；600 秒请求预算覆盖首次 CUDA Graph 编译的等待时间，不改变请求内容，且不会在超时后重发 stateful
action。维护者不需要修改父仓库或应用私有兼容补丁。

launcher exit 0 之外，还必须检查该次运行的 `_result.json`，要求 `eval_time >= 1`。
一次 episode、debug 或 import/CUDA 成功都不能证明统计性能、全任务通过或优化数值等价。

## 5. 锁文件维护与 eval-only 范围

通常不需要重新解析依赖。维护者更新 `requirements.in` 后，使用 CPython 3.11 和公开 pip
执行 `python3.11 -I environment/lock.py /path/to/new-wheelhouse`：解析、下载并保留完整
wheelhouse，再逐一核对 PyPI 元数据及 SHA256，生成完整锁。直接 URL 的 Torch 包还要求
wheel 实际哈希等于 `requirements.in` 的片段。目标 wheelhouse 和锁均不得覆盖已有文件。
随后必须通过新的空环境安装和真实 checkpoint debug，不能只检查文本。

本 adapter 为 eval-only，不提供 `process_data.sh`、`train.sh`、私有环境复制工具或训练
集群启动器。训练发布计划和维护者信息尚未提供，公开 leaderboard 提交前仍需补齐。

## 6. 当前验证边界

本轮已在一个新的独立环境中完成公共哈希锁定依赖安装、导入/依赖检查、实际 CUDA
矩阵乘法，并用显式提供的本地 120k checkpoint 和 Qwen 输入完成 raw/encoded 各
10 episodes；两种运行均 exit 0、无 traceback。`VAP_REQUIRE_NO_WAN=1` 的审计在两次
debug 中均未发现 Wan 文件访问。完整 checkpoint 的 old/new CPU 持久状态、VAE 非持久
normalization 和固定输入 A0/A1/reset 动作已比较；无 Wan 候选与参考路径六个 FP32
`[32,20]` 动作张量逐位一致。这仅是同一 GPU、固定输入和本地模型的 serving 数值/debug
证据，不是全任务数值或性能等价证明。

无 Wan 原生 simulator 已完成两次独立的 `stack_bowls` 单 episode smoke：本地已验证 Qwen 的
`sim_r2`，以及独立下载并逐文件验证的官方 Qwen 的 `sim_public_qwen`。两次 launcher 均 exit 0，
Wan 审计均无访问记录，各自 `_result.json` 均为 `eval_time=1`、`success_rate=1.0`、`score=100.0`。
后者 runtime 配置精确指向独立下载目录。这些是原生 smoke 的单 episode 结果，不是统计性能或
leaderboard 结果。原始本地证据由提交者单独归档，不是公开 adapter 的运行配置。

公开 checkpoint 发布、从远端 PR commit 在无 CPFS 的机器重建和完整 benchmark 是后续
验收，不由安装、debug 或单 episode 成功替代。公共 Qwen 的 13 个文件已完成全目录
SHA256 复验，固定 prompt 的 BF16 `[2,512,4096]` 文本特征与本地已验证副本逐位一致。
下载 provenance 位于模型目录的同级 `Qwen3.5-9B.provenance.json`，不进入 Qwen 自身的
identity 哈希范围。本轮 deterministic acceptance 已退出 0；raw/encoded debug、模型状态、
固定输入动作和上述原生 smoke 证据均通过其检查，但不扩大上述验证边界。
