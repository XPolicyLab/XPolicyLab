# Riemann-RSI-Preview

**贡献者：** [bolwang](https://huggingface.co/bolwang) | **论文：** 暂无 | **arXiv：** 暂无 | **原始代码：** 本目录。

Riemann-RSI-Preview 通过标准 XPolicyLab 接口执行 RoboDojo 任务，包含任务技能、
工作流与 [Riemann1.0](https://huggingface.co/bolwang/Riemann-RSI-Preview_robodojo) 推理。

参数含义、checkpoint 命名、分机部署和 `EVAL_ENV_TYPE` 等公共约定见
[XPolicyLab README](https://github.com/XPolicyLab/XPolicyLab/blob/main/README.md)；
官方结果见 [RoboDojo LeaderBoard](https://robodojo-benchmark.com/LeaderBoard)。

## 安装

将本目录放入 `XPolicyLab/policy/Riemann-RSI-Preview/`。

环境要求：

- Linux、Python **3.11.15**。
- RoboDojo `f57e1633ffed59761bc8e568cfe7530c4b33381c`，按[官方指南](https://robodojo-benchmark.com/doc/usage/install-and-download/)安装模拟器和资产。
- XPolicyLab 协议基线 `aa46382aa12dc46b360e2fdc63d0d3d39215b3e9`，或协议实现相同的后代版本；本地 Git 历史中需保留该基线。
- 任务工作流使用一张物理 GPU，Riemann1.0 推理使用两张不同物理 GPU。已在 RTX 5090 上测试，最低显存需求尚未确定。

设置本机路径，下载模型和资源，再创建策略环境：

```bash
cd XPolicyLab/policy/Riemann-RSI-Preview
export RIEMANN_OFFICIAL_ROOT=/absolute/path/to/RoboDojo-official
export RIEMANN_EVAL_PYTHON=/absolute/path/to/robodojo-official/bin/python
export RIEMANN_RESOURCES_BASE="$HOME/.cache/riemann-rsi/resources"
export RIEMANN_POLICY_ENV="$HOME/.venvs/riemann-rsi"

bash download_checkpoints.sh --destination "$RIEMANN_RESOURCES_BASE" --prepare \
  --official-assets "$RIEMANN_OFFICIAL_ROOT/Assets"

RIEMANN_INSTALL_PYTHON="$RIEMANN_EVAL_PYTHON" bash install.sh \
  --robodojo "$RIEMANN_OFFICIAL_ROOT" --eval-python "$RIEMANN_EVAL_PYTHON" \
  --create-policy-env "$RIEMANN_POLICY_ENV" \
  --resources-base "$RIEMANN_RESOURCES_BASE" \
  --wam-checkpoint "$RIEMANN_RESOURCES_BASE/Riemann1.0" \
  --wam-encoder "$RIEMANN_RESOURCES_BASE/encoder"
```

使用已有兼容环境时，将 `--create-policy-env` 换成
`--policy-python /existing/env/bin/python`。添加 `--plan` 可预览安装。
单独指定资源目录时使用 `--resource-root NAME=/absolute/path`，
可用名称见 `configs/resource-roots.json`。

安装后可检查配置：

```bash
RIEMANN_INSTALL_PYTHON="$RIEMANN_EVAL_PYTHON" bash install.sh --check-only
```

如果 Vulkan/X11 库不在系统加载路径中，安装时传入
`--native-library-dir /absolute/path/to/native/lib`，或用
`RIEMANN_NATIVE_LIBRARY_DIRS` 设置以路径分隔符分隔的目录列表。

## 数据处理

本次发布仅支持评测，不包含数据处理流程或训练数据集。

```bash
bash process_data.sh
```

脚本会提示不支持数据处理，并以状态码 2 退出。

## 训练

本次发布仅提供推理和评测，不包含训练代码，暂无训练代码的公开发布时间。

```bash
bash train.sh
```

脚本会提示不支持训练，并以状态码 2 退出。

## 评测

安装后使用标准十参数接口：

```bash
bash eval.sh <bench> <task> <checkpoint> <robot> <action> <seed> \
  <policy_gpu> <env_gpu> <policy_python_or_conda_env> <official_eval_python>

# Rule 示例：一张物理 GPU，joint 动作。
bash eval.sh RoboDojo insert_key rule arx_x5 joint 0 \
  0 0 "$RIEMANN_EVAL_PYTHON" "$RIEMANN_EVAL_PYTHON"

# Riemann1.0 示例：两张物理 GPU，EE 动作。
bash eval.sh RoboDojo stack_bowls "$RIEMANN_RESOURCES_BASE/Riemann1.0" arx_x5 ee 0 \
  0 1 "$RIEMANN_POLICY_ENV/bin/python" "$RIEMANN_EVAL_PYTHON"
```

环境参数可以填写解释器路径、环境目录或 Conda 名称。评测采用 SDK 默认回合数：
泛化任务的标准和随机模式各 25 条，其他模式各 50 条。

每个策略进程只支持一个活动环境，`eval_batch` 默认为 `false`。
分机部署按 XPolicyLab 公共指南操作。Riemann1.0 策略服务端的 GPU 参数填写
两个物理卡号，例如 `5,6`；环境客户端填写一张模拟器 GPU 的卡号。

### 结果

RoboDojo 本地评测结果，按回合加权。Score 为官方 reward 的均值，按 0–100 展示。

| 成功率 | Score / 100 |
|---:|---:|
| 64.14% | 69.86 |

## 模型和资源

| 组件 | 内容 |
|---|---|
| Riemann1.0 | `Riemann1.0/` 下的策略权重和归一化文件。 |
| 公用视觉模型 | SAM3、CLIP ViT-B/32、GroundingDINO Tiny，以及 Wan2.2 VAE/T5/tokenizer。 |
| 任务检测器 | 传送带和模仿任务使用的分类器。 |
| 模板 | 任务技能使用的视觉和几何模板。 |
| 机器人几何 | 从官方 SDK 的 `Assets/` 目录准备。 |
| 运行依赖 | SAM3 运行组件和 WebSocket 协议依赖。 |

模型和模板从 [Hugging Face 仓库](https://huggingface.co/bolwang/Riemann-RSI-Preview_robodojo/tree/e0fa3f2a920eb74ae105194829e13cb07b861c08)
的 `e0fa3f2a920eb74ae105194829e13cb07b861c08` 版本下载。
下载器检查文件大小和 SHA256，重复运行会复用匹配的文件。
下载内容保存在 `RIEMANN_RESOURCES_BASE` 下。

## 配置

`deploy.yml` 设置策略名称、WebSocket 主机/端口及评测默认值。
Riemann1.0 可选配置为 `wam_checkpoint`、`encoder_path`、`device` 和 `t5_device`。

SDK、Python 和资源路径通过上面的安装参数配置。
每次运行的任务、checkpoint、动作类型、seed 和 GPU 通过评测命令选择。
