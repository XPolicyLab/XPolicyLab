# WorldScape_RSI

[English](README.md) · [官网](https://r2si.worldscape.io) · [模型权重](https://huggingface.co/manifoldai-research/WorldScape-R2SI)

**贡献者：** WorldScape 团队  
**支持：** RoboDojo · ARX X5（`arx_x5`）· 关节动作

WorldScape_RSI 是结合视觉推理、通用操作 Skill 和 WorldScape Flash/Memory 策略的机器人控制 Agent。Agent 根据当前观测与执行进展，逐阶段选择原生工具或学习策略，并可在已确认的动作边界接管、丢弃剩余 Policy 动作。

本目录包含 Agent Harness、通用库、WAM 实现、RoboDojo 适配和训练脚本，可独立于外层 RSI 循环运行。

## 组件

| 路径 | 用途 |
|---|---|
| `model.py`、`server.py` | 官方 Agent 适配与单环境服务 |
| `harness/` | Agent、通用 Skill、原生工具、WAM 调用与版本 |
| `wam_model.py`、`xpolicylab_adapter/` | WorldScape 模型集成 |
| `worldscape-policy/` | 模型源码、数据管线与训练配置 |
| `deploy.yml`、`eval.sh` | 官方评测配置与入口 |
| `wam_deploy*.yml`、`serve_robodojo.sh` | 独立 WAM 部署 |

经验库包含 **34 份通用 Skill、4 个工具和 2 个原语**，在任务间共享。WAM 接收环境原始指令，动作历史随执行确认推进。

## 安装

使用 Linux、Python 3.11 和 NVIDIA CUDA。按 [INSTALLATION.md](INSTALLATION.md) 创建环境，然后运行：

```bash
bash install.sh
# 训练模型时额外安装：
python -m pip install -e './worldscape-policy[train]'
```

设置仿真资产和 Agent 服务：

```bash
export ROBODOJO_ROOT=/path/to/RoboDojo
export R2SI_API_BASE=https://your-provider.example/v1
export R2SI_API_KEY=your-api-key
```

Agent 配置位于 `harness/configs/inner_agent.json`：模型 `gpt-6-astra`、推理强度 `high`，保留完整回合上下文。官方机器人配置放在父工作目录的 `env_cfg/` 下。

## 模型权重

```bash
bash download_checkpoints.sh /data/models/WorldScape-R2SI
export WRSI_WAM_MEMORY_CHECKPOINT=/data/models/WorldScape-R2SI/wsp_2_memory_posttrain_robodojo
export WRSI_WAM_NOMEMORY_CHECKPOINT=/data/models/WorldScape-R2SI/wsp_2_flash_posttrain_robodojo
export WRSI_WAM_MEMORY_DEVICE=cuda:1
export WRSI_WAM_NOMEM_DEVICE=cuda:0
python -B preflight.py --check-runtime --check-cuda --check-robot-assets --check-checkpoints
```

权重至少预留 60 GB 存储。GPU 编号相对于服务进程的 `CUDA_VISIBLE_DEVICES`。权重文件哈希随 Harness 版本保存。

## 数据处理

RoboDojo 训练通过 XPolicyLab 图像工具读取官方 HDF5 轨迹。准备阶段生成归一化统计和时序索引：

```bash
DATA_ROOT=/data/RoboDojo ROBODOJO_PREPARED_DIR=/data/rsi-prepared \
  ROBODOJO_MODEL=flash bash process_data.sh RoboDojo baseline arx_x5 joint
```

Memory 数据管线使用 `ROBODOJO_MODEL=memory`。

## 训练

下载 Stage-1 初始化权重：

```bash
bash download_pretrained.sh /data/models/WorldScape-Policy-2
export PRETRAINED_MODEL_PATH=/data/models/WorldScape-Policy-2/wsp_2_pretrain
```

将 `DATA_ROOT`、`ROBODOJO_INDEX_PATH`、`ROBODOJO_ZSCORE_STATS` 设置为准备好的数据，然后训练：

```bash
ROBODOJO_MODEL=flash OUTPUT_DIR=/data/rsi-training/flash \
  bash train.sh RoboDojo baseline arx_x5 joint 62 0,1,2,3
```

训练 Memory 时，在数据准备和训练两处均设置 `ROBODOJO_MODEL=memory`。额外参数作为配置覆盖项传给训练配方。

[原生训练](worldscape-policy/docs/native_training.md) · [数据热加载](worldscape-policy/docs/robodojo_hotload.md)

## 评测

绑定权重和 Agent 服务后，使用官方十参数入口：

```bash
bash eval.sh RoboDojo stack_bowls public arx_x5 joint 0 0,1 0 r2si base
```

参数依次为：基准、任务、权重标签、机器人配置、动作类型、seed、策略 GPU、仿真 GPU、策略 conda 环境、仿真 conda 环境。

分机部署使用 `setup_eval_policy_server.sh` 和 `setup_eval_env_client.sh`。每个 Agent 服务处理一个环境（`eval_batch: false`）。官方评测加载 `model.py`，独立 WAM 接口由 `serve_robodojo.sh` 提供。

协议调试在同一评测命令前设置 `EVAL_ENV_TYPE=debug`；再加 `DEBUG_OBS_ENCODED=1` 可测试编码观测。

## 配置

| 设置 | 用途 |
|---|---|
| `deploy.yml` | Agent 服务及 XPolicyLab 标准字段 |
| `release_path`、`release_sha256` | 选择和绑定 Harness 版本 |
| `harness/configs/inner_agent.json` | Agent 模型与服务环境变量名 |
| `harness/configs/wam_skills.json` | Flash/Memory 能力及权重绑定 |
| `R2SI_SESSION_RELEASE_PATH` | 可选的服务会话释放接口 |
| `WRSI_POLICY_CLIENT_CONFIG` | 可选的远程 WAM 客户端配置 |
| `WRSI_INTEGRITY_MODE=full` | 动作间进行完整内容校验 |

两种 WAM 均使用公开权重的 rolling 设置：每个 chunk 24 个动作、观测间隔 3、8 段历史。Flash 权重容量为一个 chunk，因此视觉输入使用最新帧；Memory 容量为 24 个 chunk。完成 chunk 后保留连续状态，Agent 接管或切换策略时清空未执行预测和旧上下文。

## 测试

从 XPolicyLab 的父工作目录运行：

```bash
export PYTHONPATH="$PWD:$PWD/XPolicyLab"
export R2SI_TEST_XPL_ROOT="$PWD/XPolicyLab"
python -m pip install -r XPolicyLab/policy/WorldScape_RSI/requirements-test.txt
python -B -m pytest -p no:cacheprovider -q XPolicyLab/policy/WorldScape_RSI/harness/tests
```

实现结构见 [Harness 说明](harness/README.md)，组件来源见[第三方说明](harness/THIRD_PARTY.md)，版本测试记录见 `VALIDATION.json`。

