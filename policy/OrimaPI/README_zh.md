# OrimaPI

**Contributor:** Orimaton Team | **Paper:** TBD | **arXiv:** TBD | **Original code:** [Orimaton/OrimaPI](https://github.com/Orimaton/OrimaPI)

[English](README.md)

OrimaPI 在 RoboDojo-Sim 的 `arx_x5` 上运行 `orimapi/skills/code_skills/` 下的正式 skill，输出关节空间动作（`action_type: joint`）。适配器读取内置 `OrimaPI/` 源码中的 `configs/robodojo.yaml`，使用 `code-policy`、`direct` 模式，每次观测执行一个动作；GPT-6-Astra 根据任务指令和 RGB 观测选择并调度 skill。本适配器仅用于评测。

共享约定 —— 参数含义、checkpoint 命名、跨机部署、`EVAL_ENV_TYPE` —— 见 [XPolicyLab README](../../README_zh.md)。官方结果：[RoboDojo LeaderBoard](https://robodojo-benchmark.com/LeaderBoard)。

## Installation

使用配有 NVIDIA GPU 的 Linux 机器，先在名为 `RoboDojo` 的 Conda 环境里装好 RoboDojo，并将本 XPolicyLab checkout 放在 `RoboDojo/XPolicyLab`。以下命令从 RoboDojo 根目录执行。

```bash
conda create -n orimapi python=3.11 -y
conda activate orimapi
bash XPolicyLab/policy/OrimaPI/install.sh
python XPolicyLab/policy/OrimaPI/download_assets.py

export ORIMAPI_NATIVE_PYTHON="$(conda run -n RoboDojo python -c 'import sys; print(sys.executable)')"
```

`install.sh` 将 XPolicyLab 和随仓库提供的 [OrimaPI 源码](OrimaPI/) 安装到当前策略环境。`download_assets.py` 将 Grounding DINO、SAM 和 Depth Anything 下载并校验到 `policy/OrimaPI/OrimaPI/checkpoint/`。原生规划 worker 使用上面导出的 RoboDojo Python 解释器。

## Data Processing

不适用。本适配器为仅评测（eval-only）适配器，直接消费运行时观测，省略 `process_data.sh`。

## Training

不支持。本适配器不包含本地训练入口，省略 `train.sh`。

## Evaluation

标准 10 参数入口：

```bash
bash XPolicyLab/policy/OrimaPI/eval.sh \
  <bench_name> <task_name> <ckpt_name> <env_cfg_type> <action_type> <seed> \
  <policy_gpu_id> <env_gpu_id> <policy_env> <eval_env_conda_env>
```

离线 debug 闭环检查（无需启动仿真器）：

```bash
cd XPolicyLab/policy/OrimaPI
EVAL_ENV_TYPE=debug bash eval.sh RoboDojo stack_bowls code-skills arx_x5 joint 0 0 0 orimapi RoboDojo
DEBUG_OBS_ENCODED=1 EVAL_ENV_TYPE=debug bash eval.sh RoboDojo stack_bowls code-skills arx_x5 joint 0 0 0 orimapi RoboDojo
```

从 RoboDojo 根目录运行 `stack_bowls`，使用 GPU 0：

```bash
EVAL_NUM=1 bash XPolicyLab/policy/OrimaPI/eval.sh \
  RoboDojo stack_bowls code-skills arx_x5 joint 0 0 0 orimapi RoboDojo
```

按各任务的原生回合数跑完一个 seed：

```bash
bash scripts/robodojo.sh benchmark --policy-dir XPolicyLab/policy/OrimaPI \
  --policy-env orimapi --ckpt code-skills --env-cfg arx_x5 --action-type joint \
  --seed 0 --eval-num native
bash scripts/robodojo.sh summarize
```

跨机或分离启动服务端与客户端请参考 [部署流程](../../README_zh.md#-部署流程)。

## Model Assets

`python XPolicyLab/policy/OrimaPI/download_assets.py` 会从 Hugging Face 下载并校验固定的视觉权重（Grounding DINO、SAM 与 Depth Anything V2 Metric Indoor Small）至 `policy/OrimaPI/OrimaPI/checkpoint/`。评测无需 VLA checkpoint，`ckpt_name` 固定传 `code-skills`。

## Configuration

在策略机器上编辑 [OrimaPI/configs/robodojo.yaml](OrimaPI/configs/robodojo.yaml)。`supervisor.model` 是模型 ID，`supervisor.endpoint` 是完整的 Responses URL。默认使用 `gpt-6-astra`，地址为 `https://api.githubcopilot.com/responses`。修改 YAML 后重启策略服务。

默认使用 `supervisor.auth.type: copilot`。在策略机器上创建 `policy/OrimaPI/OrimaPI/configs/copilot.credentials.json`。该文件已被 Git 忽略：

```json
{
  "github_token": "<your GitHub Copilot credential>"
}
```

如果使用接受普通 API Key 的 Responses 兼容服务，将整个 `supervisor.auth` 配置替换为 `auth: {type: env}`，并填写服务对应的 `supervisor.endpoint` 和 `supervisor.model`。通过策略服务的 `OPENAI_API_KEY` 环境变量提供密钥。`supervisor.api_key_env` 指定变量名。配置会校验响应中的模型 ID 是否与请求一致。

额外 `deploy.yml` 配置键与环境变量：

| 配置键 / 环境变量 | 默认值 | 说明 |
| --- | --- | --- |
| `request_timeout_s` | `3600.0` | 客户端单次 WebSocket 推理请求超时（秒） |
| `ORIMAPI_NATIVE_PYTHON` | `OrimaPI/.venv-robodojo/bin/python` | 原生 IK/规划 worker 使用的 RoboDojo Python 解释器 |
| `ROBODOJO_ROOT` | `<XPolicyLab>/..` | 包含 `Assets/` 与 `env_cfg/` 的 RoboDojo 根目录 |
