# 使用 RLinf 启动 Pi0.5

[English](GUIDE.md)

## 获取仿真工作区

选择一个仿真环境，从存放仓库的目录执行对应命令。已有工作区可跳过克隆。切换仿真时，在新终端进入对应工作区，重新设置环境变量。

### RoboTwin `adjust_bottle`

克隆 RoboTwin 的 `RLinf_support` 分支，在其根目录克隆 XPolicyLab。Pi0.5 的 ALOHA 配置需要 `env_cfg/` 中的两个文件，用 `git restore` 从 RoboTwin 的 `origin/main` 取到当前工作区即可。

```bash
git clone --branch RLinf_support https://github.com/RoboTwin-Platform/RoboTwin.git RoboTwin
cd RoboTwin
git clone https://github.com/XPolicyLab/XPolicyLab.git
git restore --source=origin/main --worktree -- env_cfg/aloha_agilex.yml env_cfg/robot/_robot_info.json
export ROBOTWIN_ASSETS_PATH="$PWD"
export ROBOT_PLATFORM=ALOHA
preset=robotwin_adjust_bottle
```

在 RoboTwin 根目录运行 `bash script/_download_assets.sh` 下载资产。若已有资产，将 `ROBOTWIN_ASSETS_PATH` 指向包含 `assets/` 的目录。

### RoboDojo `stack_bowls`

使用含 RLinf 适配的 `RoboDojo/XPolicyLab` 子模块。首次安装时，官方脚本会创建 Python 3.11 的 `RoboDojo` Conda 环境，并安装 Isaac Sim、Isaac Lab 和 CuRobo。已有该环境和仓库时，直接进入 RoboDojo 根目录。[RoboDojo 安装与资产下载](https://robodojo-benchmark.com/doc/usage/install-and-download/)

```bash
git clone https://github.com/RoboDojo-Benchmark/RoboDojo.git
cd RoboDojo
bash scripts/install.sh -i
```

在 RoboDojo 根目录激活仿真环境。首次准备资产时，在激活后执行 `bash scripts/init_assets.sh` 和 `python utils/update_embodiment_config_path.py`。保存仿真解释器路径后，在同一终端切换到策略训练环境。

```bash
conda activate RoboDojo
export ROBODOJO_ROOT="$PWD"
export ROBODOJO_SIM_PYTHON="$CONDA_PREFIX/bin/python"
conda deactivate
preset=robodojo_stack_bowls
```

## 准备策略环境

从所选仿真仓库的根目录进入 XPolicyLab。

```bash
cd XPolicyLab
```

首次使用时，按 [Pi0.5 训练环境安装说明](../../../policy/Pi_05/README.md#rlinf-training-environment) 准备 Linux CUDA 环境。RoboTwin 的环境进程还需要该 Python 环境中的 RoboTwin 仿真依赖。[RoboTwin 依赖安装](https://rlinf.readthedocs.io/zh-cn/latest/rst_source/examples/embodied/robotwin.html)

在同一终端激活策略训练环境（示例使用上面安装的 `.venv`）。将 `checkpoints/pi05` 换成自己的 Pi0.5 PyTorch 权重目录，例如包含 `model.safetensors` 的目录。训练还需要与权重匹配的 `norm_stats.json`，用于归一化机器人状态和动作。

```bash
source .venv/bin/activate
export POLICY_MODEL_PATH="$PWD/checkpoints/pi05"
```

## 启动训练

上一步的 `preset` 选择仿真环境和任务，`algo` 选择训练算法（`ppo` 或 `grpo`）。先打印配置，核对 `actor.model.model_path` 和 `cluster.component_placement`（策略更新、仿真和采样使用的 GPU），按本机可用 GPU 调整对应的任务 YAML，再执行最后一行。默认输出目录为 `infra/RLinf/results/`。[任务 YAML](../../../policy/Pi_05/rlinf/config)

```bash
algo=ppo
config="${preset}_${algo}_xpolicylab_pi05"
bash infra/RLinf/scripts/run.sh "$config" --cfg job --resolve
bash infra/RLinf/scripts/run.sh "$config"
```

Pi0.5 使用 PyTorch 在配置的 GPU 上采样和更新策略。若要让 OpenPI 中的 JAX 运算使用 CPU，为 PyTorch 留出显存，可以在启动命令前加 `JAX_PLATFORMS=cpu`。

## 更换任务

**RoboTwin** 的 `preset` 选择训练配置。以 `lift_pot` 为例，它与 `adjust_bottle` 使用相同的环境参数。保留 RoboTwin 的 `config`，将上面的最后一条启动命令替换为：

```bash
bash infra/RLinf/scripts/run.sh "$config" \
  env.train.task_config.task_name=lift_pot \
  env.eval.task_config.task_name=lift_pot \
  runner.logger.experiment_name="lift_pot_${algo}"
```

在命令末尾加 `--cfg job --resolve` 可先核对训练和评估任务。若新任务的步数上限或机器人配置不同，复制对应算法的[训练 YAML](../../../policy/Pi_05/rlinf/config)，把 `defaults` 中 `@env.train` 和 `@env.eval` 两项都指向`infra/RLinf/RLinf/examples/embodiment/config/env/` 下新任务的环境 YAML，核对权重和归一化统计后，用新配置名启动。

**RoboDojo** 在启动命令末尾加 `task_name` 即可同时切换训练、评估和策略指令。下面以 `align_blocks` 为例，按它的原生上限运行 200 步。

```bash
bash infra/RLinf/scripts/run.sh "$config" task_name=align_blocks episode_steps=200
```

`episode_steps` 是每回合的步数预算，应为策略每次输出的动作数（`actor.model.num_action_chunks`）的整数倍。任务仍可提前成功或达到自身步数上限。任务名取自 RoboDojo 的 `task/RoboDojo/tasks/`，奖励和成功判定来自任务本身。

RoboDojo 配置默认从权重目录的 `assets/robodojo_<任务名>/` 读取 `norm_stats.json`。例如，权重中的文件是 `assets/arx_x5_sim/norm_stats.json` 时，在启动前执行 `export PI05_REPO_ID=arx_x5_sim`。若文件位于 `assets/` 以外，在训练命令末尾加 `"+actor.model.xpolicylab.norm_stats_path=<统计文件所在目录>"`。RoboTwin 示例的默认目录是 `assets/RoboTwin_lerobot_v30/`。
