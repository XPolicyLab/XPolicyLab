# Run Pi0.5 with RLinf

[中文](GUIDE_zh.md)

## Clone a simulator workspace

Choose one simulator and run its commands from the directory where you keep your repositories. Skip the clone commands if you already have the checkouts. To switch simulators, open a new terminal in the other workspace and set its environment variables.

### RoboTwin `adjust_bottle`

Clone RoboTwin’s `RLinf_support` branch, then clone XPolicyLab in the RoboTwin root. The Pi0.5 ALOHA configuration reads two files under `env_cfg/`. Use `git restore` to copy them from RoboTwin’s `origin/main` into the same checkout.

```bash
git clone --branch RLinf_support https://github.com/RoboTwin-Platform/RoboTwin.git RoboTwin
cd RoboTwin
git clone https://github.com/XPolicyLab/XPolicyLab.git
git restore --source=origin/main --worktree -- env_cfg/aloha_agilex.yml env_cfg/robot/_robot_info.json
export ROBOTWIN_ASSETS_PATH="$PWD"
export ROBOT_PLATFORM=ALOHA
preset=robotwin_adjust_bottle
```

Run `bash script/_download_assets.sh` from the RoboTwin root to download its assets. If you already have them, set `ROBOTWIN_ASSETS_PATH` to the directory containing `assets/`.

### RoboDojo `stack_bowls`

Use the `RoboDojo/XPolicyLab` submodule containing the RLinf adapter. On first install, the official script creates the `RoboDojo` Conda environment with Python 3.11 and installs Isaac Sim, Isaac Lab and CuRobo. If the environment and repository are already set up, start in the RoboDojo root. [RoboDojo installation and assets](https://robodojo-benchmark.com/doc/usage/install-and-download/)

```bash
git clone https://github.com/RoboDojo-Benchmark/RoboDojo.git
cd RoboDojo
bash scripts/install.sh -i
```

Activate the simulator environment from the RoboDojo root. On first setup, run `bash scripts/init_assets.sh` and `python utils/update_embodiment_config_path.py` after activation to prepare the assets. Save the simulator interpreter path before switching to the policy training environment in the same terminal.

```bash
conda activate RoboDojo
export ROBODOJO_ROOT="$PWD"
export ROBODOJO_SIM_PYTHON="$CONDA_PREFIX/bin/python"
conda deactivate
preset=robodojo_stack_bowls
```

## Set up the policy environment

Enter XPolicyLab from the simulator repository root.

```bash
cd XPolicyLab
```

On first use, follow the [Pi0.5 training environment setup](../../../policy/Pi_05/README.md#rlinf-training-environment) on Linux with CUDA. For RoboTwin, the environment worker also needs the RoboTwin simulator dependencies in this Python environment. [RoboTwin dependency installation](https://rlinf.readthedocs.io/en/latest/rst_source/examples/embodied/robotwin.html)

In the same terminal, activate the policy training environment (the example uses `.venv` from the setup above). Replace `checkpoints/pi05` with your Pi0.5 PyTorch checkpoint directory, for example the directory containing `model.safetensors`. Training also needs the matching `norm_stats.json` to normalize robot states and actions.

```bash
source .venv/bin/activate
export POLICY_MODEL_PATH="$PWD/checkpoints/pi05"
```

## Launch

The `preset` you set above selects the simulator and task. Set `algo` to `ppo` or `grpo`. Print the config and check `actor.model.model_path` and `cluster.component_placement` (the GPUs for policy updates, simulation and sampling). Adjust the task YAML to your available GPUs before running the final command. Outputs default to `infra/RLinf/results/`. [Task YAML files](../../../policy/Pi_05/rlinf/config)

```bash
algo=ppo
config="${preset}_${algo}_xpolicylab_pi05"
bash infra/RLinf/scripts/run.sh "$config" --cfg job --resolve
bash infra/RLinf/scripts/run.sh "$config"
```

Pi0.5 uses PyTorch on the configured GPUs for rollout and policy updates. To keep OpenPI’s JAX operations on CPU and leave GPU memory available for PyTorch, you can prefix the launch command with `JAX_PLATFORMS=cpu`.

## Change the task

**RoboTwin** uses `preset` to select a training config. For example, `lift_pot` uses the same environment settings as `adjust_bottle`. Keep the RoboTwin `config` and replace the final launch command above with this command.

```bash
bash infra/RLinf/scripts/run.sh "$config" \
  env.train.task_config.task_name=lift_pot \
  env.eval.task_config.task_name=lift_pot \
  runner.logger.experiment_name="lift_pot_${algo}"
```

Append `--cfg job --resolve` to check the training and evaluation task names first. If the new task has different step limits or robot settings, copy the [training YAML](../../../policy/Pi_05/rlinf/config) for your algorithm. Point both `@env.train` and `@env.eval` entries in `defaults` to the new environment YAML under `infra/RLinf/RLinf/examples/embodiment/config/env/`, check the weights and normalization statistics, then launch with the new config name.

**RoboDojo** accepts a single `task_name` override for training, evaluation and policy instructions. For example, run `align_blocks` with its native 200-step limit.

```bash
bash infra/RLinf/scripts/run.sh "$config" task_name=align_blocks episode_steps=200
```

Set the per-episode step budget, `episode_steps`, to a multiple of the number of actions the policy outputs at once (`actor.model.num_action_chunks`). Episodes can end earlier on success or at the task’s own step limit. Choose a task from RoboDojo’s `task/RoboDojo/tasks/`. Each task supplies its own reward and success criteria.

The RoboDojo config looks for `norm_stats.json` under `assets/robodojo_<task_name>/` in the checkpoint. For example, if the file is `assets/arx_x5_sim/norm_stats.json`, run `export PI05_REPO_ID=arx_x5_sim` before launch. For files outside `assets/`, append `"+actor.model.xpolicylab.norm_stats_path=<directory-containing-norm_stats.json>"` to the training command. The RoboTwin examples default to `assets/RoboTwin_lerobot_v30/`.
