# demo_policy

**Contributor:** RoboDojo Team | **Paper:** Not applicable — XPolicyLab demo adapter | **arXiv:** Not applicable | **Original code:** Not applicable

`demo_policy` is the minimal reference adapter for checking policy-server / environment-client wiring. It returns zero actions with the correct action keys and dimensions; it does not train a real model or load a checkpoint. `process_data.sh` and `train.sh` are stubs that only demonstrate the standard naming conventions.

Shared conventions — argument meanings, checkpoint naming, split-machine deployment, `EVAL_ENV_TYPE` — are documented in the [XPolicyLab README](../../README.md). Official results: [RoboDojo LeaderBoard](https://robodojo-benchmark.com/LeaderBoard).

## Installation

Installs XPolicyLab in editable mode so the policy server can import `XPolicyLab.policy.demo_policy` and the websocket modules:

```bash
cd XPolicyLab/policy/demo_policy
conda activate <policy_env>  # e.g. demo-policy
bash install.sh
```

`bash install.sh` does not install the RLinf runtime. After the eval install, activate that env and run the runtime script yourself.

```bash
conda activate <policy_env>
bash ../../infra/RLinf/scripts/install_rl_runtime.sh
```

## Data Processing

Stub — prints the standard output naming convention without converting data. The optional trailing argument is an episode-limit placeholder:

```bash
bash process_data.sh <bench_name> <ckpt_name> <env_cfg_type> <action_type> [expert_data_num]

# Example
bash process_data.sh RoboDojo stack_bowls arx_x5 joint
```

## Training

Stub — creates the conventional checkpoint directory `checkpoints/<bench_name>-<ckpt_name>-<env_cfg_type>-<action_type>-<seed>/` and prints where real training code should write:

```bash
bash train.sh <bench_name> <ckpt_name> <env_cfg_type> <action_type> <seed> <gpu_id>

# Example
bash train.sh RoboDojo cotrain arx_x5 joint 0 0
```

`demo_policy` does not read this directory during evaluation; real adapters load weights from their own checkpoint path.

## Evaluation

```bash
cd XPolicyLab/policy/demo_policy
bash eval.sh <bench_name> <task_name> <ckpt_name> <env_cfg_type> <action_type> <seed> \
  <policy_gpu_id> <env_gpu_id> <policy_conda_env> <eval_env_conda_env>

# Example: offline shape/IO smoke check
EVAL_ENV_TYPE=debug bash eval.sh RoboDojo stack_bowls demo arx_x5 joint 0 0 0 <policy_conda_env> <eval_env_conda_env>
```

`EVAL_ENV_TYPE=debug` runs the offline wiring check (no simulator); leave it unset or set `EVAL_ENV_TYPE=sim` for RoboDojo simulation. For split-machine deployment via `setup_eval_policy_server.sh` / `setup_eval_env_client.sh`, follow the [Deployment Flow](../../README.md#-deployment-flow).

## RL training

`rl.py` is the RLinf fill-in. `RL_MODES` is `("eval",)`, so a run with `required_mode: eval` keeps using `get_action` / `get_action_batch`.

To train with PPO or GRPO:

1. Implement `_sample_policy` and `_log_prob_value` in `rl.py`. `sample_actions_rl` and `get_log_prob_value` already return the shapes RLinf expects. Log-probs have to come from the policy sample.
2. Assign the trainable `nn.Module` to `self.model` or `self.policy` in `Model.__init__`.
3. Add `"ppo"` and/or `"grpo"` to `Model.RL_MODES`.
4. Add a Hydra training config under `policy/<policy_name>/rlinf/config/` and set `actor.model.xpolicylab.required_mode` to the mode you enabled. Pi0.5 provides PPO examples for [RoboDojo `stack_bowls`](../Pi_05/rlinf/config/robodojo_stack_bowls_ppo_xpolicylab_pi05.yaml) and [RoboTwin `adjust_bottle`](../Pi_05/rlinf/config/robotwin_adjust_bottle_ppo_xpolicylab_pi05.yaml). For custom model registration, see the [Pi0.5 extension](../Pi_05/rlinf/extension.py).
5. Install the runtime with `bash ../../infra/RLinf/scripts/install_rl_runtime.sh`, then follow [RL training on RoboDojo and RoboTwin](../../README.md#rl-training-on-robodojo-and-robotwin) to select your simulator and launch your config.

`predict_action_batch_rl` returns actions `[B, H, D]` in `pack_robot_state` order, plus `prev_logprobs`, `prev_values` (`[B, 1]`), and `forward_inputs`. `compute_logprob_value` returns `logprobs` (same shape as `prev_logprobs`), `values` (`[B]`), and `entropy` (`[B, 1]`).

## Notes

- When replacing these stubs with a real policy, keep `ckpt_name` stable between data processing, training, and evaluation.
