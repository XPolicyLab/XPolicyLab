# WorldTerminal for RoboDojo

This evaluation adapter connects RoboDojo to a WorldTerminal endpoint. It supports one environment per session and joint control. The split-process setup uses the benchmark environment for simulation and the policy environment for the adapter and SDK.

## Install

From the RoboDojo workspace containing this XPolicyLab policy, activate a Python 3.10+ policy environment and install from the submitted directory. The installer downloads the [WorldTerminal Python SDK](https://github.com/org2AI/worldterminal-sdk) wheel from [WorldTerminal SDK v0.1.0](https://github.com/org2AI/worldterminal-sdk/releases/tag/v0.1.0) and verifies its pinned SHA-256.

```sh
cd "$ROBODOJO_ROOT/XPolicyLab/policy/WorldTerminal"
conda activate policy_env
bash install.sh
```

If you are starting from the standalone SDK source instead, run `bash scripts/stage_xpolicylab.sh "$ROBODOJO_ROOT/XPolicyLab"` from its root first. This creates the policy directory above.

## Data processing

The adapter consumes live XPolicyLab observations. It requires no offline dataset conversion.

## Training

This is an evaluation-only integration. Inference runs on a hosted service.

## Debug the connection

Set `WORLDTERMINAL_API_URL` and, if required, `WORLDTERMINAL_API_TOKEN` in the launching shell. For this test, request a debug endpoint that returns ARX X5 joint actions for synthetic observations. XPolicyLab sends a placeholder instruction and synthetic camera frames. Its debug client needs the parent workspace's `env_cfg/` and can run in the policy environment.

```sh
EVAL_ENV_TYPE=debug bash eval.sh RoboDojo stack_blocks worldterminal arx_x5 joint 0 0 0 policy_env policy_env
EVAL_ENV_TYPE=debug DEBUG_OBS_ENCODED=1 bash eval.sh RoboDojo stack_blocks worldterminal arx_x5 joint 0 0 0 policy_env policy_env
```

Both commands should reach `[MAIN] eval finished`. For simulation, use the task-capable endpoint and the simulator environment as shown below.

## Evaluate

Set WORLDTERMINAL_API_URL and, for authenticated deployments, WORLDTERMINAL_API_TOKEN on the policy host. Replace the last two arguments with your policy and simulator Conda environment names.

```sh
bash eval.sh RoboDojo stack_blocks worldterminal arx_x5 joint 0 0 0 policy_env sim_env
```

The example uses seed 0 and GPU 0. The evaluator controls its task layouts. The adapter forwards observations and polls on waiting before applying joint targets. Episode completion and scoring come from the environment. To enable SDK checks for a matching endpoint, set embodiment to arx_x5/joint-v1 in deploy.yml.

[Upstream RoboDojo setup](https://robodojo-benchmark.com/doc/usage/)
