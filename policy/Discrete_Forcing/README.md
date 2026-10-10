# Discrete Forcing

**Original code:** [Discrete Forcing](https://github.com/Jbo-Wang/discrete_forcing) |
**Project:** [discrete-forcing.github.io](https://discrete-forcing.github.io/index.html)

`Discrete_Forcing` connects Discrete Forcing to the RoboTwin / XPolicyLab
evaluation stack for clean tasks with absolute joint actions. The model
implementation is installed under `source_discrete_forcing/`.

Shared conventions, checkpoint naming and split-machine deployment are documented
in the [XPolicyLab README](../../README.md).

## Installation

Place this checkout at `RoboTwin/XPolicyLab` in an XPolicyLab-enabled RoboTwin
workspace. Install RoboTwin and its assets in a separate simulator environment
following the benchmark's instructions.

```bash
conda create -n discrete_forcing python=3.10 -y
conda activate discrete_forcing
cd RoboTwin/XPolicyLab/policy/Discrete_Forcing
bash install.sh
```

Use a CUDA-compatible Linux environment for the policy and the benchmark's own
environment for simulation. Set `DF_ROOT` to use an existing model checkout.

## Data Processing

The adapter consumes prepared RoboTwin clean LeRobot v2.0/v2.1 datasets with
standard feature keys and the upstream `meta/modality.json` mapping.
`process_data.sh` validates and links the export; it does not convert raw data.
See the upstream [data preparation guide](https://github.com/Jbo-Wang/discrete_forcing/blob/acce1e9/docs/RoboTwin.md).

```bash
DF_DATA_ROOT=/path/to/robotwin-datasets \
  bash process_data.sh RoboTwin clean <env_cfg_type> joint
```

The dataset root must contain `Clean/<task>/` for all 50 clean tasks.

## Training

```bash
bash train.sh RoboTwin clean <env_cfg_type> joint 42 0,1,2,3,4,5,6,7
```

Training uses the upstream RoboTwin clean configuration. Trailing arguments are
forwarded as config overrides. Checkpoints are saved under
`checkpoints/<bench_name>-<ckpt_name>-<env_cfg_type>-<action_type>-<seed>/`.
Use a new `ckpt_name` for each run.

## Evaluation

Run from `RoboTwin/XPolicyLab/policy/Discrete_Forcing`:

```bash
# Evaluate one clean task.
bash eval.sh RoboTwin <task_name> clean <env_cfg_type> joint 42 \
  0 0 discrete_forcing <robotwin_env>
```

Use the `env_cfg_type` corresponding to the checkpoint's training robot.
`ckpt_name` accepts a run name or an explicit `.pt` path. Keep `config.yaml`
and `dataset_statistics.json` in the run directory, with weights in its
`checkpoints/` subdirectory.

The policy and RoboTwin simulator run in separate environments. Task settings
and rollout counts are configured by RoboTwin. For split-machine evaluation,
see the [deployment guide](../../README.md#-deployment-flow).

## Configuration

Adapter-specific `deploy.yml` keys: `model_root`, `checkpoint_path`,
`checkpoint_file`, and `unnorm_key`. `DF_ROOT` overrides the model checkout;
`DF_DATA_ROOT` selects the prepared training data.
