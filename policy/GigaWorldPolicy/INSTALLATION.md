# GigaWorldPolicy Installation

Install into the current Python/conda environment, or point `GIGAWORLD_CONDA_ENV` at a conda env to install into:

```bash
# Option A: install into the currently active environment
bash install.sh

# Option B: install into a specific conda env
export GIGAWORLD_CONDA_ENV=/path/to/conda/envs/gigaworld-policy
bash install.sh
```

`install.sh` first installs a CUDA-matched PyTorch build, then installs both the policy package under `policy/GigaWorldPolicy/giga_world_policy` and the XPolicyLab repo in editable mode. All other dependencies are declared in `giga_world_policy/pyproject.toml` (validated on Python 3.11 + CUDA 12.8).

PyTorch install knobs (defaults target cu128):

```bash
export TORCH_INDEX_URL=https://download.pytorch.org/whl/cu128  # match your CUDA
export TORCH_VERSION=2.8.0
export TORCHVISION_VERSION=0.23.0
export SKIP_TORCH_INSTALL=1   # skip if a suitable torch is already installed
```

The pretrained video backbone (Wan2.2-TI2V-5B-Diffusers) is not bundled. Download it and point the runtime to it via an environment variable:

```bash
export GIGAWORLD_PRETRAINED_PATH=/path/to/Wan2.2-TI2V-5B-Diffusers
# WAN22_DIFFUSERS_PATH is accepted as an alias.
```

Common runtime overrides:

```bash
export GIGAWORLD_PRETRAINED_PATH=/path/to/Wan2.2-TI2V-5B-Diffusers
export GIGAWORLD_NORM_PATH=/path/to/norm_stats_delta.json
export GIGAWORLD_CONFIG=configs.xpolicylab_gigaworld.config
export GIGAWORLD_WANDB_PROJECT=gwp-xpolicylab
```

## XPolicyLab Deploy

Run `bash install.sh` before first use. The deploy policy server env is normally `gigaworld-policy` or the conda env passed to `install.sh`; the env client uses the XPolicyLab environment.

`setup_eval_policy_server.sh` and `setup_eval_env_client.sh` support manual split-machine evaluation and are also used by `eval.sh`.

Example checkpoint layout under `policy/GigaWorldPolicy/`:

```text
checkpoints/<6-tuple>/checkpoint-<step>/model_ema.pt
```

Manual evaluation:

```bash
# terminal 1 - server
bash setup_eval_policy_server.sh XPolicyLab stack_bowls <ckpt_name> arx_x5 3500 joint 0 0 gigaworld-policy <port> localhost

# terminal 2 - client
bash setup_eval_env_client.sh XPolicyLab stack_bowls <ckpt_name> arx_x5 joint 0 0 XPolicyLab "ckpt_name=<ckpt_name>,action_type=joint" <port> localhost
```

Or run `eval.sh`, which allocates a port, starts the server, waits for readiness, and then starts the client.

## Training and Evaluation

See [README.md](README.md) for XPolicyLab data conversion, training, and evaluation usage.
