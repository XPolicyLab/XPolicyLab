# Installation

The Harness runs on Linux with NVIDIA GPUs. One conda environment, `harness-policy`,
holds everything on the RoboDojo path as separate processes on one interpreter: the CLI
and episode driver, the simulator worker, the kernel that runs model-written programs,
and the model servers behind the skills.

!!! tip "Without a simulator"
    The configuration, the CLI's `config`, `lock` and `report` commands, and the test
    suite need only Python 3.11 and the standard library. You can read, resolve and
    compare configurations on any machine, e.g. a laptop.

## 1. Checkouts

The simulator and the learned policies come from upstream repositories at pinned
revisions, recorded in `third_party/sources.toml`. Print the plan for a checkout:

```bash
python scripts/install_plan.py robodojo
```

It prints the clone, checkout and submodule commands without running them. Put the
RoboDojo checkout (with its submodules: XPolicyLab, Isaac Lab, cuRobo) at
`third_party/RoboDojo`, or anywhere and set `HARNESS_ROBODOJO_ROOT`. Its simulator
assets are a separate download (`<RoboDojo>/scripts/init_assets.sh`).

Optional checkouts, also in `third_party/sources.toml`:

| Checkout | For |
|---|---|
| SAM 3 (`SAM3_ROOT`) | the `sam3` segmentation service |

## 2. The environment

```bash
bash environment/setup.sh
```

The script creates `harness-policy` (Python 3.11) and installs it in layers: PyTorch
2.7 (CUDA 12.8), Isaac Sim 5.1, Isaac Lab and cuRobo from the RoboDojo checkout,
RoboDojo's runtime and the XPolicyLab client, this
repository (editable), the inference dependencies of the learned skills and perception,
and this documentation's tools. Every layer installs under
`environment/constraints.txt`, so nothing can move torch, numpy or Isaac Sim.

```bash
bash environment/setup.sh --from project    # resume at a step
bash environment/setup.sh --only lock       # refresh environment/lock/
```

Steps: `conda torch isaacsim isaaclab curobo robodojo project skills docs
verify lock`. The [environment reference](../reference/environment.md) lists the pins,
the lock files and the known deviations from upstream.

## 3. Machine settings and keys

Enter the environment with

```bash
source environment/activate.sh
```

It activates the conda environment, sets `HARNESS_ROBODOJO_ROOT`, accepts the
Omniverse EULA, and sources `~/.harness_env`, the place for machine-specific settings:

```bash
# ~/.harness_env  (chmod 600; never commit it)
export DEEPSEEK_API_KEY=...                 # default model, deepseek-flash
export OPENROUTER_API_KEY=...        # only for a model file that uses OpenRouter
export HF_ENDPOINT=https://hf-mirror.com
export TMPDIR=/data/me/tmp           # Isaac Lab logs under $TMPDIR/isaaclab
```

!!! warning "Keys"
    Keys never go into configuration files. A model's config names the variable that
    holds its key (`api_key_env`, default `OPENROUTER_API_KEY`).

## 4. Check

```bash
python environment/check.py          # the pinned core: versions, imports, CUDA
bash environment/smoke.sh 0          # one RoboDojo episode with XPolicyLab's demo policy on GPU 0
python -m unittest discover -s tests
```

## 5. A deploy for this machine

A deploy file says which GPUs the project may use, where model weights live and where
each service runs. Start from an example:

```bash
cp configs/deploy/example-1gpu.toml configs/deploy/mine.toml
harness doctor --deploy mine         # files, GPUs, free memory, ports, interpreters
harness fetch sam3 --deploy mine     # download the weights the deploy needs
```

The examples cover one GPU, two GPUs and no GPU for models;
[Services and deploys](../concepts/services.md) explains the choices. Next:
[Quickstart](quickstart.md).
