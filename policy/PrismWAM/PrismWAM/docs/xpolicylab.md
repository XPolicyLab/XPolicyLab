# XPolicyLab integration

RoboTwin 2.0 and RoboDojo are evaluated through [XPolicyLab](https://github.com/XPolicyLab/XPolicyLab)
with the adapter in [`integrations/xpolicylab/policy/PrismWAM`](../integrations/xpolicylab/policy/PrismWAM/README.md).

## Install the adapter

```bash
python integrations/xpolicylab/install_adapter.py --xpolicylab-root /path/to/XPolicyLab --dry-run
python integrations/xpolicylab/install_adapter.py --xpolicylab-root /path/to/XPolicyLab
```

The adapter goes to `XPolicyLab/policy/PrismWAM`. Tested with XPolicyLab
[`ddf5dcf`](https://github.com/XPolicyLab/XPolicyLab/tree/ddf5dcfff5e089a323da1d22891909b72864647e);
transport dependencies are pinned in [`environments/xpolicylab.txt`](../environments/xpolicylab.txt).

## Evaluate a released policy

```bash
export PRISMWAM_ROOT=/path/to/PrismWAM
cd "$PRISMWAM_ROOT"
hf download THUSI-Lab/PrismWAM --include "robodojo/*" --local-dir weights

cd /path/to/XPolicyLab/policy/PrismWAM
export PRISMWAM_RECIPE=robodojo PRISMWAM_DEPTH_KEY=approximate_depth
EVAL_ENV_TYPE=sim bash eval.sh RoboDojo stack_bowls "$PRISMWAM_ROOT/weights/robodojo" \
  arx_x5 joint 0 0 0 /path/to/envs/prismwam robodojo
```

Arguments follow XPolicyLab:
`BENCH TASK CHECKPOINT ROBOT joint SEED POLICY_GPU ENV_GPU POLICY_ENV EVAL_CONDA_ENV`, with
`CHECKPOINT` the model folder. Split-machine evaluation and the `deploy.yml` options are
in the [adapter README](../integrations/xpolicylab/policy/PrismWAM/README.md). `DRY_RUN=1`
prints a launcher's commands. Full-suite runs are in the [RoboTwin](benchmarks/robotwin.md#evaluate)
and [RoboDojo](benchmarks/robodojo.md#evaluate) guides.

Each camera must provide real metric depth; set `depth_scale: 1` for millimetres or `1000` for metres.

LIBERO uses 7-D delta end-effector actions and does not go through this adapter; see the
[LIBERO guide](benchmarks/libero.md#evaluate).
