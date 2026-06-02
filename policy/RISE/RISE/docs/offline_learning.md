# Offline Policy Training and Value Model

Run all below commands from `policy_and_value/policy_offline_and_value`.

## Available Release Configs

The release registers these config names in `src/openpi_value/training/config.py`:

- `Policy_offline_release`
- `value_release`
- `Compute_norm`

## Compute Normalization Stats

```bash
python scripts/compute_norm_stats_fast.py --config-name Compute_norm
```

## Train Policy or Value Models

```bash
# usage: bash train.sh <CONFIG_NAME> <NUM_GPUS> [extra args]
bash train.sh Policy_offline_release 8
bash train.sh value_release 8
```

Resume example:

```bash
bash train.sh Policy_offline_release 8 --resume
```

## Label Per-frame Value Predictions with the Value Model

* Labeling value and advantage for lerobot datasets using learned value model. 

* This step is required for both offline and online policy improvement with advantage conditioning.

```bash
# usage: bash label_value.sh <CONFIG_NAME> <CKPT_DIR>
bash label_value.sh vis_value_release_joint_T /path/to/checkpoints/value_release_joint/<exp>/<step>
```

## Visualize Value Predictions

```bash
# usage: bash vis_value.sh <CONFIG_NAME> <CKPT_DIR>
bash vis_value.sh vis_value_release_joint_T /path/to/checkpoints/value_release_joint/<exp>/<step>
```

