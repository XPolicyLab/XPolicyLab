# Installation

## Environment Setup

<!-- Use a dedicated Conda environment for training: -->

```bash
conda create -n rise python=3.11.14 -y
conda activate rise

cd /path/to/RISE
bash install.sh
```

For deployment setup, please refer to the [deployment README](deploy.md).

## Data Preparation


The training code expects LeRobot-style data.


The data is first collect on Piper in hdf5 format and then converted to the lerobot format.


### Raw Data Collection
Please refer to the [data collection README](deploy.md) for details.

### Lerobot Conversion

Convert raw HDF5 logs collected on Piper into LeRobot-format data with:

```bash
cd /path/to/RISE/policy_and_value/policy_offline_and_value

python examples/aloha_real/convert_to_lerobot.py \
  --data-dir /path/to/raw_dataset \
  --repo-ids aloha_mobile_dummy \
  --prompt "TASK_PROMPT" \   # e.g., "Pick up the block"
  --save-dir /path/to/lerobot_output_root \
  --save_repoid output_dataset_name \
```

Arguments:

- `--data-dir`: Root directory of one raw dataset collection. The script expects HDF5 files under `<data-dir>/<repo-id>/` and videos under `<data-dir>/<repo-id>/video/`.
- `--repo-ids`: One or more raw subdirectories to convert. For most RISE data, this is `aloha_mobile_dummy`.
- `--prompt`: Task description stored in the converted LeRobot dataset.
- `--save-dir`: Parent directory for converted output.
- `--save_repoid`: Name of the output dataset folder created under `--save-dir`.
- `--overwrite`: Optional. Remove an existing output folder and regenerate it.



The generated output follows the standard LeRobot layout shown below.



### Expected Lerobot Data Layout

```text
<dataset_name>/
├── data/chunk-000/episode_*.parquet
├── meta/
│   ├── info.json
│   ├── episodes.jsonl
│   ├── episodes_stats.jsonl
│   └── tasks.jsonl
└── videos/chunk-000/*.mp4
```
