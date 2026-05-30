# Converting DROID from Scratch

If you want to reproduce the DreamZero DROID dataset conversion yourself (or modify the filtering), follow the steps below. This requires the raw DROID 1.0.1 dataset in RLDS format and the idle filter ranges JSON.

> **Most users should skip this** and simply download the preprocessed dataset:
> ```bash
> huggingface-cli download GEAR-Dreams/DreamZero-DROID-Data --repo-type dataset --local-dir ./data/droid_lerobot
> ```

## Step 1: Install conversion dependencies

```bash
pip install tensorflow tensorflow-datasets polars av
```

## Step 2: Download the raw DROID 1.0.1 dataset

This requires `gsutil` ([Google Cloud CLI](https://cloud.google.com/storage/docs/gsutil_install)). The full dataset is ~1.7TB.

```bash
gsutil -m cp -r gs://gresearch/robotics/droid/1.0.1 ./data/droid/1.0.1
```

> **Important:** Use version 1.0.1, not 1.0.0. Version 1.0.1 contains the complete set of language annotations (~75k episodes).

## Step 3: Download the idle filter ranges

This JSON file maps each episode to the frame ranges that should be kept (non-idle frames). It was originally computed by [Physical Intelligence](https://github.com/Physical-Intelligence/openpi) for training pi0-DROID models.

```bash
gsutil cp gs://openpi-assets/droid/droid_sample_ranges_v1_0_1.json ./data/keep_ranges.json
```

## Step 4: Run the conversion

```bash
python scripts/data/convert_droid.py \
    ./data/droid/1.0.1 \
    ./data/droid_lerobot \
    --keep-ranges-path ./data/keep_ranges.json \
    --filter-failed \
    -n 16
```

For a quick test with a small subset:
```bash
python scripts/data/convert_droid.py \
    ./data/droid/1.0.1 \
    ./data/droid_lerobot_test \
    --keep-ranges-path ./data/keep_ranges.json \
    --filter-failed \
    --first-n 5 \
    -n 4
```

## Script reference

See [`scripts/data/convert_droid.py`](scripts/data/convert_droid.py) for full usage:

```
python scripts/data/convert_droid.py --help
```