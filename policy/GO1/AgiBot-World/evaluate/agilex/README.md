# AgileX Data Processing, Fine-tuning and Inference Guide

This README provides instructions for processing data collected from AgileX robots, fine-tuning GO-1 model, and performing real robot inference.

## Table of Contents
- [AgileX Data Processing, Fine-tuning and Inference Guide](#agilex-data-processing-fine-tuning-and-inference-guide)
  - [Table of Contents](#table-of-contents)
  - [Data Processing](#data-processing)
    - [1. Analyze Outliers](#1-analyze-outliers)
    - [2. Configure Dataset Features](#2-configure-dataset-features)
    - [3. Convert HDF5 to LeRobot Dataset](#3-convert-hdf5-to-lerobot-dataset)
  - [Model Fine-tuning](#model-fine-tuning)
  - [Model Deployment](#model-deployment)
    - [1. Start GO-1 Server](#1-start-go-1-server)
    - [2. Start AgileX Client](#2-start-agilex-client)


## Data Processing

### 1. Analyze Outliers

```bash
python agilex_dataset_filter.py --root_dir /path/to/your/hdf5/files
```

This tool performs comprehensive outlier analysis on HDF5 files by calculating Z-scores based on global mean and standard deviation across all data. It identifies data points that exceed the specified threshold (default 5 standard deviations) and outputs a detailed list of files containing these outliers. The analysis helps ensure data quality by flagging potentially problematic episodes that may contain sensor errors or abnormal robot behaviors.

**Arguments**:
- `root_dir` - Directory containing HDF5 files
- `std_threshold` - Standard deviation threshold (default: `5.0`)

### 2. Configure Dataset Features

Set data key names and shapes in `agilex_features.py`.

### 3. Convert HDF5 to LeRobot Dataset

```bash
# Activate the GO-1 environment
conda activate go1

python convert_agilex_data_to_lerobot.py \
    --src_path /path/to/your/hdf5/files \
    --tgt_path /path/to/output \
    --repo_ids <REPO_IDS> \
    --save_repoid <SAVE_REPO_ID>
```

Arguments:
- `src_path` - Path to the directory containing HDF5 files
- `tgt_path` - Path to save the converted LeRobot dataset
- `repo_ids` - List of folders to process in `src_path`
- `save_repoid` - Repo ID for saving the converted dataset (default: the first item in `repo_ids`)

**Notes:** 

- We use the `observations/qpos` as both state and action. (The default `action` key cannot replay.)

- Set the `exclude_files` list in the script to **exclude files containing outliers**.


## Model Fine-tuning

Refer to the [GO-1 Fine-tuning Guide](../../README.md#fine-tuning-on-your-own-dataset-) for detailed instructions.

## Model Deployment

### 1. Start GO-1 Server 

Start the GO-1 inference server using your fine-tuned model checkpoint and data statistics:


```bash
conda activate go1

python evaluate/deploy.py --model_path /path/to/your/checkpoint --data_stats_path /path/to/your/dataset_stats.json --port <SERVER_PORT>
```

The server will will listen on port `SERVER_PORT` and wait for observations.

### 2. Start AgileX Client

```bash
python agilex_inference.py --host <SERVER_IP> --port <SERVER_PORT> --ctrl_type joint --publish_rate 30 --chunk_size 30
```

**Arguments:**
- `host` - IP address of the server (default: `127.0.0.1`)
- `port` - Port number of the server (default: `9000`)
- `ctrl_type` - Control type of the robot (`joint` or `eef`, default: `joint`)
- `publish_rate` - Action publishing frequency (default: `30`)
- `chunk_size` - Executed action chunk size, should be smaller than the action chunk size of the model (default: `30`)

