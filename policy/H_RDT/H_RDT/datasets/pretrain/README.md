# H-RDT Pretrain Data Processing

This directory contains scripts for processing EgoDex dataset for H-RDT pretraining.

## Overview

The pretraining data processing consists of three steps:
1. **Precompute 48D Actions** - Convert hand transforms to 48-dimensional action vectors
2. **Calculate Statistics** - Compute min/max values for action normalization  
3. **Encode Language** - Generate T5 embeddings for language instructions

## Quick Start

### 1. Setup Environment

Edit the paths in `setup_pretrain.sh` according to your environment:

```bash
# Edit the script with your paths
nano datasets/pretrain/setup_pretrain.sh

# Then source it to set up environment variables
source datasets/pretrain/setup_pretrain.sh
```

**Required paths to configure:**
- `EGODEX_DATA_ROOT`: Path to your EgoDex dataset
- `T5_MODEL_PATH`: Path to your T5-v1_1-xxl model

### 2. Run Complete Pipeline

```bash
# Run all three steps automatically
./datasets/pretrain/run_pretrain_pipeline.sh
```

### 3. Run Individual Steps (Optional)

```bash
# Step 1: Precompute 48D actions
python datasets/pretrain/precompute_48d_actions.py

# Step 2: Calculate statistics
python datasets/pretrain/calc_stat.py

# Step 3: Encode language embeddings
python datasets/pretrain/encode_lang_batch.py
```

## Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `EGODEX_DATA_ROOT` | EgoDex dataset root directory | Required |
| `T5_MODEL_PATH` | T5-v1_1-xxl model path | Required |
| `HRDT_PROJECT_ROOT` | H-RDT project root | Auto-detected |
| `HRDT_CONFIG_PATH` | Config file path | `configs/hrdt_pretrain.yaml` |
| `HRDT_OUTPUT_DIR` | Output directory | `datasets/pretrain` |
| `NUM_PROCESSES` | Number of processes | 8 |
| `NUM_GPUS` | Number of GPUs | 8 |
| `PROCESSES_PER_GPU` | Processes per GPU | 4 |
| `FORCE_OVERWRITE` | Force overwrite existing data | true |

### Command Line Arguments

Each script also accepts command line arguments that override environment variables:

```bash
# Precompute 48D actions with custom settings
python datasets/pretrain/precompute_48d_actions.py \
    --data_root /path/to/egodex \
    --num_processes 16 \
    --force_overwrite \
    --test_mode

# Calculate statistics (no additional args needed)
python datasets/pretrain/calc_stat.py

# Encode language embeddings (uses environment variables)
python datasets/pretrain/encode_lang_batch.py
```

## Directory Structure

```
datasets/pretrain/
├── setup_pretrain.sh              # Environment setup script
├── run_pretrain_pipeline.sh       # Complete pipeline runner
├── precompute_48d_actions.py      # Step 1: Precompute actions
├── calc_stat.py                   # Step 2: Calculate statistics
├── encode_lang_batch.py           # Step 3: Encode language
├── egodex_dataset.py              # EgoDex dataset loader
├── egodx_stat.json                # Generated statistics file
├── egodex_large_values.txt        # Outlier detection log
└── README.md                      # This file
```

## Expected Dataset Structure

Your EgoDex dataset should be organized as:

```
$EGODEX_DATA_ROOT/
├── part1/
│   ├── task1/
│   │   ├── 0.hdf5
│   │   ├── 1.hdf5
│   │   └── ...
│   └── task2/
├── part2/
├── part3/
├── part4/
├── part5/
├── extra/
└── test/
```

## Output Files

After processing, you'll have:

1. **48D Action Data**: Added as `actions_48d` key in all HDF5 files
2. **Statistics**: `egodex_stat.json` with min/max values for normalization
3. **Language Embeddings**: `.pt` files alongside each HDF5 file
4. **Log Files**: `egodx_large_values.txt` with outlier information