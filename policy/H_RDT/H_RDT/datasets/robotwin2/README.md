# H-RDT RobotWin2 Data Processing

This directory contains scripts for processing RobotWin2 dataset for H-RDT fine-tuning.

## Quick Start

### 1. Setup Environment

Edit the paths in `setup_robotwin2.sh` according to your environment:

```bash
# Edit the script with your paths
nano datasets/robotwin2/setup_robotwin2.sh

# Then source it to set up environment variables
source datasets/robotwin2/setup_robotwin2.sh
```

**Required paths to configure (only if processing):**
- `ROBOTWIN2_DATA_ROOT`: Path to your RobotWin2 dataset
- `T5_MODEL_PATH`: Path to your T5-v1_1-xxl model

### 2. Run Pipeline (Not Required)

**Since pre-computed language embeddings are already provided in the repository, you do NOT need to run the pipeline.**

The pipeline is only needed if you want to regenerate files:

```bash
# Default: Skip all processing steps (recommended - files already provided)
./datasets/robotwin2/run_robotwin2_pipeline.sh

# Only run if you need to regenerate specific files:
ENABLE_STATS_CALCULATION=true ./datasets/robotwin2/run_robotwin2_pipeline.sh
ENABLE_LANGUAGE_ENCODING=true ./datasets/robotwin2/run_robotwin2_pipeline.sh
```

### 3. Run Individual Steps (Not Required)

**These steps are not required since pre-computed files are already provided:**

```bash
# Step 1: Calculate statistics (not needed, stats.json already provided)
python datasets/robotwin2/calc_stat.py

# Step 2: Encode language embeddings (not needed, lang_embeddings/ already provided)
python datasets/robotwin2/encode_lang_batch.py
```

## Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `ROBOTWIN2_DATA_ROOT` | RobotWin2 dataset root directory | Required |
| `T5_MODEL_PATH` | T5-v1_1-xxl model path | Required |
| `HRDT_PROJECT_ROOT` | H-RDT project root | Auto-detected |
| `HRDT_CONFIG_PATH` | Config file path | `configs/hrdt_finetune.yaml` |
| `HRDT_OUTPUT_DIR` | Output directory | `datasets/robotwin2` |
| `NUM_PROCESSES` | Number of processes | 8 |
| `NUM_GPUS` | Number of GPUs | 8 |
| `PROCESSES_PER_GPU` | Processes per GPU | 4 |

### Command Line Arguments

Scripts accept command line arguments that override environment variables:

```bash
# Calculate statistics with custom settings
python datasets/robotwin2/calc_stat.py

# Encode language embeddings (uses environment variables)  
python datasets/robotwin2/encode_lang_batch.py
```

## Directory Structure

```
datasets/robotwin2/
├── setup_robotwin2.sh              # Environment setup script
├── run_robotwin2_pipeline.sh       # Complete pipeline runner
├── calc_stat.py                    # Step 1: Calculate statistics
├── encode_lang_batch.py            # Step 2: Batch encode language embeddings
├── robotwin_agilex_dataset.py      # RobotWin2 dataset loader
├── stats.json                     # Pre-computed statistics file
├── task_instructions.csv          # Task instruction mapping
├── lang_embeddings/               # Pre-computed language embeddings
│   ├── adjust_bottle.pt
│   ├── beat_block_hammer.pt
│   ├── ...                        # All task embeddings
│   └── turn_switch.pt
└── README.md                      # This file
```

## Expected Dataset Structure

Your RobotWin2 dataset should be organized as (only needed if running processing steps):

```
$ROBOTWIN2_DATA_ROOT/
├── task1/
│   ├── demo_clean/data/
│   │   ├── episode0.hdf5
│   │   ├── episode1.hdf5
│   │   └── ...
│   └── ...
├── task2/
├── task3/
└── ...
```

## Available Files

The repository includes pre-computed files:

1. **Statistics**: `stats.json` with min/max values for action normalization (Note: Not used in RobotWin2 training)
2. **Language Embeddings**: `lang_embeddings/*.pt` files with T5 embeddings for all tasks
3. **Task Instructions**: `task_instructions.csv` with task name to instruction mapping

## Language Embedding Usage

The dataset loader automatically reads language embeddings from the centralized `lang_embeddings/` directory. Each task's embedding is stored as `{task_name}.pt` containing:

```python
{
    "name": "task_name",
    "instruction": "task instruction text", 
    "embeddings": torch.Tensor  # T5 embeddings
}
```

The embeddings are loaded using relative paths, making the system portable across different environments.