# RobotWin Data Converter

Converts RobotTwin2.0 dataset from HuggingFace to Motus format.

## Quick Start

```bash
# 1. Activate Motus environment
conda activate your_motus_env

# 2. Download dataset (uses HF mirror by default for faster speed)
python3 download_robotwin_dataset.py --output_dir /path/to/robotwin_raw_dataset

# 3. Edit config.yml
vim config.yml

# 4. Run conversion
./run_conversion.sh
```

## Download Options

```bash
# Download all tasks (default: uses HF mirror)
python3 download_robotwin_dataset.py --output_dir /path/to/save

# Download specific tasks
python3 download_robotwin_dataset.py --tasks adjust_bottle clean_mirror --output_dir /path/to/save
```

## Configuration

Edit `config.yml`:

```yaml
source_root: "/path/to/robotwin_raw_dataset"  
target_root: "/path/to/robotwin_dataset"
wan_repo_path: "/path/to/Wan2.2-TI2V-5B"
```

## Input Structure

```
source_root/
├── {task_name}/
│   ├── aloha-agilex_clean_50/
│   │   ├── data/episode*.hdf5
│   │   └── instructions/episode*.json
│   └── aloha-agilex_randomized_50/
│       ├── data/episode*.hdf5
│       └── instructions/episode*.json
```

## Output Structure

```
target_root/
├── clean/
│   └── {task_name}/
│       ├── videos/0.mp4    # Multi-camera video
│       ├── qpos/0.pt       # Robot positions (T, 14)
│       ├── metas/0.txt     # Instructions with prefix
│       └── umt5_wan/0.pt   # T5 embeddings (optional)
└── randomized/
    └── {task_name}/
        ├── videos/
        ├── qpos/
        ├── metas/
        └── umt5_wan/
```