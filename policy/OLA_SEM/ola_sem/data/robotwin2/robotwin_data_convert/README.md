# RoboTwin data conversion

These utilities convert RoboTwin 2.0 episodes into the format consumed by the
OLA-SEM loader. They are source code only; no downloaded or converted data is
included.

## Download and convert

```bash
python download_robotwin_dataset.py --output_dir ./data/robotwin_raw_dataset

# Edit source_root, target_root, and wan_repo_path first.
python robotwin_converter.py --config config.yml
```

## Generate language actions

Run the following commands from the repository root. First, extract absolute
end-effector poses from the raw HDF5 episodes into `epos/*.pt`:

```bash
python data/robotwin2/robotwin_data_convert/robotwin_generate_epos_from_raw.py \
  --raw-root ./data/robotwin_raw_dataset \
  --target-root ./data/robotwin_dataset
```

Then convert the absolute XYZ/RPY poses into 16-step language-action summaries
stored as `language_action/*.txt`:

```bash
python data/robotwin2/robotwin_data_convert/robotwin_generate_language_action.py \
  --target-root ./data/robotwin_dataset \
  --input-dir-name epos \
  --input-mode absolute_xyzrpy \
  --window-size 16
```



## Input layout

```text
robotwin_raw_dataset/
└── <task>/
    ├── aloha-agilex_clean_50/
    │   ├── data/episode*.hdf5
    │   └── instructions/episode*.json
    └── aloha-agilex_randomized_500/
        ├── data/episode*.hdf5
        └── instructions/episode*.json
```

## Output layout

```text
robotwin_dataset/
├── clean/<task>/
│   ├── videos/0.mp4
│   ├── qpos/0.pt
│   ├── metas/0.txt
│   ├── umt5_wan/0.pt
│   ├── epos/0.pt
│   └── language_action/0.txt
└── randomized/<task>/
    ├── videos/
    ├── qpos/
    ├── metas/
    ├── umt5_wan/
    ├── epos/
    └── language_action/
```


