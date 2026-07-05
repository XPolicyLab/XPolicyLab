# Data Preparation

The training code expects LeRobot-style data.

## Expected Dataset Layout

```text
<bench_name>/
├── data/chunk-000/episode_*.parquet
├── meta/
│   ├── info.json
│   ├── episodes.jsonl
│   ├── episodes_stats.jsonl
│   └── tasks.jsonl
└── videos/chunk-000/*.mp4
```

## Useful Utilities

- ALOHA conversion example:
  - `policy_and_value/policy_offline_and_value/examples/aloha_real/convert_to_lerobot.py`
- Lightweight LeRobot tools:
  - `policy_and_value/policy_offline_and_value/mini_lerobot/`
- Dynamics-model video resize helper:
  - `dynamics/dynamics_model/preprocess.sh`
