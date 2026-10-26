# Tools

This directory contains the public utility scripts used by the release
workflows.

- `check_lerobot_v3_integrity.py`: check basic LeRobot v3 dataset integrity.
- `convert_egodex_to_lerobot.py`: convert EgoDex-style data into LeRobot v3.
- `compute_norm_stats_single.py`: compute normalization statistics for one
  LeRobot dataset.
- `compute_norm_stats_multi.py`: compute and aggregate normalization statistics
  across multiple LeRobot datasets.
- `wsa_large_compute_pretrain_norm_stats.sh`: group WSA-Large pretraining
  datasets and compute per-group normalization stats with
  `compute_norm_stats_multi.py`.
- `precompute_text_embeds.py`: build WSA_Large text-embedding caches.
- `preprocess_expert_backbones.py`: prepare Wan2.2 expert backbone weights.
- `discover_robotwin_repos.py`: discover RoboTwin LeRobot repos for training.
