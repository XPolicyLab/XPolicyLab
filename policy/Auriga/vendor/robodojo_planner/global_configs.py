# Vendored from RoboDojo env/global_configs.py (commit 726e9aabfaa642203722eb126f5eaf0f37f3e1ad).
# Copyright (c) 2025 Yue Chen. MIT License, see LICENSE in this directory.
# Only the constant CuroboPlanner reads. It sizes BatchMotionPlanner, which CuroboPlanner builds and
# warms up in __init__; keep it equal to RoboDojo's value so construction (and the cuRobo sampler
# state after warmup) matches the in-simulation planner.
BATCH_NUM = 10
