#!/bin/bash
set -euo pipefail

# Usage: bash process_data.sh <bench_name> <ckpt_name> <env_cfg_type> <action_type> [expert_data_num]
#
# WallX trains on the official end-effector LeRobot export, read as published:
# it already carries what the policy acts in -- position, wxyz quaternion and
# gripper per arm. Nothing is converted here.
#
# What this does fetch is the memory annotations: which frames of a memory task
# recorded progress the policy has to remember. They are published separately,
# because they annotate the official episodes rather than replacing them.
#
# The two have to line up. An annotation bound to the wrong episode trains the
# policy to recall the wrong frames, and that failure is silent -- training
# proceeds, the loss falls. So this checks the correspondence and refuses to
# leave a mismatch behind.
bench_name=$1
ckpt_name=$2
env_cfg_type=$3
action_type=$4
expert_data_num=${5:-}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
XPL_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
DATA_ROOT="$(cd "${XPL_ROOT}/.." && pwd)/data"
EXPORT_DIR="${DATA_ROOT}/RoboDojo_ee_lerobot_v30_video"
ANNOTATION_DIR="${WALLX_ANNOTATIONS:-${DATA_ROOT}/robodojo_memory_annotations}"
ANNOTATION_REPO="${WALLX_ANNOTATION_REPO:-x-square-robot/robodojo-memory-annotations}"

echo "[WallX] episodes:    ${EXPORT_DIR}"
echo "[WallX] annotations: ${ANNOTATION_DIR}"

if [[ ! -f "${EXPORT_DIR}/meta/info.json" ]]; then
    echo "[WallX] fetching the end-effector export"
    bash "${XPL_ROOT}/scripts/RoboDojo/download_robodojo_data.sh" lerobot_ee
fi

if [[ ! -f "${ANNOTATION_DIR}/index.json" ]]; then
    echo "[WallX] fetching the memory annotations from ${ANNOTATION_REPO}"
    python - "$ANNOTATION_REPO" "$ANNOTATION_DIR" <<'FETCH'
import sys
from huggingface_hub import snapshot_download

repo, target = sys.argv[1], sys.argv[2]
snapshot_download(repo_id=repo, repo_type="dataset", local_dir=target)
print("downloaded", repo, "->", target)
FETCH
fi

echo "[WallX] checking that every annotation belongs to an episode of this export"
python - "$EXPORT_DIR" "$ANNOTATION_DIR" <<'CHECK'
"""Bind the annotations to the episodes and verify the binding.

Reads the export's own episode table, through the same code the training
path uses, and compares episode lengths: the historical image/state
off-by-one in this dataset showed up exactly as a length difference of one.
"""
import glob
import json
import os
import sys

export, annotations = sys.argv[1], sys.argv[2]
try:
    from wall_x.data.backends.lerobot.robodojo.episodes import EpisodeIndex
    from wall_x.data.backends.lerobot.robodojo.export import episodes_of
except ImportError:
    sys.exit("[WallX] install wall-x first: it owns the binding this check uses")

try:
    from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
except ImportError:
    sys.exit("[WallX] install lerobot first (see install.sh)")

meta = LeRobotDatasetMetadata(repo_id="RoboDojo-Benchmark/Robodojo", root=export)
episodes = [(e.index, e.instruction, e.length) for e in episodes_of(meta)]

index = EpisodeIndex.from_release(annotations, episodes)
frames_of = {}
for path in glob.glob(os.path.join(annotations, "*", "trajectory_*", "progress.json")):
    task, episode = path.split(os.sep)[-3:-1]
    with open(path, encoding="utf-8") as stream:
        frames_of[f"{task}/{episode}"] = int(json.load(stream)["total_frames"])
index.check_lengths(frames_of)

bound = {f"{index.binding(i).task}/{index.binding(i).episode}"
         for i, _, _ in episodes if index.binding(i) is not None}
unused = sorted(set(frames_of) - bound)
if unused:
    sys.exit(f"[WallX] {len(unused)} annotations match no episode, e.g. {unused[:3]}")
print(f"[WallX] {len(bound)} of {len(episodes)} episodes carry an annotation, "
      f"all lengths agree; the rest train without a memory, as they should")
CHECK

echo "[WallX] Normalization statistics are not measured here: they come with the"
echo "[WallX] weights. The base model ships the pair it was pretrained with, and"
echo "[WallX] stage 2 inherits them from the stage-1 checkpoint."
echo "[WallX] ready: ${bench_name}/${ckpt_name}/${env_cfg_type}/${action_type}${expert_data_num:+ (limit ${expert_data_num})}"
