#!/bin/bash


cd $(dirname $(realpath $0))

# bash ./ray_utils/start_ray_unified_multi_task.sh
bash ray_utils/start_ray_unified_multi_task.sh


CURRENT_RANK=${node_rank:--1}
CONFIG_NAME=$1

echo ">>> Current Node Rank: $CURRENT_RANK"

if [ "$CURRENT_RANK" -eq 0 ]; then
    bash ./examples/embodiment/run_embodiment.sh ${CONFIG_NAME}
fi