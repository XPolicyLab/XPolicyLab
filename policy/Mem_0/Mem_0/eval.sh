#!/bin/bash
# LEGACY (RMBench standalone eval). Use policy/Mem_0/eval.sh for XPolicyLab server/client eval.

policy_name=Mem_0

SIM_GPU=${SIM_GPU:-3}
MODEL_GPU=${MODEL_GPU:-4}

export CUDA_VISIBLE_DEVICES=${SIM_GPU},${MODEL_GPU}
echo -e "\033[33mSimulation GPU: ${SIM_GPU}; model GPU: ${MODEL_GPU} (visible as cuda:1)\033[0m"

cd ../..  # move to project root

# Example command, you can change the arguments as needed

# M(1) evaluation format
PYTHONWARNINGS=ignore::UserWarning \
python script/eval_policy.py --config policy/${policy_name}/deploy_policy.yml --overrides \
    --task_name swap_blocks \
    --execution_ckpt ./policy/Mem_0/checkpoints/model.pt \
    --state_stats_path ./policy/Mem_0/assets/model/norm_stats.json \
    --global_task "There are three traies on the table, and two blocks are placed in two different traies. You may move only one block at a time, and each tray can hold at most one block. Swap the positions of the two blocks. Finally press the button." \
    --vllm_url "http://localhost:8000" \
    --device "cuda:1" \
    --action_horizon 30 # Changeable

# M(n) evaluation format
# PYTHONWARNINGS=ignore::UserWarning \
# python script/eval_policy.py --config policy/${policy_name}/deploy_policy.yml --overrides \
#     --task_name cover_blocks \
#     --execution_ckpt ./policy/Mem_0/checkpoints/model.pt \
#     --state_stats_path ./policy/Mem_0/assets/model/norm_stats.json \
#     --global_task "On the table, red, green, and blue blocks are arranged randomly along with three lids. From the current viewpoint, cover the blocks from left to right using the lids, and then uncover them again in the sequence red, green, and blue." \
#     --vllm_url "http://localhost:8000" \
#     --device "cuda:1" \
#     --action_horizon 8 # Changeable
