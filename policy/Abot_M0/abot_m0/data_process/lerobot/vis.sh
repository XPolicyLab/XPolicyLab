# """
# 字段说明：
# --repo-id RoboCOIN \ #保存文件会以此为前缀命名
# --root /mnt/workspace/.cache/modelscope/datasets/RoboCOIN/Split_aloha_fold_the_pants/ \ #对应的可视化lerobot dataset文件地址，地址下需要包含meta/文件夹以及meta/info.json
# --mode local \ #可视化本地文件
# """
set -x
CONDA_ENV_PATH="/mnt/workspace/yangyandan/miniforge3/envs/lerobot/bin/python"

# Use a workspace-local HuggingFace datasets cache to avoid lock permission issues.
export HF_DATASETS_CACHE="$(pwd)/.hf_datasets_cache"
mkdir -p "${HF_DATASETS_CACHE}"


data_root="/mnt/xlab-nas-2/vla_dataset/lerobot/oxe/fmb_dataset_lerobot"  #here the dataset is downloaded in the local machine
repo_id="oxe_fmb"  #name to save , do not need to be related to repo name

#The output file will be saved in the directory ./vis_result/${repo_id}_episode_${episode_index}.rrd
# python -m lerobot.scripts.lerobot-dataset-viz \
PYTHONPATH="$(pwd)/src:${PYTHONPATH}" ${CONDA_ENV_PATH} -m lerobot.scripts.lerobot_dataset_viz \
    --repo-id ${repo_id} \
    --root ${data_root} \
    --mode local \
    --episode-index 1 \
    --save 1 \
    --output-dir "./vis_result/" \
    --num-workers 4
