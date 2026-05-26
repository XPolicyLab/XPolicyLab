export HDF5_USE_FILE_LOCKING=FALSE
export RAY_DEDUP_LOGS=0
export RAY_TMPDIR=/mnt/xlab-nas-2/ray_tmp
mkdir -p $RAY_TMPDIR
# 控制 Ray 进程数量的方法：
# 方法1: 通过环境变量限制 Ray 使用的 CPU 总数（可选）
# export RAY_NUM_CPUS=12  # 例如限制 Ray 最多使用 12 个 CPU

# 方法2: 通过 --max-concurrent-tasks 参数限制同时运行的任务数（在脚本参数中设置）

# 激活 conda 环境（如果使用 conda）
if command -v conda &> /dev/null; then
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate /mnt/workspace/yangyandan/miniforge3/envs/lerobot
fi

python agibot_h5.py \
    --src-path /mnt/nas-data-4/bearbee/AgiBotWorld-Beta/ \
    --output-path /mnt/xlab-nas-2/vla_dataset/lerobot/agibot_convert_3_ \
    --eef-type gripper \
    --cpus-per-task 3 \
    --task-ids task_761 task_734 task_722 task_709\
    --log-path output_aigc \
    --max-concurrent-tasks 10  # 限制同时运行的任务数为 4（可选，不设置则自动计算）   
#  --task-ids task_779 task_786 task_764 task_741 