export HDF5_USE_FILE_LOCKING=FALSE
export RAY_DEDUP_LOGS=0

export TMPDIR=/mnt/workspace/yangyandan/my_tmp
export RAY_TMPDIR=/mnt/workspace/yangyandan/ray_tmp
export RAY_OBJECT_STORE_MEMORY=4000000000
export RAY_USE_SHM=false  # 容器环境建议加上

# Step 3: 创建目录
mkdir -p $RAY_TMPDIR


python robomind_h5_v3_new.py \
    --src-path /mnt/nas-data-4/gaowo.cyz/RoboMIND \
    --output-path /mnt/xlab-nas-2/vla_dataset/lerobot/robomind_11_new \
    --benchmark benchmark1_1_compressed \
    --no-save-images \
    --embodiments ur_1rgb franka_3rgb  sim_franka_3rgb  tienkung_gello_1rgb  tienkung_xsens_1rgb \
    --cpus-per-task 6


