
# ============================================
# 激活 Conda 环境
# ============================================
if [ -n "$CONDA_ROOT" ] && [ -n "$CONDA_ENV" ]; then
  echo "[conda] 从 $CONDA_ROOT 激活环境: $CONDA_ENV"
  source "$CONDA_ROOT/etc/profile.d/conda.sh"
  conda activate "$CONDA_ENV"
fi

python a1/data/vla/robomind_datasets.py --mode build_index \
--embodiment h5_franka_1rgb h5_franka_3rgb h5_agilex_3rgb h5_ur_1rgb \
--dataset_path data/RoboMIND/benchmark1_0_compressed

python a1/data/vla/robomind_datasets.py --mode build_index \
--embodiment h5_agilex_3rgb h5_franka_3rgb h5_ur_1rgb \
--dataset_path data/RoboMIND/benchmark1_1_compressed

python a1/data/vla/robomind_datasets.py --mode build_index \
--embodiment h5_franka_3rgb \
--dataset_path data/RoboMIND/benchmark1_2_compressed
