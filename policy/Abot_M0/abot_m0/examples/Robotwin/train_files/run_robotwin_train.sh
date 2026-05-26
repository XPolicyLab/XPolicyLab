
###########################################################################################
# === Please modify the following paths according to your environment ===
Framework_name=ABot_M0
freeze_module_list=""
base_vlm=path_to_your_base_vlm_checkpoint
config_yaml=./examples/Robotwin/train_files/ABot_robotwin.yaml
libero_data_root=path_to_your_data_root_dir
data_mix=robotwin
pretrain_ckpt=path_to_your_pretrain_ckpt
run_root_dir=path_to_your_run_root_dir
run_id=your_run_id
# === End of environment variable configuration ===
###########################################################################################


export WANDB_MODE=disabled
export WANDB_MODE=offline
export WANDB_DISABLED=true
export CUDA_HOME=/usr/local/cuda-12
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export PATH=$CUDA_HOME/bin:$PATH
export PATH="$HOME/.local/bin:$PATH"
export HF_ENDPOINT=https://hf-mirror.com 


output_dir=${run_root_dir}/${run_id}
mkdir -p ${output_dir}
# mv this script to the output dir
cp $0 ${output_dir}/


accelerate launch \
  --config_file ABot/config/deepseeds/deepspeed_zero2.yaml \
  --num_processes 8 \
  ABot/training/train.py \
  --config_yaml ${config_yaml} \
  --framework.name ${Framework_name} \
  --framework.qwenvl.base_vlm ${base_vlm} \
  --datasets.vla_data.data_root_dir ${libero_data_root}\
  --datasets.vla_data.data_mix ${data_mix} \
  --trainer.pretrained_checkpoint ${pretrain_ckpt} \
  --trainer.reload_modules qwen_vl_interface,action_model \
  --datasets.vla_data.num_workers 4 \
  --datasets.vla_data.per_device_batch_size 4 \
  --datasets.vla_data.include_state false \
  --trainer.vla_data.video_backend torchvision_av \
  --trainer.freeze_modules ${freeze_module_list} \
  --trainer.max_train_steps 150000 \
  --trainer.save_interval 5000 \
  --trainer.logging_frequency 100 \
  --trainer.eval_interval 5000 \
  --run_root_dir ${run_root_dir} \
  --run_id ${run_id} \


