###########################################################################################
# === Please modify the following paths according to your environment ===
Framework_name=ABot_M0
base_vlm=path_to_your_base_vlm_checkpoint
freeze_module_list='' # just for fast debug, sota is under fully FT, i.g., freeze_module_list=""
DIT_TYPE="DiT-B"
data_root_dir=path_to_your_data_root_dir
data_mix=robocase_gr1
pretrain_ckpt=path_to_your_pretrain_ckpt
run_root_dir=path_to_your_run_root_dir
run_id=your_run_id
# === End of environment variable configuration ===
###########################################################################################


export WANDB_MODE=disabled

output_dir=${run_root_dir}/${run_id}
mkdir -p ${output_dir}
# mv this script to the output dir
cp $0 ${output_dir}/

accelerate launch \
  --config_file ABot/config/deepseeds/deepspeed_zero2.yaml \
  --num_processes 8 \
  ABot/training/train.py \
  --config_yaml ./examples/Robocasa_tabletop/train_files/ABot_robocasa_gr1.yaml \
  --framework.name ${Framework_name} \
  --framework.qwenvl.base_vlm ${base_vlm} \
  --framework.action_model.action_model_type ${DIT_TYPE} \
  --datasets.vla_data.data_root_dir ${data_root_dir} \
  --datasets.vla_data.data_mix ${data_mix} \
  --trainer.pretrained_checkpoint ${pretrain_ckpt} \
  --trainer.reload_modules qwen_vl_interface,action_model \
  --datasets.vla_data.num_workers 4 \
  --datasets.vla_data.per_device_batch_size 16 \
  --trainer.freeze_modules ${freeze_module_list} \
  --trainer.max_train_steps 50000 \
  --trainer.save_interval 5000 \
  --trainer.logging_frequency 100 \
  --trainer.eval_interval 100 \
  --run_root_dir ${run_root_dir} \
  --run_id ${run_id} \


