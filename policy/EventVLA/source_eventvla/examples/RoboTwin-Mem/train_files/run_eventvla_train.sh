export NCCL_SOCKET_IFNAME=bond0
export NCCL_IB_HCA=mlx5_2,mlx5_3

# used for check save when communication
export NCCL_BLOCKING_WAIT=1
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_TIMEOUT=1000  # timeout set to 1 hour (unit: seconds)

###########################################################################################
# === Please modify the following paths according to your environment ===
Framework_name=EventVLA
freeze_module_list=''
base_vlm=${BASE_VLM:-/mnt/inspurfs/efm_t/yangganlin/models/download_models/Qwen3-VL-4B-Instruct}
config_yaml=./examples/RoboTwin-Mem/train_files/eventvla_robotwin_mem.yaml
run_root_dir=${RUN_ROOT_DIR:-./results/Checkpoints}
data_root_dir=${EVENTVLA_DATA_ROOT:-${ROBOTWIN_MEM_DATA_ROOT:-/mnt/inspurfs/efm_t/yangganlin/workspace_tzz/final/RoboTwin-Mem/lerobotdata}}
data_mix=${EVENTVLA_DATA_MIX:-robotwin_mem8}
memory_ablation_mode=${EVENTVLA_MEMORY_ABLATION_MODE:-pure_image_keyframe_memory}
max_keyframe_images=${MAX_KEYFRAME_IMAGES:-5}
keep_recent_checkpoints=${KEEP_RECENT_CHECKPOINTS:-2}
resolved_profile=${memory_ablation_mode}
keyframe_train_memory_source=${KEYFRAME_TRAIN_MEMORY_SOURCE:-teacher_to_predict}
keyframe_train_memory_schedule=${KEYFRAME_TRAIN_MEMORY_SCHEDULE:-teacher_to_predict}
keyframe_schedule_teacher_prob_start=${KEYFRAME_SCHEDULE_TEACHER_PROB_START:-1.0}
keyframe_schedule_teacher_prob_end=${KEYFRAME_SCHEDULE_TEACHER_PROB_END:-0.0}
memory_debug=${MEMORY_DEBUG:-true}
memory_debug_interval=${MEMORY_DEBUG_INTERVAL:-1}
memory_debug_first_steps=${MEMORY_DEBUG_FIRST_STEPS:-1}

if [[ $# -gt 0 && "${1}" != --* ]]; then
  data_root_dir=${EVENTVLA_DATA_ROOT:-${1}}
  shift
fi
train_extra_args=("$@")

run_date=$(date +%Y%m%d)
run_id=${RUN_ID:-${run_date}_${data_mix}_${memory_ablation_mode}_eventvla}
# === End of environment variable configuration ===
###########################################################################################


# export WANDB_MODE=disabled

output_dir=${run_root_dir}/${run_id}
mkdir -p ${output_dir}
# mv this script to the output dir
cp $0 ${output_dir}/

echo "[train] data_mix=${data_mix}"
echo "[train] data_root_dir=${data_root_dir}"
echo "[train] memory_ablation_mode=${memory_ablation_mode}"
echo "[train] resolved_profile=${resolved_profile}"
echo "[train] keyframe_train_memory_source=${keyframe_train_memory_source}"
echo "[train] keyframe_train_memory_schedule=${keyframe_train_memory_schedule}"
echo "[train] max_keyframe_images=${max_keyframe_images}"
echo "[train] memory_debug=${memory_debug}"
echo "[train] config_yaml=${config_yaml}"
echo "[train] run_id=${run_id}"
echo "[train] keep_recent_checkpoints=${keep_recent_checkpoints}"
if [[ ${#train_extra_args[@]} -gt 0 ]]; then
  echo "[train] extra_args=${train_extra_args[*]}"
fi

accelerate launch \
  --config_file eventvla/config/deepseeds/deepspeed_zero2.yaml \
  --main_process_port 38567 \
  --num_processes 8 \
  eventvla/training/train_eventvla.py \
  --config_yaml ${config_yaml} \
  --framework.name ${Framework_name} \
  --framework.memory_ablation_mode ${memory_ablation_mode} \
  --framework.qwenvl.base_vlm ${base_vlm} \
  --framework.memory_buffer.qwen_memory_injection.keyframe_image_position after_anchor_images_before_action \
  --framework.memory_buffer.qwen_memory_injection.max_keyframe_images ${max_keyframe_images} \
  --framework.memory_buffer.qwen_memory_injection.use_image_role_text true \
  --framework.memory_buffer.keyframe_loss_weight 1.0 \
  --framework.memory_buffer.keyframe_positive_weight 7.0 \
  --framework.memory_buffer.keyframe_threshold 0.5 \
  --framework.memory_buffer.keyframe_predict_mode chunk_future \
  --framework.memory_buffer.event_future_min_offset 1 \
  --framework.memory_buffer.event_commit_threshold 0.55 \
  --framework.memory_buffer.enable_delayed_chunk_event_commit true \
  --framework.memory_buffer.keyframe_train_memory_source ${keyframe_train_memory_source} \
  --framework.memory_buffer.keyframe_eval_memory_source predict \
  --framework.memory_buffer.keyframe_train_memory_schedule ${keyframe_train_memory_schedule} \
  --framework.memory_buffer.keyframe_schedule_warmup_steps 10000 \
  --framework.memory_buffer.keyframe_schedule_transition_steps 30000 \
  --framework.memory_buffer.keyframe_schedule_teacher_prob_start ${keyframe_schedule_teacher_prob_start} \
  --framework.memory_buffer.keyframe_schedule_teacher_prob_end ${keyframe_schedule_teacher_prob_end} \
  --framework.memory_buffer.keyframe_schedule_mix_granularity sample \
  --framework.memory_buffer.debug ${memory_debug} \
  --framework.memory_buffer.debug_interval ${memory_debug_interval} \
  --framework.memory_buffer.debug_first_steps ${memory_debug_first_steps} \
  --datasets.vla_data.use_sequential_episode_sampler true \
  --datasets.vla_data.sampling_interval 50 \
  --datasets.vla_data.chunk_keyframe_target_dilation 8 \
  --datasets.vla_data.chunk_keyframe_target_kernel raised_cosine \
  --datasets.vla_data.event_future_min_offset 1 \
  --datasets.vla_data.teacher_event_threshold 0.55 \
  --datasets.vla_data.keyframe_image_memory.max_keyframes ${max_keyframe_images} \
  --datasets.vla_data.keyframe_image_memory.include_current_keyframe true \
  --datasets.vla_data.keyframe_image_memory.order chronological \
  --datasets.vla_data.keyframe_image_memory.selection latest \
  --datasets.vla_data.keyframe_image_memory.view_mode include_names \
  --datasets.vla_data.keyframe_image_memory.include_names '[cam_high,head,main]' \
  --datasets.vla_data.keyframe_image_memory.exclude_name_patterns '[wrist]' \
  --datasets.vla_data.keyframe_image_memory.strict_single_view true \
  --datasets.vla_data.per_device_batch_size 4 \
  --datasets.vla_data.data_root_dir ${data_root_dir} \
  --datasets.vla_data.data_mix ${data_mix} \
  --trainer.freeze_modules ${freeze_module_list} \
  --trainer.max_train_steps 100000 \
  --trainer.learning_rate.keyframe_head 1.0e-04 \
  --trainer.save_interval 10000 \
  --trainer.keep_recent_checkpoints ${keep_recent_checkpoints} \
  --trainer.logging_frequency 100 \
  --trainer.eval_interval 1000 \
  --run_root_dir ${run_root_dir} \
  --run_id ${run_id} \
  --wandb_project null \
  --trainer.gradient_accumulation_steps 1 \
  "${train_extra_args[@]}"



##### Multi-Server Multi-GPU training script #####
  # accelerate launch \
  #   --config_file eventvla/config/deepseeds/deepspeed_zero2.yaml \
  #   --main_process_ip $MASTER_ADDR \
  #   --main_process_port $MASTER_PORT \
  #   --machine_rank $SLURM_PROCID \
  #   --num_machines $SLURM_NNODES \
  #   --num_processes=${TOTAL_GPUS} \
  #   eventvla/training/train_eventvla.py \
  #   --config_yaml ${config_yaml} \
  #   --framework.name ${Framework_name} \
  #   --framework.qwenvl.base_vlm ${base_vlm} \
  #   --run_root_dir ${run_root_dir} \
  #   --run_id ${run_id} \
  #   --wandb_project your_project \
  #   --wandb_entity your_name
##### Multi-Server Multi-GPU training script #####
