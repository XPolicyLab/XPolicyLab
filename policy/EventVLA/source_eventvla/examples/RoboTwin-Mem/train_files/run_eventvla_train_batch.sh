#!/bin/bash
set -euo pipefail

export NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-bond0}
export NCCL_IB_HCA=${NCCL_IB_HCA:-mlx5_2,mlx5_3}

# used for check save when communication
export NCCL_BLOCKING_WAIT=${NCCL_BLOCKING_WAIT:-1}
export NCCL_ASYNC_ERROR_HANDLING=${NCCL_ASYNC_ERROR_HANDLING:-1}
export NCCL_TIMEOUT=${NCCL_TIMEOUT:-1000}  # unit: seconds

###########################################################################################
# === Please modify the following paths according to your environment ===
Framework_name=QwenOFT
freeze_module_list=''
base_vlm=./Qwen3-VL-4B-Instruct/
config_yaml=./examples/Robotwin/train_files/starvla_cotrain_robotwin_abs.yaml
run_root_dir=./results/Checkpoints
data_mix=${1:-}
memory_ablation_mode=${2:-pure_image_keyframe_memory}
keyframe_memory_policy=${3:-${KEYFRAME_MEMORY_POLICY:-teacher}}
keep_recent_checkpoints=${KEEP_RECENT_CHECKPOINTS:-2}
action_chunk_size=${ACTION_CHUNK_SIZE:-50}
sampling_interval=${SAMPLING_INTERVAL:-${action_chunk_size}}
chunk_keyframe_target_dilation=${CHUNK_KEYFRAME_TARGET_DILATION:-8}
chunk_keyframe_target_kernel=${CHUNK_KEYFRAME_TARGET_KERNEL:-raised_cosine}
memory_buffer_capacity=${MEMORY_BUFFER_CAPACITY:-8}
keep_first_slot=${KEEP_FIRST_SLOT:-true}
keep_last_slot=${KEEP_LAST_SLOT:-true}
max_keyframe_images=${MAX_KEYFRAME_IMAGES:-5}
keyframe_cluster_timestep_window=${KEYFRAME_CLUSTER_TIMESTEP_WINDOW:-8}
allow_keyframe_fifo_eviction_on_overflow=${ALLOW_KEYFRAME_FIFO_EVICTION_ON_OVERFLOW:-false}

# Multi-node launch controls.
alloc_job_id=${ALLOC_JOB_ID:-${SLURM_JOB_ID:-${SLURM_JOBID:-}}}
num_nodes=${SLURM_NNODES:-${NUM_NODES:-4}}
gpus_per_node=${GPUS_PER_NODE:-8}

if [[ -z "${SLURM_NNODES:-}" && -z "${NUM_NODES:-}" && -n "${alloc_job_id}" ]]; then
  detected_num_nodes=$(scontrol show job "${alloc_job_id}" 2>/dev/null | sed -n 's/.* NumNodes=\([0-9][0-9]*\).*/\1/p' | head -n 1 || true)
  if [[ -n "${detected_num_nodes}" ]]; then
    num_nodes=${detected_num_nodes}
  fi
fi
total_gpus=$((gpus_per_node * num_nodes))

if [[ -n "${SLURM_JOB_NODELIST:-}" ]]; then
  master_addr_default=$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | head -n 1)
elif [[ -n "${alloc_job_id}" ]]; then
  master_addr_default=$(scontrol show job "${alloc_job_id}" 2>/dev/null | sed -n 's/.* BatchHost=\([^ ]*\).*/\1/p' | head -n 1 || true)
else
  master_addr_default=$(hostname)
fi
master_addr=${MASTER_ADDR:-${master_addr_default}}
master_port=${MASTER_PORT:-38567}

if ! [[ "${action_chunk_size}" =~ ^[0-9]+$ ]] || (( action_chunk_size < 1 )); then
  echo "[train][error] ACTION_CHUNK_SIZE must be a positive integer, got ${action_chunk_size}"
  exit 1
fi
future_action_window_size=$((action_chunk_size - 1))

if [[ -z "${data_mix}" ]]; then
  echo "[train][error] Missing data_mix."
  echo "Usage: bash examples/Robotwin/train_files/run_robotwin_train_srun_multinode.sh <data_mix> <memory_ablation_mode> [teacher|predict]"
  exit 1
fi

case "${memory_ablation_mode}" in
  baseline_no_memory|raw_anchors_only|raw_anchors_first_current_only|raw_anchors_first_m30_current|pure_image_keyframe_memory|raw_anchors_token_memory|memory_tokens_only|replace_image_tokens)
    resolved_profile=${memory_ablation_mode}
    ;;
  *)
    echo "[train][error] Unsupported memory_ablation_mode=${memory_ablation_mode}"
    echo "[train][error] Supported modes: baseline_no_memory, raw_anchors_only, raw_anchors_first_current_only, raw_anchors_first_m30_current, pure_image_keyframe_memory, raw_anchors_token_memory, memory_tokens_only, replace_image_tokens"
    exit 1
    ;;
esac

case "${keyframe_memory_policy}" in
  teacher|with_teacher|teacher_to_predict)
    resolved_keyframe_memory_policy=teacher
    keyframe_train_memory_source=teacher_to_predict
    keyframe_train_memory_schedule=teacher_to_predict
    use_teacher_future_frame_write_in_train=true
    keyframe_schedule_teacher_prob_start=1.0
    keyframe_schedule_teacher_prob_end=0.0
    ;;
  predict|no_teacher|without_teacher|student)
    resolved_keyframe_memory_policy=predict
    keyframe_train_memory_source=predict
    keyframe_train_memory_schedule=predict
    use_teacher_future_frame_write_in_train=false
    keyframe_schedule_teacher_prob_start=0.0
    keyframe_schedule_teacher_prob_end=0.0
    ;;
  *)
    echo "[train][error] Unsupported keyframe_memory_policy=${keyframe_memory_policy}"
    echo "[train][error] Supported policies: teacher, predict"
    exit 1
    ;;
esac

run_date=$(date +%Y%m%d)
default_run_id=${run_date}_${data_mix}_${memory_ablation_mode}_${resolved_keyframe_memory_policy}_qwenoft
run_id=${RUN_ID:-${default_run_id}}
# === End of environment variable configuration ===
###########################################################################################

output_dir=${run_root_dir}/${run_id}
mkdir -p "${output_dir}"
cp "$0" "${output_dir}/"

echo "[train] data_mix=${data_mix}"
echo "[train] memory_ablation_mode=${memory_ablation_mode}"
echo "[train] resolved_profile=${resolved_profile}"
echo "[train] keyframe_memory_policy=${resolved_keyframe_memory_policy}"
echo "[train] keyframe_train_memory_source=${keyframe_train_memory_source}"
echo "[train] keyframe_train_memory_schedule=${keyframe_train_memory_schedule}"
echo "[train] use_teacher_future_frame_write_in_train=${use_teacher_future_frame_write_in_train}"
echo "[train] action_chunk_size=${action_chunk_size}"
echo "[train] future_action_window_size=${future_action_window_size}"
echo "[train] sampling_interval=${sampling_interval}"
echo "[train] chunk_keyframe_target_dilation=${chunk_keyframe_target_dilation}"
echo "[train] chunk_keyframe_target_kernel=${chunk_keyframe_target_kernel}"
echo "[train] memory_buffer_capacity=${memory_buffer_capacity}"
echo "[train] keep_first_slot=${keep_first_slot}"
echo "[train] keep_last_slot=${keep_last_slot}"
echo "[train] max_keyframe_images=${max_keyframe_images}"
echo "[train] keyframe_cluster_timestep_window=${keyframe_cluster_timestep_window}"
echo "[train] allow_keyframe_fifo_eviction_on_overflow=${allow_keyframe_fifo_eviction_on_overflow}"
echo "[train] config_yaml=${config_yaml}"
echo "[train] run_id=${run_id}"
echo "[train] keep_recent_checkpoints=${keep_recent_checkpoints}"
echo "[train] num_nodes=${num_nodes}"
echo "[train] gpus_per_node=${gpus_per_node}"
echo "[train] total_gpus=${total_gpus}"
echo "[train] master_addr=${master_addr}"
echo "[train] master_port=${master_port}"
echo "[train] alloc_job_id=${alloc_job_id:-<none>}"

printf -v config_yaml_q '%q' "${config_yaml}"
printf -v framework_name_q '%q' "${Framework_name}"
printf -v memory_ablation_mode_q '%q' "${memory_ablation_mode}"
printf -v future_action_window_size_q '%q' "${future_action_window_size}"
printf -v action_chunk_size_q '%q' "${action_chunk_size}"
printf -v base_vlm_q '%q' "${base_vlm}"
printf -v memory_buffer_capacity_q '%q' "${memory_buffer_capacity}"
printf -v keep_first_slot_q '%q' "${keep_first_slot}"
printf -v max_keyframe_images_q '%q' "${max_keyframe_images}"
printf -v keep_last_slot_q '%q' "${keep_last_slot}"
printf -v allow_keyframe_fifo_eviction_q '%q' "${allow_keyframe_fifo_eviction_on_overflow}"
printf -v use_teacher_future_frame_write_in_train_q '%q' "${use_teacher_future_frame_write_in_train}"
printf -v keyframe_cluster_timestep_window_q '%q' "${keyframe_cluster_timestep_window}"
printf -v keyframe_train_memory_source_q '%q' "${keyframe_train_memory_source}"
printf -v keyframe_train_memory_schedule_q '%q' "${keyframe_train_memory_schedule}"
printf -v keyframe_schedule_teacher_prob_start_q '%q' "${keyframe_schedule_teacher_prob_start}"
printf -v keyframe_schedule_teacher_prob_end_q '%q' "${keyframe_schedule_teacher_prob_end}"
printf -v sampling_interval_q '%q' "${sampling_interval}"
printf -v chunk_keyframe_target_dilation_q '%q' "${chunk_keyframe_target_dilation}"
printf -v chunk_keyframe_target_kernel_q '%q' "${chunk_keyframe_target_kernel}"
printf -v freeze_module_list_q '%q' "${freeze_module_list}"
printf -v keep_recent_checkpoints_q '%q' "${keep_recent_checkpoints}"
printf -v data_mix_q '%q' "${data_mix}"
printf -v run_root_dir_q '%q' "${run_root_dir}"
printf -v run_id_q '%q' "${run_id}"
printf -v master_addr_q '%q' "${master_addr}"
printf -v master_port_q '%q' "${master_port}"

srun_cmd=(srun --nodes "${num_nodes}" --ntasks "${num_nodes}" --ntasks-per-node=1)
if [[ -n "${alloc_job_id}" ]]; then
  srun_cmd+=(--jobid "${alloc_job_id}")
fi
srun_log_out=${SRUN_LOG_OUT:-}
srun_log_err=${SRUN_LOG_ERR:-}
if [[ -n "${srun_log_out}" ]]; then
  mkdir -p "$(dirname -- "${srun_log_out}")"
  srun_cmd+=(-o "${srun_log_out}")
fi
if [[ -n "${srun_log_err}" ]]; then
  mkdir -p "$(dirname -- "${srun_log_err}")"
  srun_cmd+=(-e "${srun_log_err}")
fi
echo "[train] srun_log_out=${srun_log_out:-<default>}"
echo "[train] srun_log_err=${srun_log_err:-<default>}"
echo "[train] srun_cmd=${srun_cmd[*]}"

"${srun_cmd[@]}" bash -c "
set -euo pipefail
echo Host=\$(hostname) SLURM_PROCID=\${SLURM_PROCID}

accelerate launch \
  --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml \
  --main_process_ip ${master_addr_q} \
  --main_process_port ${master_port_q} \
  --machine_rank \${SLURM_PROCID} \
  --num_machines ${num_nodes} \
  --num_processes ${total_gpus} \
  starVLA/training/train_starvla.py \
  --config_yaml ${config_yaml_q} \
  --framework.name ${framework_name_q} \
  --framework.memory_ablation_mode ${memory_ablation_mode_q} \
  --framework.action_model.future_action_window_size ${future_action_window_size_q} \
  --framework.action_model.action_horizon ${action_chunk_size_q} \
  --framework.qwenvl.base_vlm ${base_vlm_q} \
  --framework.memory_buffer.capacity ${memory_buffer_capacity_q} \
  --framework.memory_buffer.keep_first_slot ${keep_first_slot_q} \
  --framework.memory_buffer.qwen_memory_injection.keyframe_image_position after_anchor_images_before_action \
  --framework.memory_buffer.qwen_memory_injection.max_keyframe_images ${max_keyframe_images_q} \
  --framework.memory_buffer.qwen_memory_injection.use_image_role_text true \
  --framework.memory_buffer.keep_last_slot ${keep_last_slot_q} \
  --framework.memory_buffer.reset_each_train_forward false \
  --framework.memory_buffer.merge_similarity_weight 0.7 \
  --framework.memory_buffer.merge_coverage_weight 0.3 \
  --framework.memory_buffer.merge_coverage_lambda 0.8 \
  --framework.memory_buffer.merge_span_lambda 0.2 \
  --framework.memory_buffer.keyframe_loss_weight 1.0 \
  --framework.memory_buffer.keyframe_positive_weight 7.0 \
  --framework.memory_buffer.keyframe_threshold 0.5 \
  --framework.memory_buffer.keyframe_predict_mode chunk_future \
  --framework.memory_buffer.event_future_min_offset 1 \
  --framework.memory_buffer.event_commit_threshold 0.55 \
  --framework.memory_buffer.enable_delayed_chunk_event_commit true \
  --framework.memory_buffer.allow_keyframe_fifo_eviction_on_overflow ${allow_keyframe_fifo_eviction_q} \
  --framework.memory_buffer.use_teacher_future_frame_write_in_train ${use_teacher_future_frame_write_in_train_q} \
  --framework.memory_buffer.keyframe_cluster_timestep_window ${keyframe_cluster_timestep_window_q} \
  --framework.memory_buffer.eval_keyframe_neighbor_merge false \
  --framework.memory_buffer.eval_keyframe_compact_neighbors_on_overflow false \
  --framework.memory_buffer.keyframe_train_memory_source ${keyframe_train_memory_source_q} \
  --framework.memory_buffer.keyframe_eval_memory_source predict \
  --framework.memory_buffer.keyframe_train_memory_schedule ${keyframe_train_memory_schedule_q} \
  --framework.memory_buffer.keyframe_schedule_warmup_steps 10000 \
  --framework.memory_buffer.keyframe_schedule_transition_steps 30000 \
  --framework.memory_buffer.keyframe_schedule_teacher_prob_start ${keyframe_schedule_teacher_prob_start_q} \
  --framework.memory_buffer.keyframe_schedule_teacher_prob_end ${keyframe_schedule_teacher_prob_end_q} \
  --framework.memory_buffer.keyframe_schedule_mix_granularity sample \
  --framework.memory_buffer.debug true \
  --framework.memory_buffer.debug_interval 1 \
  --framework.memory_buffer.debug_first_steps 1 \
  --datasets.vla_data.use_sequential_episode_sampler true \
  --datasets.vla_data.sampling_interval ${sampling_interval_q} \
  --datasets.vla_data.chunk_keyframe_target_dilation ${chunk_keyframe_target_dilation_q} \
  --datasets.vla_data.chunk_keyframe_target_kernel ${chunk_keyframe_target_kernel_q} \
  --datasets.vla_data.event_future_min_offset 1 \
  --datasets.vla_data.teacher_event_threshold 0.55 \
  --datasets.vla_data.keyframe_image_memory.max_keyframes ${max_keyframe_images_q} \
  --datasets.vla_data.keyframe_image_memory.include_current_keyframe true \
  --datasets.vla_data.keyframe_image_memory.order chronological \
  --datasets.vla_data.keyframe_image_memory.selection latest \
  --datasets.vla_data.keyframe_image_memory.view_mode include_names \
  --datasets.vla_data.keyframe_image_memory.include_names '[cam_high,head,main]' \
  --datasets.vla_data.keyframe_image_memory.exclude_name_patterns '[wrist]' \
  --datasets.vla_data.keyframe_image_memory.strict_single_view true \
  --datasets.vla_data.per_device_batch_size 4 \
  --datasets.vla_data.data_root_dir ./RMBench/lerobot_data \
  --datasets.vla_data.data_mix ${data_mix_q} \
  --trainer.freeze_modules ${freeze_module_list_q} \
  --trainer.max_train_steps 150000 \
  --trainer.learning_rate.keyframe_head 1.0e-04 \
  --trainer.save_interval 10000 \
  --trainer.keep_recent_checkpoints ${keep_recent_checkpoints_q} \
  --trainer.logging_frequency 100 \
  --trainer.eval_interval 1000 \
  --run_root_dir ${run_root_dir_q} \
  --run_id ${run_id_q} \
  --wandb_project null \
  --trainer.gradient_accumulation_steps 1
"
