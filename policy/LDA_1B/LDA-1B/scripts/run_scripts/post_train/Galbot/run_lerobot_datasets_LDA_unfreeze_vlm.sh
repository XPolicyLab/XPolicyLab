source /mnt/home/liukai/miniconda3/bin/activate
conda activate /mnt/home/liukai/miniconda3/envs/LDA
export NCCL_SOCKET_IFNAME=eth0
export NCCL_IB_HCA=mlx5_2,mlx5_3
export NCCL_IB_DISABLE=1
export NCCL_DEBUG=WARN
export WANDB_API_KEY=8c8c73c90734b2e7292c02bb73313673f5167291 # replace with your wandb api key
# used for check save when communication
export NCCL_BLOCKING_WAIT=1
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_TIMEOUT=3600  # timeout set to 1 hour (unit: seconds)
# export NCCL_IB_TIMEOUT=10000
# export NCCL_IB_RETRY_CNT=10000
# export NCCL_IB_AR_THRESHOLD=0


Framework_name=QwenMMDiT
base_vlm=/mnt/home/liukai/starVLA/playground/pretrained/vlm/Qwen3-VL-4B-Instruct
vision_encoder_path=/mnt/home/liukai/World-Action-Model/pretrained # should be the parent path of vision encoder ckpt

freeze_module_list='action_model.vision_encoder' # if you would like to directly train on the robocasa dataset, unfreeze vlm could obtain better performance
DIT_TYPE="DiT-L"
# freeze_module_list="qwen_vl_interface.model.model.visual,dino_encoder" # just for fast debug, sota is under fully FT, i.g., freeze_module_list=""

llavadata="asv2_conversation_en,asv2_detailed_description_en"
data_root_dir=/mnt/project
data_mix=galbot_pick_vegetable # should be recorded in data_config.py

obs_horizon=2 # history obs length
state_dim=null
action_dim=138
max_num_embodiments=32
num_layers=16
use_delta_action=true
positional_embeddings=null # rope
TRAINING_TASK_WEIGHTS="[1,1,1,1]"

seed=42

repeated_diffusion_steps=1

future_obs_index=5
run_root_dir=/mnt/project/world_model/checkpoints/lda/post-train # replace with your own path
run_id=galbot_pick_vegetable_unfreeze_vlm_modify_gripper # replace with your own run id, e.g., galbot_pick_vegetable_2

pretrained_checkpoint=/mnt/project/world_model/checkpoints/lda/pretrain_48_node_batch_sampler/checkpoints/steps_190000_pytorch_model.pt # set to null if training from scratch
post_train=true

only_policy=false
policy_and_video_gen=false
only_wo_video_gen=false

export WANDB_MODE=online
wandb_entity=KaiLiu-Personal

output_dir=${run_root_dir}/${run_id}
mkdir -p ${output_dir}
# mv this script to the output dir
cp $0 ${output_dir}/

accelerate launch \
  --config_file /mnt/home/liukai/code/LDA/lda/config/deepseeds/deepspeed_zero2.yaml \
  --num_machines 1 \
  --num_processes 8 \
  /mnt/home/liukai/code/LDA/lda/training/train_LDA.py \
  --config_yaml /mnt/home/liukai/code/LDA/lda/config/training/LDA_pretrain.yaml \
  --seed ${seed} \
  --framework.name ${Framework_name} \
  --framework.qwenvl.base_vlm ${base_vlm} \
  --framework.action_model.vision_encoder_path ${vision_encoder_path} \
  --framework.action_model.action_model_type ${DIT_TYPE} \
  --framework.action_model.max_num_embodiments ${max_num_embodiments} \
  --framework.action_model.state_dim ${state_dim} \
  --framework.action_model.action_dim ${action_dim} \
  --framework.action_model.obs_horizon ${obs_horizon} \
  --framework.action_model.future_obs_index ${future_obs_index} \
  --framework.action_model.only_policy ${only_policy} \
  --framework.action_model.policy_and_video_gen ${policy_and_video_gen} \
  --framework.action_model.only_wo_video_gen ${only_wo_video_gen} \
  --framework.action_model.diffusion_model_cfg.num_layers ${num_layers} \
  --framework.action_model.diffusion_model_cfg.positional_embeddings ${positional_embeddings} \
  --datasets.vla_data.use_delta_action ${use_delta_action} \
  --datasets.vla_data.data_root_dir ${data_root_dir} \
  --datasets.vla_data.training_task_weights ${TRAINING_TASK_WEIGHTS} \
  --datasets.vla_data.data_mix ${data_mix} \
  --datasets.vla_data.per_device_batch_size 16 \
  --trainer.freeze_modules ${freeze_module_list} \
  --trainer.post_train ${post_train} \
  --trainer.max_train_steps 100000 \
  --trainer.save_interval 5000 \
  --trainer.logging_frequency 100 \
  --trainer.eval_interval 1000 \
  --trainer.repeated_diffusion_steps ${repeated_diffusion_steps} \
  --trainer.learning_rate.base 4e-5 \
  --trainer.pretrained_checkpoint ${pretrained_checkpoint} \
  --run_root_dir ${run_root_dir} \
  --run_id ${run_id} \
  --wandb_project lda-post-train \
  --wandb_entity ${wandb_entity} \
  --is_debug False

