from a1.data import build_rlds_dit_action_train_dataloader
from a1 import TrainConfig, DataConfig,OptimizerConfig, OptimizerType,SchedulerConfig, SchedulerType


from a1.config import ModelConfig, VisionBackboneConfig, \
    TokenizerConfig

from a1.torch_util import get_world_size

DEBUG_MODEL = ModelConfig(
    d_model=128,
    n_heads=2,
    n_layers=1,
    max_sequence_length=4096,
    additional_vocab_size=128,
    vocab_size=152064,
    rope=True,
    embedding_size=None,
    weight_tying=False,
    vision_backbone=VisionBackboneConfig(
        image_num_layers=1,
    ),
    crop_mode="resize",
    tokenizer=TokenizerConfig(
        identifier="Qwen/Qwen2-7B",
    ),
)

model_cfg = DEBUG_MODEL

seq_len=768

cfg = TrainConfig(
    # run_name=f"action_head_libero_{timestamp}",
    no_pre_train_checkpoint=True,
    # save_folder="debug_run" if debug else omegaconf.MISSING,
    seed=6198,
    dry_run=False,
    wandb=None ,
    model=model_cfg,
    data=DataConfig(
        dataset="rlds_dataset",
        rlds_dataset_name="austin_buds_dataset_converted_externally_to_rlds", ##
        rlds_data_root_dir="data/OXE", ##
        use_wrist_image=True,  # Set to True if you want to use wrist images
        use_proprio=True,  # Set to True if you want to use proprioceptive data
        for_inference=False,
        shuffle=True,
        split="train",
        drop_last=True,
        sequence_length=seq_len,
        seed=95818,
        # num_workers=2,
        num_workers=0,
        pad="to_max",
        pin_memory=True,
        shuffle_messages=False,
    ),
    ft_connector=True,
    ft_llm=True,
    ft_vit=True,

    optimizer=OptimizerConfig(
        name=OptimizerType.adamw,
        connector_learning_rate=2e-4,
        vit_learning_rate=6e-6,
        llm_learning_rate=2e-5,
        connector_weight_decay=0.0,
        vit_weight_decay=0.0,
        llm_weight_decay=0.0,
        connector_betas=[0.9, 0.95],
        vit_betas=[0.9, 0.95],
        llm_betas=[0.9, 0.95],
        connector_eps=1e-6,
        vit_eps=1e-6,
        llm_eps=1e-6,
        metrics_log_interval=20
    ),
    scheduler=SchedulerConfig(
        name=SchedulerType.multimodal,
        connector_t_warmup=200,
        vit_t_warmup=2000,
        llm_t_warmup=2000,
        alpha_f=0.1,
        warmup_min_lr=0.0
    ),
    fsdp=None,
    load_path=None,
    initial_model_checkpoint=None,
    allow_resume=False, ## add
    save_overwrite=True,
    save_dataloader_state=False,
    # save_interval="${max_duration}", # 4000
    save_interval=1000, # 4000
    save_num_checkpoints_to_keep=1,
    # save_interval_unsharded="${max_duration}",
    save_interval_unsharded=500,
    save_num_unsharded_checkpoints_to_keep = 1,
    save_interval_action_head = 500, # only save the action head checkpoints
    save_num_action_head_checkpoints_to_keep = 3,
    global_train_batch_size=96,
    device_eval_batch_size=8,
    # device_train_microbatch_size=4,
    device_train_microbatch_size=8,
    time_limit=None,
    max_duration=100000,
    stop_at="${max_duration}",
    max_grad_norm=1,
    # batch_divisor=BatchDivisor.global_batch,
    precision="amp_bf16",
    # console_log_interval=log_interval,
    # speed_monitor=SpeedMonitorConfig(window_size=20),
    softmax_auxiliary_loss=True,
    softmax_auxiliary_loss_scale=1e-4,
    # activation_checkpointing=ActivationCheckpointingStrategy.whole_layer,
    # eval_interval=eval_interval,
    # evaluators=[
    #     # Evaluate loss on data with and without the transcripts
    #     evaluator,
    #     # replace(
    #     #     evaluator,
    #     #     label="caption_val",
    #     #     data=replace(
    #     #         evaluator.data,
    #     #         dataset="pixmo_cap"
    #     #     )
    #     # )
    # ]
)

cfg.device_train_batch_size = cfg.global_train_batch_size // get_world_size()

train_dataloader = build_rlds_dit_action_train_dataloader(cfg,device='cuda:7')

for batch in train_dataloader:
    print('******** pixel values')
    print(batch['pixel_values'].shape)
    print('******** input_ids shape')
    print(batch['input_ids'].shape)
    print('******** action shape')
    print(batch['action'].shape)
    print('******** proprio shape')
    print(batch['proprio'].shape)
    print("******** text_attention_mask shape")
    print(batch['text_attention_mask'].shape)
    print(batch['text_attention_mask'].dtype)
    break