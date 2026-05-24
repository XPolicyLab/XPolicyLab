config = dict(
    runners=['wan.WanTrainer'],
    project_dir='./experiments/wan_5b_t2v_ft',
    # launch=dict(
    #     gpu_ids=[0],
    # ),
    launch=dict(
        gpu_ids=[0, 1, 2, 3, 4, 5, 6, 7],
        distributed_type='DEEPSPEED',
        deepspeed_config=dict(
            deepspeed_config_file='accelerate_configs/zero2.json',
        ),
    ),
    # launch=dict(
    #     gpu_ids=[0, 1, 2, 3, 4, 5, 6, 7],
    #     distributed_type='FSDP',
    #     fsdp_config=dict(
    #         fsdp_version='2',
    #         fsdp_auto_wrap_policy='TRANSFORMER_BASED_WRAP',
    #         fsdp_transformer_layer_cls_to_wrap='WanTransformerBlock',
    #         fsdp_cpu_ram_efficient_loading='false',
    #         fsdp_state_dict_type='FULL_STATE_DICT',
    #     ),
    # ),
    dataloaders=dict(
        train=dict(
            data_or_config=[
                './data/giga_data/',
            ],
            batch_size_per_gpu=1,
            num_workers=6,
            transform=dict(
                type='WanTransform',
                num_frames=121,
                height=704,
                width=1280,
            ),
            sampler=dict(
                type='DefaultSampler',
                shuffle=True,
            ),
        ),
    ),
    models=dict(
        pretrained='./models/Wan2.2-TI2V-5B-Diffusers/',
        flow_shift=5.0,
    ),
    optimizers=dict(
        type='AdamW',
        lr=2e-5,
        weight_decay=1e-3,
    ),
    schedulers=dict(
        type='ConstantScheduler',
    ),
    train=dict(
        resume=True,
        max_epochs=20,
        gradient_accumulation_steps=1,
        mixed_precision='bf16',  # fp16, bf16
        checkpoint_interval=500,
        checkpoint_total_limit=3,
        log_with='tensorboard',
        log_interval=1,
        with_ema=True,
        activation_checkpointing=True,
        activation_class_names=['WanTransformerBlock'],  # For DEEPSPEED
        # activation_class_names=['WanAttention', 'FeedForward'],  # For FSDP2
    ),
)
