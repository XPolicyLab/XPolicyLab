dst_size = (256, 192)
num_frames = 48
action_dim = 16

project_dir="/mnt/nfs/niantian/RoboDojo_env/XPolicyLab/policy/GigaWorldPolicy/giga_world_policy"
data_dir = "/xspark-cache/shared/lerobot/RoboDojo_sim_arx-x5_v21/"

view_keys = [
    "observation.images.cam_high",
    "observation.images.cam_left_wrist",
    "observation.images.cam_right_wrist",
]
image_frame_offsets = [0, num_frames // 4, num_frames // 2, (3 * num_frames) // 4, num_frames]

norm_path = [
    '/mnt/nfs/niantian/RoboDojo_env/XPolicyLab/policy/GigaWorldPolicy/giga_world_policy/norm_stats_delta.json'
]

config = dict(
    project_dir=project_dir,
    runners=["world_action_model.trainer.wa_casual_trainer.CasualWATrainer"],
    launch=dict(
        gpu_ids=[0,1,2,3,4,5,6,7],
        distributed_type='DEEPSPEED',
        deepspeed_config=dict(
            deepspeed_config_file='accelerate_configs/zero2.json',
        ),
        until_completion=True,
    ),
    dataloaders=dict(
        train=dict(
            data_or_config=[
                dict(
                    _class_name="LeRobotDataset",
                    data_path=data_dir,
                    data_size=None,
                    delta_info={"action": num_frames},
                    delta_frames={k: image_frame_offsets for k in view_keys}
                )
            ],
            batch_size_per_gpu=8,
            num_workers=8,
            transform=dict(
                type='WATransformsLerobot',
                robotype_to_embed_id={
                    "aloha": 0, 
                    "agilex": 0, 
                    "agibot": 1
                },
                dst_size=dst_size,
                num_frames=num_frames,
                is_train=True,
                norm_path=norm_path,
                model_action_dim=action_dim,
                num_views=3,
                t5_len=64,
                image_cfg=dict(
                    mask_generator=dict(
                        max_ref_frames=1,
                        start=1,
                        factor=4,
                    ),
                ),

            ),
            sampler=dict(
                type='DefaultSampler',
            ),
            collator=dict(
                is_equal=True,
            ),
        ),
        test=dict(),
    ),
    models=dict(
        pretrained="/mnt/xspark-data/xspark_shared/model_weights/Wan2.2-TI2V-5B-Diffusers/",
        strict=False,
        action_dim=action_dim,
        flow_shift=5.0,
        expand_timesteps=True,
        view_dir=project_dir,
        state_repeats=1,
    ),
    optimizers=dict(
        type='CAME8Bit',
        lr=2 ** (-14.5),
        weight_decay=1e-2,
    ),
    schedulers=dict(
        type='ConstantScheduler',
    ),
    train=dict(
        resume=False,
        max_epochs=0,
        max_steps=100000,
        gradient_accumulation_steps=1,
        mixed_precision='bf16', 
        checkpoint_interval=5000,
        checkpoint_total_limit=-1,
        checkpoint_safe_serialization=False,
        checkpoint_strict=False,
        log_with='tensorboard',
        log_interval=1,
        with_ema=True,
        activation_checkpointing=False,
        activation_class_names=["WanAttention"],
    ),
    test=dict(),
)
