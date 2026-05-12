export CUDA_VISIBLE_DEVICES=0,1,2,3

accelerate launch \
    --mixed_precision bf16 \
    train.py \
    --models '/vepfs-cnbje63de6fae220/xspark_shared/model_weights/X-VLA-Pt' \
    --train_metas_path /vepfs-cnbje63de6fae220/niantian/RoboDojo_env/XPolicyLab/policy/X-VLA/xvla/meta.json \
    --learning_rate 1e-4 \
    --learning_coef 0.1 \
    --iters 30000 \
    --freeze_steps 1000 \
    --warmup_steps 2000 \
    --batch_size 32 \
    --output_dir /vepfs-cnbje63de6fae220/niantian/RoboDojo_env/XPolicyLab/policy/X-VLA/checkpoints/sim_stack_bowls/ \
    --save_interval 1000 \