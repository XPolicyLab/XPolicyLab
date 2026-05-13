export CUDA_VISIBLE_DEVICES=0,1,2,3

accelerate launch \
    --mixed_precision bf16 \
    train.py \
    --models '/path/to/X-VLA-Pt' \
    --train_metas_path /path/to/meta.json \
    --learning_rate 1e-4 \
    --learning_coef 0.1 \
    --iters 30000 \
    --freeze_steps 1000 \
    --warmup_steps 2000 \
    --batch_size 32 \
    --output_dir /path/to/output_dir/ \
    --save_interval 1000 \