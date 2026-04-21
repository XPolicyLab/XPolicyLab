export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

accelerate launch \
    --mixed_precision bf16 \
    train.py \
    --models '/share_data/wangyuran/X-VLA/X-VLA-Pt' \
    --train_metas_path /share_data/wangyuran/X-VLA/meta.json \
    --learning_rate 1e-4 \
    --learning_coef 0.1 \
    --iters 30000 \
    --freeze_steps 1000 \
    --warmup_steps 2000 \
    --batch_size 32 \
    --output_dir /share_data/wangyuran/X-VLA/rmbench_rearrange_blocks