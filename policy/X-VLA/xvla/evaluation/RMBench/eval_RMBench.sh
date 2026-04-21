# Define your log directory here:
eval_log_dir=X-VLA/evaluation/rmbench/logs

# Start your RoboTwin client
python client.py \
    --host 0.0.0.0 \
    --port 4567 \
    --eval_log_dir $eval_log_dir \
    --num_episodes 100 \
    --device 0 \
    --seed 100000 \
    --task_name rearrange_blocks \
    --output_path $eval_log_dir \
    --task_config demo_clean
    
# Kill the server
PID=$(lsof -i :$port -t)
kill -9 $PID
