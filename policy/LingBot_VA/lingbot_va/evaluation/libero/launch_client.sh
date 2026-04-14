START=0
END=10

python evaluation/libero/client.py \
    --libero-benchmark libero_10 \
    --port 29056 \
    --test-num 50 \
    --task-range $START $END \
    --out-dir outputs/libero
