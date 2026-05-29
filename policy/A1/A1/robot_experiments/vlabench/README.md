```
    pip install -r robot_experiments/vlabench/VLABench/requirements.txt
    pip install -e robot_experiments/vlabench/VLABench
    export PYTHONPATH=$PWD
    python robot_experiments/vlabench/eval_point.py \
        --pretrained_checkpoint=...
```