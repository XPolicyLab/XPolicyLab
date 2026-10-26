# WSA 训练脚本

当前仓库仅保留原版 `WSA_Base` 训练路径：

- `wsa_base_pretrain.sh`：多数据集预训练。
- `wsa_base_finetune.sh`：原版微调入口。
- `wsa_base_finetune_multi.sh`：从一个目录自动发现多个 LeRobot v3 数据集并微调。

模型输入的文本就是数据集样本中的原始 `task`。

WSA-B 默认使用 `image_delta_indices=[0,0,15]`：前两个条件图像都是当前帧，第 15 帧只作为未来图像监督。

多数据集微调示例：

```bash
POLICY_INIT_PATH=/path/to/WSA-Base \
ROBOTWIN_ROOT=/path/to/dataset_collection \
USE_EXTERNAL_STATS=true \
DATASET_EXTERNAL_STATS_PATH=/path/to/stats.json \
ACTION_TYPE=delta \
PROC_PER_NODE=8 \
BATCH_SIZE=10 \
STEPS=120000 \
bash launch/wsa_base_finetune_multi.sh
```

`ROBOTWIN_ROOT` 下每个包含 `meta/info.json` 且具有 `data/` 或 `videos/` 的目录都会作为一个 LeRobot v3 数据集加入训练。
