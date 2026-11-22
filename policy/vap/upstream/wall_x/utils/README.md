# wall_x.utils

该目录包含 Wall-X 训练与推理过程中使用的一些通用工具模块。

## Activation Inspector（激活值分析工具）

`activation_inspector.py` 提供了一个**即插即用的激活值统计工具**，用于在模型前向 / 反向过程中分析各个模块产生的 activation 大小，主要用于：

- 定位显存占用最大的模块
- 排查训练过程中显存 spike 的原因
- 决定哪些模块最适合做 activation checkpoint / recompute
- 对比不同配置下的 activation 开销
  （如不同图像分辨率、不同模型结构、FA1 vs FA2 等）

---

## 基本用法

```python
from wall_x.utils.activation_inspector import ActivationInspector

inspector = ActivationInspector(
    model,
    only_requires_grad=True,   # 只统计会参与反向传播的 activation
    min_size_mb=5.0,           # 忽略较小的 activation（单位 MB）
    rank0_only=True,           # 仅在 rank0 统计，适合 DDP / FSDP
)

with inspector:
    loss = model(inputs).mean()
    loss.backward()

inspector.print_top_modules(topk=20)
