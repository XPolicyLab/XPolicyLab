# Classify Objects By Language

`classify_objects_by_language` 由 `ME_Brain_Adaptor` 统一调度，支持 `RoboDojo`、`arx_x5`、`ee`，每个进程运行一个环境，使用非批量部署入口。

## Dependencies

使用任务独立的语言分类、闭集匹配、材料证据和低释放控制。需要在环境客户端安装支持 CUDA 的 PyTorch、Transformers、NumPy、SciPy、OpenCV 和 Pillow，并提供 DINOv2 Small、CLIP ViT-B/32 与 Grounding DINO Tiny 的完整本地模型目录。原入口在分配的 GPU 上使用 FP32，保留观察器绑定、缓存清理与推理顺序。

策略循环在环境客户端执行；`ME_BRAIN_POLICY_PYTHON` 仅指定策略服务器解释器。环境客户端需要具备本任务的运行依赖。基础脚本环境可由 `install.sh code` 安装。

## Evaluation

从 `ME_Brain_RSI` 目录执行：

```bash
export ME_BRAIN_POLICY_PYTHON=./strategy_env/.venv/bin/python
export ME_BRAIN_EVIDENCE_DIR=../../../runs/ME_Brain_RSI
export PYTHONDONTWRITEBYTECODE=1
export SORTING_DINO_PATH=../../../model_assets/dinov2-small
export CLASSIFY_CLIP_PATH=../../../model_assets/clip-vit-base-patch32
export CLASSIFY_DINO_PATH=../../../model_assets/grounding-dino-tiny

EVAL_ENV_TYPE=sim ROBODOJO_NUM_ENVS=1 \
  bash eval.sh RoboDojo classify_objects_by_language none arx_x5 auto 0 0 0 uv base
```

将最后的 `base` 换成具备本任务依赖的环境客户端环境，GPU 参数按实际部署选择。`auto` 选择原始 `ee` 部署契约；内部数值 IK 与关节路径仍按原策略发送。启动脚本在两个解释器启动前将 OpenMP、OpenBLAS 和 MKL 的线程数设为 `1`。脚本策略无需训练，`ckpt_name=none`。

## Resources

三个模型变量均指向包含配置、处理器或 tokenizer 与权重的本地目录，只从本地加载。未设置时从本任务 `coap/assets/` 中对应的 `sorting_dinov2_small`、`clip-vit-base-patch32` 和 `grounding-dino-tiny` 目录加载。模型权重不包含在源码中。相机与机器人几何及第三方许可证随源码提供。

相对模型路径以本策略目录为基准。运行输出通过 `strategy_evidence_dir` 或 `ME_BRAIN_EVIDENCE_DIR` 放在源码目录外。输入为 XPolicyLab 公共图像解码函数处理后的 RGB 和机器人自身状态，不读取评测布局或评分状态。

任务模块映射在 `modules.json`，运行资源在 `resources/classify_objects_by_language/`。任务上下文相互独立，动态源变换使用原有源码与路径身份，不共享可变观察器或策略状态。
