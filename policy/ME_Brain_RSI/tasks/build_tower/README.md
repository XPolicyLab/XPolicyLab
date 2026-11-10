# Build Tower

`build_tower` 使用 SAM3 分割和当前 RGB 几何测量，按观察到的积木尺寸、支撑面和双臂可达性规划搭塔。`ME_Brain_Adaptor` 使用 `arx_x5`、`ee` 动作及单环境部署入口。

## Dependencies

策略进程使用 `install.sh code` 创建的环境。共享的 SAM3 服务使用支持 `Sam3Model` 和 `Sam3Processor` 的 Transformers、PyTorch、NumPy、Pillow 和 CUDA，权重通过 `SND_SAM_WEIGHTS` 指定为本地 Transformers 模型目录。

在 `ME_Brain_RSI` 目录使用服务环境的 Python 启动：

```bash
export SND_SAM_ROOT=../../../runs/ME_Brain_RSI/sam3_requests
export SND_SAM_WEIGHTS=model_assets/sam3-transformers
CUDA_VISIBLE_DEVICES="${SAM3_GPU:?Set the assigned GPU index}" \
  python -u capabilities/perception/sam3_file_service.py
```

服务加载完成后写出 `ready.json`。相对路径以 `ME_Brain_RSI` 目录为基准。服务和策略进程需要能访问同一请求目录，同一目录由一个服务进程处理。搭塔和标准套娃可共用该服务，各自发送分割提示并筛选结果。

## Evaluation

从 `ME_Brain_RSI` 目录运行，使用服务端的请求目录：

```bash
export SND_SAM_ROOT=../../../runs/ME_Brain_RSI/sam3_requests
export ME_BRAIN_POLICY_PYTHON=./strategy_env/.venv/bin/python
export ME_BRAIN_EVIDENCE_DIR=../../../runs/ME_Brain_RSI
export PYTHONDONTWRITEBYTECODE=1
export ROBODOJO_WS_REQUEST_TIMEOUT_S=600

EVAL_ENV_TYPE=sim ROBODOJO_NUM_ENVS=1 \
  bash eval.sh RoboDojo build_tower none arx_x5 auto 0 0 0 uv base
```

`runtime_config.example.json` 提供 `SND_SAM_ROOT` 的任务配置示例。相机标定位于 `resources/build_tower/configs/calibration.json`。策略输入为三路 RGB、任务指令和机器人自身状态。SAM3 服务通过文件请求接收 RGB 和文本提示，返回实例掩码与置信分数。
