# Build Tower

`build_tower` 使用纯 RGB 脚本策略，由 `ME_Brain_Adaptor` 调度。支持 `RoboDojo`、`arx_x5`、`ee` 动作，每个进程运行一个环境，使用非批量部署入口。

## Dependencies

使用 `install.sh code` 创建的脚本环境，需要 NumPy、SciPy、OpenCV 和 Pillow。相机标定随源码提供，不需要外部模型权重、CuRobo 或在线服务。

## Evaluation

从 `ME_Brain_RSI` 目录执行：

```bash
export ME_BRAIN_POLICY_PYTHON=./strategy_env/.venv/bin/python
export ME_BRAIN_EVIDENCE_DIR=../../../runs/ME_Brain_RSI
export PYTHONDONTWRITEBYTECODE=1

EVAL_ENV_TYPE=sim ROBODOJO_NUM_ENVS=1 \
  bash eval.sh RoboDojo build_tower none arx_x5 auto 0 0 0 uv base
```

GPU 参数和环境客户端环境名按实际部署选择。`action_type=auto` 选择 `ee`，也可显式指定 `ee`。`ckpt_name=none`，无需训练或下载权重。标准启动脚本在策略解释器启动前将 OpenMP、OpenBLAS 和 MKL 的线程数设为 `1`。

任务模块映射在 `modules.json`，静态标定在 `resources/build_tower/configs/calibration.json`。输入为服务端解码后的三路 RGB 与机器人自身状态，动作由各任务独立的感知和控制模块产生。

运行输出由 `strategy_evidence_dir` 或 `ME_BRAIN_EVIDENCE_DIR` 指定，建议放在策略目录之外。
