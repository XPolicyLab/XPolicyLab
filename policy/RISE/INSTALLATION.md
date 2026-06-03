# RISE Installation

完整步骤见 [`README.md`](README.md)（安装、权重、数据、训练与评测）。

```bash
cd policy/RISE
bash install.sh RISE
conda activate RISE
```

`install.sh` 不会自动下载 Pi0.5 权重或 checkpoint；按 README §1 准备 `weights/pi05_base_pytorch/`，评测前确保 checkpoint 可被 `setup_eval_policy_server.sh` 解析（或设置 `RISE_CHECKPOINT_PATH`）。
