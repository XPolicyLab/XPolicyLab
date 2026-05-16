
# 配置Pi 05

## 配置模型环境
```bash
UV_LINK_MODE=copy GIT_LFS_SKIP_SMUDGE=1 uv sync --group lerobot
UV_LINK_MODE=copy GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .
```
## 配置XPolicyLab环境
```bash
source .venv/bin/activate
uv pip install -e ../../.
```