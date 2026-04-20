
# 配置Pi 05

## 配置模型环境
```bash
GIT_LFS_SKIP_SMUDGE=1 uv sync
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .
```
## 配置XPolicyLab环境
```bash
source .venv/bin/activate
cd ../../
uv pip install -e .
```