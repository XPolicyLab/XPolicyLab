# 配置LingBot_VA
## 配置模型基础环境
``` bash
conda create -n lingbot_va python==3.10.6
conda activate lingbot_va

pip install torch==2.9.0 torchvision==0.24.0 torchaudio==2.9.0 --index-url https://download.pytorch.org/whl/cu126
pip install websockets einops diffusers==0.36.0 transformers==4.55.2 accelerate msgpack opencv-python matplotlib ftfy easydict

pip install packaging ninja
ninja --version; echo $?  # Verify Ninja --> should return exit code "0"
mkdir -p .pip-tmp .pip-cache
TMPDIR=$PWD/.pip-tmp PIP_CACHE_DIR=$PWD/.pip-cache MAX_JOBS=4 pip install flash-attn --no-build-isolation
```

## 配置训练环境
```bash
pip install lerobot==0.3.3 scipy wandb --no-deps
```

## 配置XPolicyLab环境
```bash
cd ../../
pip install -e .
```