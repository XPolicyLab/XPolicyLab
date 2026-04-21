# 配置OpenVLA OFT环境
## 配置模型环境
```bash
conda create -n openvla_oft python==3.10.6
conda activate openvla_oft

pip install torch torchvision torchaudio

cd openvla-oft
pip install -e .

# Install Flash Attention 2 for training (https://github.com/Dao-AILab/flash-attention)
#   =>> If you run into difficulty, try `pip cache remove flash_attn` first
pip install packaging ninja
ninja --version; echo $?  # Verify Ninja --> should return exit code "0"
mkdir -p .pip-tmp .pip-cache
TMPDIR=$PWD/.pip-tmp PIP_CACHE_DIR=$PWD/.pip-cache MAX_JOBS=4 pip install "flash-attn==2.5.5" --no-build-isolation
```

## 配置XPolicyLab环境
```bash
cd ../../
pip install -e .
```