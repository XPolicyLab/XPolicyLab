# 配置X-VLA训练环境
``` bash
conda create -n XVLA python=3.10 -y
conda activate XVLA

cd xvla/
pip install -r requirements.txt

pip show torch
# 安装对应torch版本的cuda版本, 否则训练可能报错
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 # 举例
```
