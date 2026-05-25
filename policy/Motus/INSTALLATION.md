``` bash
# Create conda environment
conda create -n motus python=3.10 -y
conda activate motus

# install torch (cuda12.8)
pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128

# install flash 
pip install flash-attn --no-build-isolation

# Install motus dependencies
pip install -r requirements.txt

# (Optinal) Install lerobot dependencies
pip install --no-deps lerobot==0.3.2
pip install -r requirements/lerobot.txt

```