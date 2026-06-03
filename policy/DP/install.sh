pip install torch==2.4.1 torchvision
pip install \
  zarr==2.12.0 wandb ipdb gpustat dm_control \
  omegaconf hydra-core==1.2.0 dill==0.3.5.1 \
  einops==0.4.1 diffusers==0.11.1 numba==0.56.4 \
  moviepy imageio av matplotlib termcolor sympy \
  h5py opencv-python numpy==1.23.5 \
  huggingface_hub==0.25.2 pandas

pip install -e .

# install XPolicyLab
cd ../../
pip install -e .