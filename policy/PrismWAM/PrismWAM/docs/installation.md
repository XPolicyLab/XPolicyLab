# Installation

Requires Linux x86-64, an NVIDIA GPU, a C++ compiler and the CUDA 12.8 toolkit (`nvcc`).

```bash
conda create -n prismwam --override-channels -c conda-forge python=3.13 pip
conda activate prismwam
pip install -r environments/torch.txt
pip install --no-build-isolation -r requirements.txt -e .
```

If the CUDA toolkit is not in the default location, set `CUDA_HOME` before the last step.

Simulators (RoboTwin 2.0, RoboDojo, LIBERO) are installed separately. RoboTwin and RoboDojo
connect to the policy through [XPolicyLab](xpolicylab.md).
