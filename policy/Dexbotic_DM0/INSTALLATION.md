# Dexbotic_DM0 Installation

The one-command installer covers the local adapter and upstream package install, but this document is kept because Dexbotic DM0 needs a CUDA-matched PyTorch environment, extra training dependencies, pretrained DM0-base weights, and raw-data path setup.

## 1. Create the Conda Environment

```bash
cd XPolicyLab/policy/Dexbotic_DM0
conda create -n DM0 python=3.10 -y
conda activate DM0

pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 \
  --index-url https://download.pytorch.org/whl/cu128
pip install 'deepspeed>=0.18.0' 'numpydantic>=1.6'
```

## 2. Run the Adapter Installer

```bash
cd XPolicyLab/policy/Dexbotic_DM0
conda activate DM0
bash install.sh
```

`install.sh` installs `dexbotic/` in editable mode, installs XPolicyLab in editable mode, and adds the data conversion dependencies `opencv-python-headless` and `tqdm`.

## 3. Manual Install Equivalent

```bash
cd XPolicyLab/policy/Dexbotic_DM0/dexbotic
pip install -e .

cd ../../..
pip install -e .
pip install h5py pyyaml opencv-python-headless tqdm
```

## 4. Prepare DM0-base Weights

```bash
pip install -U "huggingface_hub[cli]"
# Run `hf auth login` first if the model is gated.

cd XPolicyLab/policy/Dexbotic_DM0
mkdir -p dexbotic/checkpoints
hf download Dexmal/DM0-base --local-dir dexbotic/checkpoints/DM0-base
```

Or point the adapter at an existing checkpoint:

```bash
export DM0_BASE_MODEL=/path/to/DM0-base
```

## 5. Raw Data Layout

By default, `process_data.sh` expects:

```text
<DM0_RAW_DATA_ROOT>/sim_cloud/<task_name>/<env_cfg_type>/*.hdf5
```

Override it with:

```bash
export DM0_RAW_DATA_ROOT=/path/to/RoboDojo
```

## 6. Smoke Checks

```bash
python -c "import torch; print('cuda:', torch.cuda.is_available())"
python -c "import deepspeed; print('deepspeed ok')"
python -c "import dexbotic; print('dexbotic ok')"
python -c "import XPolicyLab; print('XPolicyLab ok')"
```

## 7. Common Training Variables

| Variable | Meaning |
| --- | --- |
| `DM0_RAW_DATA_ROOT` | Raw HDF5 root used by conversion. |
| `DM0_CONVERTED_DATA_ROOT` | Converted output root; defaults to `data/<4-tuple>`. |
| `DM0_BASE_MODEL` | Pretrained DM0-base model path. |
| `DM0_GLOBAL_BATCH_SIZE` | Global batch size. |
| `DM0_BATCH_SIZE` | Per-GPU micro batch size. |
| `DM0_GRAD_ACCUM` | Gradient accumulation steps; derived automatically when unset. |
| `DM0_MAX_STEPS` | Training steps. |
| `DM0_SAVE_STEPS` | Checkpoint save interval. |
| `DM0_CONVERT_WORKERS` | Data conversion workers. |
| `DM0_TRAIN_BACKEND` | Training backend: `deepspeed`, `fsdp2`, or `ddp`. |
