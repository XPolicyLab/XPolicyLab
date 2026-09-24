# Rex_M1_preview Installation

Adapted for Rex-M1-preview from the `Xiaomi_Robotics_1` adapter in XPolicyLab.

`install.sh` is the recommended path. This document provides the manual equivalent, checkpoint
preparation, and smoke checks. This is an eval-only release: the vendored `rex_m1/`
carries the inference code, and although the upstream training entrypoint is still present it does
not implement this policy's history sampling — see the Training section of `README.md`.

## 1. One-command Install

```bash
cd XPolicyLab/policy/Rex_M1_preview
bash install.sh
conda activate rex_m1_preview
```

The installer creates the `rex_m1_preview` conda environment (override with `REX_M1_PREVIEW_CONDA_ENV`) and installs
PyTorch 2.8, Flash Attention, and the other core dependencies.

## 2. Manual Install Equivalent

```bash
conda create -n rex_m1_preview python=3.12 -y
conda activate rex_m1_preview

pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 \
  --index-url https://download.pytorch.org/whl/cu128
pip install transformers==4.57.1 scipy numpy Pillow ninja psutil
# ninja and psutil are imported by flash-attn's setup.py, and --no-build-isolation
# tells pip not to fetch build requirements, so install them first.
pip install flash-attn==2.8.3 --no-build-isolation

# Vendored rex_m1 requirements (mmengine and liger-kernel are required just to
# import mibot, so inference needs them as well).
pip install -r rex_m1/assets/requirements.txt

# XPolicyLab itself: client_server.ws imports websockets, msgpack and
# msgpack_numpy, and the requirements above bring in none of them. Run this from
# this policy directory.
pip install -e ../..

pip install opencv-python-headless h5py imageio imageio-ffmpeg tqdm
```

`Qwen/Qwen3-VL-4B-Instruct` must be reachable from HuggingFace or the local cache: it is the VLM
backbone and supplies the processor used at inference.

## 3. Prepare Inference Weights

The checkpoint is distributed through ModelScope.

```bash
cd XPolicyLab/policy/Rex_M1_preview
modelscope download gzx2019/Rex-M1-preview --local-dir checkpoints/Rex_M1_preview
```

If the checkpoint is already on the machine (a shared filesystem, an earlier download, a copy from a
colleague), skip the download entirely — `model_dir` takes priority over `ckpt_name`:

```bash
# (a) link it into place
ln -sfn /abs/path/to/Rex_M1_preview checkpoints/Rex_M1_preview

# (b) or put the absolute path in deploy.yml
#     model_dir: /abs/path/to/Rex_M1_preview
```

Verify what you got before evaluating:

```bash
md5sum checkpoints/Rex_M1_preview/last.ckpt/checkpoint/mp_rank_00_model_states.pt
#   383ed6219938fd065c7d2700ada5e9f1
md5sum checkpoints/Rex_M1_preview/config.py
#   138ed1e3da91221a5af267853353c111
```

The expected layout:

```text
policy/Rex_M1_preview/checkpoints/Rex_M1_preview/
├── config.py
└── last.ckpt/
```

At evaluation time the checkpoint is resolved via the `ckpt_name` field in `deploy.yml`, or via
`model_dir` pointing at an absolute path.

`config.py` is the authority for the history settings: the adapter reads `history_frames`,
`history_stride`, `history_res`, `history_pool` and `ptp_steps` from it. `history_dropout` is read
and logged only.

## 4. Smoke Checks

```bash
conda activate rex_m1_preview
python -c "import torch; print('cuda:', torch.cuda.is_available())"
python -c "import transformers; print('transformers:', transformers.__version__)"
python -c "import XPolicyLab; print('XPolicyLab ok')"

# The vendored model package must import cleanly — this is what model.py loads.
PYTHONPATH=rex_m1 python -c \
  "from mibot.server.deploy import load_model, load_stats; print('mibot ok')"
```

Then run the offline wiring check from `README.md` (Evaluation, `EVAL_ENV_TYPE=debug`); it needs no
simulator and no GPU-resident checkpoint beyond what `model.py` loads.
