# FastWAM Installation

`install.sh` installs the Python environment, but this document is kept because
FastWAM needs two additional steps before evaluation or training: preprocessing
the ActionDiT backbone and downloading the released FastWAM checkpoint/statistics.

## 1. Install the Environment and Preprocess ActionDiT

```bash
cd XPolicyLab/policy/FastWAM
bash install.sh
conda activate fastwam

cd FastWAM
mkdir -p checkpoints
export DIFFSYNTH_MODEL_BASE_PATH="$(pwd)/checkpoints"
python scripts/preprocess_action_dit_backbone.py \
  --model-config configs/model/fastwam.yaml \
  --output checkpoints/ActionDiT_linear_interp_Wan22_alphascale_1024hdim.pt \
  --device cuda \
  --dtype bfloat16
```

The preprocessing step writes the ActionDiT backbone used by the FastWAM runtime.

## 2. Download Released Checkpoints

```bash
cd XPolicyLab/policy/FastWAM/FastWAM
huggingface-cli download yuanty/fastwam \
  robotwin_uncond_3cam_384.pt \
  robotwin_uncond_3cam_384_dataset_stats.json \
  --local-dir ./checkpoints/fastwam_release
```

## 3. Runtime Notes

- Keep `DIFFSYNTH_MODEL_BASE_PATH` pointed at the directory containing the
  preprocessed ActionDiT and any Wan/FastWAM checkpoints expected by the config.
- `robotwin_uncond_3cam_384_dataset_stats.json` must stay next to the released
  checkpoint unless `deploy.yml` or environment variables override the stats path.
