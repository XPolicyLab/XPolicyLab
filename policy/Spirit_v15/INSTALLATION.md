# Spirit_v15 Installation

`install.sh` covers the recommended `uv` setup, but this document is kept because Spirit_v15 has both `uv` and plain-`pip` setup paths plus data conversion variables that are useful for debugging.

## 1. One-command Install

```bash
cd XPolicyLab/policy/Spirit_v15
bash install.sh
```

## 2. Manual `uv` Install

```bash
cd XPolicyLab/policy/Spirit_v15/spirit_v15
uv sync --extra train
source .venv/bin/activate
uv pip install -e .

cd ../../..
pip install -e .
```

## 3. Manual `pip` Install Without `uv`

```bash
cd XPolicyLab/policy/Spirit_v15/spirit_v15
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-base.txt
pip install -r requirements-train.txt
pip install -e .

cd ../../..
pip install -e .
```

## 4. Useful Variables

| Variable | Meaning |
| --- | --- |
| `SPIRIT_PRETRAINED_PATH` | Pretrained checkpoint path or Hugging Face repo id. |
| `SPIRIT_RAW_DATA_ROOT` | RoboDojo raw HDF5 root. |
| `XPOLICYLAB_DATA_ROOT` | XPolicyLab data root; conversion defaults to `../../../data`. |
| `SPIRIT_CONVERTED_DATA_ROOT` | Converted Spirit training data directory. |
| `SPIRIT_PATTERNS_CSV` | Data matching pattern, for example `RoboDojo.stack_bowls.arx_x5`. |
| `SPIRIT_BACKBONE_PATH` | Optional local Qwen backbone directory used by upstream training. |
| `HF_HOME` / `HF_HUB_CACHE` | Optional Hugging Face cache locations used when `spirit_backbone_path` points to a model id or cache-backed snapshot. |

Run `process_data.sh` before `train.sh`; see `README.md` for the unified command interface.

For evaluation, `deploy.yml` defaults to `checkpoints/shared/Spirit-v1.5` for the Spirit base config and `checkpoints/shared/Qwen3-VL-4B-Instruct` for the Qwen backbone. Put those local checkpoints there or override `spirit_base_weights` and `spirit_backbone_path` before starting the policy server.
