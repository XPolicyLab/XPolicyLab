---
name: setup-libero-training
description: Use when the user wants to set up and run a minimal end-to-end LIBERO fine-tuning/training in this GalaxeaVLA (G0.5) repo from scratch — builds the venv, wires the checkpoints + LIBERO LeRobot data + stats into the exact layout the configs expect, then runs a single-GPU smoke to prove the data→model→loss→optimizer loop works. Triggers: "配一个最小 libero 训练", "set up libero training", "跑通 libero finetune", "minimal libero smoke".
---

# Set up a minimal LIBERO training

Goal: get `bash scripts/run/finetune.sh 1 libero --test` to run end-to-end on one GPU.
Follow the steps in order; **each step has a verification gate — do not proceed past a failing gate.**
Fail loudly (report the exact error); never paper over a missing file with a fallback.

Create a TodoWrite list mirroring steps 1–5.

## Step 1 — Environment (`.venv` + `.env`)

```bash
# China-network mirrors (harmless elsewhere)
export UV_DEFAULT_INDEX=${UV_DEFAULT_INDEX:-https://mirrors.aliyun.com/pypi/simple/}
export UV_PYTHON_INSTALL_MIRROR=${UV_PYTHON_INSTALL_MIRROR:-https://gh-proxy.com/https://github.com/astral-sh/python-build-standalone/releases/download}
export HF_ENDPOINT=${HF_ENDPOINT:-https://hf-mirror.com}

[ -d .venv ] || uv sync --index-strategy unsafe-best-match   # builds venv; flash-attn-4 is a prebuilt wheel, no nvcc needed
[ -f .env ] || cp .env.example .env                          # then fill PROJECT_ROOT / G05_OUTPUT_DIR / HF caches
```

- If `.env` was just created from the template, edit it: set `PROJECT_ROOT` to this repo's
  absolute path and `G05_OUTPUT_DIR="$PROJECT_ROOT/runs"`. Append `source "$PROJECT_ROOT/.venv/bin/activate"`
  at the end so a single `source .env` is the one-step entry.
- **Gate:** `source .env && python -c "import torch, g05; print(torch.__version__, torch.cuda.is_available())"`
  must print a `+cu128` torch and `True`.

## Step 2 — Checkpoints into the layout the configs expect

The configs hardcode these relative paths (training path → config field):

| Path | Config field |
|---|---|
| `checkpoints/g05-base/checkpoints/model_state_dict.pt` | `configs/model/g05.yaml: pretrained_ckpt` |
| `checkpoints/qwen3_5_2b_base_processor/` | `configs/model/g05.yaml: hf_processor_path` |
| `checkpoints/action_tokenizer.pt` | `configs/tokenizer/actioncodec.yaml: ckpt_dir` |

Two ways to populate them:

- **Public download:** `huggingface-cli download OpenGalaxea/G05 --repo-type model --local-dir checkpoints`
  (≈44 GB; produces exactly this layout). The public `g05-base` matches the public source's
  self-attention action expert — use it, not any visual-memory variant.
- **Reuse a prepared copy** (clusters with a shared, already-built copy): if `GALAXEA_LIBERO_SRC`
  is set to such a directory, symlink instead of downloading:
  ```bash
  S="$GALAXEA_LIBERO_SRC"
  ln -sfn "$S/data" data
  mkdir -p checkpoints/g05-base/checkpoints
  ln -sfn "$S/checkpoints/action_tokenizer.pt"          checkpoints/action_tokenizer.pt
  ln -sfn "$S/checkpoints/qwen3_5_2b_base_processor"    checkpoints/qwen3_5_2b_base_processor
  ln -sfn "$S/checkpoints/g05-base/checkpoints/model_state_dict.pt" checkpoints/g05-base/checkpoints/model_state_dict.pt
  ```
  This assumes `GALAXEA_LIBERO_SRC` is already in the canonical layout above. For a source with
  different internal names, map each file explicitly (see `ENV_SETUP_NOTES.md` §2.3 if present).
  Verify a loaded checkpoint actually matches: the first-step log must read `Loaded N / N` with
  `Missing (rand init) 0`. A partial `Loaded 838/946` means a **wrong checkpoint variant** (e.g. a
  linear-attention `mem` build vs the public self-attention model) — stop and fix the source, do not train on it.
- **Gate:** `ls checkpoints/g05-base/checkpoints/model_state_dict.pt checkpoints/action_tokenizer.pt checkpoints/qwen3_5_2b_base_processor/tokenizer_config.json` — all three exist.

## Step 3 — LIBERO training data + stats

`configs/data/libero.yaml` expects four LeRobot **v3.0 @ 512×512** suites and a stats file:

```text
data/stats/libero_stats.json
data/libero/libero_{10,goal,object,spatial}_no_noops_lerobot/
```

- If you symlinked `data/` in Step 2, this is already satisfied — skip to the gate.
- Otherwise: the `IPEC-COMMUNITY/libero_*_no_noops_1.0.0_lerobot` datasets are LeRobot **v2.1 / 256×256**
  and do **not** drop in. Re-convert to v3.0 / 512 (e.g.
  [openpi convert_libero_data_to_lerobot.py](https://github.com/Physical-Intelligence/openpi/blob/main/examples/libero/convert_libero_data_to_lerobot.py)),
  place each suite under the exact directory name above, then generate stats (it is **computed**, not downloaded):
  ```bash
  python tests/test_dataloader_batch.py --mixture configs/data/libero.yaml \
      --stats data/stats/libero_stats.json --stats-downsample-rate 1
  ```
- **Gate:** `ls data/stats/libero_stats.json data/libero/libero_10_no_noops_lerobot` both resolve.

## Step 4 — Smoke test (the real verification)

```bash
source .env
CUDA_VISIBLE_DEVICES=0 bash scripts/run/finetune.sh 1 libero --test --max_datasets 1 \
    model.max_steps=2 model.num_workers=2
```

(Add `model.pretrained_ckpt=null` to test the pipeline with random init before checkpoints are ready.)

- **Gate (must all appear in the log / `runs/libero/test/`):**
  - first-batch `pixel_values` shape log → data pipeline OK
  - ActionCodec round-trip `LOSSLESS`
  - decreasing per-step `Loss:`
  - `Max step 2 reached, stop training` and exit code 0
- The `--test` flag forces offline logging (`logger.mode=offline`); no SwanLab/W&B key needed.

## Step 5 — Report

Summarize: env (torch+cuda), how checkpoints/data were obtained (download vs symlink), the
checkpoint match ratio (`Loaded N/N`), and the smoke's loss trajectory. Then give the real
training command: `bash scripts/run/finetune.sh 1 libero` (single GPU) or `8 libero` (multi-GPU).

## Notes / gotchas
- Training needs **no** LIBERO simulator. Only evaluation rollouts (`scripts/run/eval_libero.sh`) do —
  install it separately with `uv pip install -e <LIBERO> --no-deps` (never without `--no-deps`).
- `--dry-run` still runs 1 train + 1 eval step (it is **not** config-only), so it also needs data + checkpoints.
- `flash_attn` has no `__version__`; importing it successfully is the success signal.
