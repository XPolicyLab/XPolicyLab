# Pi_05_SF

`Pi_05_SF` 是基于 OpenPI接入 Spatial Forcing。


## 环境安装

安装说明见 [INSTALLATION.md](INSTALLATION.md)。

基本入口：

```bash
export XPL_ROOT=<XPolicyLab 仓库根目录>
cd "$XPL_ROOT/policy/Pi_05_SF/openpi"

UV_LINK_MODE=copy GIT_LFS_SKIP_SMUDGE=1 uv sync --group lerobot
UV_LINK_MODE=copy GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .
```

## 保存 VGGT 离线特征

直接生成 cache：

```bash
export XPL_ROOT=<XPolicyLab 仓库根目录>
cd "$XPL_ROOT/policy/Pi_05_SF/openpi"

export HF_LEROBOT_HOME=<LeRobot 数据根目录>
export VGGT_WEIGHT_PATH=<VGGT 权重目录>
export SF_CACHE_DIR=<要写入的离线 VGGT feature cache 根目录>

uv run --no-sync python scripts/precache_vggt_sf_cache.py pi05sf_jax_robodojo_v21_offcache \
  --batch-size 256 \
  --num-workers 8
```

多进程生成：

```bash
SF_CACHE_DIR=<要写入的离线 VGGT feature cache 根目录> \
PRECACHE_NPROC=8 \
uv run --no-sync torchrun --standalone --nproc_per_node=8 \
  scripts/precache_vggt_sf_cache.py pi05sf_jax_robodojo_v21_offcache \
  --batch-size 256 \
  --num-workers 8
```

常用变量：

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `SF_CACHE_DIR` | `openpi/results/sf_cache` | cache 输出目录 |
| `SF_CACHE_SAVE_DTYPE` | `bf16` | 支持 `fp16`、`bf16`、`int8` |
| `SF_CACHE_CHUNK_SIZE` | `128` | 每个 chunk 保存的 step 数 |
| `SF_DATASET_UID` | `0` | 数据集编号，会参与 cache key |
| `PRECACHE_NUM_BATCHES` | `0` | `0` 表示遍历数据集可推断的全部 batch |
| `PRECACHE_OVERWRITE` | `0` | `1` 表示覆盖已有 slot |
| `PRECACHE_VGGT_DEVICES` | 空 | 指定单进程使用的 CUDA device，例如 `0,1` |
| `PRECACHE_VGGT_DEVICE_COUNT` | `1` | 未指定 device 时，单进程默认使用的 GPU 数 |

生成脚本会写 summary：

```text
SF_CACHE_DIR/_precache_summary.json
SF_CACHE_DIR/_precache_summary.rank0.json
```

单进程使用 `_precache_summary.json`；`torchrun` 多进程使用 rank summary。

## 一键生成 Cache 后 Strict 训练

该入口对齐原 openpi-SF 的 strict offcache 流程：先生成或补齐 VGGT cache，检查 summary 和 on-disk mask，再调用 readonly JAX 训练。

```bash
export XPL_ROOT=<XPolicyLab 仓库根目录>
cd "$XPL_ROOT/policy/Pi_05_SF/openpi"

export HF_LEROBOT_HOME=<LeRobot 数据根目录>
export PI05_BASE_PATH=<Pi05 base checkpoint 根目录>
export VGGT_WEIGHT_PATH=<VGGT 权重目录>
export SF_CACHE_DIR=<离线 VGGT feature cache 根目录>

bash scripts/run_pi05sf_jax_offcache_strict.sh
```

已有 cache 目录时，脚本默认拒绝继续，避免误复用旧 cache。可显式选择：

```bash
RESET_SF_CACHE=1 bash scripts/run_pi05sf_jax_offcache_strict.sh
REUSE_SF_CACHE=1 bash scripts/run_pi05sf_jax_offcache_strict.sh
```

strict wrapper 常用变量：

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `PRECACHE_NPROC` | `1` | 预生成 cache 使用的 torchrun 进程数 |
| `MIN_CACHE_SLOTS` | `1000` | cache 校验要求的最少已写 slot |
| `MIN_EPISODES` | `2` | cache 校验要求的最少 episode 目录数 |
| `MIN_CHUNKS` | `2` | cache 校验要求的最少 chunk 数 |
| `BATCH_SIZE` | `256` | 预生成和训练 batch size |
| `TRAIN_STEPS` | `60000` | readonly 训练步数 |
| `NUM_WORKERS` | `8` | dataloader worker 数 |

## Readonly 离线 Cache 训练

如果 cache 已经准备好，也可以只跑 readonly 训练：

```bash
export XPL_ROOT=<XPolicyLab 仓库根目录>
cd "$XPL_ROOT/policy/Pi_05_SF/openpi"

export HF_LEROBOT_HOME=<LeRobot 数据根目录>
export PI05_BASE_PATH=<Pi05 base checkpoint 根目录>
export VGGT_WEIGHT_PATH=<VGGT 权重目录>
export SF_CACHE_DIR=<已有离线 VGGT feature cache 根目录>

bash scripts/run_pi05sf_jax_offcache.sh
```

训练脚本固定使用：

```text
sf_cache_enable = True
sf_cache_mode = readonly
sf_cache_miss_policy = error
```

缺任意一帧 cache 时会直接报错。


## 当前训练配置

实际配置在 `openpi/src/openpi/external_config/pi05sf_jax_robodojo_v21_offcache.py`。

| 配置项 | 当前值 |
|--------|--------|
| `TrainConfig.name` | `pi05sf_jax_robodojo_v21_offcache` |
| `project_name` | `xpolicylab-pi05sf-jax` |
| `repo_id` | `RoboDojo_lerobot_v21_video` |
| `model` | `Pi0Config(pi05=True)` |
| `weight_loader` | `${PI05_BASE_PATH}/params` |
| `align_target_model` | `vggt` |
| `align_loss_coeff` | `0.2` |
| `use_vggt_pe` | `True` |
| `use_vlm_norm` | `True` |
| `use_camera_params` | `False` |
| `sf_cache_enable` | `True` |
| `sf_cache_mode` | `readonly` |
| `sf_cache_miss_policy` | `error` |
| `batch_size` | `256` |
| `num_train_steps` | `60000` |
| `fsdp_devices` | `8` |

## XPolicyLab 推理/评测

确认 `deploy.yml` 中的任务、checkpoint、端口和 `repo_id` 后启动。推荐使用当前 policy 目录自带的 `eval.sh`，它会自动把当前目录链接到 `XPL_ROOT/policy/Pi_05_SF`，并把本目录的 `openpi/src`、`openpi/packages/openpi-client/src` 和 `openpi/src/vggt` 加入 `PYTHONPATH`。

```bash
export XPL_ROOT=<XPolicyLab 仓库根目录>
export POLICY_DIR=<Pi_05_SF policy 目录>
export OPENPI_SF_VENV="${POLICY_DIR}/openpi/.venv"

cd "${POLICY_DIR}"
source "${OPENPI_SF_VENV}/bin/activate"

bash eval.sh
```

如果当前目录还没有拷贝进 `XPolicyLab/policy/Pi_05_SF`，也可以直接把 `POLICY_DIR` 指到独立目录，例如当前开发目录；`eval.sh` 会在 `XPL_ROOT/policy/Pi_05_SF` 创建软链接。

### Server 启动方式迁移

对应启动示例：

```bash
export XPL_ROOT=<XPolicyLab 仓库根目录>
export POLICY_DIR=<Pi_05_SF policy 目录>
export OPENPI_SF_VENV="${POLICY_DIR}/openpi/.venv"
export CKPT_NAME=<Pi05SF checkpoint step 目录>
export POLICY_PORT=5001
export POLICY_HOST=127.0.0.1
export GPU=1

export HF_LEROBOT_HOME=<LeRobot 数据根目录>
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export WANDB_MODE=offline
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9

ACTION_DIM=$(bash "${XPL_ROOT}/utils/get_action_dim.sh" "$(dirname "${XPL_ROOT}")" arx_x5)
echo "ACTION_DIM=${ACTION_DIM}"

cd "${POLICY_DIR}"
source "${OPENPI_SF_VENV}/bin/activate"

PYTHONUNBUFFERED=1 \
PYTHONWARNINGS=ignore::UserWarning \
CUDA_VISIBLE_DEVICES="${GPU}" \
XPL_ROOT="${XPL_ROOT}" \
bash eval.sh \
  --overrides \
    port="${POLICY_PORT}" \
    host="${POLICY_HOST}" \
    dataset_name="RoboDojo_lerobot_v21_video" \
    task_name="stack_bowls" \
    ckpt_name="${CKPT_NAME}" \
    env_cfg_type="arx_x5" \
    expert_data_num="all" \
    seed="0" \
    policy_name="Pi_05_SF" \
    action_type="joint" \
    action_dim="${ACTION_DIM}" \
    train_config_name="pi05sf_jax_robodojo_v21_offcache" \
    repo_id="RoboDojo_lerobot_v21_video"
```