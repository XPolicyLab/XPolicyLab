# OpenPI-SF for Spatial_Forcing

## 训练配置

默认训练配置名：

```text
pi05sf_jax_robodojo_v21_offcache
```

主要字段：

| 字段 | 当前值 |
|------|--------|
| `repo_id` | `RoboDojo_lerobot_v21_video` |
| `model` | `Pi0Config(pi05=True)` |
| `align_enabled` | `True` |
| `align_target_model` | `vggt` |
| `vla_layers_align` | `12` |
| `vggt_layers_align` | `-1` |
| `pooling_func` | `bilinear` |
| `use_vggt_pe` | `True` |
| `use_vlm_norm` | `True` |
| `align_loss_coeff` | `0.2` |
| `sf_cache_enable` | `True` |
| `sf_cache_mode` | `readonly` |
| `sf_cache_miss_policy` | `error` |
| `sf_cache_save_dtype` | `bf16` |
| `sf_cache_chunk_size` | `128` |
| `sf_dataset_uid` | `0` |
| `batch_size` | `256` |
| `num_workers` | `8` |
| `num_train_steps` | `60000` |
| `save_interval` | `5000` |

权重和路径通过环境变量覆盖：

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `PI05_BASE_PATH` | `./checkpoints/pi05_base` | Pi05 JAX base checkpoint 根目录，必须包含 `params/` 和 `assets/` |
| `VGGT_WEIGHT_PATH` | `./checkpoints/VGGT-1B` | VGGT 权重目录，必须包含 `model.pt` |
| `SF_CACHE_DIR` | `./results/sf_cache` | OpenPI-SF chunked 离线特征 cache |
| `PI05SF_ASSETS_DIR` | `./assets/pi05sf_robodojo_v21` | norm stats / assets 目录 |
| `HF_LEROBOT_HOME` | `${XPL_DATA_ROOT}` 或 `../data` | LeRobot 数据根目录 |


## Policy Server

从 checkpoint 启动 OpenPI policy server：

```bash
cd /path/to/openpi05-sf/openpi

uv run --no-sync scripts/serve_policy.py \
  policy:checkpoint \
  --policy.config=pi05sf_jax_robodojo_v21_offcache \
  --policy.dir=/path/to/checkpoint_step_dir \
  --port=8000
```