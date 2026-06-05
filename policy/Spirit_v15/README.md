# Spirit_v15

Spirit_v15 训练数据为 Spirit 自有目录结构（非 LeRobot）。安装见 [INSTALLATION.md](INSTALLATION.md)。

## 数据格式

转换后的目录：

```text
<converted_data_root>/
  meta/task_info.json
  data/episode_000000/...
```

默认原始数据根目录为 XPolicyLab 仓库的 `data/`（相对 `../../../data`），可通过 `SPIRIT_RAW_DATA_ROOT` 覆盖。

## 数据处理

```bash
bash process_data.sh <dataset_name> <ckpt_name> <env_cfg_type> <expert_data_num> <action_type>
```

## 训练

```bash
bash train.sh <dataset_name> <ckpt_name> <env_cfg_type> <expert_data_num> <action_type> <seed> <gpu_id>
```

示例：

```bash
bash process_data.sh RoboDojo sweep_blocks arx_x5 50 ee
bash train.sh RoboDojo sweep_blocks arx_x5 50 ee 0 0,1,2,3
```

### Co-train（35 任务）

```bash
bash process_data.sh RoboDojo cotrain arx_x5 100 ee
bash train.sh RoboDojo cotrain arx_x5 100 ee 0 0,1,2,3,4,5,6,7
```

输出：

```text
data/RoboDojo-cotrain-arx_x5-100-ee/
checkpoints/RoboDojo-cotrain-arx_x5-100-ee-0/
```

## 环境变量

| 变量 | 说明 |
|------|------|
| `SPIRIT_RAW_DATA_ROOT` | 原始 HDF5 数据根 |
| `SPIRIT_PATTERNS_CSV` | 匹配 pattern |
| `SPIRIT_CONVERTED_DATA_ROOT` | 转换输出目录 |
| `SPIRIT_PRETRAINED_PATH` | 预训练权重路径或 HF id |
| `SPIRIT_SKIP_CONVERT` | 设为 `1` 跳过转换 |

## 部署

环境安装见 [INSTALLATION.md](INSTALLATION.md)。首次请执行 `bash install.sh`。

推荐分别执行 `setup_eval_policy_server.sh` 与 `setup_eval_env_client.sh` 便于查看 server 报错；同机也可使用 `eval.sh`：

```bash
bash eval.sh RoboDojo stack_bowls RoboDojo_sim_arx-x5_seed_0 arx_x5 3500 joint 0 <policy_gpu> <env_gpu> uv XPolicyLab
```
