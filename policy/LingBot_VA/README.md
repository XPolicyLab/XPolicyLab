# LingBot_VA

LingBot_VA 是基于 Wan2.2 的视频动作（VA）基础模型策略，在 XPolicyLab 中以 **websocket 桥接** 方式部署：底层官方 `wan_va_server.py` 跑推理，上层 `setup_policy_server.py` 做观测编码 / 动作解码转发，对外暴露统一 eval 接口。

安装见 [INSTALLATION.md](INSTALLATION.md)。

## 架构

```
eval client ──ws──▶ forward server (setup_policy_server.py)
                        │ Model (model.py)
                        │   encode_obs → WebsocketClientPolicy
                        │   ← action chunk ←
                        ▼ ws
              backend VA server (wan_va_server.py, 0.0.0.0:<auto>)
                        │ load .merged_ckpt (base vae/tokenizer/text_encoder
                        │                    + finetuned transformer)
                        ▼ GPU
```

- **backend VA server**：`launch_wan_va_server.sh`，官方推理进程，监听 `0.0.0.0:<VA_PORT>`。
- **forward server**：`setup_eval_policy_server.sh`，XPolicyLab 桥接，订阅 backend VA，对外监听 `<EXPOSE_HOST>:<EXPOSE_PORT>`。
- 两层分离便于：backend 跑在 GPU 机，forward 可同机或跨机；obs/action 维度转换只在 forward 层做。

## 一键启动（推荐）

`start_servers.sh` 一次性在 tmux 里起好两个 server，backend VA 端口自动挑空余，只需指定 GPU 和对外暴露端口：

```bash
cd policy/LingBot_VA
bash start_servers.sh <GPU_ID> <EXPOSE_PORT>
# 例：
bash start_servers.sh 0 10002
```

可选 env 覆盖：

| 变量 | 默认 | 作用 |
|------|------|------|
| `CHECKPOINT_PATH` | launch 脚本默认 | finetune ckpt 目录（须含 `transformer/`） |
| `BASE_MODEL_PATH` | launch 脚本默认 | base 权重目录（须含 `vae/` 等） |
| `CONFIG_NAME` | `robotwin30_train` | wan_va 配置名 |
| `EXPOSE_HOST` | `0.0.0.0` | forward server 监听地址 |
| `CONDA_ENV` | `lingbot_va` | conda 环境名 |
| `TMUX_PREFIX` | `lingbot_va` | tmux session 名前缀（多实例时区分） |

启动后输出会给出两个 tmux session 名和日志路径。管理：

```bash
tmux attach -t lingbot_va_va     # 看 backend
tmux attach -t lingbot_va_fwd    # 看 forward
# 停止
tmux kill-session -t lingbot_va_va; tmux kill-session -t lingbot_va_fwd
```

多实例：换 `EXPOSE_PORT` + `TMUX_PREFIX` 即可，backend 端口和 MASTER_PORT 都自动挑空余。

```bash
TMUX_PREFIX=lingbot_va2 bash start_servers.sh 1 10003
```

## 手动分步启动

### 1. backend VA server

```bash
conda activate lingbot_va
cd policy/LingBot_VA
bash launch_wan_va_server.sh <GPU_ID> <VA_PORT>
# 例：bash launch_wan_va_server.sh 0 10001
```

首次会跑 `prepare_merged_ckpt.py` 把 base 权重和 finetune transformer 合并到 `.merged_ckpt/`（软链），耗时较长。env 覆盖 `CHECKPOINT_PATH` / `BASE_MODEL_PATH` 可换权重。

### 2. forward server

```bash
conda activate lingbot_va
cd policy/LingBot_VA

VA_SERVER_HOST=127.0.0.1 \
VA_SERVER_PORT=10001 \
bash setup_eval_policy_server.sh \
    RoboDojo <task_name> <ckpt_name> \
    <env_cfg_type> <expert_data_num> <action_type> \
    <seed> <gpu_id> lingbot_va \
    <EXPOSE_PORT> 0.0.0.0
```

`VA_SERVER_HOST/PORT` 指向 backend VA server；`$10` 是 forward 自己的监听端口，`$11` 是监听地址。

> 注：在 ws-bridge 模式下，`task_name` / `ckpt_name` / `expert_data_num` / `seed` / `action_type` 不影响 server 端推理（推理由 backend VA 驱动），它们只在 eval client 端有意义。这些参数仅为满足脚本位置参数契约而保留默认值。

### 3. eval client

```bash
bash setup_eval_env_client.sh \
    RoboDojo <task_name> <ckpt_name> \
    <env_cfg_type> <action_type> <seed> <env_gpu_id> \
    <eval_env_conda_env> "<additional_info>" \
    <EXPOSE_PORT> <forward_server_ip>
```

debug 验证（假 obs，不依赖 simulator）：`deploy.yml` 的 `eval_env: debug` 时自动走 `debug_env_client.py`，跑全零 obs 验证 `client → forward → backend` 链路。

同机闭环也可用 `eval.sh`（会等 forward 端口就绪后起 client）。

## 数据处理

```bash
cd lingbot_va
python dataset/transform.py --raw_dir <processed_data_task_dir> --repo_id <repo_id>
python dataset/add_action_config.py --dataset-root <lerobot_dataset_dir> --backup
python dataset/extract_wan_22_latents.py --dataset-root <lerobot_dataset_dir> --model-root <wan_model_dir>
python dataset/make_empty_embedding.py --model-root <wan_model_dir> --output <lerobot_dataset_dir>/empty_emb.pt
python dataset/compute_action_stat.py --dataset-root <lerobot_dataset_dir> --output <lerobot_dataset_dir>/action_norm_stats.json
```

默认 LeRobot 数据：`${XPOLICYLAB_LEROBOT_DATA_ROOT:-<robodojo_test>/data}/<repo_id>`（`arx_x5` → `RoboDojo_sim_arx-x5_v30`）。可用 `LINGBOT_VA_DATASET_PATH` 覆盖完整路径，或用 `LEROBOT_DATASET_REPO_ID` 覆盖 repo 名。

## 训练

```bash
bash train.sh <bench_name> <ckpt_name> <env_cfg_type> <expert_data_num> <action_type> <seed> <gpu_id>
```

Checkpoint：`checkpoints/<6-tuple>/`。

## 可视化合成

`visualization/make_video.py` 把 `obs_data_*.pt` 里的三视角观测合成视频（左=`cam_left_wrist`，中=`cam_high`，右=`cam_right_wrist`，按时序）：

```bash
conda activate lingbot_va
cd policy/LingBot_VA/visualization
python make_video.py "real/<episode_dir>" --fps 8
```

## 模型与数据路径

| 变量 | 说明 |
|------|------|
| `XPOLICYLAB_LEROBOT_DATA_ROOT` / `LEROBOT_DATA_ROOT` | LeRobot 根目录，默认 `<robodojo_test>/data` |
| `LEROBOT_DATASET_REPO_ID` | repo_id，默认 `RoboDojo_sim_arx-x5_v30`（`arx_x5`） |
| `LINGBOT_VA_DATASET_PATH` | LeRobot 训练数据完整目录 |
| `LINGBOT_VA_CONFIG_NAME` | 训练配置名（默认 `robotwin30_train`） |
| `CHECKPOINT_PATH` | backend VA server 加载的 finetune ckpt 目录 |
| `BASE_MODEL_PATH` | base 权重目录（vae / text_encoder / tokenizer） |

换 ckpt 必须重启 backend VA server（forward 不会热加载）。
