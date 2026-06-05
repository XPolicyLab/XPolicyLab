# RDT_1B

RDT_1B 已按 XPolicyLab policy 方式封装。环境安装见 [INSTALLATION.md](INSTALLATION.md)。

## 数据准备

默认训练脚本会把数据目录设为：

```text
policy/RDT_1B/data/<dataset_name>-<ckpt_name>-<env_cfg_type>-<expert_data_num>-<action_type>
```

如需使用已有 hdf5/tfrecord 数据目录，可通过环境变量覆盖：

```bash
RDT_HDF5_DIR=<path_to_training_data> bash train.sh ...
```

## 训练

```bash
bash train.sh <dataset_name> <ckpt_name> <env_cfg_type> <expert_data_num> <action_type> <seed> <gpu_id>
```

示例：

```bash
bash train.sh RoboDojo stack_bowls arx_x5 50 joint 0 0,1,2,3,4,5,6,7
```

训练输出固定保存到：

```text
policy/RDT_1B/checkpoints/<dataset_name>-<ckpt_name>-<env_cfg_type>-<expert_data_num>-<action_type>-<seed>
```

常用可覆盖变量：

```bash
RDT_PRETRAINED_MODEL=<path_or_hf_id>
TEXT_ENCODER_NAME=<path_or_hf_id>
VISION_ENCODER_NAME=<path_or_hf_id>
RDT_DEEPSPEED_ARGS="--hostfile=hostfile.txt --num_gpus=8"
```

## 部署

环境安装见 [INSTALLATION.md](INSTALLATION.md)。首次请执行 `bash install.sh`。

推荐分别执行 `setup_eval_policy_server.sh` 与 `setup_eval_env_client.sh` 便于查看 server 报错；同机也可使用 `eval.sh`：

```bash
bash eval.sh RoboDojo stack_bowls RoboDojo_sim_seed_0 arx_x5 3500 joint 0 <policy_gpu> <env_gpu> RDT XPolicyLab
```
