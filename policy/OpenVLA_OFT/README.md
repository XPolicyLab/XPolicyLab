# OpenVLA_OFT

OpenVLA_OFT 使用 ALOHA/TFDS 格式数据。安装见 [INSTALLATION.md](INSTALLATION.md)。

## 数据转换

```bash
# 在 XPolicyLab 根目录
python scripts/transform_aloha_hdf5_format.py <xspark_data_dir> <aloha_output_dir>

cd policy/OpenVLA_OFT/openvla_oft
TFDS_DATA_DIR=<tensorflow_datasets_dir> \
  bash scripts/build_tfds_aloha.sh <data_sample> <aloha_output_dir> <processed_dir> 0.05 0
```

默认 TFDS 名：`aloha_<bench_name>-<ckpt_name>-<env_cfg_type>-<expert_data_num>-<action_type>`

可用 `OPENVLA_TFDS_DATASET_NAME` 覆盖。

## 训练

```bash
bash train.sh <bench_name> <ckpt_name> <env_cfg_type> <expert_data_num> <action_type> <seed> <gpu_id>
```

Checkpoint：`checkpoints/<6-tuple>/`

## 部署

环境安装见 [INSTALLATION.md](INSTALLATION.md)。首次请执行 `bash install.sh`。

推荐分别执行 `setup_eval_policy_server.sh` 与 `setup_eval_env_client.sh` 便于查看 server 报错；同机也可使用 `eval.sh`：

```bash
bash eval.sh RoboDojo stack_bowls RoboDojo-cotrain-arx_x5-3500-joint-0 arx_x5 3500 joint 0 <policy_gpu> <env_gpu> openvla_oft XPolicyLab
```
