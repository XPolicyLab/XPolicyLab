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

默认 TFDS 名：`aloha_<dataset_name>-<ckpt_name>-<env_cfg_type>-<expert_data_num>-<action_type>`

可用 `OPENVLA_TFDS_DATASET_NAME` 覆盖。

## 训练

```bash
bash train.sh <dataset_name> <ckpt_name> <env_cfg_type> <expert_data_num> <action_type> <seed> <gpu_id>
```

Checkpoint：`checkpoints/<6-tuple>/`

## 评估

```bash
bash eval.sh <task_name> <env_cfg> <expert_data_num> joint <gpu_id> <seed> <policy_conda_env> <eval_env_conda_env> <checkpoint_path>
```
