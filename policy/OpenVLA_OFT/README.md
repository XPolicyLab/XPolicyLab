# OpenVLA_OFT

OpenVLA_OFT 使用 ALOHA/TFDS 格式数据训练，已封装为 XPolicyLab policy。

## 数据转换

先将 XPolicyLab/RoboDojo hdf5 转为 ALOHA，再构建 TFDS：

```bash
cd /mnt/nfs/niantian/robodojo_test/XPolicyLab
python scripts/transform_aloha_hdf5_format.py <xspark_data_dir> <aloha_output_dir>

cd policy/OpenVLA_OFT/openvla_oft
TFDS_DATA_DIR=<tensorflow_datasets_dir> \
  bash scripts/build_tfds_aloha.sh <data_sample> <aloha_output_dir> <processed_dir> 0.05 0
```

`train.sh` 默认使用的 TFDS 名称为：

```text
aloha_<dataset_name>-<ckpt_name>-<env_cfg_type>-<expert_data_num>-<action_type>
```

如需使用已有 TFDS 名称，可设置 `OPENVLA_TFDS_DATASET_NAME`。

## 训练

```bash
bash train.sh <dataset_name> <ckpt_name> <env_cfg_type> <expert_data_num> <action_type> <seed> <gpu_id>
```

示例：

```bash
bash train.sh RoboDojo stack_bowls arx_x5 50 joint 0 0,1,2,3
```

训练产物默认保存到：

```text
policy/OpenVLA_OFT/checkpoints/<dataset_name>-<ckpt_name>-<env_cfg_type>-<expert_data_num>-<action_type>-<seed>
```

底层 `openvla_oft/scripts/finetune.sh` 的 `--run_root_dir` 已由根目录 `train.sh` 固定为上述 checkpoint 目录。

## 评估

```bash
bash eval.sh <task_name> <env_cfg> <expert_data_num> joint <gpu_id> <seed> <policy_conda_env> <eval_env_conda_env> <checkpoint_path>
```

当前模型默认使用 `joint` 动作类型。
