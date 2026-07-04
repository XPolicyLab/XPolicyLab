# TinyVLA

TinyVLA 基于 `tinyvla` 接入 XPolicyLab

## 环境安装

```bash
conda activate <your_env>
bash install.sh
```

## 数据处理

```bash
bash process_data.sh <dataset_name> <ckpt_name> <env_cfg_type> <expert_data_num> <action_type>
```

> `process_data.sh` 会调用 `process_data.py`，严格遵守 XPolicyLab README 的 5 元组约定，
> 自动扫描 `data/<dataset_name>/` 下所有 task 子目录，把每个 task 在 `<env_cfg_type>/data/`
> 下前 `expert_data_num` 个 episodes 转成 TinyVLA 官方 HDF5 schema，并把三路图像预先解码/resize
> 为 `480x640` 的 `uint8` RGB 帧，写入：
> 
> ```
> XPolicyLab/policy/TinyVLA/data/<dataset_name>-<ckpt_name>-<env_cfg_type>-<expert_data_num>-<action_type>/
>     episode_0000000.hdf5
>     episode_0000001.hdf5
>     ...
> ```


## 训练

训练入口遵循 XPolicyLab 统一 7 参数：

```bash
bash train.sh <dataset_name> <ckpt_name> <env_cfg_type> <expert_data_num> <action_type> <seed> <gpu_id>
```

训练权重默认保存在 `TinyVLA/checkpoints` 下；子目录名采用 XPolicyLab 约定的 6 元组
`<dataset_name>-<ckpt_name>-<env_cfg_type>-<expert_data_num>-<action_type>-<seed>`。

## 评测

评测入口遵循 XPolicyLab 统一 11 参数：

```bash
bash eval.sh <dataset_name> <task_name> <ckpt_name> <env_cfg_type> <expert_data_num> <action_type> <seed> <policy_gpu_id> <env_gpu_id> <policy_conda_env> <eval_env_conda_env>
```
默认使用最新的checkpoint
