## 数据转化
使用默认的转化的lerobot数据集即可.


## 计算norm stat
首先在`assets/norm/`里面根据自己训练的数据集, 编写好一个yaml.
示例:
```yaml
fold_clothes.yaml

data:
  datasets_type: vla
  train_path: /mnt/pfs/user/lerobot/fold_clothes
  norm_path: assets/norm_stats/fold_clothes_0_customized.json

train:
  global_batch_size: 512
  output_dir: output/norm
```
然后计算norm:

```bash
cd lingbot_vla
bash compute_norm_stat.sh /path/to/*.yml
```
由于默认保存的norm stat不能直接用于训练, 训练需要使用另外格式, 可以使用脚本:
```bash
# 示例, 后四维度填的是arm, effort, arm,effort对应的维度
python scripts/conver_norm_stat.py  assets/norm_stats/robotwin_5_customized.json assets/norm_stats/robotwin_5.json 6 1 6 1
```

## 训练
首先在`configs/vla/`编辑一个训练用的配置, 示例:
注意, global_batch_size建议设置为 micro_batch_size * GPU_NUM, 里面所有路径建议写为绝对路径, 方便部署时的索引.

```yaml
fold_clothes.yml

model:
  model_path: /mnt/pfs/user/model_weights/lingbot-vla-4b
  tokenizer_path: /mnt/pfs/user/model_weights/Qwen2.5-VL-3B-Instruct/
  post_training: true
  adanorm_time: true
  old_adanorm: true

data:
  datasets_type: vla
  data_name: robotwin_fold_clothes_0
  train_path: /mnt/pfs/user/lerobot/fold_clothes
  num_workers: 8
  norm_type: bounds_99_woclip
  norm_stats_file: /mnt/pfs/user/assets/norm_stats/fold_clothes_0.json # 转化后的norm路径

train:
  output_dir: /mnt/pfs/user/lingbot-vla/output/fold_clothes_01
  loss_type: L1_fm
  data_parallel_mode: fsdp2
  enable_full_shard: false
  module_fsdp_enable: true
  use_compile: true
  use_wandb: true
  rmpad: false
  rmpad_with_pos_ids: false
  ulysses_parallel_size: 1
  freeze_vision_encoder: false
  tokenizer_max_length: 24
  action_dim: 14
  max_action_dim: 75
  max_state_dim: 75
  lr: 1.0e-4
  lr_decay_style: constant
  num_train_epochs: 69
  micro_batch_size: 8
  global_batch_size: 16
  max_steps: 220000
  ckpt_manager: dcp
  save_steps: 220000
  save_epochs: 1
  enable_fp32: true
  enable_resume: true
```

```bash
# bash finetune.sh configs/vla/fold_clothes.yml /mnt/pfs/user/lerobot/fold_clothes output/fold_clothes/
bash finetune.sh /path/to/vla/train_config /path/to/lerobot/dataset_name /path/to/output/
```

## 部署
首先要在`/path/to/output/`下面copy一份对应的`lingbotvla_cli.yaml`, 然后就能正常部署.