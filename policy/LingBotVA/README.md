## 数据转化
### step1 转化lerobot数据
转化lerobot数据需要使用特定的脚本, 在`XPolicyLab/policy/LingBotVA/lingbot_va/dataset/transofrm.py`.  
注意, 这里的action是30维度, 直接对应了lingbot-va中的30维度, 如果你没有对应维度, 可以选择填0, 或者不填, 如果不填, 则需要修改后续config配置.  
脚本转化用的video而非image, 因为后续操作要用到视频, 并且图像被resize到了256*256.  

```bash
# 这里的数据存放结构位robotwin中pi05运行过process_data_pi05.sh后的文件格式
# 如果开启--is_multi参数, 那么将会遍历文件夹下所有的hdf5文件, 只要路径指定到/path/to/processed_data/
# instruction固定为task/instructions.json中第一条语言指令, 方便转化
python dataset/transform.py --raw_dir /path/to/processed_data/task --repo_id ... 
```

### step2 添加额外的参数给到info.json
这里给到添加的`action_config`使用的是对应的`instruction`.

```bash
python scripts/add_action_config.py --dataset-root /path/to/leroobt/dataset/ --backup
```

### step3 Wan2.2生成latents编码
计算生成latents, 使用的模型为`Wan2.2-TI2V-5B-Diffusers`.
```bash
python scripts/extract_wan_22_latents.py --dataset-root /path/to/leroobt/dataset/ --model-root /path/to/Wan2.2-TI2V-5B-Diffusers
```

### step4 生成empty embedding

``` bash
python scripts/make_empty_embedding.py --model-root /path/to/Wan2.2-TI2V-5B-Diffusers --output /path/to/leroobt/dataset/empty_emb.pt
```

## 计算norm stat
``` bash
python scripts/compute_action_stat.py --dataset-root /path/to/leroobt/dataset/ --output /any/where/
```
然后编辑`wan_va/configs/va_robotwin30_train_cfg.py`, 这里有个norm stat, 填进去.
注意, 如果你在步骤1中action和state不是30维度, 需要模仿`va_robotwin_cfg.py`中的`used_action_channel_ids`相关参数, 进行映射.

## 训练

```bash
# 编辑里面的参数, 注意, 在va_robotwin30_train_cfg.py中开启了gradient_accumulation_steps, 作者建议global batch size到32或者64
# 计算方法为: GPU_NUM * 1 * py中开启了gradient_accumulation_steps, 注意不能修改batch_size参数, 否则会报错
bash finetune.sh
```

## 部署
TODO