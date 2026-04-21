# 数据转化
## robotwin数据
``` bash
mkdir processed_data && mkdir training_data
bash process_data_rdt.sh ${task_name} ${task_config} ${expert_data_num} ${gpu_id}
```
## 其他
TODO

# 训练
## 生成训练配置文件
```bash
cd policy/RDT
bash generate.sh ${model_name}
# bash generate.sh RDT_demo_clean
```
## 训练
修改生成的`model_config/${model_name}.json`, 设置GPU.
然后将所有要训练的数据放入`training_data/${model_name}/`.

```bash
bash train.sh ${model_name}
```

