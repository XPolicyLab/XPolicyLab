
---

# Customized Downstream Task Dataset Construction and Deployment

This guide explains how to construct a custom downstream task dataset for post-training and how to deploy it on corresponding downstream tasks. We use the 5 tasks from **RoboTwin 2.0** ("open_microwave", "click_bell", "stack_blocks_three", "place_shoe", "put_object_cabinet") as an example.

## 1. Compute Normalization Statistics

First, you need to calculate the normalization statistics for your custom dataset:

```bash
CUDA_VISIBLE_DEVICES=0 bash train.sh \
    scripts/compute_norm_robotwin_5.py \
    configs/norm/robotwin_5.yaml \
    --model.model_path /path/to/LingBot-VLA \
    --model.tokenizer_path /path/to/Qwen2.5-VL-3B-Instruct/ \
    --data.train_path /path/to/mixed_robotwin_5tasks \
    --data.norm_path assets/norm_stats/robotwin_5_customized.json
```

> **Note:**  
> In [`scripts/compute_norm_robotwin_5.py`](../../../scripts/compute_norm_robotwin_5.py) (lines 71–75), specify the original keys of **action** and **state** in the lerobot-formatted data.  
> For RoboTwin2.0, these correspond to:
> - `action`
> - `observation.state`

---

## 2. Construct Custom Dataset

The `assets/norm_stats/robotwin_5_customized.json` generated in the previous step stores your normalization statistics. To use this file:

1.  **Specify the path** in your [Run Command](../../../README.md) via: `--data.norm_stats_file assets/norm_stats/robotwin_5_customized.json`.
2.  **Replace the Dataset Class:** In [tasks/vla/train_lingbotvla.py](../../../tasks/vla/train_lingbotvla.py), replace the default `RobotwinDataset` with the provided `CustomizedRobotwinDataset`.

### Implementation Details:
When constructing a custom dataset similar to `CustomizedRobotwinDataset`, ensure the following:
*   **Key Mapping:** When instantiating `self.normalizer` and obtaining `normalized_item`, you must modify the original keys for actions, states, and images from all views. For RoboTwin 2.0, the keys are:
    *   **Action:** `'action'`
    *   **State:** `'observation.state'`
    *   **Images:** `'observation.images.cam_high'`, `'observation.images.cam_left_wrist'`, `'observation.images.cam_right_wrist'`
*   **Data Type:** When instantiating `self.normalizer`, set the parameter **`data_type='customized'`**.

---

## 3. Deployment

To ensure correct results, the data processing logic during the testing phase must **be identical to the training phase**.

Taking [deploy/lingbot_robotwin_policy.py](../../../deploy/lingbot_robotwin_policy.py) as an example:
You should use the same **action**, **state**, and **image keys** as in the training phase.  
Also, when constructing `policy.normalizer` in line 323, make sure the attribute `data_type` is consistent with your training setup.

For example, if you used `CustomizedRobotwinDataset` during training,  
then line **323** in `deploy/lingbot_robotwin_policy.py` should be **`data_type='customized'`**.

---