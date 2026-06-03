# RoboTwin2 Inference Setup

## Setup Steps

1. Copy H-RDT folder to RoboTwin/policy/
```bash
ROBOTWIN_POLICY_DIR=./policy
cp -r H-RDT "${ROBOTWIN_POLICY_DIR}/"
```

2. Copy bak folder to H-RDT/
```bash
cp -r H-RDT/bak "${ROBOTWIN_POLICY_DIR}/H-RDT/"
```

3. Create checkpoints directory and copy model files
```bash
CKPT_NAME=<your_checkpoint_name>
mkdir -p "${ROBOTWIN_POLICY_DIR}/H-RDT/checkpoints/${CKPT_NAME}/"
cp <path_to_config.json> "${ROBOTWIN_POLICY_DIR}/H-RDT/checkpoints/${CKPT_NAME}/config.json"
cp <path_to_pytorch_model.bin> "${ROBOTWIN_POLICY_DIR}/H-RDT/checkpoints/${CKPT_NAME}/pytorch_model.bin"
```

## Run Inference

1. Modify eval.sh configuration
```bash
cd "${ROBOTWIN_POLICY_DIR}/H-RDT"
# Edit eval.sh parameters:
# - ckpt_setting="checkpoints/<your_checkpoint_name>"  # Set checkpoint path
# - task_name="your_task"                 # Set task name
# - task_config="your_config"             # Set task config
```

2. Run inference
```bash
bash eval.sh
```
