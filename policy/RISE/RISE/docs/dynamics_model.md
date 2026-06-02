# Dynamics Model

```
cd RISE/dynamics/dynamics_model 
# Navigate to the dynamics model directory before running the following commands
```

## Generated Samples


<p align="center">
   <img src="../rise_assets/conveyor.gif" alt="Conveyor demo" width="49%" />
   <img src="../rise_assets/backpack.gif" alt="Backpack demo" width="49%" />
</p>






## Data Format

The framework expects data in the LeRobot format. For optimal training performance, we strongly recommend pre-resizing videos to `[256, 192]` resolution.

### Directory Structure

All tasks should be organized in the `dataset` directory with the following structure:

```bash
# copy your dataset under the dataset directory
cp -r path/to/your/dataset dataset/
```

Each dataset is organized as follows:
```
task_A/
├── data/
│   └── chunk-000/
│       ├── episode_000000.parquet
│       ├── episode_000001.parquet
│       ├── episode_000002.parquet
│       └── ...
├── meta/
│   ├── info.json              
│   ├── episodes.jsonl        
│   ├── episodes_stats.jsonl   
│   └── tasks.jsonl        
└── videos/
    └── chunk-000/
        └── [video files]
```

### Video Preprocessing

The `preprocess.sh` script resizes all videos in the dataset to `256x192` resolution using ffmpeg, preserving aspect ratio with center padding. Processed videos are saved in `videos_small` while maintaining the original directory structure.

**Usage:**

```bash
# Process specific datasets
./preprocess.sh dataset1 [dataset2](optional)
```

The output would be as follows with **videos_small**:
```
task_A/
├── data/
├── meta/
└── videos/
└── videos_small/
│    └── chunk-000/
│        └── [video files]
```




## Model Checkpoints

### Base LTX Backbone

Download the LTX-Video backbone components (Text Encoder, Tokenizer, and VAE) using the provided script:

```bash
./download.sh
```

This script automatically downloads all required components from the [LTX-Video HuggingFace repository](https://huggingface.co/Lightricks/LTX-Video) to the `checkpoints` directory.

Alternatively, you can manually download the following components:
1. **Text Encoder**: [text_encoder](https://huggingface.co/Lightricks/LTX-Video/tree/main/text_encoder)
2. **Tokenizer**: [tokenizer](https://huggingface.co/Lightricks/LTX-Video/tree/main/tokenizer)
3. **VAE**: [vae](https://huggingface.co/Lightricks/LTX-Video/tree/main/vae)
4. **Pre-trained dynamics model**: [dynamics_model](https://huggingface.co/OpenDriveLab-org/RISE_Assets/tree/main/dynamics_model/pretrained), pretrained on Galaxea Open World and AgiBot World Alpha jointly.

Place all downloaded weights in the same directory and update the `pretrained_model_name_or_path` field in your configuration file.


## Training

### Pre-training

Pre-training is performed on large-scale robotic datasets to learn general dynamics priors. We utilize the following datasets:


- **Galaxea Open World Dataset**: [Galaxea-Open-World-Dataset](https://huggingface.co/datasets/OpenGalaxea/Galaxea-Open-World-Dataset/tree/main)
- **AgiBot World Alpha**: [AgiBotWorld-Alpha](https://huggingface.co/datasets/agibot-world/AgiBotWorld-Alpha)


#### Steps

1. **Prepare Data**: Convert your datasets to the LeRobot format as described above.

2. **Configure Training**: Edit `configs/ltx_model/pretrain.yaml` according to the comments:
   - Set `pretrained_model_name_or_path` to your LTX backbone checkpoint directory
   - Set `diffusion_model.model_path` to your pre-trained diffusion checkpoint
   - Configure `data.train.data_roots` and `data.val.data_roots` to point to your dataset directories

3. **Launch Training**:
   ```bash
   bash train_task_centric.sh
   ```

### Fine-tuning

Fine-tuning adapts the pre-trained model to specific task domains using domain-specific datasets.

#### Steps

1. **Prepare Task-Specific Data**: Organize your fine-tuning dataset in the LeRobot format.

2. **Compute Action Normalization Statistics**: Use `norm.py` to compute and save normalization statistics:
   ```bash
   python norm.py --datasets <your_finetune_dataset> --save-config data/utils/action_norm.json
   ```
   This automatically computes min and max values for each dataset and saves them to a JSON configuration file.

3. **Configure Fine-tuning**: Edit `configs/ltx_model/finetune.yaml`:
   - Set `pretrained_model_name_or_path` to your LTX backbone checkpoint directory
   - Set `diffusion_model.model_path` to your diffusion checkpoint
   - Configure `data.train.data_roots` and `data.val.data_roots` for your fine-tuning dataset
   - Add `norm_config_path: data/utils/action_norm.json` to both `data.train` and `data.val` sections
   - The data loader will automatically use the normalization values from the config file based on dataset names

4. **Launch Fine-tuning**:
   ```bash
   bash task_finetune.sh
   ```


## Inference

The inference pipeline generates future video sequences conditioned on initial observations and action sequences.

### Steps

1. **Configure Inference**: Edit `configs/ltx_model/infer.yaml`:
   - Set `pretrained_model_name_or_path` your LTX backbone checkpoint directory
   - Set `diffusion_model.model_path` to your diffusion checkpoint

2. **Update Inference Script**: Edit `infer.sh` with appropriate paths

3. **Run Inference**:
   ```bash
   bash infer.sh
   ```

### Inference Parameters

- `--config_file`: Path to inference configuration file
- `--image_root`: Directory containing input observation images
- `--output_path`: Directory to save generated videos
- `--act_tokens_path`: Path to action token file (`.pt` format)
- `--norm_constant`: Normalization constant for action tokens (e.g., `FINETUNE_TASK`)
