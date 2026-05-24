# A step-by-step guide to fine-tuning Wan

[WAN](https://github.com/Wan-Video/Wan2.2) is an excellent video generation model, but the training code is missing.
We provide the following example for fine-tuning the WAN model.

## 0. Installation

Install `giga-train`, `giga-datasets` from PyPi:

```bash
pip3 install giga-train
pip3 install giga-datasets
```

## 1. Download model

Download wan2.2 models using huggingface-cli:

```bash
pip3 install "huggingface_hub[cli]"
huggingface-cli Wan-AI/Wan2.2-TI2V-5B-Diffusers --local-dir ./models/Wan2.2-TI2V-5B-Diffusers
```

## 2. Prepare dataset

The raw_data directory contains the following structure:

```
raw_data/
├── 0.mp4          # Video file 0
├── 0.txt          # Prompt for Video 0
├── 1.mp4          # Video file 1
├── 1.txt          # Prompt for Video 1
├── ...
```

Run the following code to process it into the giga training format:

```python
video_paths = glob(os.path.join(video_dir, '*.mp4'))
label_writer = PklWriter(os.path.join(save_dir, 'labels'))
video_writer = FileWriter(os.path.join(save_dir, 'videos'))
for idx in tqdm(range(len(video_paths))):
    anno_file = video_paths[idx].replace('.mp4', '.txt')
    prompt = open(anno_file, 'r').read().strip()
    label_dict = dict(data_index=idx, prompt=prompt)
    label_writer.write_dict(label_dict)
    video_writer.write_video(idx, video_paths[idx])
label_writer.write_config()
video_writer.write_config()
label_writer.close()
video_writer.close()
label_dataset = load_dataset(os.path.join(save_dir, 'labels'))
video_dataset = load_dataset(os.path.join(save_dir, 'videos'))
dataset = Dataset([label_dataset, video_dataset])
dataset.save(save_dir)
```

The full code is available [here](scripts/pack_data.py) and run it as follows:

```bash
python scripts/pack_data.py --video_dir ./data/raw_data/ --save_dir ./data/giga_data/
```

## 3. Define dataloader

First, write a transform class to handle data preprocessing, enabling configurable loading via a registry mechanism.
The full code is available [here](wan/wan_transforms.py).

```python
@TRANSFORMS.register
class WanTransform:
    def __init__(self, num_frames, height, width):
        self.num_frames = num_frames
        self.height = height
        self.width = width

    def __call__(self, data_dict):
        video = data_dict['video']
        prompt = data_dict['prompt']
        indexes = np.linspace(0, len(video) - 1, self.num_frames, dtype=int)
        images = video.get_batch(indexes)
        if not isinstance(images, torch.Tensor):
            images = torch.from_numpy(images.asnumpy())
        images = images.permute(0, 3, 1, 2).contiguous()
        images = F.resize(images, (self.height, self.width), InterpolationMode.BILINEAR)
        images = images / 255.0 * 2.0 - 1.0
        new_data_dict = dict(images=images, prompt=prompt)
        return new_data_dict
```

After integrating the `Trainer` class, you can implement the `get_dataloaders` interface to create your own dataloader.
Of course, the Trainer already comes with a default dataloader that you can use directly.

```python
class WanTrainer(Trainer):
    def get_dataloaders(self, data_config):
        dataset = load_dataset(data_config.data_or_config)
        transform = WanTransform(**data_config.transform)
        dataset.set_transform(transform)
        batch_sampler = ...
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_sampler=batch_sampler,
            collate_fn=DefaultCollator(),
            num_workers=data_config.num_workers,
        )
        return dataloader
```

## 4. Define model

Three models are defined for the WAN: text_encoder, VAE, and transformer, all of which are implemented via the `get_models` interface:

```python
class WanTrainer(Trainer):
    def get_models(self, model_config):
        pretrained = model_config.pretrained
        # text_encoder
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained, subfolder='tokenizer')
        self.text_encoder = UMT5EncoderModel.from_pretrained(pretrained, subfolder='text_encoder', torch_dtype=self.dtype)
        self.text_encoder.requires_grad_(False)
        self.text_encoder.to(self.device)
        # vae
        self.vae = AutoencoderKLWan.from_pretrained(pretrained, subfolder='vae')
        self.vae.requires_grad_(False)
        self.vae.to(self.device)
        # transformer
        transformer = WanTransformer3DModel.from_pretrained(pretrained, subfolder='transformer', torch_dtype=self.dtype)
        transformer.train()
        ...
        return transformer
```

Then use the `forward_step` interface to implement the model's forward pass:

```python
class WanTrainer(Trainer):
    def forward_step(self, batch_dict):
        with torch.no_grad():
            images = batch_dict['images'].to(self.vae.dtype)
            images = rearrange(images, 'b t c h w -> b c t h w')
            latents = self.vae.encode(images).latent_dist.sample()
            latents = (latents - self.latents_mean) * self.latents_std
            prompt_embeds = self.get_t5_prompt_embeds(batch_dict['prompt'])
        timestep, sigma = self.get_timestep_and_sigma(latents.shape[0], latents.ndim)
        noise = torch.randn_like(latents)
        target = noise - latents
        noisy_latents = noise * sigma + latents * (1 - sigma)
        model_pred = self.model(
            hidden_states=noisy_latents.to(self.dtype),
            timestep=timestep,
            encoder_hidden_states=prompt_embeds.to(self.dtype),
            return_dict=False,
        )[0]
        loss = (model_pred.float() - target.float()) ** 2
        loss = loss.mean()
        return loss
```

The full code is [here](./wan/wan_trainer.py).

## 5. Define config file

Define all the necessary modules for training via a configuration file. A complete configuration is provided [here](./configs/wan_5b_t2v_ft.py).

```python
config = dict(
    project_dir='./experiments/wan/wan_5b_t2v_ft',  # Path to save logs, config, and models
    launch=dict(
        gpu_ids=[0],  # GPU IDs for training
    ),
    dataloaders=dict(  # Dataloader Configuration
        train=dict(
            data_or_config=[  # Paths to the training data
                './data/giga_data/',
            ],
            ...
        ),
    ),
    models=dict(  # Model Configuration
        pretrained='./models/Wan2.2-TI2V-5B-Diffusers/',  # Path to wan model
        ...
    ),
    optimizers=dict(  # Optimizer Configuration
        ...
    ),
    schedulers=dict(  # Scheduler Configuration
        ...
    ),
    train=dict(  # Other train Configuration
        ...
    ),
)

```

## 6. Launch training

Run the following code to start finetuning the WAN model:

```python
config = load_config(config_file)
torch.cuda.set_device(config.launch.gpu_ids[0])
runner = WanTrainer.load(config)
runner.train()
```

We also provide a launch utility for multi-node, multi-GPU training:

```bash
python scripts/train.py --launch --config configs.wan_5b_t2v_ft.config
```

## 7. Training strategies

### Launching training using DeepSpeed

GigaTrain supports training on single/multiple GPUs using DeepSpeed.
To use it, you just need to change `config.launch.distributed_type` to `DEEPSPEED` and set `config.launch.deepspeed_config` to your DeepSpeed config file path.
The supported DeepSpeed config options can be found in [here](https://github.com/open-gigaai/giga-train/giga_train/distributed/accelerate_configs).
Below is an example of launching training using DeepSpeed ZeRO-0 or ZeRO-2:

```python
launch=dict(
    gpu_ids=[0, 1, 2, 3, 4, 5, 6, 7],  # set available GPUs
    distributed_type='DEEPSPEED',  # set distributed type to DEEPSPEED
    deepspeed_config=dict(
        deepspeed_config_file='accelerate_configs/zero0.json',  # relative path to deepspeed config directory
        # deepspeed_config_file='accelerate_configs/zero2.json',
    ),
),
```

### Launching training using FSDP2

GigaTrain supports training on single/multiple GPUs using FSDP2.
To use it, you just need to change `config.launch.distributed_type` to `FSDP` and set `config.launch.fsdp_config`.
Below is an example of launching training using FSDP2:

```python
launch=dict(
    gpu_ids=[0, 1, 2, 3, 4, 5, 6, 7],  # set available GPUs
    distributed_type='FSDP',  # set distributed type to FSDP
    fsdp_config=dict(
        fsdp_version='2',  # specify fsdp version
        fsdp_auto_wrap_policy='TRANSFORMER_BASED_WRAP',
        fsdp_transformer_layer_cls_to_wrap='WanTransformerBlock',  # specify the transformer block class in your model
        fsdp_cpu_ram_efficient_loading='false',
        fsdp_state_dict_type='FULL_STATE_DICT',
    ),
),
```

### Mixed Precision Training

GigaTrain supports mixed precision training using fp16, bf16 and fp8.
To use it, you just need to set `config.train.mixed_precision` to `fp16`, `bf16`, `fp8`.

### Gradient Accumulation

You can set `config.train.gradient_accumulation_steps` to a value greater than 1 to enable gradient accumulation for large batch size training.

### Gradient Checkpointing

You can enable gradient checkpointing for saving memory during training. For it, you just need to set `config.train.activation_checkpointing` to `True` and
specify the list of module names to checkpoint in `config.train.activation_class_names`.

### Exponential Moving Average (EMA)

You can enable EMA to store the EMA model for higher accuracy. For it, you just need to set `config.train.with_ema` to `True`.

### Torch Dynamo

Torch Dynamo can be configured to optimize your training. For it, you just need to set `config.train.dynamo_config` to `dict(backend='inductor')`.
See https://pytorch.org/docs/stable/torch.compiler.html for more details.

### Resume Training

To resume training after an unexpected interruption midway through, simply set `config.train.resume` to `True` and re-run your code.

## 8. Model Inference

You can use the official [code](https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B-Diffusers) or run the following code to perform model inference：

```python
vae = AutoencoderKLWan.from_pretrained(pretrained, subfolder='vae', torch_dtype=torch.float32)
kwargs = dict(vae=vae, torch_dtype=torch.bfloat16)
if transformer_model_path is not None:
    kwargs['transformer'] = WanTransformer3DModel.from_pretrained(transformer_model_path, torch_dtype=torch.bfloat16)
pipe = WanPipeline.from_pretrained(pretrained, **kwargs)
pipe.to(device)
negative_prompt = (
    '色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走'
)
output = pipe(
    prompt=prompt,
    negative_prompt=negative_prompt,
    height=704,
    width=1280,
    num_frames=121,
    num_inference_steps=50,
    guidance_scale=5.0,
).frames[0]
export_to_video(output, save_path, fps=24)
```

The full code is available [here](scripts/inference.py) and run it as follows:

```bash
python scripts/inference.py --pretrained ./models/Wan2.2-TI2V-5B-Diffusers/ --prompt PROMPT --save_path ./result.mp4 --transformer_model_path TRASFORER_MODEL_PATH
```
