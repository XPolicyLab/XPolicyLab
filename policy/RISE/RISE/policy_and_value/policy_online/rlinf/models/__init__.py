# Copyright 2025 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import os

import torch
from omegaconf import DictConfig
from transformers import (
    AutoConfig,
    AutoImageProcessor,
    AutoProcessor,
    AutoTokenizer,
)

from rlinf.config import torch_dtype_from_precision


def get_model(model_path, cfg: DictConfig, override_config_kwargs=None):
    torch_dtype = torch_dtype_from_precision(cfg.precision)
    
    if cfg.model_name == "openpi":

        import openpi_value.shared.download as download
        import openpi_value.transforms as transforms
        from openpi_value.training import checkpoints as _checkpoints
        from openpi_value.training import config as _config

        from .embodiment.openpi_action_model import (
            OpenPi0Config,
            OpenPi0ForRLActionPrediction,
        )

        import safetensors
        actor_train_config = _config.get_config(cfg.openpi.config_name)

        actor_model_config = actor_train_config.model
                
        default_prompt = actor_train_config.data.default_prompt
        assert default_prompt is not None, "Default prompt must be provided in the data config."
        
        openpi0_config = {**actor_model_config.__dict__, "default_prompt": default_prompt}


        # Turn off these for online learning
        openpi0_config = {**openpi0_config,
                          "state_noise_snr": None,
                          "apply_shape_visual_aug": False,
                          "apply_blur_visual_aug": False}

        actor_model_config = OpenPi0Config(**openpi0_config)

        
        override_config_kwargs = cfg.openpi
        if override_config_kwargs is not None:
            for key, val in override_config_kwargs.items():
                actor_model_config.__dict__[key] = val
        # load model
        checkpoint_dir = download.maybe_download(str(model_path))
        weight_path = os.path.join(checkpoint_dir, "model.safetensors")
        
        model: OpenPi0ForRLActionPrediction = OpenPi0ForRLActionPrediction(
            actor_model_config
        )
        
        # * train actionexpert only
        if actor_model_config.train_expert_only:
            model.freeze_vlm()
        
        safetensors.torch.load_model(model, weight_path, strict=False)
        model.paligemma_with_expert.to_bfloat16_for_selected_params("bfloat16")
        # fsdp replace
        # model.paligemma_with_expert.replace_gemma_decoder_layers()
        # load data stats
        data_config = actor_train_config.data.create(
            actor_train_config.assets_dirs, actor_model_config
        )

        norm_stats = None
        if norm_stats is None:
            # We are loading the norm stats from the checkpoint instead of the config assets dir to make sure
            # that the policy is using the same normalization stats as the original training process.
            if data_config.asset_id is None:
                raise ValueError("Asset id is required to load norm stats.")
            
            try:
                norm_stats = _checkpoints.load_norm_stats(
                    checkpoint_dir /  "assets", data_config.asset_id
                )
            except:
                norm_stats = _checkpoints.load_norm_stats(
                    checkpoint_dir, data_config.asset_id
                )
        # wrappers
        repack_transforms = transforms.Group()
        # default_prompt = None

        # 'repo_id', 'asset_id', 'norm_stats', 'repack_transforms', 'data_transforms', 
        # 'model_transforms', 'use_quantile_norm', 'action_sequence_keys', 'prompt_from_task', 'rlds_data_dir', 'action_space', 'filter_dict_path'

        model.setup_wrappers(
            transforms=[
                *repack_transforms.inputs,
                transforms.InjectDefaultPrompt(default_prompt),
                *data_config.data_transforms.inputs,
                transforms.Normalize(
                    norm_stats, use_quantiles=data_config.use_quantile_norm
                ),
                
                *data_config.model_transforms.inputs,   
            ],
            # data_config.model_transforms.inputs
            # [InjectDefaultPrompt(prompt='Insert the memory stick.'), ResizeImages(height=224, width=224), TokenizePrompt(tokenizer=<openpi_value.models.tokenizer.PaligemmaTokenizer object at 0x7f8f00686c10>, discrete_state_input=True), PadStatesAndActions(model_action_dim=32)]

            output_transforms=[
                *data_config.model_transforms.outputs,
                transforms.Unnormalize(
                    norm_stats, use_quantiles=data_config.use_quantile_norm
                ),
                *data_config.data_transforms.outputs,
                *repack_transforms.outputs,
            ],
        )

    elif cfg.model_name == "mlp_policy":
        from .embodiment.mlp_policy import MLPPolicy

        model = MLPPolicy(
            cfg.obs_dim,
            cfg.action_dim,
            cfg.hidden_dim,
            num_action_chunks=cfg.num_action_chunks,
            add_value_head=cfg.add_value_head,
        )
    else:
        return None
    if torch.cuda.is_available():
        model = model.cuda()

    if cfg.is_lora:
        from peft import LoraConfig, PeftModel, get_peft_model

        if not hasattr(cfg, "lora_path") or cfg.lora_path is None:
            lora_config = LoraConfig(
                r=cfg.lora_rank,
                lora_alpha=cfg.lora_rank,
                lora_dropout=0.0,
                target_modules=[
                    "proj",
                    "qkv",
                    "fc1",
                    "fc2",  # vision
                    "q",
                    "kv",
                    "fc3",
                    "out_proj",  # project
                    "q_proj",
                    "k_proj",
                    "v_proj",
                    "o_proj",
                    "gate_proj",
                    "up_proj",
                    "down_proj",
                    "lm_head",  # llm
                ],
                init_lora_weights="gaussian",
            )
            if cfg.model_name == "openpi":
                module_to_lora = model.paligemma_with_expert.paligemma
                module_to_lora = get_peft_model(module_to_lora, lora_config)
                tag_vlm_subtree(model, False)
                tag_vlm_subtree(module_to_lora, True)
                model.paligemma_with_expert.paligemma = module_to_lora
            else:
                model = get_peft_model(model, lora_config)
        else:
            model = PeftModel.from_pretrained(model, cfg.lora_path, is_trainable=True)

        if hasattr(model, "value_head"):
            for param in model.value_head.parameters():
                param.requires_grad = True

    if hasattr(cfg, "ckpt_path") and cfg.ckpt_path is not None:
        model_dict = torch.load(cfg.ckpt_path)
        model.load_state_dict(model_dict)
    return model


def tag_vlm_subtree(model, is_vlm: bool):
    for n, m in model.named_modules():
        setattr(m, "_to_lora", is_vlm)
