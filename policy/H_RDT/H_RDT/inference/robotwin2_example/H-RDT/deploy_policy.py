import os
import sys
import subprocess
import numpy as np
import torch
import pandas as pd
from PIL import Image
import yaml
import json
from pathlib import Path
from torchvision import transforms
import torch.nn.functional as F
import safetensors.torch
import cv2

current_file_path = os.path.abspath(__file__)
current_folder_path = os.path.dirname(current_file_path)
parent_folder_path = os.path.dirname(os.path.dirname(current_folder_path))
sys.path.append(parent_folder_path)
sys.path.append(current_folder_path)

from models.hrdt_runner import HRDTRunner
from models.encoder.dinosiglip_vit import DinoSigLIPViTBackbone


class MyModel:
    def __init__(self, ckpt_folder, task_name):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.dtype = torch.bfloat16
        
        # Load config
        config_file_path = os.path.join(current_folder_path, 'utils', 'hrdt.yaml')
        with open(config_file_path, 'r') as config_file:
            self.config = yaml.safe_load(config_file)

        # Load action normalization stats
        # Note: In H-RDT practice, we do not perform action normalization
        stat_file_path = os.path.join(current_folder_path, 'utils', 'stats.json')
        with open(stat_file_path, 'r') as file:
            stat = json.load(file)
        self.action_min = np.array(stat['robotwin_agilex']['min'])
        self.action_max = np.array(stat['robotwin_agilex']['max'])

        ckpt_file = os.path.join(ckpt_folder, 'pytorch_model.bin')
        print(f"load model from: {ckpt_file}")
        
        # Load vision encoder
        vision_backbone_path = os.path.join(current_folder_path, 'bak', 'dino-siglip')
        print(f"Loading vision encoder from: {vision_backbone_path}")
        self.vision_encoder = DinoSigLIPViTBackbone(
            vision_backbone_id="dino-siglip",
            image_resize_strategy="letterbox" 
                if self.config["dataset"]["image_aspect_ratio"] == "pad" 
                else "resize-naive",
            default_image_size=384
        )
        self.vision_encoder.to(self.device, dtype=self.dtype)
        self.vision_encoder.eval()
        self.image_transform = self.vision_encoder.get_image_transform()
        
        state_dim = self.config["common"]["state_dim"]
        action_dim = self.config["common"]["action_dim"]
        pred_horizon = self.config["common"]["action_chunk_size"]
        
        # Create H-RDT model with specified training mode
        self.policy = HRDTRunner.from_pretrained(
            pretrained_model_name_or_path=ckpt_folder,
            state_dim=state_dim,
            action_dim=action_dim,
            pred_horizon=pred_horizon,
            config=self.config["model"],
            act_pos_emb_config=[
                ("state", 1),
                ("action", pred_horizon),
            ],
            img_pos_emb_config=[
                ("image", (self.config["common"]["img_history_size"], 
                          self.config["common"]["num_cameras"], 
                          -self.vision_encoder.num_patches)),
            ],
            lang_pos_emb_config=[
                ("lang", -self.config["dataset"]["tokenizer_max_length"]),
            ],
            max_img_len=self.config["common"]["img_history_size"] * self.config["common"]["num_cameras"] * self.vision_encoder.num_patches,
            max_lang_len=self.config["dataset"]["tokenizer_max_length"],
            dtype=self.dtype,
        )

        self.policy.to(self.device, dtype=self.dtype).eval()
        
        # Initialize image cache
        self.img_cache = []
        self.max_img_cache_size = self.config['common']['img_history_size']

        # Load embeddings based on training mode
        self.task_name = task_name
        self.lang_tokens = None
        self.lang_attn_mask = None
        
        # Load pre-encoded language embeddings
        lang_embed_path = os.path.join(current_folder_path, 'utils', 'lang_embeddings', f'{task_name}.pt')
        if os.path.exists(lang_embed_path):
            # Load embedding data (it's a dictionary with 'embeddings' key)
            embedding_data = torch.load(lang_embed_path, map_location=self.device)
                
            # Extract embeddings tensor from dictionary
            embeddings = embedding_data.get('embeddings', None)
            if embeddings is None:
                print(f"Warning: No 'embeddings' key found in {lang_embed_path}")
                self.lang_tokens = None
            else:
                # Remove batch dimension if present (convert from 3D to 2D)
                if embeddings.dim() == 3:
                    embeddings = embeddings.squeeze(0)
                    
                self.lang_tokens = embeddings.to(dtype=self.dtype)
                print(f"Loaded language embeddings from: {lang_embed_path}")
                # Create attention mask (all tokens are valid)
                self.lang_attn_mask = torch.ones(self.lang_tokens.shape[:1], dtype=torch.bool, device=self.device)
        else:
            print(f"Warning: Language embedding not found for task {task_name}")

    def update_obs(self, observation):
        self.img_cache.append(observation)
        if len(self.img_cache) > self.max_img_cache_size:
            self.img_cache.pop(0)

    @torch.no_grad()
    def get_action(self):
        if len(self.img_cache) == 0:
            return []
        
        current_obs = self.img_cache[-1]
        
        # Process state tokens
        state_tokens = None
        if 'agent_pos' in current_obs:
            normalized_pos = current_obs['agent_pos']
            state_tokens = torch.tensor(
                normalized_pos, 
            ).unsqueeze(0).unsqueeze(0).to(self.device, dtype=self.dtype)

        # Process image tokens from separate camera views
        image_tokens = None
        if all(key in current_obs for key in ['head_cam', 'left_cam', 'right_cam']):
            camera_images = [
                current_obs['head_cam'],
                current_obs['right_cam'],
                current_obs['left_cam'],
            ]
            
            # Process images with dino-siglip encoding
            img_transform_list = []
            for img_array in camera_images:
                img = Image.fromarray(img_array)
                transformed = self.image_transform(img)
                img_transform_list.append(transformed)
            
            # Organize image data structure - process according to training code format
            # From training code, image data should be a dictionary containing dino and siglip keys
            image_inputs = {}
            pv_example = img_transform_list[0]
            for k in pv_example.keys():
                # Ensure conversion to same data type as model (bfloat16)
                image_inputs[k] = torch.stack([img[k] for img in img_transform_list], dim=0).unsqueeze(0).to(self.device, dtype=self.dtype)
            
            # Use vision encoder
            with torch.no_grad():
                # Process batch_size, sequence_length, channels, height, width
                k = next(iter(image_inputs))
                batch_size, seq_len, C, H, W = image_inputs[k].shape
                # Reshape to (batch_size * seq_len, C, H, W) to fit encoder input
                for k in image_inputs:
                    image_inputs[k] = image_inputs[k].view(-1, C, H, W)
                # Use vision encoder to get features
                image_features = self.vision_encoder(image_inputs)
                # Reshape features to correct dimensions
                image_tokens = image_features.view((batch_size, -1, self.vision_encoder.embed_dim))

        # Prepare inputs based on training mode
        lang_tokens = None
        lang_attn_mask = None
        
        lang_tokens = self.lang_tokens.unsqueeze(0)  # Add batch dimension
        lang_attn_mask = self.lang_attn_mask.unsqueeze(0) if self.lang_attn_mask is not None else None
        
        # Predict action
        action_pred = self.policy.predict_action(
            state_tokens=state_tokens,
            image_tokens=image_tokens,
            lang_tokens=lang_tokens,
            lang_attn_mask=lang_attn_mask,
        )
        
        normalized_actions = action_pred.float().cpu().numpy()[0]
        joint_actions = normalized_actions
        
        return joint_actions


def encode_obs(observation):
    obs = {}

    # Process separate camera views
    obs['head_cam'] = observation['observation']['head_camera']['rgb']
    obs['left_cam'] = observation['observation']['left_camera']['rgb']
    obs['right_cam'] = observation['observation']['right_camera']['rgb']
    obs['agent_pos'] = observation['joint_action']['vector']

    return obs


def get_model(args):
    ckpt_folder = args.get('ckpt_setting')
    task_name = args.get('task_name')

    if ckpt_folder.startswith('./'):
        ckpt_folder = os.path.join(current_folder_path, ckpt_folder[2:])
    elif not os.path.isabs(ckpt_folder):
        ckpt_folder = os.path.join(current_folder_path, ckpt_folder)

    print('ckpt_folder: ', ckpt_folder)
    model = MyModel(ckpt_folder, task_name)
    return model


def eval(TASK_ENV, model, observation):
    torch.cuda.empty_cache()

    obs = encode_obs(observation)
    if len(model.obs_cache) == 0:
        model.update_obs(obs)
    
    actions = model.get_action()
    
    for i in range(len(actions)):
        TASK_ENV.take_action(actions[i])
        observation = TASK_ENV.get_obs()
        obs = encode_obs(observation)
        model.update_obs(obs)


def reset_model(model):
    model.obs_cache = []