# Copyright 2025 MemoryMatters Team. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License");
# Implemented by [Yuran Wang / Peking University] in [2025]. 
# Design and Merged by [Yuran Wang / Peking University] in [2025].
"""
Qwen3-VL Encapsulation,
A lightweight encapsulation of Qwen3-VL.
"""
import os
import sys
from typing import List, Optional, Tuple, Dict
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
from termcolor import cprint

from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

class Qwen3VL_Encapsulation(nn.Module):
    def __init__(self, model_path: str, device: Optional[torch.device] = None, **kwargs):
        """
        Initialize the Qwen3-VL wrapper.
        """
        super().__init__()

        # Force single-device placement to stay compatible with DDP; avoid device_map="auto" sharding.
        target_device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        device_map = {"": target_device} if target_device.type == "cuda" else None

        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_path,
            dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            device_map=device_map,
        )
        self.processor = AutoProcessor.from_pretrained(model_path)
        self.processor.tokenizer.padding_side = "right"
        
        self.hidden_size = self.model.config.vision_config.out_hidden_size
        
        # Get special token IDs for feature extraction
        self.vision_start_token_id = self.processor.vision_start_token_id
        self.vision_end_token_id = self.processor.vision_end_token_id
        self.im_end_token_id = self.processor.tokenizer.convert_tokens_to_ids("<|im_end|>")
        self.image_token_id = self.processor.image_token_id

        
    def generate(self, **kwargs):
        """
        Generation interface (auto-regressive decoding) for Qwen-VL.
        Recommended input format:
        * output_hidden_states=True,   
        * return_dict_in_generate=True,  
        """
        with torch.autocast("cuda", dtype=torch.float16):
            generation_output = self.model.generate(
                **kwargs,
            )
        return generation_output
    
    def forward(
        self,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        """
        Forward interface for Qwen-VL.
        Recommended input format:
        * output_attentions=False, 
        * output_hidden_states=True, 
        * return_dict=True,
        """
        with torch.autocast("cuda", dtype=torch.bfloat16):
            outputs = self.model(
                **kwargs,
            )
        return outputs
    
    def build_qwenvl_inputs(
        self, 
        images, 
        instructions, 
        system_prompt=None, 
        add_summary_token=False, 
        add_generation_prompt=False, 
        max_length:int=None, 
        **kwargs
    ):
        """
        Build model inputs from raw data (images + instructions).
        Follow Official Qwen3-VL Instruct format: https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct
        """
        # Create user messages and system messages ( if system_prompt is not None ) 
        messages = []
        assert len(images) == len(instructions), "Images and instructions must have the same length"
        for imgs, instruction in zip(images, instructions):
            # add image
            content = [{"type": "image", "image": img} for img in imgs]
            # add instruction
            if add_summary_token:
                self.summary_token = "🔍"
                content.append({"type": "text", "text": instruction+self.summary_token})
            else:
                content.append({"type": "text", "text": instruction})
            # add system prompt
            if system_prompt is not None:
                msg = [{"role": "system", "content": [{"type": "text", "text": system_prompt}]}, {"role": "user", "content": content}]
            else:
                msg = [{"role": "user", "content": content}]
            
            messages.append(msg)
            
        # Preparation for inference
        batch_inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            add_generation_prompt=add_generation_prompt,
            padding='max_length' if max_length is not None else True,
            max_length=max_length if max_length is not None else 128,
        )
        
        return batch_inputs.to(self.model.device)
    
    def extract_features(
        self,
        input_ids: torch.Tensor,
        last_hidden_state: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Extract image and text features from last_hidden_state based on special tokens.
        
        Args:
            input_ids: Token IDs tensor of shape (batch_size, sequence_length)
            last_hidden_state: Hidden states tensor of shape (batch_size, sequence_length, hidden_size)
        
        Returns:
            Dictionary containing:
                - image_features: Features between <|vision_start|> and <|vision_end|>
                  Shape: (batch_size, num_image_tokens, hidden_size)
                - text_features: Features between <|vision_end|> and <|im_end|>
                  Shape: (batch_size, num_text_tokens, hidden_size)
        """
        batch_size, seq_len, hidden_size = last_hidden_state.shape
        device = last_hidden_state.device
        
        image_features_list = []
        text_features_list = []
        
        for batch_idx in range(batch_size):
            batch_input_ids = input_ids[batch_idx]  # (seq_len,)
            batch_hidden = last_hidden_state[batch_idx]  # (seq_len, hidden_size)
            
            # Find positions of special tokens
            vision_start_positions = (batch_input_ids == self.vision_start_token_id).nonzero(as_tuple=True)[0]
            vision_end_positions = (batch_input_ids == self.vision_end_token_id).nonzero(as_tuple=True)[0]
            im_end_positions = (batch_input_ids == self.im_end_token_id).nonzero(as_tuple=True)[0]
            
            # Extract image features: between <|vision_start|> and <|vision_end|>
            if len(vision_start_positions) > 0 and len(vision_end_positions) > 0:
                # Take the first occurrence of each token
                vision_start_idx = vision_start_positions[0].item()
                vision_end_idx = vision_end_positions[0].item()
                
                # Extract features between vision_start and vision_end (exclusive of the tokens themselves)
                # Include image tokens but exclude the boundary tokens
                image_start = vision_start_idx + 1  # Skip <|vision_start|>
                image_end = vision_end_idx  # Up to but not including <|vision_end|>
                
                if image_end > image_start:
                    image_features = batch_hidden[image_start:image_end]  # (num_image_tokens, hidden_size)
                else:
                    image_features = torch.empty((0, hidden_size), device=device, dtype=batch_hidden.dtype)
            else:
                image_features = torch.empty((0, hidden_size), device=device, dtype=batch_hidden.dtype)
            
            # Extract text features: between <|vision_end|> and <|im_end|>
            if len(vision_end_positions) > 0 and len(im_end_positions) > 0:
                vision_end_idx = vision_end_positions[0].item()
                im_end_idx = im_end_positions[0].item()
                
                # Extract features between vision_end and im_end (exclusive of the tokens themselves)
                text_start = vision_end_idx + 1  # Skip <|vision_end|>
                text_end = im_end_idx  # Up to but not including <|im_end|>
                
                if text_end > text_start:
                    text_features = batch_hidden[text_start:text_end]  # (num_text_tokens, hidden_size)
                else:
                    text_features = torch.empty((0, hidden_size), device=device, dtype=batch_hidden.dtype)
            else:
                text_features = torch.empty((0, hidden_size), device=device, dtype=batch_hidden.dtype)
            
            # Mean pool each sample's features to avoid padding issues
            # Since downstream code uses .mean(dim=1) anyway, we can do it here directly
            if image_features.shape[0] > 0:
                image_features = image_features.mean(dim=0, keepdim=True)  # (1, hidden_size)
            else:
                image_features = torch.zeros((1, hidden_size), device=device, dtype=batch_hidden.dtype)
            
            if text_features.shape[0] > 0:
                text_features = text_features.mean(dim=0, keepdim=True)  # (1, hidden_size)
            else:
                text_features = torch.zeros((1, hidden_size), device=device, dtype=batch_hidden.dtype)
            
            image_features_list.append(image_features)
            text_features_list.append(text_features)
        
        # Stack mean-pooled features (all have shape (1, hidden_size))
        image_features = torch.cat(image_features_list, dim=0)  # (batch_size, hidden_size)
        text_features = torch.cat(text_features_list, dim=0)   # (batch_size, hidden_size)
        
        # Add sequence dimension to match expected shape: (batch_size, num_tokens, hidden_size)
        # Since we've already mean-pooled, num_tokens=1
        image_features = image_features.unsqueeze(1)  # (batch_size, 1, hidden_size)
        text_features = text_features.unsqueeze(1)     # (batch_size, 1, hidden_size)
        
        return image_features, text_features
    
if __name__ == "__main__":
    # * model load test *
    model_path = "checkpoints/Qwen3-VL-2B-Instruct"
    model = Qwen3VL_Encapsulation(model_path)
    print("model load success", "\n")
    
    # * input test *
    image1 = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
    image2 = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
    # images = [[image1, image2], [image1]]
    images = [[image1], [image2]]
    instructions = ["Left: Pick the middle block and push it to the left tray.", "Right: Pick the middle block and push it"]
    # system_prompt = "You are a helpful assistant specialized in image analysis."
    inputs = model.build_qwenvl_inputs(images, instructions, system_prompt=None, add_summary_token=False, max_length=128)
    print("--------inputs.input_ids.shape:\n", inputs.input_ids.shape, "\n")
    text_sample1 = model.processor.tokenizer.decode(inputs.input_ids[0])
    # text_sample2 = model.processor.tokenizer.decode(inputs.input_ids[1])
    print("--------text_sample1:\n", text_sample1)
    # print("--------text_sample2:\n", text_sample2)
    tokens_sample1 = model.processor.tokenizer.convert_ids_to_tokens(inputs.input_ids[0])
    # tokens_sample2 = model.processor.tokenizer.convert_ids_to_tokens(inputs.input_ids[1])
    print("--------tokens_sample1:\n", tokens_sample1, "\n")
    # print("--------tokens_sample2:\n", tokens_sample2)
    
    # * forward function test *
    qwenvl_forward_outputs = model.forward(
        **inputs,
        output_attentions=False,
        output_hidden_states=True,
        return_dict=True,
    )
    last_hidden_state = qwenvl_forward_outputs.hidden_states[-1]    # (batch_size, sequence_length, hidden_size). final layer hidden state.
    print("--------last_hidden_state.shape:\n", last_hidden_state.shape, "\n")
    
    # Extract image and text features
    image_features, text_features = model.extract_features(
        inputs.input_ids,
        last_hidden_state,
    )
    print("--------image_features.shape:\n", image_features.shape, "\n")
    print("--------text_features.shape:\n", text_features.shape, "\n")
