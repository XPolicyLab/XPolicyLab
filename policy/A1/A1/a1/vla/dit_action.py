import re

import numpy as np
from typing import Dict, Any, Optional
import torch
import torch.nn as nn

# from transformers import AutoTokenizer, SiglipTextModel
# from transformers import AutoProcessor, SiglipVisionModel

# from a1.vla.projectors import ProprioProjector

from a1.vla.constants import (NUM_ACTIONS_CHUNK,ACTION_DIM,)
                                # PROPRIO_DIM,ACTION_PROPRIO_NORMALIZATION_TYPE,NormalizationType)
from a1.config import DiTActionConfig

from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.schedulers.scheduling_dpmsolver_multistep import DPMSolverMultistepScheduler

from a1.vla.dit.model import TextImageConditionDiT

class DiffusionTransformerRunner(nn.Module):
    def __init__(
        self,
        # input_dim=4096,
        max_lang_cond_len, 
        lang_cond_dim,
        img_cond_len,
        img_cond_dim,
        hidden_dim=2048,
        depth=14,
        num_heads=16,
        action_dim=7,
        action_horizon = 8,
        use_proprio=False,
        num_diffusion_steps=100,
        num_diffusion_inference_steps=5,
        pred_type="sample", # epsilon
        ):
        super().__init__()
        self.action_dim = action_dim
        self.action_horizon = action_horizon

        self.hidden_dim = hidden_dim
        # self.cond_dim = cond_dim
        self.max_lang_cond_len = max_lang_cond_len
        self.img_cond_len = img_cond_len
        self.lang_cond_dim = lang_cond_dim
        self.img_cond_dim = img_cond_dim

        self.use_proprio = use_proprio

        self.num_diffusion_steps = num_diffusion_steps
        self.num_diffusion_inference_steps = num_diffusion_inference_steps
        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=num_diffusion_steps,
            beta_schedule="squaredcos_cap_v2",
            prediction_type=pred_type,
            clip_sample=False,
        )

        self.noise_scheduler_sample = DPMSolverMultistepScheduler(
            num_train_timesteps=num_diffusion_steps,
            beta_schedule="squaredcos_cap_v2",
            prediction_type=pred_type,
        )

        self.model = TextImageConditionDiT(output_dim=action_dim,horizon=action_horizon,hidden_size=hidden_dim,depth=depth,
                            num_heads=num_heads,max_lang_cond_len=max_lang_cond_len,img_cond_len=img_cond_len,use_proprio=use_proprio,)
        
        self.action_adaptor = self.build_condition_adapter(
            "mlp2x_gelu", # mlp3x_gelu 
            in_features=action_dim ,out_features=hidden_dim)
        
        self.image_adaptor = self.build_condition_adapter("mlp2x_gelu", in_features=img_cond_dim ,out_features=hidden_dim)
        self.lang_adaptor = self.build_condition_adapter("mlp2x_gelu", in_features=lang_cond_dim ,out_features=hidden_dim)


        self._verify_parameters()

    def _verify_parameters(self):
        """验证参数是否正确初始化"""
        for name, param in self.named_parameters():
            if param.numel() == 0:
                raise ValueError(f"Parameter {name} is empty: {param.shape}")
            if torch.isnan(param).any():
                raise ValueError(f"Parameter {name} contains NaN values")
        

    def sample_noisy_actions(self, target_actions):
        batch_size = target_actions.shape[0]
        device = target_actions.device
        # Sample noise that we'll add to the actions
        noise = torch.randn(target_actions.shape, dtype=target_actions.dtype, device=device)
        # Sample random diffusion timesteps
        timesteps = torch.randint(0, self.num_diffusion_steps, (batch_size,), device=device).long()
        # Add noise to the clean actions according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        noisy_action = self.noise_scheduler.add_noise(target_actions, noise, timesteps)
        return noise, noisy_action, timesteps


    def predict_noise_or_sample(self, noise_action, timestep,text_embeds,image_embeds,proprio,text_attn_mask=None):
        # noise_action shape: (B, T, D) where B=batch_size, T=action_horizon, D=action_dim
        # target_dtype = proprio_feat.dtype
        # proprio暂时没用到, cat 到x？

        if self.use_proprio:
            # Concatenate the proprio and action tokens to form the input sequence
            assert proprio.shape[2] == noise_action.shape[2], \
                f"Proprio shape {proprio.shape} does not match action shape {noise_action.shape}"
            # proprio = proprio.unsqueeze(1)  # (B, 1, D)
            proprio_noise_action = torch.cat([proprio, noise_action], dim=1)
            noise_action = proprio_noise_action
        noise_action_adapted = self.action_adaptor(noise_action)  # (B, T, D)
        img_c = self.image_adaptor(image_embeds)  
        lang_c = self.lang_adaptor(text_embeds)
        # model_dtype = next(self.model.parameters()).dtype

        pred = self.model(noise_action_adapted, timestep,lang_c,img_c,lang_mask=text_attn_mask)
        # pred = pred.to(target_dtype)
        return pred
    

    def condition_sampling(self, text_embeds, image_embeds,proprio=None):
        device = text_embeds.device
        dtype = text_embeds.dtype
        
        img_c = self.image_adaptor(image_embeds)  
        lang_c = self.lang_adaptor(text_embeds)
        
        noise_action = torch.randn(
            size=(text_embeds.shape[0], self.action_horizon, self.action_dim), 
            dtype=dtype, device=device)
        # action_mask = action_mask.expand(-1, self.action_horizon, -1)
    
        # Set step values
        self.noise_scheduler_sample.set_timesteps(self.num_diffusion_inference_steps)
        
        for t in self.noise_scheduler_sample.timesteps:
            if self.use_proprio:
                # Concatenate the proprio and action tokens to form the input sequence
                assert proprio.shape[2] == noise_action.shape[2], \
                    f"Proprio shape {proprio.shape} does not match action shape {noise_action.shape}"
                noise_proprio_action = torch.cat([proprio, noise_action], dim=1)
            # noise_proprio_action = noise_proprio_action.to(model_dtype)
            noise_action_adapted = self.action_adaptor(noise_proprio_action)
            
            m_dtype = next(self.model.parameters()).dtype
            # Predict the model output
            model_output = self.model(noise_action_adapted.to(m_dtype), t.unsqueeze(-1).to(device),lang_c.to(m_dtype),img_c.to(m_dtype))
            
            
            # Compute previous actions: x_t -> x_t-1
            noise_action = self.noise_scheduler_sample.step(model_output, t, noise_action).prev_sample
            noise_action = noise_action.to(dtype)
        
        # Finally apply the action mask to mask invalid action dimensions
        # noise_action = noise_action * action_mask
        return noise_action


    # adapter to action input
    def build_condition_adapter(
        self, projector_type, in_features, out_features):
        projector = None
        if projector_type == 'linear':
            projector = nn.Linear(in_features, out_features)
        else:
            mlp_gelu_match = re.match(r'^mlp(\d+)x_gelu$', projector_type)
            if mlp_gelu_match:
                mlp_depth = int(mlp_gelu_match.group(1))
                modules = [nn.Linear(in_features, out_features)]
                for _ in range(1, mlp_depth):
                    modules.append(nn.GELU(approximate="tanh"))
                    modules.append(nn.Linear(out_features, out_features))
                projector = nn.Sequential(*modules)

        if projector is None:
            raise ValueError(f'Unknown projector type: {projector_type}')

        return projector


class DiffusionTransformerAction(nn.Module):
    def __init__(self, config: DiTActionConfig):
        '''
        Example of using text model, toknizer and vision model, processor

        # self.text_model = SiglipTextModel.from_pretrained("google/siglip-so400m-patch14-384") # google/siglip-base-patch16-224
        # self.tokenizer = AutoTokenizer.from_pretrained("google/siglip-so400m-patch14-384")
        # # important: make sure to set padding="max_length" as that's how the model was trained
        # inputs = tokenizer(["a photo of a cat", "a photo of a dog"], padding="max_length", return_tensors="pt")

        # outputs = model(**inputs)
        # last_hidden_state = outputs.last_hidden_state
        # pooled_output = outputs.pooler_output  # pooled (EOS token) states

        # self.vision_model = SiglipVisionModel.from_pretrained("google/siglip-so400m-patch14-384")
        # self.processor = AutoProcessor.from_pretrained("google/siglip-so400m-patch14-384")
        # image = Image.open(requests.get(url, stream=True).raw)

        # inputs = processor(images=image, return_tensors="pt")

        # outputs = model(**inputs)
        # last_hidden_state = outputs.last_hidden_state
        # pooled_output = outputs.pooler_output  # pooled features
        '''
        super(DiffusionTransformerAction, self).__init__()
        self.config = config
        self.use_proprio = config.use_proprio

        # choose small text encoder model [siglip text moddel; T5; Qwen3; Phi-3;]

        self.diff_pred_type = 'sample'  # 'epsilon' or 'sample'
        # num_patches = (self.vision_model.config.image_size // self.vision_model.config.patch_size) ** 2
        num_cameras = 2 if config.use_wrist_image else 1  # primary + wrist

        img_cond_len = (num_cameras * config.num_patches)
        self.dit_model = DiffusionTransformerRunner(action_dim=ACTION_DIM,action_horizon=NUM_ACTIONS_CHUNK,hidden_dim=self.config.dit_hidden_dim,
                                                            depth=self.config.dit_depth,
                                                            num_heads=self.config.dit_num_heads,
                                                            # cond_len=ACTION_DIM*NUM_ACTIONS_CHUNK,cond_dim=self.config.d_model,
                                                            max_lang_cond_len=self.config.sequence_length,
                                                            lang_cond_dim=config.lang_cond_dim,
                                                            img_cond_len=img_cond_len,
                                                            img_cond_dim=config.img_cond_dim,
                                                            use_proprio=config.use_proprio,
                                                            num_diffusion_steps=self.config.num_diffusion_steps,
                                                            num_diffusion_inference_steps=self.config.num_diffusion_inference_steps,
                                                            pred_type=self.diff_pred_type,)


    def forward(self,text_embeds,image_embeds,target_actions=None,proprio=None,text_attn_mask=None):
        # print(f"****** DiffusionTransformerAction forward: proprio shape: {proprio.shape if proprio is not None else None}")
        # img_attn_mask?
        # target_actions (16, 7) (B, NUM_ACTIONS_CHUNK, ACTION_DIM)


        # bs, seq_len = input_ids.shape
        # pv_shape = pixel_values.shape
        # C,H,W = pv_shape[-3],pv_shape[-2],pv_shape[-1]
        # pixel_values = pixel_values.reshape(-1,C,H,W)
        # (bs, seq_len, d_model)
        # text_embeds = self.text_model(input_ids=input_ids,attention_mask=text_attention_mask).last_hidden_state.detach()  
        # image_embeds = self.vision_model(pixel_values=pixel_values).last_hidden_state.detach()
        # image_embeds = image_embeds.reshape(bs,-1,self.config.img_cond_dim,) #self.vision_model.config.hidden_size
        
        if not self.training:
            return text_embeds, image_embeds
        
        if self.training:
            noise, noisy_actions, timesteps = self.dit_model.sample_noisy_actions(target_actions)
            pred = self.dit_model.predict_noise_or_sample(noisy_actions,timesteps,text_embeds,image_embeds, proprio,text_attn_mask)
            
            pred_type = self.diff_pred_type 
            if pred_type == 'epsilon':
                target = noise
            elif pred_type == 'sample':
                target = target_actions

            predicted_actions = None

            return {  
                'predicted_actions': predicted_actions,  
                'diffusion_target': target ,
                'diffusion_pred': pred,
                'noisy_actions': noisy_actions ,
                'diff_timesteps': timesteps ,
            }

    
    def predict_actions(self,text_embeds,image_embeds,proprio=None,text_attn_mask=None,**kwargs):
        
        text_embeds, image_embeds = self.forward(text_embeds, image_embeds,proprio=proprio, text_attn_mask=text_attn_mask,**kwargs)
        
        normalized_actions = self.dit_model.condition_sampling(text_embeds,image_embeds,proprio)
        
        # Unnormalize predicted actions
        # actions = self._unnormalize_actions(normalized_actions, unnorm_key)

        return normalized_actions

    