import os
import sys
from typing import Any, Dict, List, Optional, Tuple
import argparse
import pandas as pd

from yaml import load, Loader
import numpy as np
import torch
from einops import rearrange

import cv2

from dynamics_model.utils.model_utils import (
    load_condition_models,
    load_latent_models,
    load_vae_models,
    load_diffusion_model,
    count_model_parameters,
)
from dynamics_model.utils import import_custom_class, save_video


class DynamicsModel:
    """
        DynamicsModel: Given the current frame [t-3,...,t] and its corresponding action, predict frames [t+1,...,t+20].
    """
    
    def __init__(self, config_file: str, device="cuda"):
        """
            Initialize DynamicsModel

            Args:

                config_file: Configuration file path
        """
        self.config_file = config_file
        self.args = self.load_config(config_file)
        self.tokenizer = None
        self.text_encoder = None
        self.vae = None
        self.diffusion_model = None
        self.scheduler = None
        self.pipe = None
        self.valid_cams = ['observation.images.front_color', 'observation.images.left_color', 'observation.images.right_color']
        self.video_fps = 10
        self.device = device
        
        self._load_models()
    
    def load_config(self, config_file: str) -> argparse.Namespace:
        """
            Load configuration file

            Args:

                config_file: Configuration file path

            Returns:

                args: Configuration parameter namespace
        """
        cd = load(open(config_file, "r"), Loader=Loader)
        args = argparse.Namespace(**cd)
        return args
    
    def _load_models(self) -> None:
        """
            Load model components (Tokenizer, TextEncoder, VAE, Diffusion Model, etc.)
        """
        # Loading Tokenizer and TextEncoder
        tokenizer_class = import_custom_class(
            self.args.tokenizer_class, getattr(self.args, "tokenizer_class_path", "transformers")
        )
        textenc_class = import_custom_class(
            self.args.textenc_class, getattr(self.args, "textenc_class_path", "transformers")
        )
        cond_models = load_condition_models(
            tokenizer_class, textenc_class,
            self.args.pretrained_model_name_or_path if not hasattr(self.args, "tokenizer_pretrained_model_name_or_path") else self.args.tokenizer_pretrained_model_name_or_path,
            load_weights=self.args.load_weights
        )
        self.tokenizer, self.text_encoder = cond_models["tokenizer"], cond_models["text_encoder"]
        self.text_encoder = self.text_encoder.to(self.device, dtype=torch.bfloat16).eval()
        
        # Loading VAE
        vae_class = import_custom_class(
            self.args.vae_class, getattr(self.args, "vae_class_path", "transformers")
        )
        if getattr(self.args, 'vae_path', False):
            self.vae = load_vae_models(vae_class, self.args.vae_path).to(self.device, dtype=torch.bfloat16).eval()
        else:
            self.vae = load_latent_models(vae_class, self.args.pretrained_model_name_or_path)["vae"].to(self.device, dtype=torch.bfloat16).eval()
        
        if isinstance(self.vae.latents_mean, List):
            self.vae.latents_mean = torch.FloatTensor(self.vae.latents_mean)
        if isinstance(self.vae.latents_std, List):
            self.vae.latents_std = torch.FloatTensor(self.vae.latents_std)
        
        if self.vae is not None:
            self.vae.enable_slicing()
            self.vae.enable_tiling()
        self.vae = self.vae
        
        # Loading Diffusion Model
        diffusion_model_class = import_custom_class(
            self.args.diffusion_model_class, getattr(self.args, "diffusion_model_class_path", "transformers")
        )
        self.diffusion_model = load_diffusion_model(
            model_cls=diffusion_model_class,
            model_dir=self.args.diffusion_model['model_path'],
            load_weights=self.args.load_weights and getattr(self.args, "load_diffusion_model_weights", True),
            **self.args.diffusion_model['config']
        ).to(self.device, dtype=torch.bfloat16)
        total_params = count_model_parameters(self.diffusion_model)
        print(f'Total parameters for transformer model: {total_params}')
        self.diffusion_model = self.diffusion_model
        
        # Loading Diffuser Scheduler
        diffusion_scheduler_class = import_custom_class(
            self.args.diffusion_scheduler_class, getattr(self.args, "diffusion_scheduler_class_path", "diffusers")
        )
        if hasattr(self.args, "diffusion_scheduler_args"):
            self.scheduler = diffusion_scheduler_class(**self.args.diffusion_scheduler_args)
        else:
            self.scheduler = diffusion_scheduler_class()
        
        # Import and create Pipeline
        pipeline_class = import_custom_class(
            self.args.pipeline_class, getattr(self.args, "pipeline_class_path", "diffusers")
        )
        
        self.pipe = pipeline_class(
            self.scheduler, self.vae, self.text_encoder, self.tokenizer, self.diffusion_model
        )
    
    def load_images(self, image_root: str, size: Tuple[int, int] = (256, 192)) -> torch.Tensor:
        """
            Load images from the specified path

            Args:

                image_root: Image root directory

                size: Image size

            Returns:

                mv_images: Loaded image data, shape=(v,c,t,h,w)
        """
        n_mem = 4
        mv_images = []
        for cam in self.valid_cams:
            images = []
            for i in range(n_mem):
                img_path = os.path.join(image_root, cam, f"{i}.png")
                if not os.path.exists(img_path):
                    raise FileNotFoundError(f"Image file not found: {img_path}")
                
                img = cv2.imread(img_path)[:, :, ::-1]  # BGR to RGB
                img = cv2.resize(img, size)
                img = img.astype(np.float32) / 255.0 * 2.0 - 1.0  # Normalize to [-1, 1]
                img = torch.from_numpy(np.transpose(img, (2, 0, 1)))
                images.append(img)
            
            # Stack to (c, t, h, w)
            images = torch.stack(images, dim=1)
            mv_images.append(images)
        
        # Stack to (v, c, t, h, w)
        mv_images = torch.stack(mv_images, dim=0)
        return mv_images
    
    def infer(
        self,
        obs: Optional[torch.Tensor] = None,
        act_tokens: Optional[torch.Tensor] = None,
        image_root: Optional[str] = None,
        prompt: str = "",
        save_path: Optional[str] = None,
        n_chunk: int = 1,
        normed_state: Optional[np.ndarray] = None,
        num_denois_steps: int = 10,
        seed: int = 42,
        default_fps: int = 30
    ) -> Dict[str, Any]:
        """
            Inference Function

            Args:

                obs: Observation data, shape=(3, 3, 4, 192, 256), used if provided

                act_tokens: Action tokens, shape=(bs, 25, 14), used if provided

                image_root: Image root directory, loaded from here if obs is not provided

                prompt: Text prompt

                save_path: Save path

                n_chunk: Number of predicted chunks

                normed_state: Normalized state

                num_denois_steps: Number of denoising steps

                seed: Random seed

                default_fps: Default frame rate

            Returns:

                preds: Dictionary of inference results
        """
        # Validate input
        if obs is None and image_root is None:
            raise ValueError("Either 'obs' or 'image_root' must be provided")
        
        # If obs is not provided, load from image_root.
        if obs is None:
            obs = self.load_images(image_root, size=(256, 192))
        

        # obs.shape -> (24, 3, 4, 192, 256)
        # act_tokens.shape -> (8, 25, 30)
        bv, c, t, h, w = obs.shape

        if act_tokens is None:
            act_tokens = torch.rand(1, 25, 30, self.device, dtype=torch.bfloat16)
        
        # Verify the shape of act_tokens
        if act_tokens.shape[1:] != (25, 30):
            raise ValueError(f"act_tokens must have shape (bs, 25, 30), got {act_tokens.shape}")
        
        num_inference_steps = num_denois_steps
        
        # inferring
        preds = self.pipe.infer(
            image=obs,
            prompt=[prompt for _ in range(int(bv/3))],
            negative_prompt='',
            num_inference_steps=num_inference_steps,
            decode_timestep=0.03,
            decode_noise_scale=0.025,
            height=h,
            width=w,
            n_view=3,
            guidance_scale=1.0,
            return_action=False,
            n_prev=4,
            chunk=4,
            return_video=True,
            noise_seed=seed,
            action_chunk=25,
            history_action_state=normed_state,
            pixel_wise_timestep=True,
            n_chunk=n_chunk,
            act_tokens=act_tokens,
        )[0]

        # preds[0].shape -> (24 [8x3], 3, 29, 192, 256)
        
        
        # saving predicted results
        if save_path is not None:
            print(f"!!!!! Saving results to {save_path}...")

            os.makedirs(save_path, exist_ok=True)
            if self.args.return_video:
                video = preds['video'].data.cpu()
                video = rearrange(video, '(b v) c t h w -> b c t h (v w)', v=3)
                current_time = pd.Timestamp.now().strftime("%d_%H%M%S")
                # save_path = os.path.join(save_path, f"inference_{current_time}")
                for i in range(video.shape[0]):
                    save_video(
                        video[i],
                        os.path.join(save_path, f"dynamics_model_pred_{current_time}.mp4"),
                        fps=self.video_fps
                    )
        
        return preds
    
    def get_default_act_tokens(self, parquet_path) -> torch.Tensor:
        """
            Get the default action tokens

            Args:

                parquet_path: Parquet file path

            Returns:

                act_tokens: Action tokens, shape=(1, 25, 14)
        """
        act_tokens_index = [181 + i for i in range(25)]
        data = pd.read_parquet(parquet_path)
        action = np.stack([data['action'][i] for i in range(data['action'].shape[0])])
        action_tokens_need = action[act_tokens_index].astype(np.float32)
        action_tokens_need = torch.FloatTensor(action_tokens_need)
        act_tokens = action_tokens_need.unsqueeze(0)
        return act_tokens

