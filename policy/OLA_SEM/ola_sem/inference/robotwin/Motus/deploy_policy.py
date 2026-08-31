# Motus Policy for RoboTwin

import torch
import torch.nn as nn
import numpy as np
import cv2
from pathlib import Path
import sys
import os
import json
import logging
from typing import List, Dict, Any, Optional
from collections import deque
import yaml
from PIL import Image
from transformers import AutoProcessor

# Add model paths
POLICY_ROOT = str(Path(__file__).parent)
MODELS_ROOT = str(Path(__file__).parent / "models")
for model_path in (MODELS_ROOT, POLICY_ROOT):
    if model_path in sys.path:
        sys.path.remove(model_path)
    sys.path.insert(0, model_path)

from models.motus import Motus, MotusConfig

# Add bak path for T5EncoderModel
BAK_ROOT = str((Path(__file__).parent / "bak").resolve())
if BAK_ROOT not in sys.path:
    sys.path.insert(0, BAK_ROOT)

from wan.modules.t5 import T5EncoderModel
from utils.image_utils import resize_with_padding

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class MotusPolicy:
    """
    Motus Policy wrapper for RoboTwin evaluation.
    Implements the joint video-action diffusion model for robotic control.
    """
    
    def __init__(
        self,
        checkpoint_path: str,
        config_path: str,
        wan_path: str,
        vlm_path: str,
        inference_mode: str = "legacy",
        num_inference_timesteps: Optional[int] = None,
        history_action_noise_std: Optional[float] = None,
        future_video_denoise_fraction: Optional[float] = None,
        device: str = "cuda",
    ):
        self.device = device
        self.checkpoint_path = checkpoint_path
        self.wan_path = wan_path
        self.vlm_path = vlm_path
        self.inference_mode = inference_mode
        self.num_inference_timesteps_override = num_inference_timesteps
        self.history_action_noise_std_override = history_action_noise_std
        self.future_video_denoise_fraction = (
            1.0
            if future_video_denoise_fraction is None
            else float(future_video_denoise_fraction)
        )
        if not 0.0 <= self.future_video_denoise_fraction <= 1.0:
            raise ValueError(
                "future_video_denoise_fraction must be in [0, 1], got "
                f"{self.future_video_denoise_fraction}"
            )
        if self.inference_mode not in {"legacy", "history_flow"}:
            raise ValueError(
                "inference_mode must be 'legacy' or 'history_flow', "
                f"got {self.inference_mode!r}"
            )
        
        # Load configuration
        with open(config_path, 'r') as f:
            self.config_dict = yaml.safe_load(f)
        
        # Initialize model WITHOUT loading pretrained backbones
        self.model = self._load_model()

        # Initialize T5 encoder for language embeddings (WAN text encoder)
        self.t5_encoder = T5EncoderModel(
            text_len=512,
            dtype=torch.bfloat16,
            device=device,
            checkpoint_path=os.path.join(self.wan_path, 'models_t5_umt5-xxl-enc-bf16.pth'),
            tokenizer_path=os.path.join(self.wan_path, 'google', 'umt5-xxl'),
        )

        # Initialize VLM processor from vlm_path (for tokenization only, weights from checkpoint)
        self.vlm_processor = AutoProcessor.from_pretrained(self.vlm_path, trust_remote_code=True)
        
        # Initialize observation cache
        self.obs_cache = deque(maxlen=1)
        self.action_cache = deque()
        
        # Model state
        self.current_state = None
        self.current_state_norm = None
        self.is_first_step = True
        self.prev_action = None
        self.real_qpos_history = deque(maxlen=self.model.config.action_chunk_size)

        # Load normalization stats
        self._load_normalization_stats()
        
        if "lap" in str(self.checkpoint_path).lower():
            self.use_language_action = True
        else:
            self.use_language_action = False
        logger.info(f"Use language action: {self.use_language_action}")
        logger.info(f"Inference mode: {self.inference_mode}")
        logger.info(
            "Future video denoise fraction: %s",
            self.future_video_denoise_fraction,
        )
        logger.info("Motus Policy initialized successfully")

    def set_instruction(self, instruction: str):
        """Set the current instruction for the policy."""
        self.current_instruction = instruction
        logger.info(f"Instruction set: {instruction}")

    def _load_model(self) -> Motus:
        """Load the Motus model without pretrained backbones, then load checkpoint."""
        logger.info(f"Initializing Motus model from config (no pretrained backbones)")

        config = self._create_model_config()
        
        # Initialize model from config WITHOUT loading pretrained weights
        model = Motus(config)
        model = model.to(self.device)
        
        # Load checkpoint weights
        try:
            logger.info(f"Loading checkpoint from {self.checkpoint_path}")
            model.load_checkpoint(self.checkpoint_path, strict=False)
            logger.info("Model checkpoint loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
            raise
        
        model.eval()
        return model
    
    def _create_model_config(self) -> MotusConfig:
        """Create model configuration from yaml config - inference mode."""
        common = self.config_dict['common']
        model_cfg = self.config_dict['model']

        # Use paths passed to constructor
        vae_path = os.path.join(self.wan_path, "Wan2.2_VAE.pth")
        vlm_checkpoint_path = self.vlm_path

        hidden_size = model_cfg['action_expert']['hidden_size']
        ffn_multiplier = model_cfg['action_expert']['ffn_dim_multiplier']
        flow_source_cfg = self._load_history_flow_config(common)

        config = MotusConfig(
            # Paths for config loading only (no weights loaded)
            wan_checkpoint_path=self.wan_path,
            vae_path=vae_path,
            wan_config_path=self.wan_path,
            video_precision='bfloat16',
            vlm_checkpoint_path=vlm_checkpoint_path,
            
            # Understanding expert config
            und_expert_hidden_size=512,
            und_expert_ffn_dim_multiplier=4,
            und_expert_norm_eps=1e-5,
            und_layers_to_extract=None,
            vlm_adapter_input_dim=2048,
            vlm_adapter_projector_type="mlp3x_silu",
            
            # Model architecture
            num_layers=30,
            action_state_dim=common['state_dim'],
            action_dim=common['action_dim'],
            action_expert_dim=hidden_size,
            action_expert_ffn_dim_multiplier=ffn_multiplier,
            action_expert_norm_eps=1e-6,
            
            # Training config
            global_downsample_rate=common['global_downsample_rate'],
            video_action_freq_ratio=common['video_action_freq_ratio'],
            num_video_frames=common['num_video_frames'],
            video_loss_weight=1.0,
            action_loss_weight=1.0,
            flow_source_mode=flow_source_cfg['mode'],
            flow_source_video_mode=flow_source_cfg['video_mode'],
            flow_source_action_noise_std=flow_source_cfg['action_noise_std'],
            
            # Inference config
            batch_size=1,
            video_height=common['video_height'],
            video_width=common['video_width'],
            
            # Don't load pretrained backbones - will load full model from checkpoint
            load_pretrained_backbones=False,
            training_mode='finetune',
        )

        return config

    def _load_history_flow_config(self, common: Dict[str, Any]) -> Dict[str, Any]:
        """Load and validate history-flow metadata only when explicitly enabled."""
        if self.inference_mode == "legacy":
            return {
                "mode": "gaussian",
                "video_mode": "gaussian",
                "action_noise_std": 0.0,
            }

        checkpoint_path = Path(self.checkpoint_path)
        config_path = checkpoint_path.parent / "config.json"
        if not config_path.is_file():
            raise FileNotFoundError(
                "history_flow requires checkpoint metadata at "
                f"{config_path}"
            )

        with open(config_path, "r", encoding="utf-8") as f:
            checkpoint_config = json.load(f)

        flow_source = checkpoint_config.get("flow_source")
        if not isinstance(flow_source, dict):
            raise ValueError(
                f"history_flow requires flow_source metadata in {config_path}"
            )
        if flow_source.get("mode") != "history":
            raise ValueError(
                "history_flow requires checkpoint flow_source.mode='history', "
                f"got {flow_source.get('mode')!r}"
            )
        video_mode = flow_source.get("video_mode", flow_source["mode"])
        if video_mode not in {"gaussian", "history"}:
            raise ValueError(
                "history_flow checkpoint flow_source.video_mode must be "
                f"'gaussian' or 'history', got {video_mode!r}"
            )

        action_chunk_size = (
            int(common['num_video_frames'])
            * int(common['video_action_freq_ratio'])
        )
        history_length = flow_source.get("history_length")
        if history_length is None or int(history_length) != action_chunk_size:
            raise ValueError(
                "history_flow checkpoint history_length must match action chunk "
                f"size {action_chunk_size}, got {history_length!r}"
            )

        action_noise_std = float(flow_source.get("action_noise_std", 0.0))
        if self.history_action_noise_std_override is not None:
            action_noise_std = float(self.history_action_noise_std_override)
        if action_noise_std < 0:
            raise ValueError("flow_source.action_noise_std must be non-negative")

        logger.info(
            "Loaded history-flow config from %s: history_length=%d, "
            "video_mode=%s, action_noise_std=%s",
            config_path,
            history_length,
            video_mode,
            action_noise_std,
        )
        return {
            "mode": "history",
            "video_mode": video_mode,
            "action_noise_std": action_noise_std,
        }
    
    def update_obs(self, observation: Dict[str, Any]):
        """Update observation cache with new observation."""
        # Extract visual observations
        if 'observation' in observation:
            obs_data = observation['observation']
            if 'head_camera' in obs_data and 'left_camera' in obs_data and 'right_camera' in obs_data:
                head_img = obs_data['head_camera']['rgb']
                left_img = obs_data['left_camera']['rgb']
                right_img = obs_data['right_camera']['rgb']
                
                left_img_resized = cv2.resize(left_img, (160, 120))
                right_img_resized = cv2.resize(right_img, (160, 120))
                bottom_row = np.concatenate([left_img_resized, right_img_resized], axis=1)
                image = np.concatenate([head_img, bottom_row], axis=0)
            else:
                raise ValueError("Missing camera data")
        elif 'head_camera' in observation:
            image = observation['head_camera']
        elif 'image' in observation:
            image = observation['image']
        else:
            raise ValueError("No visual observation found")

        target_size = (self.config_dict['common']['video_height'],
                      self.config_dict['common']['video_width'])

        if isinstance(image, np.ndarray):
            image_tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)
        else:
            image_tensor = image

        if image_tensor.shape[-2:] != target_size:
            image_np = image_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
            resized_np = resize_with_padding(image_np, target_size)
            if resized_np.dtype == np.uint8:
                resized_np = resized_np.astype(np.float32) / 255.0
            image_tensor = torch.from_numpy(resized_np).permute(2, 0, 1).unsqueeze(0)
        
        self.obs_cache.append(image_tensor.to(self.device))

        self.current_state = self._extract_qpos(observation)
        self.current_state_norm = self._normalize_actions(self.current_state).to(self.device)

        if self.inference_mode == "history_flow" and not self.real_qpos_history:
            self.real_qpos_history.append(
                self.current_state.squeeze(0).detach().clone()
            )

    def _extract_qpos(self, observation: Dict[str, Any]) -> torch.Tensor:
        """Extract the current simulator qpos as a batched tensor."""
        state = observation['joint_action']['vector']
        if isinstance(state, np.ndarray):
            state_tensor = torch.from_numpy(state).float().unsqueeze(0)
        else:
            state_tensor = state.float().unsqueeze(0) if state.dim() == 1 else state.float()

        state_tensor = state_tensor.to(self.device)
        expected_dim = self.model.config.action_dim
        if state_tensor.shape != (1, expected_dim):
            raise ValueError(
                f"Expected qpos shape (1, {expected_dim}), got "
                f"{tuple(state_tensor.shape)}"
            )
        return state_tensor

    def record_executed_qpos(self, observation: Dict[str, Any]):
        """Record simulator feedback after executing one qpos command."""
        if self.inference_mode != "history_flow":
            return
        qpos = self._extract_qpos(observation)
        self.real_qpos_history.append(qpos.squeeze(0).detach().clone())

    def _build_action_source(self) -> Optional[torch.Tensor]:
        """Build the next source chunk from executed simulator qpos feedback."""
        if self.inference_mode == "legacy":
            return None

        history = list(self.real_qpos_history)
        if not history:
            history = [self.current_state.squeeze(0)]

        chunk_size = self.model.config.action_chunk_size
        if len(history) < chunk_size:
            history = [history[0]] * (chunk_size - len(history)) + history

        return torch.stack(history[-chunk_size:], dim=0).unsqueeze(0)
    
    def get_action(self, instruction: str = None) -> List[np.ndarray]:
        """Get action predictions from the model."""
        if len(self.obs_cache) == 0:
            raise ValueError("No observations in cache. Call update_obs first.")
        
        if self.current_state is None:
            raise ValueError("No robot state available. Call update_obs first.")
        
        current_frame = self.obs_cache[-1]

        # Encode instruction with T5
        scene_prefix = ("The whole scene is in a realistic, industrial art style with three views: "
                        "a fixed rear camera, a movable left arm camera, and a movable right arm camera. "
                        "The aloha robot is currently performing the following task: ")
        instruction = f"{scene_prefix}{self.current_instruction}"
        t5_out = self.t5_encoder([instruction], self.device)
        if isinstance(t5_out, torch.Tensor):
            t5_list = [t5_out.squeeze(0)] if t5_out.dim() == 3 else [t5_out]
        elif isinstance(t5_out, list):
            t5_list = t5_out
        else:
            raise ValueError("Unexpected T5 encoder output format")

        # Build VLM inputs
        first_frame_pil = self._tensor_to_pil_image(current_frame.squeeze(0).cpu())
        vlm_inputs = self._preprocess_vlm_messages(instruction, first_frame_pil)

        # Run inference
        num_inference_steps = self.num_inference_timesteps_override
        if num_inference_steps is None:
            num_inference_steps = self.config_dict['model']['inference']['num_inference_timesteps']
        num_inference_steps = int(num_inference_steps)
        if num_inference_steps <= 0:
            raise ValueError(
                f"num_inference_timesteps must be positive, got {num_inference_steps}"
            )
        action_source = self._build_action_source()

        with torch.no_grad():
            _predicted_frames, predicted_actions = self.model.inference_step(
                first_frame=current_frame,
                state=self.current_state,
                action_source=action_source,
                num_inference_steps=num_inference_steps,
                future_video_denoise_fraction=self.future_video_denoise_fraction,
                language_embeddings=t5_list,
                vlm_inputs=[vlm_inputs],
            )

        actions_real = predicted_actions.squeeze(0).cpu().numpy()
        self.prev_action = actions_real[-1].copy()
        self.action_cache.extend(actions_real)

        return actions_real

    def _tensor_to_pil_image(self, tensor_chw: torch.Tensor) -> Image.Image:
        """Convert [C, H, W] tensor to PIL Image."""
        if tensor_chw.dtype != torch.float32:
            tensor_chw = tensor_chw.float()
        tensor_chw = tensor_chw.clamp(0, 1)
        np_img = (tensor_chw.permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)
        return Image.fromarray(np_img, mode='RGB')

    def _preprocess_vlm_messages(self, instruction: str, image: Image.Image) -> Dict[str, torch.Tensor]:
        """Build VLM inputs."""
        if self.use_language_action:
            instruction = f"任务：{instruction}\n请给出下一步动作语言描述。"
        messages = [
            {
                'role': 'user',
                'content': [
                    {'type': 'text', 'text': instruction},
                    {'type': 'image', 'image': image},
                ]
            }
        ]
        # text = self.vlm_processor.apply_chat_template(messages, add_generation_prompt=False, tokenize=False)
        text = self.vlm_processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        encoded = self.vlm_processor(text=[text], images=[image], return_tensors='pt')
        vlm_inputs = {
            'input_ids': encoded['input_ids'].to(self.device),
            'attention_mask': encoded['attention_mask'].to(self.device), 
            'pixel_values': encoded['pixel_values'].to(self.device),
            'image_grid_thw': encoded.get('image_grid_thw', None)
        }
        if vlm_inputs['image_grid_thw'] is not None:
            vlm_inputs['image_grid_thw'] = vlm_inputs['image_grid_thw'].to(self.device)
        return vlm_inputs

    def _load_normalization_stats(self):
        """Load action normalization stats."""
        try:
            stat_path = Path(__file__).parent / 'utils' / 'stat.json'
            with open(stat_path, 'r') as f:
                stat_data = yaml.safe_load(f) if stat_path.suffix in ['.yml', '.yaml'] else None
        except Exception:
            stat_data = None
        if stat_data is None:
            import json as _json
            with open(Path(__file__).parent / 'utils' / 'stat.json', 'r') as f:
                stat_data = _json.load(f)

        stats = stat_data.get('robotwin2')
        if stats is None:
            raise ValueError('Normalization stats not found')
        self.action_min = torch.tensor(stats['min'], dtype=torch.float32, device=self.device)
        self.action_max = torch.tensor(stats['max'], dtype=torch.float32, device=self.device)
        self.action_range = self.action_max - self.action_min

    def _normalize_actions(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize to [0,1]."""
        shape = x.shape
        x_flat = x.reshape(-1, shape[-1])
        norm = (x_flat - self.action_min.unsqueeze(0)) / self.action_range.unsqueeze(0)
        return norm.reshape(shape)

    def _denormalize_actions(self, y: torch.Tensor) -> torch.Tensor:
        """Denormalize from [0,1]."""
        shape = y.shape
        y_flat = y.reshape(-1, shape[-1])
        denorm = y_flat * self.action_range.unsqueeze(0) + self.action_min.unsqueeze(0)
        return denorm.reshape(shape)
    
def encode_obs(observation):
    """Post-Process Observation"""
    return observation


def get_model(usr_args):
    """
    Initialize Motus model.
    
    Args:
        usr_args: Arguments from eval script (must include wan_path and vlm_path)
    """
    checkpoint_path = usr_args.get('ckpt_setting')
    wan_path = usr_args.get('wan_path')  # Passed from eval.sh or auto_eval.sh
    vlm_path = usr_args.get('vlm_path')  # Passed from eval.sh or auto_eval.sh
    inference_mode = usr_args.get('inference_mode', 'legacy')
    num_inference_timesteps = usr_args.get('num_inference_timesteps')
    history_action_noise_std = usr_args.get('history_action_noise_std')
    future_video_denoise_fraction = usr_args.get(
        'future_video_denoise_fraction'
    )
    if not wan_path:
        raise ValueError("wan_path not provided in usr_args")
    
    if not vlm_path:
        raise ValueError("vlm_path not provided in usr_args")
    
    policy_dir = Path(__file__).parent
    config_path = policy_dir / "utils" / "robotwin.yml"
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    policy = MotusPolicy(
        checkpoint_path=checkpoint_path,
        wan_path=wan_path,
        vlm_path=vlm_path,
        inference_mode=inference_mode,
        num_inference_timesteps=num_inference_timesteps,
        history_action_noise_std=history_action_noise_std,
        future_video_denoise_fraction=future_video_denoise_fraction,
        config_path=str(config_path),
        device=device,
    )
    
    return policy


def eval(TASK_ENV, model, observation):
    """Evaluation function."""
    obs = encode_obs(observation)
    
    instruction = TASK_ENV.get_instruction()
    model.set_instruction(instruction)
    model.update_obs(obs)

    actions = model.get_action()
    
    for action in actions:
        if TASK_ENV.eval_success or TASK_ENV.take_action_cnt >= TASK_ENV.step_lim:
            break
        TASK_ENV.take_action(action, action_type='qpos')
        if model.inference_mode == "history_flow":
            feedback = TASK_ENV.get_obs()
            model.record_executed_qpos(feedback)


def reset_model(model):  
    """Reset model cache at episode start."""
    model.obs_cache.clear()
    model.action_cache.clear()
    model.current_state = None
    model.is_first_step = True
    model.prev_action = None
    model.real_qpos_history.clear()
    logger.info("Model reset completed")
