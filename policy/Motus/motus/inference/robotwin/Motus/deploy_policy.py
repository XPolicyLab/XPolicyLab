# Motus Policy for RoboTwin

import inspect
import torch
import torch.nn as nn
import numpy as np
import cv2
from pathlib import Path
import sys
import os
import logging
import json
import hashlib
from typing import List, Dict, Any, Optional
from collections import deque
import yaml
from PIL import Image
from transformers import AutoProcessor
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend

# Add model paths
sys.path.append(str(Path(__file__).parent))
sys.path.append(str(Path(__file__).parent / "models"))

from models.motus import Motus, MotusConfig

# Add bak path for T5EncoderModel
BAK_ROOT = str((Path(__file__).parent / "bak").resolve())
if BAK_ROOT not in sys.path:
    sys.path.insert(0, BAK_ROOT)

from wan.modules.t5 import T5EncoderModel
from utils.image_utils import resize_with_padding
from utils.vlm_utils import preprocess_vlm_messages

logger = logging.getLogger(__name__)

class MotusPolicy:
    """
    Motus Policy wrapper for RoboTwin evaluation.
    Implements the joint video-action diffusion model for robotic control.
    """
    
    def __init__(self, checkpoint_path: str, config_path: str, wan_path: str, vlm_path: str, device: str = "cuda", log_dir: Optional[str] = None, task_name: Optional[str] = None, embodiment_type: str = "aloha_agilex_2", use_scene_prefix: bool = True, t5_device: Optional[str] = None, t5_cache_dir: Optional[str] = None):
        self.device = device
        self.t5_device = t5_device or device
        self.t5_cache_dir = Path(t5_cache_dir).expanduser().resolve() if t5_cache_dir else None
        self.t5_cache_index = self._load_t5_cache_index(self.t5_cache_dir)
        self.checkpoint_path = checkpoint_path
        self.wan_path = wan_path
        self.vlm_path = vlm_path
        # Normalization embodiment must match the one used at training time.
        # The RoboDojo checkpoint was trained with the LeRobot pipeline, which
        # normalizes both state and actions to [0, 1] using the embodiment stats
        # (default: aloha_agilex_2). It must NOT be left at robotwin2.
        self.embodiment_type = embodiment_type
        # Whether to prepend the scene-description prefix to the instruction before
        # encoding it with T5/VLM. This MUST match the training pipeline:
        #   - robotwin custom pipeline (robotwin_converter + robotwin_agilex_dataset):
        #       instructions are stored WITH the prefix -> use_scene_prefix=True
        #   - LeRobot pipeline (add_t5_cache + lerobot_dataset): T5 cache and VLM text
        #       both use the RAW task string -> use_scene_prefix=False
        self.use_scene_prefix = use_scene_prefix
        
        # Load configuration
        with open(config_path, 'r') as f:
            self.config_dict = yaml.safe_load(f)
        
        # Initialize model WITHOUT loading pretrained backbones
        self.model = self._load_model()

        # Initialize T5 encoder only when no pre-encoded cache is configured.
        # Loading the full T5 encoder at runtime needs ~41GB VRAM; cached embeddings
        # keep deployment within a 24GB GPU.
        self.t5_encoder = None
        if self.t5_cache_index is None:
            self.t5_encoder = T5EncoderModel(
                text_len=512,
                dtype=torch.bfloat16,
                device=self.t5_device,
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

        # Load normalization stats
        self._load_normalization_stats()
        
        # Initialize image saving
        self.save_images = True
        base_log_dir = log_dir or os.environ.get('LOG_DIR') or str(Path(__file__).resolve().parent.parent / "logs")
        task_dir_name = task_name or os.environ.get('TASK_NAME') or "default_task"
        self.save_dir = Path(base_log_dir) / "images" / task_dir_name
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.episode_count = 0
        self.step_count = 0

        logger.info("Motus Policy initialized successfully")

    def set_instruction(self, instruction: str):
        """Set the current instruction for the policy."""
        self.current_instruction = instruction
        logger.info(f"Instruction set: {instruction}")

    def _load_t5_cache_index(self, cache_dir: Optional[Path]) -> Optional[Dict[str, Any]]:
        if cache_dir is None:
            return None

        manifest_path = cache_dir / "manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(f"t5_cache_dir is set but manifest.json was not found: {manifest_path}")

        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)

        entries = manifest.get("entries", {})
        if not isinstance(entries, dict):
            raise ValueError(f"Invalid T5 cache manifest: expected dict at entries in {manifest_path}")

        logger.info(f"Loaded {len(entries)} pre-encoded T5 embeddings from {cache_dir}")
        return entries

    @staticmethod
    def _prompt_cache_key(prompt_text: str) -> str:
        return hashlib.sha256(prompt_text.encode("utf-8")).hexdigest()

    def _get_t5_embeddings(self, prompt_text: str) -> List[torch.Tensor]:
        if self.t5_cache_index is not None:
            cache_key = self._prompt_cache_key(prompt_text)
            entry = self.t5_cache_index.get(cache_key)
            if entry is None:
                preview = prompt_text[:200].replace("\n", "\\n")
                raise KeyError(
                    f"T5 embedding cache miss for prompt sha256={cache_key}. "
                    f"Preview: {preview!r}. Rebuild the cache with this instruction."
                )

            embed_path = self.t5_cache_dir / entry["file"]
            loaded = torch.load(embed_path, map_location="cpu")
            if isinstance(loaded, torch.Tensor):
                return [loaded]
            if isinstance(loaded, list) and all(isinstance(t, torch.Tensor) for t in loaded):
                return loaded
            raise ValueError(f"Unsupported cached T5 embedding format in {embed_path}")

        if self.t5_encoder is None:
            raise RuntimeError("No T5 encoder or T5 cache is available.")

        t5_out = self.t5_encoder([prompt_text], self.t5_device)
        if isinstance(t5_out, torch.Tensor):
            return [t5_out.squeeze(0)] if t5_out.dim() == 3 else [t5_out]
        if isinstance(t5_out, list):
            return t5_out
        raise ValueError("Unexpected T5 encoder output format")

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
            
            # Inference config
            batch_size=1,
            video_height=common['video_height'],
            video_width=common['video_width'],
            
            # Don't load pretrained backbones - will load full model from checkpoint
            load_pretrained_backbones=False,
            training_mode='finetune',
        )

        return config
    
    def update_obs(self, observation: Dict[str, Any]):
        """Update observation cache with new observation."""
        # Extract visual observations
        if 'observation' in observation:
            obs_data = observation['observation']
            if 'head_camera' in obs_data and 'left_camera' in obs_data and 'right_camera' in obs_data:
                head_img = obs_data['head_camera']['rgb']
                left_img = obs_data['left_camera']['rgb']
                right_img = obs_data['right_camera']['rgb']
                image = self._stitch_three_cameras(head_img, left_img, right_img)
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

        # Convert any input to a HWC numpy array first.
        if isinstance(image, np.ndarray):
            image_np = image
        else:
            t = image
            if t.dim() == 4:
                t = t.squeeze(0)
            image_np = t.permute(1, 2, 0).cpu().numpy()

        # Aspect-preserving resize + center pad to the training resolution.
        # resize_with_padding is a no-op resize when already at target size, so it is
        # safe to always call (this also guarantees the [0, 1] conversion below runs).
        resized_np = resize_with_padding(image_np, target_size)

        # ALWAYS normalize to float in [0, 1] to match training (frames are float/255).
        if resized_np.dtype == np.uint8:
            resized_np = resized_np.astype(np.float32) / 255.0
        else:
            resized_np = resized_np.astype(np.float32)
            if float(resized_np.max()) > 1.0:
                resized_np = resized_np / 255.0
        image_tensor = torch.from_numpy(resized_np).permute(2, 0, 1).unsqueeze(0)

        self.obs_cache.append(image_tensor.to(self.device))

        # Extract robot state
        state = observation['joint_action']['vector']

        if isinstance(state, np.ndarray):
            state_tensor = torch.from_numpy(state).float().unsqueeze(0)
        else:
            state_tensor = state.float().unsqueeze(0) if state.dim() == 1 else state.float()

        self.current_state = state_tensor.to(self.device)
        self.current_state_norm = self._normalize_actions(self.current_state).to(self.device)
    
    def get_action(self, instruction: str = None) -> List[np.ndarray]:
        """Get action predictions from the model."""
        if len(self.obs_cache) == 0:
            raise ValueError("No observations in cache. Call update_obs first.")
        
        if self.current_state is None:
            raise ValueError("No robot state available. Call update_obs first.")
        
        current_frame = self.obs_cache[-1]

        # Build the text prompt. T5 and VLM MUST receive the exact same string that
        # was used at training time (see use_scene_prefix note in __init__).
        if self.use_scene_prefix:
            scene_prefix = ("The whole scene is in a realistic, industrial art style with three views: "
                            "a fixed rear camera, a movable left arm camera, and a movable right arm camera. "
                            "The aloha robot is currently performing the following task: ")
            prompt_text = f"{scene_prefix}{self.current_instruction}"
        else:
            prompt_text = self.current_instruction

        # Load pre-encoded T5 embeddings, or encode with T5 if no cache is configured.
        t5_list = self._get_t5_embeddings(prompt_text)

        # Build VLM inputs (same text as T5)
        first_frame_pil = self._tensor_to_pil_image(current_frame.squeeze(0).cpu())
        vlm_inputs = self._preprocess_vlm_messages(prompt_text, first_frame_pil)

        # Run inference
        # IMPORTANT: training (LeRobot pipeline) feeds the model NORMALIZED state in [0, 1].
        # Feed the normalized state here to match training; feeding the raw state makes the
        # robot diverge ("fly away").
        num_inference_steps = self.config_dict['model']['inference']['num_inference_timesteps']
        with torch.no_grad():
            predicted_frames, predicted_actions = self.model.inference_step(
                first_frame=current_frame,
                state=self.current_state_norm,
                num_inference_steps=num_inference_steps,
                language_embeddings=t5_list,
                vlm_inputs=[vlm_inputs],
            )

        # Save frame grid
        if predicted_frames is not None:
            if predicted_frames.dim() == 5:
                if predicted_frames.shape[1] == 3:
                    predicted_frames_viz = predicted_frames.permute(0, 2, 1, 3, 4)
                else:
                    predicted_frames_viz = predicted_frames
                
                condition_frame_viz = current_frame.squeeze(0)
                predicted_frames_viz = predicted_frames_viz.squeeze(0)
                
                self._save_frame_grid(condition_frame_viz, predicted_frames_viz)
                self.step_count += 1

        # The model predicts actions in the NORMALIZED [0, 1] space (training target).
        # Denormalize back to the real joint scale before returning; skipping this makes
        # the robot receive ~[0, 1] values as absolute joint targets and "fly away".
        predicted_actions = self._denormalize_actions(predicted_actions.to(self.device))
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

    def _stitch_three_cameras(self, head_img, left_img, right_img) -> np.ndarray:
        """Stitch head/left/right cameras into the concatenated layout used at training.

        This mirrors data/lerobot/add_cam_concatenated_to_lerobot_dataset._stitch_frames
        (and data/utils/multi_camera_concat) exactly:
          - top   = head camera at its native resolution
          - bottom height = head_h // 2
          - each wrist camera occupies half of the head width
        Computing the wrist sizes dynamically (instead of hardcoding 160x120) keeps the
        proportions identical to training regardless of the RoboTwin camera resolution.
        """
        head = np.asarray(head_img)
        left = np.asarray(left_img)
        right = np.asarray(right_img)

        h_high, w_high = head.shape[:2]
        bottom_h = h_high // 2
        split_w = w_high // 2
        right_w = w_high - split_w

        left_resized = cv2.resize(left, (split_w, bottom_h))
        right_resized = cv2.resize(right, (right_w, bottom_h))
        bottom_row = np.concatenate([left_resized, right_resized], axis=1)
        return np.concatenate([head, bottom_row], axis=0)

    def _preprocess_vlm_messages(self, instruction: str, image: Image.Image) -> Dict[str, torch.Tensor]:
        """Build VLM inputs using the SAME preprocessing as training.

        Reuses utils.vlm_utils.preprocess_vlm_messages so the message ordering
        (image first, then text), add_generation_prompt=True and qwen process_vision_info
        all match the training dataloader exactly.
        """
        encoded = preprocess_vlm_messages(instruction, image, self.vlm_processor)
        vlm_inputs = {
            k: (v.to(self.device) if isinstance(v, torch.Tensor) else v)
            for k, v in encoded.items()
        }
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

        stats = stat_data.get(self.embodiment_type)
        if stats is None:
            raise ValueError(
                f"Normalization stats for embodiment '{self.embodiment_type}' not found in stat.json. "
                f"Available: {list(stat_data.keys())}"
            )
        logger.info(f"Using normalization stats for embodiment: {self.embodiment_type}")
        self.action_min = torch.tensor(stats['min'], dtype=torch.float32, device=self.device)
        self.action_max = torch.tensor(stats['max'], dtype=torch.float32, device=self.device)
        self.action_range = self.action_max - self.action_min
        # Guard against zero-range dimensions (matches data/utils/norm.py behavior).
        self.action_range = torch.where(
            self.action_range == 0,
            torch.ones_like(self.action_range),
            self.action_range,
        )

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
    
    def _create_frame_grid(self, condition_frame: torch.Tensor, predicted_frames: torch.Tensor) -> Image.Image:
        """Create horizontal grid."""
        def tensor_to_numpy(tensor):
            if tensor.dim() == 3:
                tensor = tensor.permute(1, 2, 0)
            tensor = tensor.detach().cpu().float()
            tensor = torch.clamp(tensor, 0, 1)
            return (tensor.numpy() * 255).astype(np.uint8)
        
        condition_np = tensor_to_numpy(condition_frame)
        predicted_np = []
        num_pred_frames = predicted_frames.shape[0]
        for i in range(num_pred_frames):
            frame_np = tensor_to_numpy(predicted_frames[i])
            predicted_np.append(frame_np)
        
        while len(predicted_np) < 4:
            predicted_np.append(predicted_np[-1] if predicted_np else condition_np)
        
        all_frames = [condition_np] + predicted_np[:4]
        grid_image = np.concatenate(all_frames, axis=1)
        
        return Image.fromarray(grid_image)
    
    def _save_frame_grid(self, condition_frame: torch.Tensor, predicted_frames: torch.Tensor):
        """Save frame grid to disk."""
        if not self.save_images:
            return
        
        try:
            grid_image = self._create_frame_grid(condition_frame, predicted_frames)
            filename = f"episode_{self.episode_count:04d}_step_{self.step_count:04d}.png"
            save_path = self.save_dir / filename
            grid_image.save(save_path)
            logger.info(f"Saved frame grid to {save_path}")
        except Exception as e:
            logger.warning(f"Failed to save frame grid: {e}")


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
    
    if not wan_path:
        raise ValueError("wan_path not provided in usr_args")
    
    if not vlm_path:
        raise ValueError("vlm_path not provided in usr_args")
    
    policy_dir = Path(__file__).parent
    config_path = policy_dir / "utils" / "robotwin.yml"
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    use_scene_prefix = usr_args.get('use_scene_prefix')
    if use_scene_prefix is None:
        use_scene_prefix = True

    policy = MotusPolicy(
        checkpoint_path=checkpoint_path,
        wan_path=wan_path,
        vlm_path=vlm_path,
        config_path=str(config_path),
        device=device,
        log_dir=usr_args.get('log_dir'),
        task_name=usr_args.get('task_name'),
        embodiment_type=usr_args.get('embodiment_type') or "aloha_agilex_2",
        use_scene_prefix=bool(use_scene_prefix),
        t5_device=usr_args.get('t5_device'),
        t5_cache_dir=usr_args.get('t5_cache_dir'),
    )

    # Keep standalone robotwin entrypoints compatible with Qwen3-VL rope indexing.
    vlm_model = getattr(getattr(getattr(policy, "model", None), "vlm_model", None), "model", None)
    if vlm_model is not None and not getattr(vlm_model, "_xpolicylab_rope_index_patched", False):
        original_get_rope_index = vlm_model.get_rope_index
        if "mm_token_type_ids" not in inspect.signature(original_get_rope_index).parameters:

            def _get_rope_index_compat(*args, **kwargs):
                kwargs.pop("mm_token_type_ids", None)
                return original_get_rope_index(*args, **kwargs)

            vlm_model.get_rope_index = _get_rope_index_compat
            vlm_model._xpolicylab_rope_index_patched = True
    
    return policy


def eval(TASK_ENV, model, observation):
    """Evaluation function."""
    obs = encode_obs(observation)
    
    instruction = TASK_ENV.get_instruction()
    model.set_instruction(instruction)
    model.update_obs(obs)

    actions = model.get_action()
    
    for action in actions:
        TASK_ENV.take_action(action, action_type='qpos')


def reset_model(model):  
    """Reset model cache at episode start."""
    model.obs_cache.clear()
    model.action_cache.clear()
    model.current_state = None
    model.is_first_step = True
    model.prev_action = None
    model.episode_count += 1
    model.step_count = 0
    logger.info(f"Model reset completed for episode {model.episode_count}")