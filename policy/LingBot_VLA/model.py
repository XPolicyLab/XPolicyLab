import json
import os
import time
import random
import numpy as np
from collections import deque
import torchvision
import yaml
from types import SimpleNamespace
from packaging.version import Version
from typing import Callable, Dict, List, Optional, Type, Union, Tuple, Any, Sequence
from glob import glob
from tqdm import tqdm
from safetensors import safe_open
from safetensors.torch import load_file
from pathlib import Path
from PIL import Image
import torch
import torch.nn.functional as F
from torch import Tensor, nn


import transformers
from transformers.models.auto.tokenization_auto import AutoTokenizer
from transformers import (
    AutoConfig,
    PretrainedConfig,
    PreTrainedModel,
    AutoProcessor,
)

from lerobot.configs.policies import PreTrainedConfig

from .lingbot_vla.lingbotvla.models.vla.pi0.modeling_pi0 import PI0Policy
from .lingbot_vla.lingbotvla.models.vla.pi0.modeling_lingbot_vla import LingbotVlaPolicy
from .lingbot_vla.lingbotvla.data.vla_data.transform import Normalizer, prepare_images, prepare_language, prepare_state
from .lingbot_vla.lingbotvla.models import build_processor

from XPolicyLab.model_template import ModelTemplate
from XPolicyLab.utils.process_data import get_robot_action_dim_info, pack_robot_state, unpack_robot_state

def set_seed_everywhere(seed: int):
    """Sets the random seed for Python, NumPy, and PyTorch functions."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)

set_seed_everywhere(42)

BASE_MODEL_PATH = {
    'pi0': os.environ.get('PALIGEMMA_PATH', './paligemma-3b-pt-224/'),
    'lingbotvla': os.environ.get('QWEN25_PATH', './Qwen2.5-VL-3B-Instruct/'),
}

def load_model_weights(policy, checkpoint_path, strict=True):
    all_safetensors = glob(os.path.join(checkpoint_path, "*.safetensors"))
    merged_weights = {}

    for file_path in tqdm(all_safetensors):
        with safe_open(file_path, framework="pt", device="cpu") as f:
            for key in f.keys():
                merged_weights[key] = f.get_tensor(key)
    policy.load_state_dict(merged_weights, strict=strict)


def center_crop_image(image: Union[np.ndarray, Image.Image]) -> Image.Image:
    crop_scale = 0.9
    side_scale = float(np.sqrt(np.clip(crop_scale, 0.0, 1.0)))  # side length scale
    out_size = (224, 224)

    # Convert input to PIL Image
    if isinstance(image, np.ndarray):
        arr = image
        if arr.dtype.kind == "f":
            # If floats likely in [0,1], map to [0,255]
            if arr.max() <= 1.0 and arr.min() >= 0.0:
                arr = (np.clip(arr, 0.0, 1.0) * 255.0).astype(np.uint8)
            else:
                arr = np.clip(arr, 0.0, 255.0).astype(np.uint8)
        elif arr.dtype == np.uint16:
            # Map 16-bit to 8-bit
            arr = (arr / 257).astype(np.uint8)
        elif arr.dtype != np.uint8:
            arr = arr.astype(np.uint8)
        pil = Image.fromarray(arr)
    elif isinstance(image, Image.Image):
        pil = image
    else:
        raise TypeError("image must be a numpy array or PIL.Image.Image")

    # Force RGB for consistent output
    pil = pil.convert("RGB")
    W, H = pil.size

    # Compute centered crop box (integer pixels)
    crop_w = max(1, int(round(W * side_scale)))
    crop_h = max(1, int(round(H * side_scale)))
    left = (W - crop_w) // 2
    top = (H - crop_h) // 2
    right = left + crop_w
    bottom = top + crop_h

    cropped = pil.crop((left, top, right, bottom))
    resized = cropped.resize(out_size, resample=Image.BILINEAR)
    return resized

def resize_with_pad(img, width, height, pad_value=-1):
    # assume no-op when width height fits already
    if img.ndim != 4:
        raise ValueError(f"(b,c,h,w) expected, but {img.shape}")
    
    # channel last to channel first if necessary
    if img.shape[1] not in (1, 3) and img.shape[-1] in (1, 3):
        img = img.permute(0, 3, 1, 2)

    cur_height, cur_width = img.shape[2:]

    ratio = max(cur_width / width, cur_height / height)
    resized_height = int(cur_height / ratio)
    resized_width = int(cur_width / ratio)
    resized_img = F.interpolate(
        img, size=(resized_height, resized_width), mode="bilinear", align_corners=False
    )

    pad_height = max(0, int(height - resized_height))
    pad_width = max(0, int(width - resized_width))

    # pad on left and top of image
    padded_img = F.pad(resized_img, (pad_width, 0, pad_height, 0), value=pad_value)
    return padded_img

class PolicyPreprocessMixin:

    @torch.no_grad
    def select_action(
        self, observation: dict[str, Tensor], use_bf16: bool = False, vlm_causal: bool = False, noise: Tensor | None = None
    ):
        self.eval()
        device = 'cuda'
        if use_bf16:
            dtype = torch.bfloat16
        else:
            dtype = torch.float32
        s1 = time.time()

        if len(observation['images'].shape) == 4:
            observation['images'] = observation['images'].unsqueeze(0)
            observation['img_masks'] = observation['img_masks'].unsqueeze(0)
        
        if 'expert_imgs' in observation:
            actions = self.model.sample_actions(
                observation['images'].to(dtype=dtype, device=device), 
                observation['img_masks'].to(device=device), 
                observation['lang_tokens'].unsqueeze(0).to(device=device), 
                observation['lang_masks'].unsqueeze(0).to(device=device), 
                observation['state'].unsqueeze(0).to(dtype=dtype, device=device), 
                observation['expert_imgs'].to(dtype=dtype, device=device), 
                vlm_causal = vlm_causal
            )
        else:
            actions = self.model.sample_actions(
                observation['images'].to(dtype=dtype, device=device), 
                observation['img_masks'].to(device=device), 
                observation['lang_tokens'].unsqueeze(0).to(device=device), 
                observation['lang_masks'].unsqueeze(0).to(device=device), 
                observation['state'].unsqueeze(0).to(dtype=dtype, device=device), 
                vlm_causal = vlm_causal
            )
        delta_time = time.time() - s1
        print(f'sample_actions cost {delta_time} s')
        action_dim = getattr(self.config, "action_dim", actions.shape[-1])
        observation['action'] = actions.squeeze(0)[:, :action_dim].to(dtype=torch.float32, device='cpu')
        if use_bf16:
            observation['state'] = observation['state'].to(dtype=torch.float32)
        data = self.normalizer.unnormalize(observation)
        return data

class LingBotVlaInferencePolicy(PolicyPreprocessMixin, LingbotVlaPolicy):
    pass # Only combine necessary functions

class PI0InfernecePolicy(PolicyPreprocessMixin, PI0Policy):
    pass # Only combine necessary functions


def merge_qwen_config(policy_config, qwen_config):
    if hasattr(qwen_config, 'to_dict'):
        config_dict = qwen_config.to_dict()
    else:
        config_dict = qwen_config

    text_keys = {
        "hidden_size",
        "intermediate_size",
        "num_hidden_layers",
        "num_attention_heads",
        "num_key_value_heads",
        "rms_norm_eps",
        "rope_theta",
        "vocab_size",
        "max_position_embeddings",
        "hidden_act",
        "tie_word_embeddings",
        "tokenizer_path",
    }

    for key in text_keys:
        if key in config_dict:
            setattr(policy_config, key, config_dict[key])
            print(f"✅ Merged LLM: {key} = {config_dict[key]}")

    if "vision_config" in config_dict:
        policy_config.vision_config = qwen_config.vision_config
    else:
        print("⚠️ Warning: 'vision_config' not found in qwen_config!")

    return policy_config

def extract_image(observation, candidate_names):
    vision = observation.get("vision", {})
    for candidate_name in candidate_names:
        if candidate_name not in vision:
            continue
        image = vision[candidate_name]
        if isinstance(image, dict):
            for image_key in ("color", "rgb"):
                if image_key in image:
                    return image[image_key]
        else:
            return image
    raise KeyError(f"Could not find any image for candidates: {candidate_names}")

def extract_prompt(observation, default_prompt):
    for key in ("prompt", "instruction", "task_instruction", "instructions"):
        value = observation.get(key)
        if value is None:
            continue
        if isinstance(value, (list, tuple)):
            value = value[0] if value else None
        if value is None:
            continue
        if hasattr(value, "item"):
            value = value.item()
        if isinstance(value, bytes):
            value = value.decode("utf-8", errors="replace")
        text = str(value).strip()
        if text:
            return text
    return default_prompt

def encode_obs(observation, action_type, robot_action_dim_info, default_prompt):    
    if robot_action_dim_info is None:
        raise ValueError("env_cfg is required when encoding raw environment observations.")
    
    images = {
        "cam_high": extract_image(observation, ["cam_high", "cam_head", "head_camera", "top_camera"]),
        "cam_left_wrist": extract_image(observation, ["cam_left_wrist", "left_camera", "left_wrist", "wrist_left"]),
        "cam_right_wrist": extract_image(observation, ["cam_right_wrist", "right_camera", "right_wrist", "wrist_right"]),
    }

    state = pack_robot_state(observation, action_type, robot_action_dim_info, source_type="obs").astype(np.float32)
    prompt = extract_prompt(observation, default_prompt)

    return {
            "observation.images.cam_high": images["cam_high"],
            "observation.images.cam_left_wrist": images["cam_left_wrist"],
            "observation.images.cam_right_wrist": images["cam_right_wrist"],
            "observation.state": state,
            "task": prompt,
        }

class Model(ModelTemplate):
    '''
    policy wrapper to support action ensemble or chunk execution
    '''
    def __init__(
        self,
        model_cfg,
    ) -> None:
        self.adaptive_ensemble_alpha = model_cfg.get("adaptive_ensemble_alpha", 0.1)
        checkpoint_path = model_cfg.get("checkpoint_path")
        
        self.task_name = model_cfg["task_name"]
        self.action_type = model_cfg["action_type"]
        self.default_prompt = model_cfg.get("prompt", self.task_name)
        self.robot_action_dim_info = (
            get_robot_action_dim_info(model_cfg["env_cfg"]) if model_cfg.get("env_cfg") is not None else None
        )
        self.observation_window: dict[str, Any] | None = None
        self._latest_env_idx_list: list[int] = [0]

        use_bf16 = True
        use_fp32 = False

        if model_cfg.get("use_fp32", False):
            use_bf16 = False
            use_fp32=True
        
        self.vla = self.get_model(checkpoint_path)
        
        self.vla = self.vla.cuda().eval()
        if use_bf16:
            self.vla = self.vla.to(torch.bfloat16)
        elif use_fp32:
            self.vla.model.float()
        
        self.use_bf16 = use_bf16
        self.use_fp32 = use_fp32

    def get_model(self, checkpoint_path) -> LingbotVlaPolicy:
        # load model
    
        print(f"loading model from: {checkpoint_path}")
        config = PreTrainedConfig.from_pretrained(checkpoint_path)
        
        # load training config
        training_config_path = Path(checkpoint_path).parent.parent.parent/'lingbotvla_cli.yaml'
        with open(training_config_path, 'r') as f:
            training_config = yaml.safe_load(f)
        f.close()

        # update model config according to training config
        training_model_config = training_config['model']
        training_model_config.update(training_config['train'])
        for k, v in training_model_config.items():
            v = getattr(config, k, training_model_config[k])
            setattr(config, k, v)

        # The underlying modeling_lingbot_vla.py only accepts 'flex'/'eager'/'fa2'/'xformer',
        # and 'fa2'/'xformer' raise NotImplementedError. 'flex' (flex_attention_forward) is
        # the fast path; 'eager' is the slow fallback. Do NOT use the HF 'flash_attention_2' alias here.
        config.attention_implementation = 'flex'
        
        # set base model according to training config
        training_base_model = training_config['model']['tokenizer_path']
        if 'paligemma' in training_base_model:
            model_name = 'pi0'
            config.vocab_size = 257152 # set vocab size for paligamma
        elif 'qwen2' in training_base_model.lower():
            model_name = 'lingbotvla'
        else:
            raise ValueError(f"Unsupported base model of {checkpoint_path}")
        base_model_path = BASE_MODEL_PATH[model_name]
        
        config.tokenizer_path = base_model_path
        self.model_name = model_name
        
        qwen_config = AutoConfig.from_pretrained(base_model_path)
        config = merge_qwen_config(config, qwen_config)

        if 'vocab_size' in training_config['model'] and training_config['model']['vocab_size'] != 0:
            config.vocab_size = training_config['model']['vocab_size']
        # load processors
        self.processor = build_processor(base_model_path)
        self.language_tokenizer = self.processor.tokenizer
        self.image_processor = self.processor.image_processor
        data_config = SimpleNamespace(**training_config['data'])
        
        print('Initializing model ... ')

        if 'paligemma' in training_base_model:
            policy = PI0InfernecePolicy(config, tokenizer_path=base_model_path)
        else:
            policy = LingBotVlaInferencePolicy(config, tokenizer_path=base_model_path)

        load_model_weights(policy, checkpoint_path, strict=True)
        
        policy.feature_transform = None
        self.data_config = data_config
        self.config = config
        self.joint_max_dim = training_config['train']['max_action_dim']
        self.action_dim = training_config['train']['action_dim']
        self.chunk_size = training_config['train']['chunk_size']
        policy.action_dim = self.action_dim
        policy.chunk_size = self.chunk_size
        norm_stats_file = Path(data_config.norm_stats_file)
        if not norm_stats_file.is_absolute():
            norm_stats_file = Path(__file__).resolve().parent / "lingbot_vla" / norm_stats_file
        self.norm_stats_file = norm_stats_file
        if 'align_params' in training_config['train']:
            self.use_depth_align = True
        else: self.use_depth_align = False
        if not self.norm_stats_file.exists():
            raise FileNotFoundError(
                f"Norm stats file not found: {self.norm_stats_file}. "
                f"Expected lingbot_vla assets path for: {data_config.norm_stats_file}"
            )
        with open(self.norm_stats_file) as f:
            self.norm_stats = json.load(f)

        policy.normalizer = Normalizer(
            norm_stats=self.norm_stats['norm_stats'],
            from_file=True,
            data_type='robotwin',
            norm_type={
                "observation.images.cam_high": "identity",
                "observation.images.cam_left_wrist": "identity",
                "observation.images.cam_right_wrist": "identity",
                "observation.state": self.data_config.norm_type,
                "action": self.data_config.norm_type,
            },
        )

        print('Model initialized ... ')

        return policy

    def resize_image(self, observation):
        for image_feature in ['observation.images.cam_high', 'observation.images.cam_left_wrist', 'observation.images.cam_right_wrist']:
            assert image_feature in observation
            assert len(observation[image_feature].shape)==3 and observation[image_feature].shape[-1] == 3
            image = observation[image_feature]
            img_pil = Image.fromarray(image)
            image_size = getattr(self.data_config, 'img_size', 224)
            img_pil = img_pil.resize((image_size, image_size), Image.BILINEAR)

            # img_resized shape: C*H*W
            img_resized = np.transpose(np.array(img_pil), (2,0,1))  # (3,224,224)
            observation[image_feature] = img_resized / 255.

    def infer(self, observation, center_crop=True):
        """Generates an action with the VLA policy."""

        # (If trained with image augmentations) Center crop image and then resize back up to original size.
        # IMPORTANT: Let's say crop scale == 0.9. To get the new height and width (post-crop), multiply
        #            the original height and width by sqrt(0.9) -- not 0.9!
        if 'reset' in observation and observation['reset']:
            self.reset(checkpoint_path=observation['checkpoint_path'] if 'checkpoint_path' in observation else None)
            return dict(action = None)
        
        self.resize_image(observation)
        for k, v in observation.items():
            if isinstance(v, np.ndarray):
                observation[k] = torch.from_numpy(v)
            
        joint_max_dim = getattr(self, 'joint_max_dim')
        action_dim = getattr(self, 'action_dim')
        chunk_size = getattr(self, 'chunk_size')
        normalized_observation = self.vla.normalizer.normalize(observation)
        base_image = (normalized_observation["observation.images.cam_high"] * 255).to(torch.uint8)
        left_wrist_image = (normalized_observation["observation.images.cam_left_wrist"] * 255).to(
            torch.uint8
        )
        right_wrist_image = (normalized_observation["observation.images.cam_right_wrist"] * 255).to(
            torch.uint8
        )
        obs_dict =  {
            "image": {"base_0_rgb": base_image, "left_wrist_0_rgb": left_wrist_image, "right_wrist_0_rgb": right_wrist_image},
            "state": normalized_observation["observation.state"].to(torch.float32),
            "prompt": [observation["task"]],
        }
        state = prepare_state(self.config, obs_dict)
        lang_tokens, lang_masks = prepare_language(self.config, self.language_tokenizer, obs_dict)
        images, img_masks, _ = prepare_images(self.config, self.image_processor, obs_dict)
        observation = {
            'images': images,
            'img_masks': img_masks,
            'state': state,
            'lang_tokens': lang_tokens,
            'lang_masks': lang_masks,
        }

        if self.use_bf16:
            observation['state'] = observation['state'].to(torch.bfloat16)

        # Standard XPolicyLab deployment mode: one forward pass produces the full
        # action chunk (chunk_size frames); deploy.py iterates over it frame by
        # frame and breaks early on is_episode_end(). No server-side chunk reuse
        # state is needed, matching LingBot_VA / Pi_0.
        org_actions = ['action']
        assert len(org_actions) == 1, "Only support single action feature"
        action = self.vla.select_action(observation, self.use_bf16, self.config.vlm_causal)[org_actions[0]]
        action = action.float().cpu().numpy()
        action = action[..., :self.action_dim]

        return dict(action=action)

    def update_obs(self, obs):
        self.update_obs_batch([obs])

    def update_obs_batch(self, obs_list):
        self._latest_env_idx_list = [obs.get("env_idx", index) for index, obs in enumerate(obs_list)]
        encoded_obs_list = [
            encode_obs(obs, self.action_type, self.robot_action_dim_info, self.default_prompt) for obs in obs_list
        ]
        self.observation_window = encoded_obs_list
    
    def get_action(self, **kwargs):
        action_list = self.get_action_batch(env_idx_list=[self._latest_env_idx_list[0]], **kwargs)

        return action_list[0]
    
    def get_action_batch(self, env_idx_list=None, **kwargs):
        if self.observation_window is None:
            raise AssertionError("update_obs or update_obs_batch first!")

        if env_idx_list is None and "obs" in kwargs:
            env_idx_list = kwargs["obs"]
        env_idx_list = env_idx_list or self._latest_env_idx_list

        action_list = []

        for batch_index, _ in enumerate(env_idx_list):
            action_chunk = self.infer(self.observation_window[batch_index])["action"]
            
            if self.robot_action_dim_info is None:
                action_list.append(action_chunk)
            else:
                action_list.append(
                    unpack_robot_state(
                        action_chunk,
                        self.action_type,
                        self.robot_action_dim_info,
                        source_type="obs",
                    )
                )
        
        return action_list


    def reset(self, checkpoint_path = None) -> None:

        if checkpoint_path is not None:
            self.vla = self.get_model(checkpoint_path)
            self.vla = self.vla.cuda().eval()
            if self.use_bf16:
                self.vla = self.vla.to(torch.bfloat16)
            elif self.use_fp32:
                self.vla.model.float()

        if getattr(self.data_config, 'norm_type', None) is None:
            self.data_config.norm_type = 'meanstd'
        if getattr(self.config, 'vlm_causal', None) is None:
            self.config.vlm_causal = False
        if getattr(self.config, 'qwenvl_bos', None) is None:
            self.config.qwenvl_bos = False

        # if update ckpt path
        if checkpoint_path is not None:
            all_safetensors = glob(os.path.join(checkpoint_path, "*.safetensors"))
            merged_weights = {}

            for file_path in tqdm(all_safetensors):
                with safe_open(file_path, framework="pt", device="cpu") as f:
                    for key in f.keys():
                        merged_weights[key] = f.get_tensor(key)
                
            self.vla.load_state_dict(merged_weights, strict=True)
