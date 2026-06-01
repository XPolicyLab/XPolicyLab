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

import math
import random
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

import jax
import numpy as np
import torch

from openpi_value import transforms as _transforms
from openpi_value.models import model as _model
from openpi_value.models.pi0_config import Pi0Config_Custom
from openpi_value.models_pytorch.pi0_pytorch import PI0Pytorch, make_att_2d_masks

from rlinf.models.embodiment.modules.explore_noise_net import ExploreNoiseNet
from rlinf.models.embodiment.modules.value_head import ValueHead
from rlinf.models.embodiment.modules.dynamics_model import DynamicsModel
from rlinf.models.embodiment.modules.reward_model import RewardModel, resize_with_pad_torch, process_view_torch_rm, write_episode_video_rm
from rlinf.utils.wm_utils import process_observations_dual_arm, process_actions, concat_obs_and_original
from einops import rearrange
from openpi_client import image_tools

import torch.distributed as dist
from dataclasses import dataclass, replace
from collections import deque
import torch.nn.functional as F

def sample_beta(alpha, beta, bsize, device):
    alpha_t = torch.as_tensor(alpha, dtype=torch.float32, device=device)
    beta_t = torch.as_tensor(beta, dtype=torch.float32, device=device)
    dist = torch.distributions.Beta(alpha_t, beta_t)
    return dist.sample((bsize,))

def sample_time(bsize, inx_len, device):
    """
        Modified sample_time function

        Args:

            bsize: batch size

            inx_len: number of time values ​​in each batch

            device: device

        Returns:

            time: a tensor of shape (bsize, inx_len), where inx_len values ​​in each bsize dimension are arranged in descending order
    """
    # Generate Beta distribution samples of shape (bsize*inx_len)
    time_beta = sample_beta(1.5, 1.0, bsize * inx_len, device)
    
    # Adjust the shape to (bsize, inx_len)
    time_beta = time_beta.view(bsize, inx_len)
    
    time = time_beta * 0.999 + 0.001
    
    # Sort the inx_len values ​​of each bsize dimension in descending order
    time, _ = torch.sort(time, dim=1, descending=True)
    
    return time.to(dtype=torch.float32, device=device)

@dataclass(frozen=True)
class OpenPi0Config(Pi0Config_Custom):
    # config for rl
    config_name: str = (
        "pi0_libero"  # pi0_libero, pi05_libero, pi0_metaworld, pi05_metaworld
    )
    num_images_in_input: int = 2  # number of images in input
    noise_method: str = "flow_sde"  # flow_sde, flow_noise, flow_cps
    # noise config for flow-sde
    noise_level: float = 0.5
    noise_anneal: bool = False
    noise_params: list = field(
        default_factory=lambda: [0.7, 0.3, 400]
    )  # noise_start, noise_end, noise_anneal_steps
    # noise config for flow-noise
    noise_logvar_range: list = field(
        default_factory=lambda: [0.08, 0.16]
    )  # [min_std, max_std]
    # hyper-parameters
    action_chunk: int = 5  # action chunk
    action_env_dim: int = 7  # for environment action dim
    num_steps: int = 10  # denoise steps
    num_steps_get_action: int = 10  # denoise steps for action prediction
    
    
    
    # training config
    train_expert_only: bool = False
    safe_get_logprob: bool = False
    joint_logprob: bool = False  # designed for flow-noise
    double_layer: bool = False  # designed for flow-sde without acceleration
    ignore_last: bool = False  # ignore the last action for noise injection
    # critic
    detach_critic_input: bool = False  # detach critic input with the action expert
    chunk_critic_input: bool = False  # use only the action chunk for critic estimation
    add_value_head: bool = False  # add value head for ppo
    value_after_vlm: bool = False  # value after vlm, pi05 mode
    value_vlm_mode: str = "mean_token"  # last_token, mean_token, first_token

    with_advantage_condition: bool = False
    default_prompt: str = None

    IL_sde_tgt: bool = False  # * Default using deterministic target.

class OpenPi0ForRLActionPrediction(PI0Pytorch):
    """
    Pi0 model for reinforcement learning action prediction.
    """

    config: OpenPi0Config

    @property
    def _no_split_modules(self) -> list[str]:
        if self.config.train_expert_only:
            no_split_modules = [
                "GemmaDecoderLayer",
                "SiglipVisionEmbeddings",
                "GemmaRMSNorm",
                "GemmaRotaryEmbedding",
            ]
        else:
            no_split_modules = [
                "GemmaMLP",
                "SiglipVisionEmbeddings",
                "GemmaRMSNorm",
                "GemmaRotaryEmbedding",
            ]
        if self.config.noise_method == "flow_noise":
            no_split_modules.append("ExploreNoiseNet")
        return no_split_modules

    @property
    def _no_split_names(self) -> list[str]:
        return [
            "action_in_proj",
            "action_out_proj",
            "lm_head",
            # --pi0 only--
            "state_proj",
            "action_time_mlp_in",
            "action_time_mlp_out",
            # --pi05 only--
            "time_mlp_in",
            "time_mlp_out",
        ]

    def __init__(
        self,
        config: OpenPi0Config,
    ):
        # Override `sample_actions` to prevent parent class polymorphic call
        sample_actions_func = self.sample_actions
        super().__init__(config)
        self.sample_actions = sample_actions_func
        self.global_step = 0
        # assert
        assert not (self.config.double_layer and self.config.joint_logprob), (
            "double_layer and joint_logprob can not be set at the same time"
        )

        # rl model init
        if self.config.value_after_vlm:
            proj_width = 2048
        else:
            proj_width = 1024
        # value head
        if self.config.add_value_head:
            self.value_head = ValueHead(
                input_dim=proj_width,
                hidden_sizes=(512, 256, 128),
                output_dim=1,
                activation="relu",
                bias_last=True,
            )
        self.use_vlm_value = getattr(self.config, "value_after_vlm", False) and getattr(
            self.config, "add_value_head", False
        )
        
        use_torch_compile = getattr(self.config, "use_torch_compile", True)  # * Default using torch.compile.

        self.reward_mode = getattr(self.config, "reward_mode", "v1")
        
        if self.config.add_dynamics_model:
            self.dynamics_model = DynamicsModel(self.config.dynamics_model_config)
            if use_torch_compile:
                self.dynamics_model.pipe.transformer = torch.compile(self.dynamics_model.pipe.transformer)
                
                # * No improvement.
                # self.dynamics_model.pipe.transformer = torch.compile(self.dynamics_model.pipe.transformer, mode="max-autotune")


                # * No improvement.
                # self.dynamics_model.pipe.transformer = torch.compile(self.dynamics_model.pipe.transformer, mode="reduce-overhead")

                # self.dynamics_model.pipe.compile_repeated_blocks(fullgraph=True)

            # self.dynamics_model.pipe.vae = torch.compile(self.dynamics_model.pipe.vae)

        if self.config.add_reward_model:
            self.reward_model = RewardModel(self.config.reward_model_config, self.config.reward_model_ckpt)
            if use_torch_compile:
                self.reward_model.model.sample_values = torch.compile(self.reward_model.model.sample_values, mode="reduce-overhead")

                # TODO: try max-autotune
                # self.reward_model.model.sample_values = torch.compile(self.reward_model.model.sample_values, mode="max-autotune")

        # noise head for flow-noise
        if self.config.noise_method == "flow_noise":
            self.noise_head = ExploreNoiseNet(
                in_dim=1024,
                out_dim=self.config.action_dim,
                hidden_dims=[128, 64],
                activation_type="tanh",
                noise_logvar_range=self.config.noise_logvar_range,
                noise_scheduler_type="learn",
            )
        
        for name, module in self.named_modules():
            # Set _fsdp_wrap_name to the last part of the path (e.g., "model.action_in_proj" -> "action_in_proj")
            path_parts = name.split(".")
            setattr(module, "_fsdp_wrap_name", path_parts[-1] if path_parts else name)
            
        self.need_infer = False

    def set_global_step(self, global_step):
        self.global_step = global_step

    def setup_wrappers(
        self,
        transforms: Sequence[_transforms.DataTransformFn] = (),
        output_transforms: Sequence[_transforms.DataTransformFn] = (),
    ):
        self._input_transform = _transforms.compose(transforms)
        self._output_transform = _transforms.compose(output_transforms)

    def input_transform(self, obs: dict, transpose=True):
        inputs = jax.tree.map(lambda x: x, obs)
        # process input
        first_process = "prompt" in inputs.keys()
        # * What does first_process mean?
        input_keys_if_exist = [
            "observation/image", "observation/wrist_image", "observation/state",
            "observation/image_top_head", "observation/image_hand_left", "observation/image_hand_right",
            "action_advantage"
        ]
        if first_process:
            inputs.pop("prompt")
        else:
            inputs = {
                value: inputs[value]
                for value in inputs.keys()
                if value in input_keys_if_exist
            }
        # tensor -> numpy
        inputs = jax.tree.map(
            lambda x: np.asarray(x.detach().cpu()) if torch.is_tensor(x) else x, inputs
        )
        
        batch_size = next(v.shape[0] for v in inputs.values() if hasattr(v, "shape"))

        # split & transform
        transformed_samples = []
        for i in range(batch_size):
            sample = jax.tree.map(lambda x: x[i], inputs)
            # convert from [3,256,256] -> [256,256,3]
            if transpose:
                sample = jax.tree.map(
                    lambda x: x.transpose(1, 2, 0)
                    if len(x.shape) == 3 and transpose
                    else x,
                    sample,
                )
            else:
                sample = jax.tree.map(lambda x: x if len(x.shape) == 3 else x, sample)
            if first_process:
                sample["prompt"] = obs["prompt"][i]
            else:
                sample["prompt"] = "xxxx"

            if 'observation/image_top_head' in sample:
                sample['images'] = {
                    "top_head": sample.pop('observation/image_top_head'),
                    "hand_left": sample.pop('observation/image_hand_left'),
                    "hand_right": sample.pop('observation/image_hand_right'),
                }

            
            if 'observation/state' in sample:
                sample['state'] = sample['observation/state']


            transformed_sample = self._input_transform(sample)
            if isinstance(transformed_sample, dict) and "prompt" in transformed_sample and self.config.offline_rl:
                transformed_sample.pop("prompt", None)

            transformed_samples.append(transformed_sample)
        
        inputs = jax.tree.map(
            lambda *torch_arr: torch.from_numpy(np.asarray(torch_arr).copy()),
            *transformed_samples,
        )

        # * Verified.

        if not first_process:
            inputs["tokenized_prompt"] = obs["tokenized_prompt"]
            inputs["tokenized_prompt_mask"] = obs["tokenized_prompt_mask"]
        return inputs

    def output_transform(self, outputs):
        # split & transform
        batch_size = outputs["actions"].shape[0]
        transformed_samples = []
        for i in range(batch_size):
            sample = jax.tree.map(lambda x: np.asarray(x[i].detach().cpu()), outputs)
            sample = self._output_transform(sample)
            transformed_samples.append(sample)
        # recombine
        outputs = jax.tree.map(
            lambda *torch_arr: torch.from_numpy(np.asarray(torch_arr).copy()),
            *transformed_samples,
        )

        #  * outputs['actions'].shape  torch.Size([8, 50, 14])
        outputs["actions"] = outputs["actions"][:, : self.config.action_chunk]

        return outputs

    # * if state from wm generation, use wm conditional action (policy inference) as ground-truth, also the reward
    # * if state from real-world data, use offline action as ground-truth, also the pre-labeled reward

    def forward(
        self,
        data: dict[str, torch.Tensor],
        **kwargs,
    ) -> dict[str, Any]:
        # get kwargs
        compute_values = kwargs.get("compute_values", False)
        gt_actions = kwargs.get("gt_actions", None)
        noise = kwargs.get("noise", None)
        timesteps = kwargs.get("timesteps", None)
        conditional_advantage = kwargs.get("conditional_advantage", None)

        if self.config.with_advantage_condition:
            data['action_advantage'] = conditional_advantage
        else:
            data.pop('action_advantage', None)
        
        
        # * Bad: -1,  Good: 1, need to convert to bins.
        
        chains = data["chains"]
        denoise_inds = data["denoise_inds"]
        # input transform
        # data['action_advantage'] = tensor([0.1721, 0.3051, 0.2347, 1.0000, 0.1560, 0.4660, 0.7044, 1.0000], device='cuda:0')
        
        observation = self.input_transform(data)
        # * if with_advantage_condition == False:  
        # *     observation['action_advantage'] is None, 
        # *     NO advantage-related info included in text prompt
        
        # * with_advantage_condition = True
        # observation['action_advantage']
        # tensor([ 6,  8, 10,  7, 10,  6,  1,  8])
        
        
        observation = _model.Observation.from_dict(observation)
        if self.config.offline_rl:
            
            # * replace original data --> avoid doing extra input_transform (which is incorrect.)
            observation.images['base_0_rgb'] = data['observation/image_top_head']
            observation.images['left_wrist_0_rgb'] = data['observation/image_hand_left']
            observation.images['right_wrist_0_rgb'] = data['observation/image_hand_right']
            observation = replace(observation, state=data['observation/state'])
        
        images, img_masks, lang_tokens, lang_masks, state = (
            self._preprocess_observation(observation, train=False)
        )

        # transfer to device
        device = chains.device
        images = [img.to(device) for img in images]
        img_masks = [img_mask.to(device) for img_mask in img_masks]
        state = state.to(device)
        
        # * get log prob, and model prediction
        log_probs, value_t, entropy, v_t = self.get_log_prob_value(
            images,
            img_masks,
            lang_tokens,
            lang_masks,
            state,
            chains,
            denoise_inds,
            compute_values,
            noise,
            gt_actions,
            timesteps,
        )
        log_probs = log_probs[
            :, :, : self.config.action_chunk, : self.config.action_env_dim
        ]
        entropy = entropy[
            :, :, : self.config.action_chunk, : self.config.action_env_dim
        ]
        # post process
        log_probs = log_probs.mean(dim=1)
        entropy = entropy.mean(dim=[1, 2, 3], keepdim=False)[
            :, None
        ]  # [:,None] to align with loss-mask shape
        value_t = value_t.mean(dim=-1, keepdim=False)
        return {
            "logprobs": log_probs,
            "values": value_t,
            "entropy": entropy,
            "v_t": v_t
        }
    

    def input_processor(self, env_processed_obs):
        to_process_obs = {
            "observation/image": env_processed_obs["images"],
            "observation/state": env_processed_obs["states"],
            "prompt": env_processed_obs["task_descriptions"],
        }
        if env_processed_obs["wrist_images"] is not None:
            to_process_obs["observation/wrist_image"] = env_processed_obs[
                "wrist_images"
            ]
        processed_obs = self.input_transform(to_process_obs)
        device = next(self.parameters()).device
        for key, value in processed_obs.items():
            if isinstance(value, list):
                processed_obs[key] = [
                    item.to(device=device).contiguous()
                    if torch.is_tensor(item)
                    else item
                    for item in value
                ]
            elif torch.is_tensor(value):
                processed_obs[key] = value.to(device=device).contiguous()
            elif isinstance(value, dict):
                for sub_key, sub_value in value.items():
                    processed_obs[key][sub_key] = sub_value.to(
                        device=device
                    ).contiguous()
        return processed_obs

    # * predict action
    # * [optional] rollout dynamics model and reward model --> rm_value.
    def predict_action_batch(
        self, 
        env_obs, 
        mode: Literal["train", "eval"] = "train", 
        compute_values=True,
        need_infer=False,
        need_wm=False,
        state_mode=None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        if not self.config.offline_rl:

            # * No world model rollout
            processed_obs = self.input_processor(env_obs)
                   
            observation = _model.Observation.from_dict(processed_obs)

            outputs = self.sample_actions(
                observation, mode=mode, compute_values=compute_values
            )
            actions_tensor = self.output_transform(
                {"actions": outputs["actions"], "state": observation.state}
            )["actions"]

            # * -------------------------------------------------
            actions_decoded_np = actions_tensor.numpy()
            action_advantage_original = None

            forward_inputs = {
                "chains": outputs["chains"],
                "denoise_inds": outputs["denoise_inds"],
                "observation/wrist_image": env_obs["wrist_images"],
                "observation/state": env_obs["states"],
                "tokenized_prompt": processed_obs["tokenized_prompt"],
                "tokenized_prompt_mask": processed_obs["tokenized_prompt_mask"],
            }
            
        else:
            # * env_obs maybe from world model rollout, where advantage is already processed in text prompt.
            
            observation = env_obs['lerobot_obs']
            gt_actions = env_obs['lerobot_gt_actions'].to(dtype=torch.float32)
            device = next(self.parameters()).device
            observation = jax.tree.map(lambda x: x.to(device), observation)
            action_advantage_original = observation.action_advantage_original
            
            if action_advantage_original is not None:
                action_advantage_original = action_advantage_original[:, None]  # * [8, 1]
            
            # * image_original: b h w c
            image_original = observation.image_original
            
            noise = None
            if hasattr(env_obs['lerobot_obs'], 'noise'):
                if env_obs['lerobot_obs'].noise is not None:
                    noise = env_obs['lerobot_obs'].noise.squeeze()
            
            self.need_infer = need_infer
            outputs = self.sample_actions(
                observation, 
                mode=mode, 
                noise=noise,
                compute_values=compute_values, 
                gt_actions=gt_actions.to(device)
            )

            action_pred = outputs["actions"]
            actions_decoded = self.output_transform(
                {"actions": action_pred, "state": observation.state}
            )["actions"]
            actions_decoded_np = actions_decoded.numpy()

            forward_inputs = {
                "chains": outputs["chains"],
                "denoise_inds": outputs["denoise_inds"],

                "observation/image_top_head": observation.images["base_0_rgb"],
                "observation/image_hand_left": observation.images["left_wrist_0_rgb"],
                "observation/image_hand_right": observation.images["right_wrist_0_rgb"],

                "observation/state": observation.state,
                
                "tokenized_prompt": observation.tokenized_prompt,
                "tokenized_prompt_mask": observation.tokenized_prompt_mask,
                
                "gt_actions": gt_actions,
                "noise": outputs['noises'],
                "timesteps": outputs['timesteps'],
                
                "actions": outputs["actions"]
            }
            
        result = {
            "prev_logprobs": outputs["prev_logprobs"],
            "prev_values": outputs["prev_values"],
            "forward_inputs": forward_inputs,
        }

        if  action_advantage_original is not None:
            result["conditional_advantage"] = action_advantage_original.cpu().contiguous()
        
        if self.config.add_dynamics_model and need_wm:

            if self.config.use_his_obs:
                device = next(self.parameters()).device
                wm_input = concat_obs_and_original(env_obs, image_original, self.config.wm_action_interval, device)
            else:
                wm_input = image_original
           # Processing observation data
            wm_obs = process_observations_dual_arm(wm_input, (192, 256), self.config.use_his_obs)

           # Processing motion data
            wm_act_tokens = process_actions(
                actions_decoded, 
                target_length=25, 
                feature_dim=30, 
                action_interval=self.config.wm_action_interval,
                min_val=self.dynamics_model.args.min_val,
                max_val=self.dynamics_model.args.max_val)


            # self.dynamics_model.device 'cuda'
            wm_preds = self.dynamics_model.infer(obs=wm_obs.to(dtype=torch.bfloat16),
                                                act_tokens=wm_act_tokens.to(device=wm_obs.device, dtype=torch.bfloat16),
                                                num_denois_steps=25,
                                                save_path=None,
                                                )

            # * wm_pred: [24, 3, 29, 192, 256], tensor

            if self.config.add_reward_model:
                
                # assert not self.config.chunk_reward, "Chunk reward is not supported for reward model."

                history_obs = deque(maxlen=3*self.config.wm_action_interval)
                pred_obs = rearrange(wm_preds['video'], '(b v) c t h w -> b v c t h w', v=3)
                # pred_next_obs: torch.Size([8, 3, 3, 29, 192, 256])

                
                reward = torch.zeros(len(pred_obs), self.config.action_chunk, dtype=torch.float32)  # * [8, 25]


                pred_start, pred_end = 4, pred_obs.shape[3]

                if self.config.wm_action_interval == 1:
                    assert pred_end - pred_start == self.config.action_chunk, \
                    f"Dynamics model prediction batch size does not match action batch size, {pred_end - pred_start} vs {actions_decoded.shape[0]}"   
                    # {pred_end - pred_start} vs {actions_decoded.shape[0]}, 25 vs 8
                    
                pred_top_all_steps = []
                pred_left_all_steps = []
                pred_right_all_steps = []

                # * rm input
                start_top = observation.images["base_0_rgb"].clone()
                start_left = observation.images["left_wrist_0_rgb"].clone()
                start_right = observation.images["right_wrist_0_rgb"].clone()
                
                
                # * TODO: check history?
                if self.config.use_his_obs:
                    for t in range(history_obs.maxlen):
                        pred_next_obs = pred_obs[:,:,:, pred_end-2-t] # (b, v, c, h, w)
                        # danamicys model generated history
                        t_base_0_rgb =  pred_next_obs[:,0].cpu().contiguous()
                        t_left_wrist_0_rgb =  pred_next_obs[:,1].cpu().contiguous()
                        t_right_wrist_0_rgb =  pred_next_obs[:,2].cpu().contiguous()
                        history_obs.append({"base_0_rgb":t_base_0_rgb, "left_wrist_0_rgb": t_left_wrist_0_rgb, "right_wrist_0_rgb": t_right_wrist_0_rgb})                   

                rm_init_observation = {
                    "state": torch.zeros((pred_obs.shape[0], 32), dtype=torch.float32).to(self.reward_model.device),
                    "images": {
                        "base_0_rgb": start_top,
                        "left_wrist_0_rgb": start_left,
                        "right_wrist_0_rgb": start_right,
                    },
                    "image_masks":{}
                }
                prompt = [self.config.default_prompt for _ in range(pred_obs.shape[0])]
                rm_value_init = self.reward_model.predict_reward(rm_init_observation, prompt)  # len -> bs

                rm_value_init = torch.tensor(rm_value_init, dtype=torch.float32).unsqueeze(1)  # [bs, 1]



                for pred_i in range(pred_start, pred_end):
                    
                    if self.config.chunk_reward:
                        if pred_i != pred_end-1:
                            continue

                    if self.reward_mode == 'v2':
                        # * only need to predict the first and last chunk of frames
                        window_size = 3
                        if pred_i in range(pred_start + window_size, pred_end - window_size):
                            continue

                    pred_next_obs = pred_obs[:,:,:, pred_i] # (b, v, c, h, w)

                    pred_top = resize_with_pad_torch(pred_next_obs[:, 0], height=224, width=224, out_mode='CHW')  # (b, c, h, w)
                    pred_left = resize_with_pad_torch(pred_next_obs[:, 1], height=224, width=224, out_mode='CHW')
                    pred_right = resize_with_pad_torch(pred_next_obs[:, 2], height=224, width=224, out_mode='CHW')

                    pred_top_all_steps.append(pred_top)
                    pred_left_all_steps.append(pred_left)
                    pred_right_all_steps.append(pred_right)


                    rm_observation = {
                        "state": torch.zeros((pred_top.shape[0], 32), dtype=torch.float32).to(self.reward_model.device),
                        "images": {
                            "base_0_rgb": pred_top,
                            "left_wrist_0_rgb": pred_left,
                            "right_wrist_0_rgb": pred_right,
                        },
                        "image_masks":{} 
                    }
                    
                    prompt = [self.config.default_prompt for _ in range(pred_top.shape[0])]
                    
                    rm_value = self.reward_model.predict_reward(rm_observation, prompt)
                    
                    reward[:, pred_i - pred_start] = torch.tensor(rm_value, dtype=torch.float32)
                    

                pred_top_all_steps = torch.stack(pred_top_all_steps, dim=1)  # [b, t, c, h, w]
                pred_left_all_steps = torch.stack(pred_left_all_steps, dim=1)  # [b, t, c, h, w]
                pred_right_all_steps = torch.stack(pred_right_all_steps, dim=1)  # [b, t, c, h, w]

                result["rm_value"] = (reward - rm_value_init).cpu().contiguous()   # [b, t]
                # * [8, 50]
                
                if self.config.chunk_reward:
                    reward = reward - rm_value_init

                    advantage_index = self.config.action_chunk // self.config.wm_action_interval - 1
                    reward_model_value = reward[:, advantage_index][:, None].contiguous().cpu()
                else:
                    # * reward_mode ['v1', 'v2']
                    if self.reward_mode == 'v1':
                        # * r = mean[(r_t+1 - r_t) + (r_t+2 - r_t) + (r_t+3 - r_t) + ...+ (r_t+chunk_size - r_t)]
                        reward = reward[:, :pred_end-pred_start]
                        reward = reward - rm_value_init
                        reward_model_value = (reward.sum(dim=1)/(pred_end-pred_start))[:, None].contiguous().cpu()
                        # * No bug here, because zeros would not affect the sum.

                    elif self.reward_mode == 'v2':
                        # * mean of last few frames - mean of first few frames
                        # * r = mean[r+chunk_size + r_cs-1 + r_cs-2] - mean[r_0 + r_1 + r_2]
                        window_size = 3

                        last_ind = pred_end - pred_start
                        last_mean = reward[:, last_ind-window_size: last_ind].mean(dim=1)

                        first_mean = torch.cat([rm_value_init, reward[:, :window_size]], dim=1).mean(dim=1)

                        reward_model_value = (last_mean - first_mean)[:, None].contiguous().cpu()

                if need_infer:                
                    result["conditional_advantage"] = (reward_model_value * self.config.advantage_scale).clamp(-1.0, 1.0)  # * [b, 1]

                # * results["rm_value"][:, 24]  # * only the 25th column has reward if chunk_reward is True

                if self.config.visualize_wm_pred:
                    # * visualization only, image_original came from observation or wm prediction.
                    start_top = image_original["base_0_rgb"].clone()
                    start_left = image_original["left_wrist_0_rgb"].clone()
                    start_right = image_original["right_wrist_0_rgb"].clone()

                    start_top = resize_with_pad_torch(start_top, height=224, width=224, out_mode='CHW')  # (b, c, h, w)
                    start_left = resize_with_pad_torch(start_left, height=224, width=224, out_mode='CHW')  # (b, c, h, w)
                    start_right = resize_with_pad_torch(start_right, height=224, width=224, out_mode='CHW')  # (b, c, h, w)

                    n_bs = pred_top_all_steps.shape[0]
                    n_steps = pred_top_all_steps.shape[1]
                    for i_bs in range(n_bs):

                        frames = []
                        start_top_img = process_view_torch_rm(start_top[i_bs])  # [h, w, c]
                        start_top_one_vid = [start_top_img for _ in range(n_steps)]

                        start_left_img = process_view_torch_rm(start_left[i_bs])  # [h, w, c]
                        start_left_one_vid = [start_left_img for _ in range(n_steps)]

                        start_right_img = process_view_torch_rm(start_right[i_bs])  # [h, w, c]
                        start_right_one_vid = [start_right_img for _ in range(n_steps)]

                        
                        pred_top_one_vid = pred_top_all_steps[i_bs]  # [t, c, h, w]
                        pred_left_one_vid = pred_left_all_steps[i_bs]  # [t, c, h, w]
                        pred_right_one_vid = pred_right_all_steps[i_bs]  # [t, c, h, w]
                        pred_reward_one_vid = reward[i_bs]

                        pred_top_one_vid = [
                            process_view_torch_rm(pred_top_one_vid[_]) for _ in range(n_steps)
                        ]

                        pred_left_one_vid = [
                            process_view_torch_rm(pred_left_one_vid[_]) for _ in range(n_steps)
                        ]

                        pred_right_one_vid = [
                            process_view_torch_rm(pred_right_one_vid[_]) for _ in range(n_steps)
                        ]
                        
                        pred_reward_one_vid = [
                            float(pred_reward_one_vid[_].item()) for _ in range(n_steps)
                        ]

                        frames = list(zip(
                            start_top_one_vid,
                            start_left_one_vid,
                            start_right_one_vid,
                            pred_top_one_vid,
                            pred_left_one_vid,
                            pred_right_one_vid,
                        ))


                        write_episode_video_rm(
                            episode_id=i_bs,
                            value_list = pred_reward_one_vid,
                            img_list = frames,
                            output_dir=f"./visualization/rm_preds_{self.global_step}",
                            fig_w=12.0,
                            fig_h=4.0,
                            dpi=100,
                            fps=5,
                            state_mode=state_mode,
                            advantage=float(result["conditional_advantage"][i_bs].item()),
                            ori_reward=float(reward_model_value[i_bs].item()),
                        )

                # * New obs --> Use last predicted frame
                result["next_top_pad"] = pred_top.cpu().contiguous()
                result["next_left_pad"] = pred_left.cpu().contiguous()
                result["next_right_pad"] = pred_right.cpu().contiguous()

                result["next_top"] = pred_next_obs[:,0].cpu().contiguous()
                result["next_left"] = pred_next_obs[:,1].cpu().contiguous()
                result["next_right"] = pred_next_obs[:,2].cpu().contiguous()

                # result["next_state"] = actions_decoded[:,-1].cpu().contiguous()
                result["next_state"] = action_pred[:,-1].cpu().contiguous()  # * Bugfixed.
                
                result["history_obs"] = history_obs
                


        # * go to huggingface_worker.py, generate func
        return actions_decoded_np, result

    @torch.no_grad()
    def sample_actions(
        self,
        observation: _model.Observation,
        noise=None,
        mode="train",
        compute_values=True,
        gt_actions=None,
    ) -> torch.Tensor:
        """Do a full inference forward and compute the action (batch_size x num_steps x num_motors)"""
        bsize = observation.state.shape[0]
        device = observation.state.device
        num_steps = self.config.num_steps   # * 5
        num_steps_get_action = self.config.num_steps_get_action
        
        if noise is None:
            actions_shape = (bsize, self.config.action_horizon, self.config.action_dim)
            noise = self.sample_noise(actions_shape, device)
        else:
            pad_width = (0, 18) 
            noise_padded = F.pad(noise, pad_width, mode='constant', value=0)
            noise = noise_padded.to(device)
        
        x_t = noise
        # add sde sample and traj collect
        chains = []
        log_probs = []
        values = []
        chains.append(x_t)
        
        state=None
        past_key_values=None
        prefix_pad_masks=None
        values_vlm=None
        
        if self.config.train_mode!="IL" or self.need_infer:

            images, img_masks, lang_tokens, lang_masks, state = (
                self._preprocess_observation(observation, train=False)
            )
            # * real-world -> world model -> world model
            prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(
                images, img_masks, lang_tokens, lang_masks,
            )
            prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
            prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1

            # Compute image and language key value cache
            prefix_att_2d_masks_4d = self._prepare_attention_masks_4d(prefix_att_2d_masks)
            self.paligemma_with_expert.paligemma.language_model.config._attn_implementation = "eager"  # noqa: SLF001
            
            #  defaultdict(<class 'list'>, {'paligemma_with_expert.paligemma.lm_head.weight': ['paligemma_with_expert.paligemma.model.language_model.embed_tokens.weight']})
            # check whether params are tied -- Tied, verified.

            (prefix_output, _), past_key_values = self.paligemma_with_expert.forward(
                attention_mask=prefix_att_2d_masks_4d,
                position_ids=prefix_position_ids,
                past_key_values=None,
                inputs_embeds=[prefix_embs, None],
                use_cache=True,
            )

            # add value based on the vlm for pi05, expert for pi0
            if self.use_vlm_value:
                values_vlm = self.get_value_from_vlm(prefix_output)
            if self.config.joint_logprob:
                initial_log_prob = self.get_logprob_norm(
                    x_t, torch.zeros_like(noise), torch.ones_like(noise)
                )
                log_probs.append(initial_log_prob)

        # In the joint logprob mode, we need to sample the logprob for each denoise step
        # In the non-joint logprob mode, only one denoise step is sampled and ode-sde mix sampling is used
        # denoise index
        if mode == "train":
            if self.config.joint_logprob:
                # * not going here
                denoise_inds = torch.arange(num_steps)
            else:
                if self.config.ignore_last:
                    # * not going here.
                    denoise_inds = torch.tensor(
                        [random.randint(0, num_steps - 2)] * num_steps
                    )
                else:
                    # * going here
                    denoise_inds = torch.tensor(
                        [random.randint(0, num_steps - 1)] * num_steps
                    )   # random_k --> [k, k, k, k]
        else:
            denoise_inds = torch.tensor([-1] * num_steps)
        denoise_inds = denoise_inds[None].repeat(bsize, 1)
    
        
        if self.config.offline_rl:
            # * True, going here.
            timesteps = sample_time(bsize, self.config.num_steps - 1, device)
            # Insert 1.0 before each batch
            ones = torch.ones(timesteps.shape[0], 1, device=timesteps.device)
            zeros = torch.zeros(timesteps.shape[0], 1, device=timesteps.device)
            timesteps = torch.cat([ones, timesteps, zeros], dim=1)
        else:
            timesteps = None
        
        # need_infer: whether to do policy inference 
        if self.config.train_mode=="IL" and self.need_infer:
            x_t_infer = x_t
            state_infer = state
            state = None
        
        # * denoised_inds.shape --> [8, 5], all 4
        # * denoise_inds[0]: tensor([4, 4, 4, 4, 4])
        # * denoise_inds[1]: tensor([4, 4, 4, 4, 4])

        # denoise step
        for idx in range(num_steps):
            # sample mean var val
            if idx == denoise_inds[0][idx]:
                sample_mode = 'train'
            else:
                sample_mode = 'eval'
            # * Originally in RLinf, 'train' mode introduces noises, while 'eval' mode does not -- pure ODE denoising.
                
                
            # * At first
            # * - sample_mode = 'eval   
            # * - check state. --> state = None
            # * - check self.config.offline_rl --> True
            
            x_t_mean, x_t_std, value_t, _ = self.sample_mean_var_val(
                    x_t,
                    idx,
                    state,
                    prefix_pad_masks,
                    past_key_values,
                    sample_mode,
                    num_steps,
                    compute_values,
                    noise,
                    gt_actions,
                    timesteps
            )

            # Euler step - use new tensor assignment instead of in-place operation
            x_t = x_t_mean + self.sample_noise(x_t.shape, device) * x_t_std
            log_prob = self.get_logprob_norm(x_t, x_t_mean, x_t_std)
            
            # store
            values.append(value_t)
            chains.append(x_t)
            log_probs.append(log_prob)
        
        if self.config.train_mode=="IL" and self.need_infer:
            # * Full-traj denoising to generate gt_action for action-conditioned IL.
            for idx_act in range(num_steps_get_action):
                x_t_infer, x_t_infer_std, _, _ = self.sample_mean_var_val(
                    x_t_infer,
                    idx_act,
                    state_infer,
                    prefix_pad_masks,
                    past_key_values,
                    "eval_wm",
                    num_steps_get_action,
                    compute_values,
                    noise,
                    gt_actions,
                    timesteps=None,  # * Bugfixed, need uniform timesteps here.
                )
                x_t_infer = x_t_infer + self.sample_noise(x_t_infer.shape, device) * x_t_infer_std
        
        if self.config.train_mode=="IL" and self.need_infer:
            # * Use the inferred x_0 as the gt action for action-conditioned IL.
            x_0 = x_t_infer
        else:
            x_0 = x_t
        
        chains = torch.stack(chains, dim=1)
        if self.config.train_mode!="IL":
            log_probs = torch.stack(log_probs, dim=1)[
                :, :, : self.config.action_chunk, : self.config.action_env_dim
            ]
            if self.use_vlm_value:
                values = values_vlm[:, None]
            else:
                values = torch.stack(values, dim=1).mean(dim=-1, keepdim=True)
        else:
            log_probs = torch.zeros([bsize,num_steps,self.config.action_horizon, self.config.action_dim])
            values = torch.zeros([bsize,1])
        
        return {
            "actions": x_0,
            "chains": chains,  # * x_t
            "prev_logprobs": log_probs,
            "prev_values": values,
            "denoise_inds": denoise_inds,
            "noises": noise,
            "timesteps": timesteps
        }

    def sample_mean_var_val(
        self,
        x_t,
        idx,
        state,
        prefix_pad_masks,
        past_key_values,
        mode,  # * ['train', 'eval', 'eval_wm']
        denoise_steps,
        compute_values=True,
        noise=None,
        gt_actions=None,
        timesteps=None
    ):
        """
        Sample the mean, variance and value of the action at a given timestep.
        Rollout sample (idx is int) and actor get_log_prob_value (idx is tensor) will load this function.
        """
        # expand the shape
        if state is not None:
            bsize = state.shape[0]
            device = state.device
        else:
            bsize = gt_actions.shape[0]
            device = gt_actions.device            
        if isinstance(idx, int):
            idx = torch.tensor(idx).expand(bsize)
        # build parameters
        if self.config.noise_anneal:
            # noise annealing
            noise_start, noise_end, anneal_steps = self.config.noise_params
            noise_level = (
                noise_start
                + (noise_end - noise_start)
                * min(self.global_step, anneal_steps)
                / anneal_steps
            )
            noise_level = torch.tensor(noise_level).to(device)
        else:
            # fixed noise level
            noise_level = torch.tensor(self.config.noise_level).to(device)
        
        if timesteps is None:
            assert mode == 'eval_wm', "for full-trajectory denoising, using uniform timesteps."
            timesteps = torch.linspace(1, 1 / denoise_steps, denoise_steps, device=device)
            timesteps = torch.cat([timesteps, torch.tensor([0.0], device=device)])
            t_input = timesteps[idx]
            delta = timesteps[idx] - timesteps[idx + 1]
            
        else:
            t_input = timesteps[torch.arange(timesteps.shape[0]), idx]
            delta = timesteps[torch.arange(timesteps.shape[0]), idx] - timesteps[torch.arange(timesteps.shape[0]), idx+1]
    
        # velocity prediction
        v_t     = None
        value_t = None


        if state is not None:
            suffix_out = self.get_suffix_out(
                state,
                prefix_pad_masks,
                past_key_values,
                x_t,
                t_input,
            )
            v_t = self.action_out_proj(suffix_out)  # [bs,n_action_steps,max_action_dim]
            # value prediction
            if (
                self.config.add_value_head
                and compute_values
                and not self.config.value_after_vlm
            ):
                # use chunk critic input
                if self.config.chunk_critic_input:
                    suffix_out_value = torch.mean(
                        suffix_out[:, : self.config.action_chunk], dim=1, keepdim=False
                    )
                else:
                    suffix_out_value = torch.mean(suffix_out, dim=1, keepdim=False)
                # detach critic input
                if self.config.detach_critic_input:
                    suffix_out_value = suffix_out_value.detach()
                value_t = self.value_head(suffix_out_value)[:, 0]
            else:
                value_t = torch.zeros((bsize), device=device)
        # ode sde mix sampling
        delta = delta[:, None, None].expand_as(x_t)
        t_input = t_input[:, None, None].expand_as(x_t)

        if noise is not None and gt_actions is not None \
            and self.config.train_mode=="IL" \
            and mode != "eval_wm":
                
            # * First step, going here. For both 'train' and 'eval' mode.
            x0_pred = gt_actions
            x1_pred = noise
            
        else:
            x0_pred = x_t - v_t * t_input
            x1_pred = x_t + v_t * (1 - t_input)
            
        
        IL_stochastic = self.config.train_mode=="IL" and mode == "eval_wm" and self.config.IL_sde_tgt
        
        if mode == "eval" or self.config.train_mode =="IL":
            
            if IL_stochastic:
                # * use flow_sde noise schedule for stochastic denoising.
                if self.config.train_mode =="RL":    # ! why determine this here?                   
                    denominator = 1 - torch.where(
                        timesteps == 1, 
                        timesteps[:, 1:2],  # Use the next time step to avoid division by zero
                        timesteps
                    )
                    sigmas = noise_level * torch.sqrt(timesteps / denominator)[:, :-1]  # (8,5)
                    sigma_i = sigmas[torch.arange(timesteps.shape[0]), idx][:, None, None].expand_as(x_t)
                else:
                    sigmas = (
                        noise_level
                        * torch.sqrt(
                            timesteps
                            / (1 - torch.where(timesteps == 1, timesteps[1], timesteps))
                        )[:-1]
                    )
                    sigma_i = sigmas[idx][:, None, None].expand_as(x_t)
                                        
                x0_weight = torch.ones_like(t_input) - (t_input - delta)
                x1_weight = t_input - delta - sigma_i**2 * delta / (2 * t_input)
                x_t_std = torch.sqrt(delta) * sigma_i
            
            else:
                x0_weight = 1 - (t_input - delta)
                x1_weight = t_input - delta
                x_t_std = torch.zeros_like(t_input)
            
        
        elif mode == "train" and self.config.train_mode !="IL":
            if self.config.noise_method == "flow_sde":
                if self.config.train_mode =="RL":                   
                    denominator = 1 - torch.where(
                        timesteps == 1, 
                        timesteps[:, 1:2],  # Use the next time step to avoid division by zero
                        timesteps
                    )
                    sigmas = noise_level * torch.sqrt(timesteps / denominator)[:, :-1]  # (8,5)
                    sigma_i = sigmas[torch.arange(timesteps.shape[0]), idx][:, None, None].expand_as(x_t)
                else:
                    sigmas = (
                        noise_level
                        * torch.sqrt(
                            timesteps
                            / (1 - torch.where(timesteps == 1, timesteps[1], timesteps))
                        )[:-1]
                    )
                    sigma_i = sigmas[idx][:, None, None].expand_as(x_t)
                                        
                x0_weight = torch.ones_like(t_input) - (t_input - delta)
                x1_weight = t_input - delta - sigma_i**2 * delta / (2 * t_input)
                x_t_std = torch.sqrt(delta) * sigma_i
            elif self.config.noise_method == "flow_cps":
                pi = torch.pi
                cos_term = torch.cos(pi * noise_level / 2).to(device)
                sin_term = torch.sin(pi * noise_level / 2).to(device)
                x0_weight = torch.ones_like(t_input) - (t_input - delta)
                x1_weight = (t_input - delta) * cos_term
                x_t_std = (t_input - delta) * sin_term
            elif self.config.noise_method == "flow_noise":
                x0_weight = 1 - (t_input - delta)
                x1_weight = t_input - delta
                x_t_std = self.noise_head(suffix_out)
            else:
                raise ValueError(f"Invalid noise method: {self.config.noise_method}")
        
        
        x_t_mean = x0_pred * x0_weight + x1_pred * x1_weight
        
        
        return x_t_mean, x_t_std, value_t, v_t

    def get_suffix_out(
        self,
        state,
        prefix_pad_masks,
        past_key_values,
        x_t,
        timestep,
    ):
        """Apply one denoising step of the noise `x_t` at a given timestep."""
        suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = (
            self.embed_suffix(state, x_t, timestep)
        )

        suffix_len = suffix_pad_masks.shape[1]
        batch_size = prefix_pad_masks.shape[0]
        prefix_len = prefix_pad_masks.shape[1]

        prefix_pad_2d_masks = prefix_pad_masks[:, None, :].expand(
            batch_size, suffix_len, prefix_len
        )

        suffix_att_2d_masks = make_att_2d_masks(suffix_pad_masks, suffix_att_masks)

        full_att_2d_masks = torch.cat([prefix_pad_2d_masks, suffix_att_2d_masks], dim=2)

        prefix_offsets = torch.sum(prefix_pad_masks, dim=-1)[:, None]
        position_ids = prefix_offsets + torch.cumsum(suffix_pad_masks, dim=1) - 1

        # Prepare attention masks
        full_att_2d_masks_4d = self._prepare_attention_masks_4d(full_att_2d_masks)
        self.paligemma_with_expert.gemma_expert.model.config._attn_implementation = (
            "eager"  # noqa: SLF001
        )

        outputs_embeds, _ = self.paligemma_with_expert.forward(
            attention_mask=full_att_2d_masks_4d,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=[None, suffix_embs],
            use_cache=False,
            adarms_cond=[None, adarms_cond],
        )

        suffix_out = outputs_embeds[1]
        suffix_out = suffix_out[:, -self.config.action_horizon :]
        suffix_out = suffix_out.to(dtype=torch.float32)
        return suffix_out

    # TODO: to check potential nan here
    def get_logprob_norm(self, sample, mu, sigma):
        # logprob = log p(x|mu,sigma) = -log(sigma) - 0.5 * log(2 * pi) - 0.5 * ((x - mu) / sigma) ** 2
        if self.config.safe_get_logprob:
            log_prob = -torch.pow((sample - mu), 2)
        else:
            mask = sigma == 0
            sigma_safe = torch.where(mask, torch.ones_like(sigma), sigma)
            constant_term = -torch.log(sigma_safe) - 0.5 * torch.log(
                2 * torch.pi * torch.ones_like(sample)
            )
            exponent_term = -0.5 * torch.pow((sample - mu) / sigma_safe, 2)
            log_prob = constant_term + exponent_term
            log_prob = torch.where(mask, torch.zeros_like(log_prob), log_prob)
        return log_prob

    def preprocess_for_train(self, data):
        return data

    def get_log_prob_value(
        self,
        images,
        img_masks,
        lang_tokens,
        lang_masks,
        state,
        chains,
        denoise_inds,
        compute_values=False,
        noise=None,
        gt_actions=None,
        timesteps=None,
    ):
        bsize = state.shape[0]
        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(
            images, img_masks, lang_tokens, lang_masks, 
        )
        prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1

        # Compute image and language key value cache
        prefix_att_2d_masks_4d = self._prepare_attention_masks_4d(prefix_att_2d_masks)
        self.paligemma_with_expert.paligemma.language_model.config._attn_implementation = "eager"  # noqa: SLF001

        # Compute image and language key value cache
        [prefix_output, _], past_key_values = self.paligemma_with_expert.forward(
            attention_mask=prefix_att_2d_masks_4d,
            position_ids=prefix_position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=True,
        )
        chains_log_probs = []
        chains_values = []
        chains_entropy = []

        # get log prob
        if self.config.joint_logprob:
            num_steps = self.config.num_steps
            initial_log_prob = self.get_logprob_norm(
                chains[:, 0],
                torch.zeros_like(chains[:, 0]),
                torch.ones_like(chains[:, 0]),
            )
            initial_entropy = self.gaussian_entropy(torch.ones_like(chains[:, 0]))
            chains_log_probs.append(initial_log_prob)
            chains_entropy.append(initial_entropy)
        else:
            num_steps = 1
        for idx in range(num_steps):
            denoise_ind = denoise_inds[:, idx]
            if self.config.train_mode=="IL":
                t_input = timesteps[torch.arange(bsize), denoise_ind]
                x0_weight = (1 - t_input)[:, None, None]
                x1_weight = t_input[:, None, None]
                x0_pred = gt_actions
                x1_pred = noise
                chains_pre = x0_weight * x0_pred + x1_weight * x1_pred
                
                delta = timesteps[torch.arange(bsize), denoise_ind] - timesteps[torch.arange(bsize), denoise_ind+1]
                x0_weight_next = (1 - (t_input - delta))[:, None, None]
                x1_weight_next = (t_input - delta)[:, None, None]
                chains_next = x0_weight_next * x0_pred + x1_weight_next * x1_pred 
            else:
                chains_pre = chains[torch.arange(bsize), denoise_ind]
                chains_next = chains[torch.arange(bsize), denoise_ind + 1]
            x_t_mean, x_t_std, value_t, v_t = self.sample_mean_var_val(
                chains_pre,
                denoise_ind,
                state,
                prefix_pad_masks,
                past_key_values,
                "train",
                self.config.num_steps,
                compute_values,
                noise=noise,
                gt_actions=gt_actions,
                timesteps=timesteps
            )
            log_probs = self.get_logprob_norm(chains_next, x_t_mean, x_t_std)
            entropy = self.gaussian_entropy(x_t_std)
            chains_log_probs.append(log_probs)
            chains_entropy.append(entropy)
            if self.use_vlm_value:
                chains_values.append(self.get_value_from_vlm(prefix_output))
            else:
                chains_values.append(value_t)
        chains_log_probs = torch.stack(chains_log_probs, dim=1)
        chains_values = torch.stack(chains_values, dim=1)

        # entropy is only available for flow-noise method
        if self.config.noise_method == "flow_noise":
            chains_entropy = torch.stack(chains_entropy, dim=1)
        else:
            chains_entropy = torch.zeros_like(chains_log_probs)
        return chains_log_probs, chains_values, chains_entropy, v_t

    def get_value_from_vlm(self, prefix_output):
        # prefix_output:
        # pi05: [bs, (256 * 3 + 200) = 968, 2048]
        # pi0: [bs, (256 * 3 + 48) = 816, 1024]
        assert self.config.pi05, \
                "Torch version only supports pi05 !! No pretrained pi0_torch"

        if self.config.pi05:
            lang_length = 200
            all_length = 968
        else:
            lang_length = 48
            all_length = 816
        if self.config.value_vlm_mode == "mean_token":
            prefix_mask = (
                [True] * 256 * self.config.num_images_in_input
                + [False] * 256 * (3 - self.config.num_images_in_input)
                + [True] * lang_length
            )
        elif self.config.value_vlm_mode == "last_token":
            prefix_mask = [False] * (all_length - 1) + [True] * 1
        elif self.config.value_vlm_mode == "first_token":
            prefix_mask = [True] * 1 + [False] * (all_length - 1)
        prefix_out_value = prefix_output[:, prefix_mask, :]
        prefix_out_value = prefix_out_value.mean(dim=1, keepdim=False)
        prefix_out_value = prefix_out_value.to(dtype=torch.float32)
        values_vlm = self.value_head(prefix_out_value)[:, 0]
        return values_vlm

    def gaussian_entropy(self, sigma):
        mask = sigma == 0
        sigma_safe = torch.where(mask, torch.ones_like(sigma), sigma)
        entropy = 0.5 * torch.log(2 * math.pi * math.e * (sigma_safe**2))
        return entropy

    def freeze_vlm(self):
        if self.config.train_expert_only:
            self.paligemma_with_expert.paligemma.eval()
            for params in self.paligemma_with_expert.paligemma.parameters():
                params.requires_grad = False
