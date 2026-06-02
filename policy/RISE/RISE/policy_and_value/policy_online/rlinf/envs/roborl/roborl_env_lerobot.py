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

import copy
import os
from typing import List, Optional, Union

import gym
import numpy as np
import torch

from openpi_value.training import config as _config
# import openpi_value.training.data_loader as _data
import openpi_value.training.episode_base_data_loader as _data
from dataclasses import replace
import random
from collections import deque

from types import SimpleNamespace
from openpi_value.models import model as _model
from dataclasses import replace

def to_tensor(
    array: Union[dict, torch.Tensor, np.ndarray, list], device: str = "cpu"
) -> Union[dict, torch.Tensor]:
    """
    Copied from ManiSkill!
    Maps any given sequence to a torch tensor on the CPU/GPU. If physx gpu is not enabled then we use CPU, otherwise GPU, unless specified
    by the device argument

    Args:
        array: The data to map to a tensor
        device: The device to put the tensor on. By default this is None and to_tensor will put the device on the GPU if physx is enabled
            and CPU otherwise

    """
    if isinstance(array, (dict)):
        return {k: to_tensor(v, device=device) for k, v in array.items()}
    elif isinstance(array, torch.Tensor):
        ret = array.to(device)
    elif isinstance(array, np.ndarray):
        if array.dtype == np.uint16:
            array = array.astype(np.int32)
        elif array.dtype == np.uint32:
            array = array.astype(np.int64)
        ret = torch.tensor(array).to(device)
    else:
        if isinstance(array, list) and isinstance(array[0], np.ndarray):
            array = np.array(array)
        ret = torch.tensor(array, device=device)
    if ret.dtype == torch.float64:
        ret = ret.to(torch.float32)
    return ret

def build_datasets(config: _config.TrainConfig, shuffle=True):
    """Build datasets using the unified data loader with PyTorch framework."""
    # data_loader = _data.create_data_loader(config, framework="pytorch", shuffle=shuffle)
    data_loader = _data.create_dataloader_with_sequential_episode(config, framework="pytorch", shuffle=shuffle)
    return data_loader, data_loader.data_config()

class RoborlEnv(gym.Env):
    """
    A Gym environment that reads states from local JSON files instead of connecting to a physical simulator.
    
    This environment uses pre-recorded state data stored in JSON files. It supports multiple environments in parallel and can handle different tasks.
    """
    def __init__(self, cfg, seed_offset, total_num_processes):
        """
        Initialize the RoborlEnv.
        
        Args:
            cfg: Configuration object containing environment parameters
            seed_offset: Offset for the seed to ensure different processes have different seeds
            total_num_processes: Total number of parallel processes
        """
        self.seed_offset = seed_offset
        self.cfg = cfg
        self.total_num_processes = total_num_processes
        self.seed = self.cfg.seed + seed_offset
        self._is_start = True
        self.num_envs = self.cfg.num_envs       # * num_group_envs
        self.group_size = self.cfg.group_size
        self.num_group = self.cfg.num_group

        self.ignore_terminations = cfg.ignore_terminations
        self.auto_reset = cfg.auto_reset
        
        # self.inter_sample = cfg.inter_sample
        self.wm_action_interval = cfg.wm_action_interval

        # Random number generators
        self._generator = np.random.default_rng(seed=self.seed)

        # Load lerobot dataset
        self.policy_config = _config.get_config(
            cfg.policy_config_name
        )
        
        self.policy_config = replace(self.policy_config, batch_size=self.num_envs)
        self.policy_config = replace(self.policy_config, num_workers=0)

        
        self.with_advantage_condition = cfg.with_advantage_condition
        
        if not self.with_advantage_condition:
            # * If no with_advantage_condition, pop out the action_advantage in repack_transform.
            self.policy_config.data.repack_transforms.inputs[0].structure.pop('action_advantage', None)
        
        self.data_loader, self.data_config = build_datasets(self.policy_config)
        self.data_iter = iter(self.data_loader)
        self.chunk_size = cfg.chunk_size
        # Initialize the episode and state (mainly for GRPO)
        self.update_reset_state_ids()

        # Metrics initialization
        self.prev_step_reward = np.zeros(self.num_envs)
        self.use_rel_reward = cfg.use_rel_reward
        self._init_metrics()
        self._elapsed_steps = np.zeros(self.num_envs, dtype=np.int32)

        # Current state tracking
        self.current_lerobot_obs = None
        self.current_gt_actions = None
        
        self.train_mode = cfg.train_mode
        self.offline_rl = cfg.offline_rl
        self.use_end_reward = True
        
        # History observation (for dynamics model)
        self.history_obs = deque(maxlen=3*self.wm_action_interval) 
        self.wm_history_obs = None
        
        
        # * Custom
        self.action_dim = 14  # * agilex
        self.policy_prompt = self.policy_config.data.default_prompt
        self.tokenize_transform = self.data_config.model_transforms.inputs[2]

    @property
    def elapsed_steps(self):
        """Get the elapsed steps for each environment."""
        return self._elapsed_steps

    @property
    def info_logging_keys(self):
        """Get the keys for logging information."""
        return []

    @property
    def is_start(self):
        """Check if the environment is in the start state."""
        return self._is_start

    @is_start.setter
    def is_start(self, value):
        """Set the start state flag."""
        self._is_start = value

    def _init_metrics(self):
        """Initialize metrics for tracking environment performance."""
        self.success_once = np.zeros(self.num_envs, dtype=bool)
        self.fail_once = np.zeros(self.num_envs, dtype=bool)
        self.returns = np.zeros(self.num_envs)

    def _reset_metrics(self, env_idx=None):
        """Reset metrics for the specified environment indices."""
        if env_idx is not None:
            mask = np.zeros(self.num_envs, dtype=bool)
            mask[env_idx] = True
            self.prev_step_reward[mask] = 0.0
            self.success_once[mask] = False
            self.fail_once[mask] = False
            self.returns[mask] = 0
            self._elapsed_steps[env_idx] = 0
        else:
            self.prev_step_reward[:] = 0
            self.success_once[:] = False
            self.fail_once[:] = False
            self.returns[:] = 0.0
            self._elapsed_steps[:] = 0

    def _record_metrics(self, step_reward, terminations, infos):
        """Record metrics for the current step."""
        episode_info = {}
        self.returns += step_reward
        self.success_once = self.success_once | terminations
        episode_info["success_once"] = self.success_once.copy()
        episode_info["return"] = self.returns.copy()
        episode_info["episode_len"] = self.elapsed_steps.copy()
        episode_info["reward"] = (episode_info["return"] / episode_info["episode_len"]) if (episode_info["episode_len"] > 0).any() else 0
        infos["episode"] = to_tensor(episode_info)
        return infos
    

    def _wrap_obs(self, lerobot_obs, gt_actions):
        """Wrap the lerobot observation into a dictionary."""
        
        obs = {
            "lerobot_obs": lerobot_obs,
            "gt_actions": gt_actions,
            "history_obs": self.history_obs if self.wm_history_obs is None else self.wm_history_obs
        }
        
        return obs


    def reset(
        self,
        env_idx: Optional[Union[int, List[int], np.ndarray]] = None
    ):
        """
        Reset the environment to the initial state.
        
        Args:
            env_idx: Indices of environments to reset
            reset_state_ids: Specific state IDs to reset to
            options: Additional options for reset
        
        Returns:
            obs: Initial observation
            infos: Additional information
        """
        if env_idx is None:
            env_idx = np.arange(self.num_envs)

        # Reset metrics
        self._reset_metrics(env_idx)

        # Get initial lerobot observation
        self.history_obs.clear()
        if hasattr(self.data_loader._data_loader, 'reset'):
            self.data_loader._data_loader.reset(self.reset_state_ids, env_idx)
        
        for _ in range(self.history_obs.maxlen): # Step N times to obtain the history obs at the first time
            self.current_lerobot_obs, self.current_gt_actions = self.get_obs_action()
            self.history_obs.append(self.current_lerobot_obs.image_original)
        
        self.current_lerobot_obs, self.current_gt_actions = self.get_obs_action()

        # Wrap observations
        obs = self._wrap_obs(self.current_lerobot_obs, self.current_gt_actions)
        
        infos = {}
        return obs, infos


    def get_obs_action(self):
        try:
            # Get the next batch
            observation, actions = next(self.data_iter)
            
        except StopIteration:
            # * Reach the end of the dataset, need to reset for a new epoch
            
            # 1. Update epoch for DDP sampler
            current_epoch = self._elapsed_steps // len(self.data_loader)
            
            if hasattr(self.data_loader, "set_epoch"):
                self.data_loader.sampler.set_epoch(current_epoch)
                
            # 2. Re-create the iterator to start the new epoch
            self.data_iter = iter(self.data_loader)
            
            # 3. Get the first batch from the new iterator
            observation, actions = next(self.data_iter)

        return observation, actions

    # * Simulation rollout for one timestep (NOT a chunk, but a step of a chunk).
    def step(self, actions=None, action_step = None, pred_result=None, auto_reset=True):
        """
        Take a step in the environment.
        
        Args:
            actions: Actions to take
            auto_reset: Whether to automatically reset the environment when done
        
        Returns:
            obs: Observation after the step
            reward: Reward for the step
            terminations: Whether the episode has terminated
            truncations: Whether the episode has been truncated
            infos: Additional information
        """
        if actions is None:
            assert self._is_start, "Actions must be provided after the first reset."
        
        if self.is_start:
            obs, infos = self.reset()
            self._is_start = False
            terminations = np.zeros(self.num_envs, dtype=bool)
            truncations = np.zeros(self.num_envs, dtype=bool)
            
            # Return the initial observation
            return obs, None, to_tensor(terminations), to_tensor(truncations), infos

        if isinstance(actions, torch.Tensor):
            actions = actions.detach().cpu().numpy()
            
        terminations = np.zeros(self.num_envs, dtype=bool)
        
        if pred_result != None:
            # * WM prediction with inferred reward

            action_indice = action_step // self.wm_action_interval
            step_reward = self.cfg.reward_coef * np.array(pred_result["rm_value"][:, action_indice])
                          
            # self.current_lerobot_obs.__dict__.keys()
            # dict_keys(['images', 'image_masks', 'state', 'tokenized_prompt', 'tokenized_prompt_mask', 'token_ar_mask', 'token_loss_mask', 'episode_index', 'frame_index', 'episode_length', 'action_advantage', 'action_advantage_original', 'image_original'])
            # - 'token_ar_mask', 'token_loss_mask' are None 
            # self.current_lerobot_obs.action_advantage tensor([ 6,  6,  6,  1, 10,  8,  7,  1])
            self.current_lerobot_obs.images["base_0_rgb"] = pred_result['next_top_pad']
            self.current_lerobot_obs.images["left_wrist_0_rgb"] = pred_result['next_left_pad']
            self.current_lerobot_obs.images["right_wrist_0_rgb"] = pred_result['next_right_pad']

            self.current_lerobot_obs.image_original["base_0_rgb"] = pred_result['next_top']
            self.current_lerobot_obs.image_original["left_wrist_0_rgb"] = pred_result['next_left']
            self.current_lerobot_obs.image_original["right_wrist_0_rgb"] = pred_result['next_right']
            
            
            self.current_lerobot_obs.state[:,:self.action_dim] = pred_result['next_state'][:, :self.action_dim]
            
            
            if self.with_advantage_condition:
                pseudo_action_advantage = torch.tensor(1.0)  
                # * used to generate action as IL ground-truth
                
                def process_prompt_batch(prompt, action_advantage, state):
                    
                    bs = state.shape[0]
                    batch_samples = [
                        {
                            "prompt": prompt,  # * single string
                            "action_advantage": action_advantage,  # * scarla tensor
                            "state": state[i]
                        } for i in range(bs)
                    ]
                    
                    for i in range(bs):
                        batch_samples[i] = self.tokenize_transform(batch_samples[i])
                        
                    # Re-pack the batch
                    batch_samples_dict = {}

                    for key in batch_samples[0].keys():
                        # Only process keys where the first element is not None
                        if batch_samples[0][key] is not None:
                            
                            # 1. Gather the samples for the current key
                            samples_for_key = [batch_samples[i][key] for i in range(bs)]
                            
                            # 2. Check the type of the first sample
                            first_sample = samples_for_key[0]
                            
                            # If the sample is NOT already a Tensor, convert all samples to Tensor
                            if not torch.is_tensor(first_sample):
                                # Convert each sample to a Tensor, inferring dtype and device
                                # This handles numpy.int64, numpy.float32, and other types
                                samples_for_key = [torch.tensor(s) for s in samples_for_key]
                            
                            # 3. Stack the Tensors
                            batch_samples_dict[key] = torch.stack(samples_for_key, dim=0)
                            
                            # 4. Align with state device 
                            # * NOTE: NOT align dtype (tokenized prompt type and state type may differ)
                            batch_samples_dict[key] = batch_samples_dict[key].to(state.device)
                            
                        else:
                            # If the key is None, set the output to None
                            batch_samples_dict[key] = None
                            

                    
                    return batch_samples_dict
                    
                
                tokenized_prompt_dict = process_prompt_batch(
                    self.policy_prompt,
                    pseudo_action_advantage,
                    self.current_lerobot_obs.state
                )
                
                self.current_lerobot_obs = replace(self.current_lerobot_obs, 
                                    tokenized_prompt=tokenized_prompt_dict['tokenized_prompt'], 
                                    tokenized_prompt_mask=tokenized_prompt_dict['tokenized_prompt_mask'],
                                    action_advantage=tokenized_prompt_dict['action_advantage'])
            
            
        else:
            if self.data_loader._data_loader.sampler.step_chunk_size > 1:
                ### Try to obtain offline data with significant differences (after ~50 steps). ###
                self.current_lerobot_obs, self.current_gt_actions = self.get_obs_action() # First, change "next step" to 1 + 43 frames.
                self._elapsed_steps += self.data_loader._data_loader.sampler.step_chunk_size
                
                self.data_loader._data_loader.sampler.step_chunk_size = 1
                # Start again from step 1 + 43, take 6 steps.
                for _ in range(self.history_obs.maxlen): # Step N times to obtain the history obs
                    self.current_lerobot_obs, self.current_gt_actions = self.get_obs_action()
                    self.history_obs.append(self.current_lerobot_obs.image_original)
                    self._elapsed_steps += 1
                    
            # * GT obs with (pre-annotated) reward
            self.current_lerobot_obs, self.current_gt_actions = self.get_obs_action()

            # ignore the termination reward
            step_reward = self._calc_step_reward(terminations) * 0
            
            
        self._elapsed_steps += 1
        # Check for truncations based on max episode steps
        truncations = self.elapsed_steps >= self.cfg.max_episode_steps
        
        terminations = self.termination_check(terminations, step_reward)
        
        # Wrap observations
        obs = self._wrap_obs(self.current_lerobot_obs, self.current_gt_actions)
        # Prepare for the next iter
        if pred_result == None:
            self.history_obs.append(self.current_lerobot_obs.image_original)
            self.wm_history_obs = None
        else:
            self.wm_history_obs = pred_result["history_obs"]
        # Record metrics
        infos = {}
        infos = self._record_metrics(step_reward, terminations, infos)
        
        if self.ignore_terminations:
            infos["episode"]["success_at_end"] = to_tensor(terminations)
            terminations[:] = False

        # Handle auto-reset if needed
        dones = terminations | truncations
        _auto_reset = auto_reset and self.auto_reset
        if dones.any() and _auto_reset:
            obs, infos = self._handle_auto_reset(dones, obs, infos)
        
        return (
            obs,                     # * obs for next step
            to_tensor(step_reward),  # * reward of this step
            to_tensor(terminations),
            to_tensor(truncations),
            infos
        )  # * go to func chunk_step in roborl_env_lerobot.py

    def termination_check(self, terminations, step_reward):
        for env_idx in range(self.num_envs):
            current_step = self.data_loader._data_loader.sampler.element_states[env_idx]['current_step']
            current_episode = self.data_loader._data_loader.sampler.element_states[env_idx]['current_episode']
            current_episode_max_length = self.data_loader._data_loader.sampler.episode_lengths[current_episode]
            
            if current_step >= current_episode_max_length or current_step / current_episode_max_length + (step_reward[env_idx]/self.cfg.reward_coef) > 0.99:
                terminations[env_idx] = True
        
        return terminations
    
    def chunk_check(self, auto_reset=False):
        chunk_rewards = raw_chunk_terminations = raw_chunk_truncations = torch.zeros(self.num_envs, self.chunk_size)
        for env_idx in range(self.num_envs):
            current_step = self.data_loader._data_loader.sampler.element_states[env_idx]['current_step']
            current_episode = self.data_loader._data_loader.sampler.element_states[env_idx]['current_episode']
            current_episode_max_length = self.data_loader._data_loader.sampler.episode_lengths[current_episode]
            
            step_reward = np.zeros(self.num_envs, dtype=float)
            terminations = np.zeros(self.num_envs, dtype=bool)
            truncations = np.zeros(self.num_envs, dtype=bool)
            
            termination_index = current_episode_max_length - current_step - 1
            
            if termination_index < self.chunk_size:
                if self.use_end_reward:
                    is_failure = self.current_lerobot_obs.is_failure_data[env_idx]
                    step_reward[env_idx] = chunk_rewards[env_idx, termination_index] = torch.where(
                        is_failure, 
                        torch.tensor(-0.1, device=chunk_rewards.device), 
                        torch.tensor(1.0, device=chunk_rewards.device)
                    ) * self.cfg.reward_coef
                    
                raw_chunk_terminations[env_idx, termination_index] = True
                terminations[env_idx] = True
            
            truncation_index = self.cfg.max_episode_steps - self.elapsed_steps[0] - 1
            if truncation_index < self.chunk_size:
                raw_chunk_truncations[env_idx, truncation_index] = True
                truncations[env_idx] = True
                
                
            # Record metrics
            infos = {}
            infos = self._record_metrics(step_reward, terminations, infos)
            
            if self.ignore_terminations:
                infos["episode"]["success_at_end"] = to_tensor(terminations)
                terminations[:] = False

            # Handle auto-reset if needed
            dones = terminations | truncations
            _auto_reset = auto_reset and self.auto_reset
            if dones.any() and _auto_reset:
                obs, infos = self._handle_auto_reset(dones, obs, infos)
        
        return chunk_rewards, raw_chunk_terminations, raw_chunk_truncations
                
    
    def update_reset_state_ids(self):
        """
            Generate `self.num_group` random episode groups.

            Each group contains: episode number (can be repeated) and the corresponding episode starting step.

            Returns:

            list: A list containing `self.num_group` tuples, each tuple being (episode_idx, start_step)
        """
        # Get all available episode numbers
        available_episodes = list(self.data_loader._data_loader.sampler.episode_lengths.keys())   
             
        # Generate random groups
        self.reset_state_ids = []
        for _ in range(self.num_group):
            # Randomly select an episode number (can be repeated)
            episode_idx = random.choice(available_episodes)
            
            # Get the length of this episode
            episode_length = self.data_loader._data_loader.sampler.episode_lengths[episode_idx]
            
            # Randomly generate the starting step (0 <= start_step < episode_length - 110), taking the previous 110 to prevent history errors.
            start_step = random.randint(0, max(0, episode_length - 1 - self.chunk_size - 10))
            
            self.reset_state_ids.extend([(episode_idx, start_step)] * self.group_size)
        

    # * Simulation rollout for multiple timesteps.
    def chunk_step(self, chunk_actions_and_wm_results):
        """
        Take multiple steps in the environment.
        
        Args:
            chunk_actions: Batch of actions to take
        
        Returns:
            obs: Observation after the steps
            chunk_rewards: Rewards for the steps
            chunk_terminations: Terminations for the steps
            chunk_truncations: Truncations for the steps
            infos: Additional information
        """
        # chunk_actions: [num_envs, chunk_step, action_dim]
        chunk_actions = chunk_actions_and_wm_results[0][0]
        pred_results = chunk_actions_and_wm_results[0][1]
        chunk_size = chunk_actions.shape[1]

        chunk_rewards = []
        raw_chunk_terminations = []
        raw_chunk_truncations = []

        self.data_loader._data_loader.sampler.step_chunk_size = 1
        if pred_results is None and (self.train_mode == 'IL' or (self.train_mode == 'RL' and self.offline_rl)):
            self.data_loader._data_loader.sampler.step_chunk_size = chunk_size - 7
            if self.train_mode == 'RL' and self.offline_rl:
                chunk_rewards, raw_chunk_terminations, raw_chunk_truncations = self.chunk_check()
            else:
                chunk_rewards = raw_chunk_terminations = raw_chunk_truncations = torch.zeros(self.num_envs, chunk_size)
            # Step one time
            extracted_obs, _, _, _, infos = self.step(
                    chunk_actions[:, 0], auto_reset=False
                )
        else:
            for i in range(chunk_size):
                actions = chunk_actions[:, i]
                extracted_obs, step_reward, terminations, truncations, infos = self.step(
                    actions, i, pred_results, auto_reset=False
                )

                chunk_rewards.append(step_reward)
                raw_chunk_terminations.append(terminations)
                raw_chunk_truncations.append(truncations)

            chunk_rewards = torch.stack(chunk_rewards, dim=1)  # [num_envs, chunk_steps]
            raw_chunk_terminations = torch.stack(
                raw_chunk_terminations, dim=1
            )  # [num_envs, chunk_steps]
            raw_chunk_truncations = torch.stack(
                raw_chunk_truncations, dim=1
            )  # [num_envs, chunk_steps]
        
        past_terminations = raw_chunk_terminations.any(dim=1)
        past_truncations = raw_chunk_truncations.any(dim=1)
        past_dones = torch.logical_or(past_terminations, past_truncations)

        if past_dones.any() and self.auto_reset:
            extracted_obs, infos = self._handle_auto_reset(
                past_dones.cpu().numpy(), extracted_obs, infos
            )

        if self.auto_reset or self.ignore_terminations:
            chunk_terminations = torch.zeros_like(raw_chunk_terminations)
            chunk_terminations[:, -1] = past_terminations

            chunk_truncations = torch.zeros_like(raw_chunk_truncations)
            chunk_truncations[:, -1] = past_truncations
        else:
            chunk_terminations = raw_chunk_terminations.clone()
            chunk_truncations = raw_chunk_truncations.clone()

        return (
            extracted_obs,
            chunk_rewards,
            chunk_terminations,
            chunk_truncations,
            infos
        )  # * go to func env_interact_step in env_worker.py

    def _handle_auto_reset(self, dones, _final_obs, infos):
        """
        Handle automatic reset of environments that are done.
        
        Args:
            dones: Indicator of which environments are done
            _final_obs: Final observation before reset
            infos: Additional information
        
        Returns:
            obs: Observation after reset
            infos: Additional information including final observation
        """
        final_obs = copy.deepcopy(_final_obs)
        env_idx = np.arange(0, self.num_envs)[dones]
        final_info = copy.deepcopy(infos)
        obs, infos = self.reset(
            env_idx=env_idx
        )
        # gymnasium calls it final observation but it really is just o_{t+1} or the true next observation
        infos["final_observation"] = final_obs
        infos["final_info"] = final_info
        infos["_final_info"] = dones
        infos["_final_observation"] = dones
        infos["_elapsed_steps"] = dones
        return obs, infos

    def _calc_step_reward(self, terminations):
        """
        Calculate the reward for the current step.
        
        Args:
            terminations: Indicator of which environments have terminated
        
        Returns:
            reward: Reward for the current step
        """
        reward = self.cfg.reward_coef * terminations
        reward_diff = reward - self.prev_step_reward
        self.prev_step_reward = reward

        if self.use_rel_reward:
            return reward_diff
        else:
            return reward

    def flush_video(self, video_sub_dir: Optional[str] = None):
        """
        Flush the recorded video frames to a file.
        
        Args:
            video_sub_dir: Subdirectory to save the video
        """
        # In a real environment, you would save the video here
        # For this local state environment, we'll just reset the render images
        self.render_images = []
