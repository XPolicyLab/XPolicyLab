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

import gc

import torch
from omegaconf import DictConfig, OmegaConf, open_dict
from tqdm import tqdm

from rlinf.data.io_struct import EmbodiedRolloutResult, put_tensor_cpu
from rlinf.models import get_model
from rlinf.scheduler import Cluster, Worker
from rlinf.utils.metric_utils import compute_split_num
from rlinf.utils.placement import HybridComponentPlacement
import random


class MultiStepRolloutWorker(Worker):
    def __init__(self, cfg: DictConfig):
        Worker.__init__(self)

        self.cfg = cfg
        self._env_group_name = cfg.env.group_name
        self._actor_group_name = cfg.actor.group_name
        self.device = torch.cuda.current_device()

        self._obs_queue_name = cfg.env.channel.queue_name
        self._action_queue_name = cfg.rollout.channel.queue_name
        self._replay_buffer_name = cfg.actor.channel.queue_name
        self.stage_num = cfg.rollout.pipeline_stage_num

        self._component_placement = HybridComponentPlacement(cfg, Cluster())
        self.channel = self.connect_channel(cfg.rollout.channel.name)

    def init_worker(self):
        # NOTE:
        # because pi series have some different dtype params, we can not call `to`
        # after get_model, here we simply change actor.model.precision to rollout.precision
        # and after get_model we change it back. THIS CODE SHOULD BE REFACTORED SOON.
        with open_dict(self.cfg):
            original_precision = self.cfg.actor.model.precision
            self.cfg.actor.model.precision = self.cfg.rollout.precision
        self.hf_model = get_model(self.cfg.rollout.model_dir, self.cfg.actor.model)
        with open_dict(self.cfg):
            self.cfg.actor.model.precision = original_precision

        self.hf_model.eval()

        self.setup_sample_params()
        if self.cfg.rollout.get("enable_offload", False):
            self.offload_model()

    def setup_sample_params(self):
        # length parameters for rollout
        self._length_params = OmegaConf.to_container(
            self.cfg.algorithm.length_params, resolve=True
        )
        # sampling parameters for rollout
        self._sampling_params = OmegaConf.to_container(
            self.cfg.algorithm.sampling_params, resolve=True
        )
        self._train_sampling_params = {
            "temperature": self._sampling_params["temperature_train"],
            "top_k": self._sampling_params["top_k"],
            "top_p": self._sampling_params["top_p"],
            "max_new_tokens": self._length_params["max_new_token"],
            "use_cache": True,
        }

        self._eval_sampling_params = {
            "temperature": self._sampling_params["temperature_eval"],
            "top_k": self._sampling_params["top_k"],
            "top_p": self._sampling_params["top_p"],
            "max_new_tokens": self._length_params["max_new_token"],
        }

    def predict(self, env_obs, do_sample=True, mode="train", need_infer=False,  need_wm = False, state_mode = None):
        kwargs = (
            self._train_sampling_params
            if mode == "train"
            else self._eval_sampling_params
        )
        kwargs["do_sample"] = do_sample

        if self.cfg.actor.model.model_name in ["openpi"]:
            kwargs = {"mode": mode}
            
        kwargs["need_infer"] = need_infer
        kwargs["need_wm"] = need_wm
        kwargs["state_mode"] = state_mode

        with torch.no_grad():
            actions, result = self.hf_model.predict_action_batch(
                env_obs=env_obs,
                **kwargs,
            )

        return actions, result

    def update_env_output(self, i, env_output):
        # first step for env_batch
        if self.model_generated_count >= self.model_generated_max_count:
            wm_steps = torch.ones_like(env_output["dones"])
        else:
            wm_steps = torch.zeros_like(env_output["dones"])
        
        self.buffer_list[i].wm_steps.append(wm_steps.contiguous().cpu())
        
        if env_output["rewards"] is None:
            self.buffer_list[i].dones.append(env_output["dones"].contiguous().cpu())
            return

        self.buffer_list[i].rewards.append(env_output["rewards"].cpu().contiguous())         
        
        self.buffer_list[i].dones.append(env_output["dones"].bool().cpu().contiguous())

        # Note: currently this is not correct for chunk-size>1 with partial reset
        if env_output["dones"].any() and self.cfg.env.train.auto_reset:
            if hasattr(self.hf_model, "value_head"):
                dones = env_output["dones"]

                final_obs = env_output["final_obs"]
 
                with torch.no_grad():

                    actions, result = self.predict(final_obs)
                    if "prev_values" in result:
                        _final_values = result["prev_values"]
                    else:
                        _final_values = torch.zeros_like(actions[:, 0])
                final_values = torch.zeros_like(_final_values[:, 0])  # [bsz, ]
                last_step_dones = dones[:, -1]  # [bsz, ]

                final_values[last_step_dones] = _final_values[:, 0][last_step_dones]

                self.buffer_list[i].rewards[-1][:, -1] += (
                    self.cfg.algorithm.gamma * final_values.cpu()
                )

    def generate(self):
        if self.cfg.rollout.get("enable_offload", False):
            self.reload_model()
        self.buffer_list = [EmbodiedRolloutResult() for _ in range(self.stage_num)]

        self.state_mode = 'real_world'               # The initial state is real_world, with possible values: 'real_world' or 'model_generated'
        self.model_generated_count = 0               # Record the number of times the model_generated state is used.
        self.model_generated_max_count = 1           # The number of times model_generated follows each real_world instance.
        self.get_next_data = True                    # Whether get data from offline dataset.
        self.template_lerobot_data = None
        self.use_gt_action = False # In the real-world state, whether input offline actions to the world model?
        self.use_expert_data_prob = 0.6 # the probabilty of only using offline samples to train the model.

        for _ in tqdm(
            range(self.cfg.algorithm.rollout_epoch),
            desc="Generating Rollout Epochs",
            disable=(self._rank != 0),
        ):
            env_step_num =0
            while env_step_num < self.cfg.algorithm.n_chunk_steps: # There must be n_chunk_steps interactions, and match the env.interact.
                for i in range(self.stage_num):  # * stage_num is 1
                    env_output = self.recv_env_output()
                    
                    # Determine the environment output based on the current state mode
                    if self.state_mode == 'model_generated':
                        self.model_generated_count += 1
                        
                    self.update_env_output(i, env_output)

                    need_infer= True
                    need_wm= True
                    
                    if self.state_mode == 'real_world' and self.cfg.actor.model.openpi.train_mode == "IL":
                        if self.use_gt_action:
                            need_infer= False
                        
                        if env_step_num == self.cfg.algorithm.n_chunk_steps - 1: # If it is the final step, and if it's the real world state, then there's no need to go through the world model.
                            need_wm = False
                            need_infer= False
                        
                        if torch.rand(()) < self.use_expert_data_prob:
                            need_wm = False # Without using world model, that is, using Lerobot to collect actions and advantages offline.
                            need_infer= False # Without world model, there's no need for reasoning policies.

                    actions, result = self.predict(env_output["obs"], need_infer=need_infer, need_wm=need_wm, state_mode=self.state_mode)

                    self.buffer_list[i].append_result(result)

                    # State transition logic
                    if self.state_mode == 'real_world':
                        result['forward_inputs']['is_expert_step'] = torch.ones(env_output["dones"].shape[0])
                        # * need_infer = False
                        # After obtaining the image from the real_world state, switch to the model_generated state.
                        if any(key.startswith('next') for key in result.keys()) and self.model_generated_max_count > 0:
                            
                            # Switch to the model_generated state and reset the counter.
                            self.state_mode = 'model_generated'
                            self.model_generated_count = 0

                            # real_world -> model_generated
                            self.get_next_data = False
                        else:
                            # no using world model
                            self.get_next_data = True
                        
                        if need_infer:
                            # * replace ground-truth actions --> policy inference actions.
                            result['forward_inputs']["gt_actions"] = \
                                result['forward_inputs']["actions"]

                    elif self.state_mode == 'model_generated':
                        result['forward_inputs']['is_expert_step'] = torch.zeros(env_output["dones"].shape[0])
                        # model_generated state logic: Check if the maximum number of model_generated iterations has been reached.
                        if self.model_generated_count >= self.model_generated_max_count:
                            # Switch back to real_world state
                            self.state_mode = 'real_world'
   
                            # model_generated -> real_world
                            self.get_next_data = True
                        else:
                            # keep model_generated

                            # model_generated -> model_generated
                            self.get_next_data = False
                        

                        # * replace ground-truth actions --> policy inference actions.
                        result['forward_inputs']["gt_actions"] = \
                               result['forward_inputs']["actions"]
                        
                    else:
                        raise NotImplementedError(f"Unknown state mode: {self.state_mode}")

                    if 'conditional_advantage' in result.keys():
                        result['forward_inputs']["conditional_advantage"] = \
                                    result['conditional_advantage']
                    
                    
                    # whether need to get offline data
                    if self.get_next_data:
                        result = None
                        # * result is None     --> get data from offline dataset
                        # * result is not None --> get data from world model rollout
                    
                    env_step_num += 1
                    self.send_chunk_actions([actions, result])

            for i in range(self.stage_num):
                env_output = self.recv_env_output()
                self.update_env_output(i, env_output)
                if self.cfg.actor.model.openpi.train_mode != "IL":
                    actions, result = self.predict(env_output["obs"])
                        
                    if "prev_values" in result:
                        self.buffer_list[i].prev_values.append(
                            result["prev_values"].cpu().contiguous()
                        )
                else: 
                    if self.cfg.actor.model.openpi.train_mode == "IL" and self.cfg.actor.model.openpi.replace_wm_sample:
                        actions, result = self.predict(env_output["obs"], need_infer=False, env_step_num=env_step_num, state_mode=self.state_mode)
                        
                        result['forward_inputs']['is_expert_step'] = torch.ones(env_output["dones"].shape[0])
                        
                        result['forward_inputs']["conditional_advantage"] = result['conditional_advantage']
                        
                        expert_step_zero_indices = []
                        for idx, forward_inputs in enumerate(self.buffer_list[i].forward_inputs):
                            is_expert_step = forward_inputs.get('is_expert_step', None)
                            if is_expert_step is not None:
                                first_value = is_expert_step[0].item() if hasattr(is_expert_step[0], "item") else is_expert_step[0]
                                if first_value == 0:
                                    expert_step_zero_indices.append(idx)
                        if len(expert_step_zero_indices) > 0:
                            num_replace = random.choice([i for i in range(len(expert_step_zero_indices))])
                            sampled_indices = random.sample(expert_step_zero_indices, num_replace)
                        else:
                            sampled_indices = []
                        for idx in sampled_indices:
                            self.buffer_list[i].forward_inputs[idx] = put_tensor_cpu(result["forward_inputs"])
                    
                    self.buffer_list[i].prev_values.append(torch.zeros_like(self.buffer_list[i].prev_values[0]))
                
        for i in range(self.stage_num):
            self.send_rollout_batch(i)

        if self.cfg.rollout.get("enable_offload", False):
            self.offload_model()

    def evaluate(self):
        if self.cfg.rollout.get("enable_offload", False):
            self.reload_model()

        for _ in range(self.cfg.algorithm.n_eval_chunk_steps):
            for _ in range(self.stage_num):
                env_output = self.recv_env_output()
                actions, _ = self.predict(env_output["obs"], mode="eval")
                self.send_chunk_actions(actions)

        if self.cfg.rollout.get("enable_offload", False):
            self.offload_model()

    def offload_model(self):
        self.hf_model = self.hf_model.to("cpu")
        # if self.hf_model.config.add_reward_model:
        #     self.hf_model.reward_model.model = self.hf_model.reward_model.model.to("cpu")
        if self.hf_model.config.add_dynamics_model:
            self.hf_model.dynamics_model.diffusion_model = self.hf_model.dynamics_model.diffusion_model.to("cpu")
            self.hf_model.dynamics_model.vae = self.hf_model.dynamics_model.vae.to("cpu")
            self.hf_model.dynamics_model.textencoder = self.hf_model.dynamics_model.text_encoder.to("cpu")
        gc.collect()
        torch.cuda.empty_cache()

    def reload_model(self):
        self.hf_model = self.hf_model.to(self.device)
        # if self.hf_model.config.add_reward_model:
        #     self.hf_model.reward_model.model = self.hf_model.reward_model.model.to(self.device)
        if self.hf_model.config.add_dynamics_model:
            self.hf_model.dynamics_model.diffusion_model = self.hf_model.dynamics_model.diffusion_model.to(self.device)
            self.hf_model.dynamics_model.vae = self.hf_model.dynamics_model.vae.to(self.device)
            self.hf_model.dynamics_model.text_encoder = self.hf_model.dynamics_model.text_encoder.to(self.device)

    # def sync_model_from_actor(self):
    #     param_state_dict = self.recv(self._actor_group_name, src_rank=self._rank)

    #     self.hf_model.load_state_dict(param_state_dict)
    #     del param_state_dict
    #     gc.collect()
    #     torch.cuda.empty_cache()
    
    
    def sync_model_from_actor(self):
        # Receive the actor's state dict
        
        if self.hf_model.config.rollout_ema_decay is False:
            param_state_dict = self.recv(self._actor_group_name, src_rank=self._rank)

            self.hf_model.load_state_dict(param_state_dict)
            del param_state_dict
        
        else:
            actor_state_dict = self.recv(self._actor_group_name, src_rank=self._rank)

            # EMA Decay Factor (tau)
            # A value closer to 1 (e.g., 0.995) makes the rollout policy more stable/slow-changing.
            # You can fetch this from your config: self.cfg.rollout.get("ema_decay", 0.995)
            # ema_decay = 0.995
            ema_decay = self.hf_model.config.rollout_ema_decay
            
            missing_keys = []

            with torch.no_grad():
                # 1. Apply EMA to Trainable Parameters
                for name, param in self.hf_model.named_parameters():
                    if name in actor_state_dict:
                        # Move source weight to correct device (e.g., if actor sent from CPU)
                        actor_param_val = actor_state_dict[name].to(param.device)
                        
                        # In-place update: param = decay * param + (1 - decay) * actor_param
                        param.data.mul_(ema_decay).add_(actor_param_val, alpha=(1 - ema_decay))
                    else:
                        # Case: Parameter exists in Rollout but Actor didn't send it.
                        # Strategy: Keep current Rollout weight (effectively decay=1.0).
                        # This is common if Actor only sends trainable parameters (e.g., LoRA).
                        missing_keys.append(name)
                        
                
                # 2. Hard-copy Buffers (e.g., BatchNorm running stats)
                # It is standard practice to copy buffers directly rather than averaging them
                for name, buffer in self.hf_model.named_buffers():
                    if name in actor_state_dict:
                        buffer.data.copy_(actor_state_dict[name])
                        
            if len(missing_keys) > 0:
                # Only print this if you expect a FULL state dict every time.
                # If using LoRA/Adapters, this list will be large (frozen layers), so you might want to suppress it.
                print(f"[Warning] {len(missing_keys)} parameters were not updated (missing from Actor dict).")

            del actor_state_dict
        
        gc.collect()
        torch.cuda.empty_cache()

    def recv_env_output(self):
        env_output = self.channel.get(
            key=f"{self._obs_queue_name}_{self._rank}",
        )
        return env_output

    def send_chunk_actions(self, chunk_actions):
        self.channel.put(
            item=chunk_actions,
            key=f"{self._action_queue_name}_{self._rank}",
        )

    def send_rollout_batch(self, stage_id):
        # send rollout_batch to actor
        send_num = self._component_placement.get_world_size("rollout") * self.stage_num
        recv_num = self._component_placement.get_world_size("actor")
        split_num = compute_split_num(recv_num, send_num)
        splited_rollout_result = self.buffer_list[stage_id].to_splited_dict(split_num)
        for i in range(split_num):
            self.channel.put(
                item=splited_rollout_result[i],
                key=self._replay_buffer_name,
            )

    def set_global_step(self, global_step):
        if hasattr(self.hf_model, "set_global_step"):
            self.hf_model.set_global_step(global_step)
