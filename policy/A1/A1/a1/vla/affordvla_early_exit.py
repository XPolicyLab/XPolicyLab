import torch
import torch.nn as nn
import torch.distributed as dist

from typing import Optional, Sequence, Tuple, List, NamedTuple, Callable
import math
import logging
import random

from os.path import join
from pathlib import Path

from a1.torch_util import ensure_finite_
from a1.model import (
    Molmo, get_causal_attention_bias, should_checkpoint_block, BufferCache,OLMoBlock,MolmoVisionBackbone,
    OLMoBlockGroup
    )
from a1.config import (
    ModelConfig,FSDPWrapStrategy,CheckpointType
)

from a1.vla.action_heads import (L1RegressionActionHead, DiffusionTransformerActionHead,DiffusionActionHead,
                                    FlowMatchingActionHead)
from a1.vla.projectors import ProprioProjector
from a1.vla.dit.blocks import DiTBlock,FinalLayer,TimestepEmbedder
from a1.vla.dit.model import DiT

from a1.tokenizer import DEFAULT_ACT_START_TOKEN, DEFAULT_ACT_END_TOKEN, get_special_token_ids 

from a1.image_vit import ResidualAttentionBlock,VisionTransformer
from a1.exceptions import OLMoConfigurationError
# from torch.distributed.fsdp import FullyShardedDataParallel as FSDP  
from a1.aliases import PathOrStr

from a1.safetensors_util import safetensors_file_to_state_dict
from a1.util import resource_path

# from a1.vla.util import FIXED_ACTION_DIM 

log = logging.getLogger(__name__)


class OLMoWithLastOutput(NamedTuple):
    logits: torch.FloatTensor
    """
    A tensor of shape `(batch_size, seq_len, vocab_size)` representing the log probabilities
    for the next token *before* normalization via (log) softmax.
    """

    attn_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]]
    """
    Attention keys and values from each block.
    """

    hidden_states: Optional[Tuple[torch.Tensor]]
    """
    Hidden states from each block.
    """

    last_hidden_state: Optional[torch.Tensor]
    """
    The last hidden state of the model, which is the output of the final layer.
    Shape: `(batch_size, seq_len, d_model)`.
    """

    exit_layer: Optional[int]
    """
    The layer at which the model exited.
    """

    exit_action: Optional[torch.Tensor] = None
    """
    The action at which the model exited.
    """


class AffordVLAEarlyExit(Molmo):
    def __init__(self, config: ModelConfig):
        
        super().__init__(config)
        self.action_head_type = config.action_head
        if self.action_head_type == 'l1_regression':
            self.action_head = L1RegressionActionHead(input_dim=self.config.d_model,hidden_dim=self.config.d_model,action_dim=config.action_dim)
        elif self.action_head_type == 'diffusion_openvla':
            self.action_head = DiffusionActionHead(
                input_dim=self.config.d_model,hidden_dim=self.config.d_model,action_dim=config.action_dim,num_diffusion_steps=config.num_diffusion_steps
            )

        elif self.action_head_type == 'diffusion':

            self.diff_pred_type = 'sample'  # 'epsilon' or 'sample'
            self.action_head = DiffusionTransformerActionHead(action_dim=config.action_dim,action_horizon=config.num_actions_chunk,hidden_dim=self.config.action_head_dit_hidden_size,
                                                                depth=self.config.action_head_dit_depth,
                                                                num_heads=self.config.action_head_dit_num_heads,
                                                                cond_len=config.action_dim*config.num_actions_chunk,cond_dim=self.config.d_model,
                                                                num_diffusion_steps=self.config.num_diffusion_steps,
                                                                num_diffusion_inference_steps=self.config.num_diffusion_inference_steps,
                                                                pred_type=self.diff_pred_type,)
        elif self.action_head_type == 'flow_matching':
            # flow matching expert head that cross-attends to prefix from main LLM
            self.action_head = FlowMatchingActionHead(
                llm_dim=self.config.d_model,
                action_dim=config.fixed_action_dim,  
                proprio_dim=config.proprio_dim,
                horizon=config.num_actions_chunk,
                qwen2_hidden_size=getattr(self.config, 'action_head_flow_matching_dim', 896),
                qwen2_num_layers=getattr(self.config, 'action_head_flow_matching_layers', 18),
                qwen2_num_heads=getattr(self.config, 'action_head_flow_matching_heads', 8),
                qwen2_intermediate_size=getattr(self.config, 'action_head_flow_matching_intermediate_size', 4096),
                qwen2_num_kv_heads=4,
            )

        if config.use_proprio:
            if config.proprio_dim != config.action_dim:
                print(f"config.proprio_dim {config.proprio_dim} does not match config.action_dim {config.action_dim} for AffordVLA")
            self.proprio_projector = ProprioProjector(llm_dim=config.d_model,proprio_dim=config.proprio_dim)
        else:
            self.proprio_projector = None

        self.__cache = BufferCache()
        # self.__num_fwd_flops: Optional[int] = None

        self.config = config
        self.tokenizer = config.get_tokenizer() 
       

    @staticmethod
    def get_act_head_parameters():
        return tuple(["action_head",])

    @staticmethod
    def get_proprio_proj_parameters():
        return tuple(["proprio_projector",])

    # use forward method rather than wrap as forward_with_actions because of the bug of FSDP
    def forward_with_actions(self, input_ids, target_actions,images=None, **kwargs):  
        assert target_actions is not None, "target_actions must be provided for action prediction"
        # batch_size = target_actions.shape[0]
        # device = target_actions.device

        # self.action_head.to(input_ids.device)  
        # Remove the output_hidden_states parameter to avoid storing all intermediate layers
        kwargs.pop('output_hidden_states', None)  
        
        outputs = self.forward(input_ids, images=images, **kwargs)  
        
        # output of transformer last layer, obtained before the logits calculation
        last_hidden_state = outputs.last_hidden_state  
        
        # extract action hidden states  (batch_size, chunk_len * action_dim, hidden_dim)
        action_hidden_states = self.extract_action_hidden_states(last_hidden_state, input_ids,kwargs["proprio_token_idx"])  
        # 打印 actions_hidden_states 在 action 段是否不同：
        # print(action_hidden_states[0, :7, :4])      # 第0步7维的前4个hidden
        # print(action_hidden_states[0, 7:14, :4])    # 第1步7维的前4个hidden

        # with FSDP.summon_full_params(self.action_head):  
        if self.action_head_type == 'l1_regression':
            predicted_actions = self.action_head.predict_action(action_hidden_states)  

        # elif self.action_head_type == 'diffusion':
        #     noisy_dict = self.action_head.sample_noisy_actions(target_actions)
        #     noise, noisy_actions, diffusion_timestep_embeddings = (
        #         noisy_dict["noise"],
        #         noisy_dict["noisy_actions"],
        #         noisy_dict["diffusion_timestep_embeddings"],
        #     )

        #     noise_pred = self.action_head.predict_noise(action_hidden_states)
        #     assert noise_pred.shape == noise.shape, f"Noise prediction shape {noise_pred.shape} does not match noise shape {noise.shape}. "
        #     # Get diffusion noise prediction MSE loss
        #     noise_pred = noise_pred.reshape(noise.shape)
        #     predicted_actions = None
        
        # use dit model
        elif self.action_head_type == 'diffusion':
            noise, noisy_actions, timesteps = self.action_head.sample_noisy_actions(target_actions)
            pred = self.action_head.predict_noise_or_sample(noisy_actions,timesteps,action_hidden_states)
            
            pred_type = self.diff_pred_type 
            if pred_type == 'epsilon':
                target = noise
            elif pred_type == 'sample':
                target = target_actions

            predicted_actions = None

        return {  
            # 'logits': outputs.logits,  
            'outputs': outputs,
            'predicted_actions': predicted_actions,  
            'diffusion_target': target if self.action_head_type == 'diffusion' else None,
            'diffusion_pred': pred if self.action_head_type == 'diffusion' else None,
            'noisy_actions': noisy_actions if self.action_head_type == 'diffusion' else None,
            'diff_timesteps': timesteps if self.action_head_type == 'diffusion' else None,
            # 'last_hidden_state': last_hidden_state  
        }

    def predict_actions_by_hidden_states_by_idx(self, hidden_states, start_idx, end_idx):
        action_hidden_states = hidden_states[:, start_idx+1:end_idx, :]  
        if self.action_head_type == 'l1_regression':
            predicted_actions = self.action_head.predict_action(action_hidden_states)  
        elif self.action_head_type == 'diffusion':
            predicted_actions = self.action_head.condition_sampling(action_hidden_states)
        else:
            raise ValueError(f"Unsupported action head type: {self.action_head_type}")
        # elif self.action_head_type == 'flow_matching':
        #     # Build prefix hidden from main LLM, then run expert-only Euler steps
        #     # prefix hidden = tokens before action start (inclusive)
        #     assert start_idx >= 0, "Action start token not found"
        #     prefix_hidden = hidden_states[:, : start_idx + 1, :]  # (B,P,H)

        #     # Euler steps with expert only
        #     device = hidden_states.device
        #     dtype = hidden_states.dtype
        #     B = hidden_states.shape[0]
        #     steps = getattr(self.config, 'num_diffusion_inference_steps', 10)
        #     dt = -1.0 / float(steps)
        #     x = torch.randn((B, self.config.num_actions_chunk, self.config.action_dim), device=device, dtype=dtype)
        #     t_float = 1.0
        #     # use proprio as state if configured
        #     assert self.config.use_proprio, "flow_matching requires use_proprio=True for state token"
        #     # the caller must pass action_proprio via kwargs in predict stage
        #     state = proprio
        #     assert state is not None, "action_proprio is required for flow_matching inference"
        #     for _ in range(steps):
        #         t = torch.full((B,), t_float, device=device, dtype=dtype)
        #         # 无缓存场景，传入 past_key_values=None，并使用起始位置作为 position 偏移
        #         pos_offset = torch.full((B,), start_idx + 1, dtype=torch.long, device=device)
        #         v = self.action_head.predict_vector_field(None, state, x, t, pos_offset=pos_offset)
        #         x = x + dt * v
        #         t_float += dt
        #     predicted_actions = x
        return predicted_actions
    
    def predict_actions_flow_matching(self, attn_key_values, proprio, pos_offset,input_x=None):
        if not attn_key_values or len(attn_key_values) == 0:
            raise ValueError("attn_key_values is empty; flow_matching requires KV cache from forward()")
        if not isinstance(attn_key_values[0], (tuple, list)) or len(attn_key_values[0]) == 0:
            raise ValueError("attn_key_values[0] is invalid; expected (key, value) tuple")
        device = attn_key_values[0][0].device
        dtype = attn_key_values[0][0].dtype
        B = attn_key_values[0][0].shape[0]
        steps = getattr(self.config, 'num_diffusion_inference_steps', 10)
        dt = -1.0 / float(steps)
        # 使用上一层传递过来的noise action作为初始状态
        if input_x is None:
            x = torch.randn((B, self.config.num_actions_chunk, self.config.fixed_action_dim), device=device, dtype=dtype)
        else:
            x = input_x.to(device, dtype)
        t_float = 1.0
        # use proprio as state if configured
        assert self.config.use_proprio, "flow_matching requires use_proprio=True for state token"
        # the caller must pass action_proprio via kwargs in predict stage
        state = proprio
        assert state is not None, "action_proprio is required for flow_matching inference"

        for _ in range(steps):
            t = torch.full((B,), t_float, device=device, dtype=dtype)
            v = self.action_head.predict_vector_field(attn_key_values, state, x, t, pos_offset=pos_offset)
            x = x + dt * v
            t_float += dt
        predicted_actions = x
        return predicted_actions

        
    def predict_actions(self,input_ids,images=None, **kwargs):
        if self.action_head_type == 'diffusion_openvla':
            return self.run_diffusion_sampling(input_ids,images, **kwargs)

        kwargs.pop('output_hidden_states', None)  


        if self.action_head_type == 'l1_regression':
            outputs = self.forward(input_ids, images=images, **kwargs)  
            exit_action = outputs.exit_action
            if exit_action is not None:
                return exit_action
            else:
                last_hidden_state = outputs.last_hidden_state  
                # extract action hidden states  (batch_size, chunk_len * action_dim, hidden_dim)
                action_hidden_states = self.extract_action_hidden_states(last_hidden_state, input_ids,kwargs["proprio_token_idx"])  
                predicted_actions = self.action_head.predict_action(action_hidden_states)  
        elif self.action_head_type == 'diffusion':
            outputs = self.forward(input_ids, images=images, **kwargs)  
            last_hidden_state = outputs.last_hidden_state  
            action_hidden_states = self.extract_action_hidden_states(last_hidden_state, input_ids,kwargs["proprio_token_idx"])  
            predicted_actions = self.action_head.condition_sampling(action_hidden_states)
        elif self.action_head_type == 'flow_matching':
            # Build prefix hidden from main LLM, then run expert-only Euler steps
            kwargs.pop('use_cache', None)
            outputs = self.forward(input_ids, images=images,use_cache=True, **kwargs)
            exit_action = outputs.exit_action
            if exit_action is not None:
                return exit_action
            else:
                raise ValueError("No exit action found")
            # past_key_values = outputs.attn_key_values

            # # Euler steps with expert only
            # device = input_ids.device
            # dtype = outputs.last_hidden_state.dtype
            # B = input_ids.shape[0]
            # steps = getattr(self.config, 'num_diffusion_inference_steps', 10)
            # dt = -1.0 / float(steps)
            # x = torch.randn((B, self.config.num_actions_chunk, FIXED_ACTION_DIM), device=device, dtype=dtype)
            # t_float = 1.0
            # # use proprio as state if configured
            # assert self.config.use_proprio, "flow_matching requires use_proprio=True for state token"
            # # the caller must pass action_proprio via kwargs in predict stage
            # state = kwargs.get('action_proprio', None)
            # assert state is not None, "action_proprio is required for flow_matching inference"
            # # 基于 prefix_pad_masks 计算每个样本的有效前缀长度，避免 padding 干扰位置编码
            # if 'prefix_pad_masks' in kwargs and kwargs['prefix_pad_masks'] is not None:
            #     ppm = kwargs['prefix_pad_masks']  # bool[B, P]
            #     pos_offset = ppm.to(torch.int64).sum(dim=1)  # (B,)
            # else:
            #     # 回退：直接用 input_ids 的非 -1 个数作为有效前缀长度
            #     pos_offset = (input_ids != -1).to(torch.int64).sum(dim=1)
            # for _ in range(steps):
            #     t = torch.full((B,), t_float, device=device, dtype=dtype)
            #     v = self.action_head.predict_vector_field(past_key_values, state, x, t, pos_offset=pos_offset)
            #     x = x + dt * v
            #     t_float += dt
            # predicted_actions = x
        
        return predicted_actions

    def get_action_idx(self, input_tokens, proprio_token_idx ):
        # batch_size, seq_len = input_tokens.shape  
        start_idx = proprio_token_idx + 2
        end_idx = start_idx + self.config.num_actions_chunk * self.config.action_dim + 1
        # assert end_idx>0 and start_idx>=0 and end_idx <= seq_len, f"Action start pos {start_idx} and end positions {end_idx} must be valid indices in the input sequence"
        return start_idx, end_idx

    def extract_action_hidden_states_by_idx(self, hidden_states, start_idx, end_idx):
        """Extract the vectors corresponding to action tokens from the last layer's hidden states in the transformer"""  
        batch_size, seq_len, hidden_dim = hidden_states.shape  
        assert end_idx>0 and start_idx>=0 and start_idx<end_idx and start_idx<seq_len, \
                        f"Action start pos {start_idx} and end positions {end_idx} must be valid indices in the input sequence"
        actions_hidden_states = hidden_states[:, start_idx+1:end_idx, :]  
        return actions_hidden_states

    def extract_action_hidden_states(self, hidden_states, input_tokens,proprio_token_idx):  
        """Extract the vectors corresponding to action tokens from the last layer's hidden states in the transformer"""  
        # self.find_action_token_positions(input_tokens)
        # batch_size, seq_len, hidden_dim = hidden_states.shape  

        # Assert that the first token is the action start token  
        special_tokens = get_special_token_ids(self.tokenizer)  
        action_start_token_id = special_tokens[DEFAULT_ACT_START_TOKEN]  
        action_end_token_id = special_tokens[DEFAULT_ACT_END_TOKEN]  

        start_idx, end_idx = self.get_action_idx(input_tokens,proprio_token_idx)

        # Extract action hidden states for all batches at once  
        actions_hidden_states = hidden_states[:, start_idx+1:end_idx, :]  
        action_input_tokens = input_tokens[:, start_idx:end_idx+1]

        # print(f"*******************action input tokens shape: {action_input_tokens.shape}")
        assert (action_input_tokens[:,0] == action_start_token_id).all(), "Expected action start token in the first position of action tokens"
        assert (action_input_tokens[:,-1] == action_end_token_id).all(), "Expected action end token in the last position of action tokens"

        return actions_hidden_states


    def find_action_token_positions(self, input_tokens):  
        """查找 action start 和 end token 在输入序列中的位置"""  
        
        # 获取特殊标记的 ID  
        tokenizer = self.config.get_tokenizer()  
        special_tokens = get_special_token_ids(tokenizer)  
        action_start_token_id = special_tokens[DEFAULT_ACT_START_TOKEN]  
        action_end_token_id = special_tokens[DEFAULT_ACT_END_TOKEN]  
        
        batch_size, seq_len = input_tokens.shape  
        
        # 查找每个样本中的 action start 和 end 位置  
        start_positions = []  
        end_positions = []  
        
        for batch_idx in range(batch_size):  
            # 查找 action_start_token_id 的位置  
            start_mask = (input_tokens[batch_idx] == action_start_token_id)  
            start_pos = torch.nonzero(start_mask, as_tuple=False)  
            
            # 查找 action_end_token_id 的位置  
            end_mask = (input_tokens[batch_idx] == action_end_token_id)  
            end_pos = torch.nonzero(end_mask, as_tuple=False)  
            
            if len(start_pos) > 0:  
                print('! len(start_pos) > 0',start_pos) if len(start_pos) > 1 else None
                start_positions.append(start_pos[-1].item())  # 取最后一个 start token  
            else:  
                start_positions.append(-1)  # 未找到  
                
            if len(end_pos) > 0:  
                print('! len(end_pos) > 0',end_pos) if len(end_pos) > 1 else None
                end_positions.append(end_pos[-1].item())  # 取最后一个 end token  
            else:  
                end_positions.append(-1)  # 未找到  

        
        return start_positions, end_positions  

    def debug_module_hierarchy(self):  
        print("=== Module Hierarchy Debug ===")  
        # for name, module in self.named_modules():  
        #     print(f"Module: {name} -> {type(module).__name__}")  
        #     if hasattr(module, '_is_root'):  
        #         print(f"Module: {name}  _is_root: {module._is_root}")  
        #     if isinstance(module, torch.distributed.fsdp.FullyShardedDataParallel):  
        #         print(f"Module: {name}  FSDP wrapped: True")  
        for name, module in self.named_modules():
            print(f"Module: {name} -> {type(module).__name__}")
            # print the module's  dtype

        print("=" * 50)


    # def extract_action_hidden_states_vectorized(self, hidden_states, action_start_pos, action_end_pos):  
    #     batch_size, seq_len, hidden_dim = hidden_states.shape  
        
    #     # 假设所有样本的 action 长度相同  
    #     action_length = action_end_pos[0] - action_start_pos[0]  
        
    #     # 创建索引矩阵  
    #     batch_indices = torch.arange(batch_size, device=hidden_states.device).unsqueeze(1)  # (batch_size, 1)  
    #     position_offsets = torch.arange(action_length, device=hidden_states.device).unsqueeze(0)  # (1, action_length)  
        
    #     # 计算每个样本的绝对位置索引  
    #     absolute_positions = action_start_pos.unsqueeze(1) + position_offsets  # (batch_size, action_length)  
        
    #     # 使用高级索引提取  
    #     actions_hidden_states = hidden_states[batch_indices, absolute_positions]  # (batch_size, action_length, hidden_dim)  
        
    #     return actions_hidden_states

    def set_size_module_is_root_false(self):
        size_based_modules = {self.transformer.wte, self.transformer.ff_out, self.transformer.ln_f}
        if self.vision_backbone is not None and self.config.vision_backbone.fsdp_wrap:
            size_based_modules.add(self.vision_backbone.image_pooling_2d)
            size_based_modules.add(self.vision_backbone.image_projector)

        size_based_modules.add(self.vision_backbone)
        size_based_modules.add(self.transformer.blocks[0])

        for module in size_based_modules:
            if hasattr(module, "_is_root"):
                module._is_root = False

    def get_fsdp_wrap_policy(self, wrap_strategy: Optional[FSDPWrapStrategy] = None):
        if wrap_strategy is None:
            return None

        # The 'recurse' mode for the wrap function does not behave like you'd expect.
        # Even if we return False, it may still recurse because PyTorch does what it wants,
        # not what you want. This causes issues when, for example, we want to wrap 'ff_out' (a linear layer)
        # but not other linear layers within a block.
        # So we have to explicitly tell PyTorch which linear layers to wrap, and we also just
        # return True in 'recurse' mode for simplicity.
        size_based_module_to_wrap = {self.transformer.wte}
        if hasattr(self.transformer, "ff_out"):
            size_based_module_to_wrap.add(self.transformer.ff_out)
        if hasattr(self.transformer, "ln_f"):
            size_based_module_to_wrap.add(self.transformer.ln_f)
        if self.vision_backbone is not None and self.config.vision_backbone.fsdp_wrap:
            size_based_module_to_wrap.add(self.vision_backbone.image_pooling_2d)
            size_based_module_to_wrap.add(self.vision_backbone.image_projector)
        
            
        # if hasattr(self, 'action_head') and self.action_head_type == 'l1_regression':
        #     size_based_module_to_wrap.add(self.action_head.model)

        if hasattr(self, 'action_head') and self.action_head_type == 'diffusion':
            size_based_module_to_wrap.add(self.action_head.action_adaptor)
            size_based_module_to_wrap.add(self.action_head.condition_adaptor)

        if hasattr(self, 'action_head') and self.action_head_type == 'flow_matching':
            size_based_module_to_wrap.add(self.action_head.qwen2.model)
            size_based_module_to_wrap.add(self.action_head.action_out)
            size_based_module_to_wrap.add(self.action_head.state_proj)
            size_based_module_to_wrap.add(self.action_head.action_in_proj)
            size_based_module_to_wrap.add(self.action_head.action_time_in)
            size_based_module_to_wrap.add(self.action_head.action_time_out)
        
        
        # if hasattr(self, 'proprio_projector'):
        #     size_based_module_to_wrap.add(self.proprio_projector.fc1)
        #     size_based_module_to_wrap.add(self.proprio_projector.fc2)


        # print('*' * 50, 'FSDP Wrap Policy', '*' * 50)
        # 重要：正确处理action_head的包装
        # 不要将action_head添加到size_based_module_to_wrap中，而是在策略函数中明确处理
        # action_head_modules = set()
        # if hasattr(self, 'action_head'):
        #     action_head_modules = set(dict(self.action_head.named_modules()).values())
        # action_head_module_names = {
        #     name for name, _ in self.named_modules()
        #     if name.startswith("action_head")
        # }
        
        #     # 收集action_head及其所有子模块
        #     def collect_action_head_modules(module, modules_set):
        #         modules_set.add(module)
        #         for child in module.children():
        #             collect_action_head_modules(child, modules_set)
        #     collect_action_head_modules(self.action_head, action_head_modules)


        wrap_layer_names = (ResidualAttentionBlock, MolmoVisionBackbone, VisionTransformer,
                            ProprioProjector, DiT)
                            # DiTBlock,ProprioProjector,DiffusionTransformerActionHead,)
                            # ,L1RegressionActionHead, MLPResNet, MLPResNetBlock)

        if wrap_strategy == FSDPWrapStrategy.by_block:

            def fsdp_wrap_fn(module, recurse: bool = True, nonwrapped_numel: int = 0):
                del nonwrapped_numel
                # # 明确排除action_head相关模块
                # if module in action_head_modules:
                #     return False
                # module_name = getattr(module, "_fsdp_wrap_module_name", None)
                # if any(module_name.startswith(name) for name in action_head_module_names):
                #     return False
            
                wrap = isinstance(module, wrap_layer_names + (OLMoBlock,))
                if recurse:
                    return True
                else:
                    return wrap

            return fsdp_wrap_fn
        elif wrap_strategy == FSDPWrapStrategy.by_block_and_size:

            def fsdp_wrap_fn(module, recurse: bool = True, nonwrapped_numel: int = 0):
                del nonwrapped_numel
                
                # # 明确排除action_head相关模块，避免冲突的包装策略
                # if module in action_head_modules:
                #     return False
                # module_name = getattr(module, "_fsdp_wrap_module_name", None)
                # if any(module_name.startswith(name) for name in action_head_module_names):
                #     print(f"[FSDP wrap policy] Skipping wrap for {module_name}")
                #     return False
                wrap = isinstance(module, wrap_layer_names + (OLMoBlock,)) or module in size_based_module_to_wrap
                if recurse:
                    return True
                else:
                    return wrap

            return fsdp_wrap_fn
        elif wrap_strategy == FSDPWrapStrategy.by_block_group:
            if self.config.block_group_size <= 1:
                raise OLMoConfigurationError(
                    "'by_block_group' FSDP wrapping strategy requires block group size greater than 1"
                )

            def fsdp_wrap_fn(module, recurse: bool = True, nonwrapped_numel: int = 0):
                del nonwrapped_numel
                wrap = isinstance(module, wrap_layer_names + (OLMoBlockGroup,))
                if recurse:
                    return True
                else:
                    return wrap

            return fsdp_wrap_fn
        elif wrap_strategy == FSDPWrapStrategy.by_block_group_and_size:
            if self.config.block_group_size <= 1:
                raise OLMoConfigurationError(
                    "'by_block_group_and_size' FSDP wrapping strategy requires block group size greater than 1"
                )

            def fsdp_wrap_fn(module, recurse: bool = True, nonwrapped_numel: int = 0):
                del nonwrapped_numel
                wrap = isinstance(module, wrap_layer_names + (OLMoBlockGroup,)) or module in size_based_module_to_wrap
                if recurse:
                    return True
                else:
                    return wrap

            return fsdp_wrap_fn
        elif wrap_strategy == FSDPWrapStrategy.size_based:
  
            from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy

            return size_based_auto_wrap_policy
        elif wrap_strategy in {
            FSDPWrapStrategy.one_in_two,
            FSDPWrapStrategy.one_in_three,
            FSDPWrapStrategy.one_in_four,
            FSDPWrapStrategy.one_in_five,
        }:
            c = {
                FSDPWrapStrategy.one_in_two: 2,
                FSDPWrapStrategy.one_in_three: 3,
                FSDPWrapStrategy.one_in_four: 4,
                FSDPWrapStrategy.one_in_five: 5,
            }[wrap_strategy]

            def fsdp_wrap_fn(module, recurse: bool = True, nonwrapped_numel: int = 0):
                del nonwrapped_numel

                wrap = isinstance(module, OLMoBlock) and module.layer_id % c == 0
                if recurse:
                    return True
                else:
                    return wrap

            return fsdp_wrap_fn
        else:
            raise NotImplementedError(wrap_strategy)


    def _init_weights(self, module):  
        """初始化权重的辅助方法"""  
        if isinstance(module, torch.nn.Linear):  
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)  
            if module.bias is not None:  
                torch.nn.init.zeros_(module.bias)  
        elif isinstance(module, torch.nn.LayerNorm):  
            torch.nn.init.ones_(module.weight)  
            torch.nn.init.zeros_(module.bias)


    # def get_action_head_parameters(self):  
    #     """获取 action_head 的参数名称，用于参数冻结"""  
    #     return [name for name, _ in self.action_head.named_parameters()]


    def reset_with_pretrained_weights(self):
        if self.config.llm_load_path is None:
            self.reset_non_vision_parameters()
        else:
            state_dict_path = resource_path(
                Path(self.config.llm_load_path).parent, Path(self.config.llm_load_path).name,
                local_cache=Path(self.config.llm_load_path).parent,
            )
            assert state_dict_path.is_file(), f"Model file {str(state_dict_path)} not found"
            if state_dict_path.name.endswith("safetensors"):
                state_dict = safetensors_file_to_state_dict(state_dict_path, map_location="cpu")
            else:
                state_dict = torch.load(state_dict_path, map_location="cpu")
            if all(x.startswith("transformer.") for x in state_dict.keys()):
                state_dict = {k[len("transformer."):]: v for k, v in state_dict.items()}
            if "wte.weight" in state_dict and self.config.additional_vocab_size:
                state_dict["wte.embedding"] = state_dict.pop("wte.weight")
            transformer_keys = set(x[0] for x in self.transformer.named_parameters())

            print(f"****** AffordVLA reset_with_pretrained_weights")
            print(f"***** transformer_keys - set(state_dict.keys()): {transformer_keys - set(state_dict.keys())}")
            assert transformer_keys - set(state_dict.keys()) <= {"wte.new_embedding"}, \
                "Unexpected keys in the model file"
            self.transformer.load_state_dict(state_dict, strict=False)

            if hasattr(self.transformer.wte, "new_embedding"):
                # This is the only parameter not initialized from the LLM weights
                nn.init.normal_(self.transformer.wte.new_embedding, std=self.config.new_embedding_init_range)

        if self.vision_backbone is not None:
            self.vision_backbone.reset_with_pretrained_weights() # need to change for lora

    # early exit related methods
    def get_all_exit_idx(self, exit_interval):
        # must contain the last layer, and do not append the last layer twice 
        exit_idx = list(range(1, self.config.n_layers, exit_interval)) # self.config.exit_interval
        if exit_idx[-1] != self.config.n_layers - 1:
            exit_idx.append(self.config.n_layers - 1)
        return exit_idx

    def get_exit_num(self):
        return len(self.get_all_exit_idx())


    def get_action_start_idx(self,input_ids,action_start_token_id):
        seq_len = input_ids.shape[1]

        start_mask = (input_ids == action_start_token_id)
        start_positions = seq_len - 1 - torch.argmax(start_mask.flip(dims=[1]).float(), dim=1)
        start_found = start_mask.any(dim=1)
        action_start_pos = torch.where(start_found, start_positions, torch.tensor(-1, device=input_ids.device))
        start_idx = action_start_pos[0].item()
        assert start_idx >= 0, "Action start token not found"
        return start_idx


    def forward(
        self,
        input_ids: torch.LongTensor,
        target_actions: torch.FloatTensor = None,
        input_embeddings: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        attention_bias: Optional[torch.Tensor] = None,
        response_mask: Optional[torch.Tensor] = None,
        images: Optional[torch.Tensor] = None,
        image_masks: Optional[torch.Tensor] = None,
        image_input_idx: Optional[torch.Tensor] = None,
        subsegment_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_values: Optional[Sequence[Tuple[torch.Tensor, torch.Tensor]]] = None,
        use_cache: bool = False,
        last_logits_only: bool = False,
        output_hidden_states: Optional[bool] = None,
        append_last_valid_logits: Optional[torch.Tensor] = None,
        action_proprio= None,
        proprio_token_idx = None,
        exit_controller = None, 
        exit_id = None,
        train_exit_random_layer = None,
        log_fn: Optional[Callable[[str], None]] = None,
    ) -> OLMoWithLastOutput:
        """
        :param input_ids: A tensor of shape `(batch_size, seq_len)`.
        :param input_embeddings: A tensor of shape `(batch_size, seq_len, d_model)` with input
            embeddings. When provided, it is treated as the output of the input embedding layer.
        :param attention_mask: A tensor of shape `(batch_size, seq_len)` that indicates
            which input IDs are masked. A `1` value in the mask means that
            the corresponding input ID should *not* be ignored. A `0` means
            that the corresponding input ID is masked.

            This has the same meaning as the `attention_mask` in HuggingFace's `transformers`
            library.
        :param attention_bias: A tensor of shape `(batch_size, 1, seq_len, seq_len)`,
            `(1, 1, seq_len, seq_len)`, or `(seq_len, seq_len)`. This is used
            to introduce causal or other biases.

            If the tensor is a bool or byte tensor, a `True` or `1` at `attention_bias[:, :, i, j]`
            indicates that the i-th element in the sequence is allowed to attend to the j-th
            element in the sequence.

            If the tensor is a float tensor, it will just be added to the attention
            scores before the softmax.

            The default is causal, which corresponds to a lower-diagonal byte matrix of ones.
        :param response_mask: A tensor of shape `(batch_size, seq_len)` that indicates
            the response mask. A `1` value in the mask means that the corresponding token
            is a response token. A `0` means that the corresponding token is not
            a response token.
        :param past_key_values: Pre-computed keys and values for each attention block.
            Can be used to speed up sequential decoding. The `input_ids` which have
            their past given to this model should not be passed as `input_ids` as they have already been computed.
        :param use_cache: If `True`, return key and value tensors for each block.
        :param last_logits_only: If `True`, only compute the logits for the last token of each sequence.
            This can speed up decoding when you only care about the next token.
        """
        # assert target_actions is not None, "target_actions must be provided for action prediction"
        # generate random layer id to train early exit
        train_exit_random_layer_id = random.randint(0, self.config.n_layers - 1)
        # 保证多卡/多进程下各 rank 选择相同的随机层，避免通信不一致
        if train_exit_random_layer is not None and dist.is_available() and dist.is_initialized():
            t = torch.tensor([train_exit_random_layer_id], device=input_ids.device, dtype=torch.int64)
            dist.broadcast(t, src=0)
            train_exit_random_layer_id = int(t.item())


        # origin forward method of super class Molmo
        output_hidden_states = output_hidden_states if output_hidden_states is not None else False

        if past_key_values:
            assert len(past_key_values) == self.config.n_layers

        has_image = images is not None

        assert not (has_image and input_embeddings is not None), "Cannot provide both images and input embeddings."
        assert not (has_image and past_key_values is not None), "Cached key and values should not be used with images."

        batch_size, seq_len = input_ids.size() if input_embeddings is None else input_embeddings.size()[:2]
        if past_key_values is None:
            past_length = 0
        else:
            past_length = past_key_values[0][0].size(-2)

        if self.config.use_position_ids and attention_mask is None:
            attention_mask = input_ids != -1
        
        if subsegment_ids is not None:
            assert not use_cache, "Subsegment_ids cannot be used with cache."
            subsegment_mask = subsegment_ids.unsqueeze(2) <= subsegment_ids.unsqueeze(1)
            attention_mask = (
                subsegment_mask.to(attention_mask.dtype) *
                attention_mask.unsqueeze(2) *
                attention_mask.unsqueeze(1))
            if position_ids is None:
                raise ValueError(f"Positioned ids must be given if using subsegment_ids")
        else:
            if self.config.use_position_ids and position_ids is None:
                position_ids = torch.clamp(
                    torch.cumsum(attention_mask.to(torch.int32), dim=-1) - 1,
                    min=0,
                ).broadcast_to((batch_size, attention_mask.shape[-1]))

        # Get embeddings of input.
        # shape: (batch_size, seq_len, d_model)
        if input_ids is not None:
            input_ids_zero_mask = input_ids * (input_ids != -1).to(input_ids.dtype)
        x = self.transformer.wte(input_ids_zero_mask) if input_embeddings is None else input_embeddings  # type: ignore

        # print('**** AffordVLA forward, embeddings of input, shape: ', x.shape)

        # process proprio embeddings
        proprio_bool = action_proprio is not None and proprio_token_idx is not None and self.proprio_projector is not None
        # print(f"****** AffordVLA forward, action_proprio: {action_proprio is not None}, proprio_token_idx: {proprio_token_idx},self.proprio_projector: {self.proprio_projector is not None}")
        if self.config.use_proprio:
            assert proprio_bool, "Proprioceptive state must be provided when use_proprio is True."

            # make the action_proprio same dtype as self.proprio_projector
            # if action_proprio.dtype != self.proprio_projector.fc1.weight.dtype:
            action_proprio = action_proprio.to(self.proprio_projector.fc1.weight.dtype)
            proprio_features = self.proprio_projector(action_proprio)  # (batch_size, d_model)
            # print(f"****** AffordVLA forward, proprio_features shape: {proprio_features.shape},x shape: {x.shape}")
            x[:,proprio_token_idx[0],:] = proprio_features.squeeze()

        num_image: Optional[int] = None
        if images is not None:
            # shape: (batch_size, num_image, num_patch, d_model)
            # cls_embed: (batch_size, num_image, d_model)
            image_features = self.vision_backbone(images, image_masks)
            num_image, num_patch = image_features.shape[1:3]
            assert image_input_idx.shape == (batch_size, num_image, num_patch)

            # inster the image feature into the embedding.
            image_features = image_features.view(batch_size, num_image * num_patch, -1)
            image_input_idx = image_input_idx.view(batch_size, num_image * num_patch)

            valid = image_input_idx >= 0
            batch_idx = torch.arange(batch_size, device=x.device)
            batch_idx = torch.tile(batch_idx[:, None], [1, image_features.shape[1]])

            # For hf demo/endpoint
            image_features = image_features.to(x.device)

            x[batch_idx[valid], image_input_idx[valid]] += image_features[valid]

        if not self.config.rope:
            # Get positional embeddings.
            # shape: (1, seq_len)
            pos = torch.arange(past_length, past_length + seq_len, dtype=torch.long, device=x.device).unsqueeze(0)
            # shape: (1, seq_len, d_model)
            pos_emb = self.transformer.wpe(pos)  # type: ignore
            x = pos_emb + x

        # Add input + positional embeddings and apply dropout.
        # shape: (batch_size, seq_len, d_model)
        x = self.transformer.emb_drop(x)  # type: ignore

        # normalized
        if self.config.normalize_input_embeds:
            x = x * (self.config.d_model ** 0.5)

        # Transform the attention mask into what the blocks expect.
        if attention_mask is not None:
            # shape: (batch_size, 1, 1, seq_len)
            if len(attention_mask.shape) == 2:
                attention_mask = attention_mask[:, :past_length + seq_len]
                attention_mask = attention_mask.to(dtype=torch.float).view(batch_size, -1)[:, None, None, :]
            else:
                attention_mask = attention_mask.unsqueeze(1).to(dtype=torch.float)
            attention_mask = (1.0 - attention_mask) * torch.finfo(attention_mask.dtype).min

        # Merge attention mask with attention bias.
        if (
            attention_bias is not None
            or attention_mask is not None
            # NOTE (epwalsh): we need to initialize the attn bias in order for attn to work properly
            # with key+value cache. Otherwise `F.scaled_dot_product_attention()` doesn't seem to compute
            # scores correctly.
            or past_key_values is not None
        ):
            if attention_bias is None:
                attention_bias = get_causal_attention_bias(self.__cache, past_length + seq_len, x.device)
            elif attention_bias.dtype in (torch.int8, torch.bool):
                attention_bias = attention_bias.to(dtype=torch.float)
                attention_bias.masked_fill_(attention_bias == 0.0, torch.finfo(attention_bias.dtype).min)

            # Transform to the right shape and data type.
            mask_len = seq_len
            if attention_mask is not None:
                mask_len = attention_mask.shape[-1]
            elif past_key_values is not None:
                mask_len = past_key_values[0][0].shape[-2] + seq_len
            attention_bias = attention_bias[:, :, :mask_len, :mask_len].to(dtype=torch.float)
            
            # no causal bias, just normal self attention @ jian
            if not self.config.llm_causal_attention:
                if self.action_head_type == 'l1_regression':
                    attention_bias = torch.zeros(1, 1, mask_len, mask_len,device=x.device, dtype=torch.float)

            # Add in the masking bias.
            if attention_mask is not None:
                attention_bias = attention_bias + attention_mask
                # Might get -infs after adding attention mask, since dtype.min + dtype.min = -inf.
                # `F.scaled_dot_product_attention()` doesn't handle -inf like you'd expect, instead
                # it can produce NaNs.
                ensure_finite_(attention_bias, check_neg_inf=True, check_pos_inf=False)

        attn_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = [] if use_cache else None
        if self.action_head_type == 'flow_matching':
            attn_key_values = [] 

        if self.action_head_type == 'flow_matching':
            start_idx, end_idx = len(input_ids)-2, len(input_ids)-1 # dummy value
        else:
            start_idx, end_idx = self.get_action_idx(input_ids,proprio_token_idx)
        

        # decoder layers
        all_hidden_states = []

        # Apply blocks one-by-one.
        if self.config.block_group_size == 1:
            for block_idx, block in enumerate(self.transformer.blocks):
                if output_hidden_states:
                    # add hidden states
                    all_hidden_states.append(x)

                layer_past = None if past_key_values is None else past_key_values[block_idx]
                if should_checkpoint_block(self.activation_checkpointing_strategy, block_idx):
                    # shape: (batch_size, seq_len, d_model)
                    x, cache = self._activation_checkpoint_fn(
                        block, x, attention_bias=attention_bias, position_ids=position_ids, drop_mask=response_mask, layer_past=layer_past, use_cache=use_cache
                    )
                else:
                    # shape: (batch_size, seq_len, d_model)
                    x, cache = block(x, attention_bias=attention_bias, position_ids=position_ids, drop_mask=response_mask, layer_past=layer_past, use_cache=use_cache)

                if attn_key_values is not None:
                    assert cache is not None
                    attn_key_values.append(cache)
                
                # early exit for training: 在随机层（含该层）之后提前停止后续层的计算
                if train_exit_random_layer is not None and block_idx >= train_exit_random_layer_id:
                    break

                # early exit for inference
                if exit_id is not None:
                    if exit_id == block_idx:
                        if log_fn is not None:
                            log_fn(f"Exit by exit_id, block_idx: {block_idx}")
                        return OLMoWithLastOutput(logits=None, attn_key_values=attn_key_values, last_hidden_state=x,exit_layer=block_idx,
                                hidden_states=tuple(all_hidden_states) if output_hidden_states else None) 
                
                if exit_controller is not None:
                    pos_offset = (input_ids != -1).to(torch.int64).sum(dim=1)
                    feats_for_exit = attn_key_values if self.action_head_type == 'flow_matching' else all_hidden_states
                    exit_flag, exit_action = exit_controller(feats_for_exit, block_idx, action_proprio, start_idx, end_idx, pos_offset)
                    if exit_flag:
                        # print(f"*** Exit by exit_controller, block_idx: {block_idx}")
                        if log_fn is not None:
                            # 纯文本日志用于外部解析统计
                            log_fn(f"Exit by exit_controller, block_idx: {block_idx}")
                        # 这里可以直接返回action，因为已经在exit_controller的value_net中计算了action
                        return OLMoWithLastOutput(logits=None, attn_key_values=attn_key_values, last_hidden_state=x,exit_layer=block_idx,
                                hidden_states=tuple(all_hidden_states) if output_hidden_states else None, exit_action=exit_action) 
        else:
            for group_idx, block_group in enumerate(self.transformer.block_groups):
                if output_hidden_states:
                    # add hidden states
                    all_hidden_states.append(x)

                layers_past = (
                    None
                    if past_key_values is None
                    else past_key_values[
                        group_idx * self.config.block_group_size : (group_idx + 1) * self.config.block_group_size
                    ]
                )
                x, cache = block_group(
                    x, attention_bias=attention_bias, position_ids=position_ids, drop_mask=response_mask, layers_past=layers_past, use_cache=use_cache
                )
                if attn_key_values is not None:
                    assert cache is not None
                    attn_key_values.extend(cache)
        # Apply final layer norm.
        # shape: (batch_size, seq_len, d_model)
        x = self.transformer.ln_f(x)  # type: ignore

        # record the last hidden state
        last_hidden_state = x 

        if output_hidden_states:
            # add final hidden state post-final-layernorm, following HuggingFace's convention
            all_hidden_states.append(x)

        if last_logits_only:
            # shape: (batch_size, 1, d_model)
            if append_last_valid_logits is not None:
                last_valid_output = x[
                    torch.arange(x.shape[0], device=x.device), append_last_valid_logits.to(x.device)]
                x = last_valid_output.unsqueeze(1)
            else:
                x = x[:, -1, :].unsqueeze(1)

        # Get logits.
        # shape: (batch_size, seq_len or 1, vocab_size)
        logits = None
        # if self.config.weight_tying:
        #     logits = F.linear(x, self.transformer.wte.weight, None)  # type: ignore
        # else:
        #     logits = self.transformer.ff_out(x)  # type: ignore
        # if self.config.scale_logits:
        #     logits.mul_(1 / math.sqrt(self.config.d_model))

        # if not last_logits_only and append_last_valid_logits is not None:
        #     last_valid_logit = logits[
        #         torch.arange(logits.shape[0], device=logits.device), append_last_valid_logits]
        #     logits = torch.cat([logits[:, :-1], last_valid_logit[:, None]], dim=1)


        is_training = self.training or target_actions is not None
        # print('******** AffordVLA forward, is_training: ',is_training)
        # print('******** AffordVLA forward, target_actions: ', target_actions)
        
        # 推理阶段
        if not is_training:
            return OLMoWithLastOutput(logits=logits, attn_key_values=attn_key_values, last_hidden_state=last_hidden_state,exit_layer=-1,
                                      hidden_states=tuple(all_hidden_states) if output_hidden_states else None) 
        # 训练阶段
        if is_training:
            
            outputs = OLMoWithLastOutput(logits=logits, attn_key_values=attn_key_values, last_hidden_state=last_hidden_state,exit_layer=-1,
                                    hidden_states=tuple(all_hidden_states) if output_hidden_states else None) 
            # end of origin forward method of super class Molmo

            def generate_action(action_hidden_states):
                if self.action_head_type == 'l1_regression':
                    predicted_actions = self.action_head.predict_action(action_hidden_states)  
            
                # use dit model
                elif self.action_head_type == 'diffusion':
                    noise, noisy_actions, timesteps = self.action_head.sample_noisy_actions(target_actions)
                    pred = self.action_head.predict_noise_or_sample(noisy_actions,timesteps,action_hidden_states)
                    
                    pred_type = self.diff_pred_type 
                    if pred_type == 'epsilon':
                        target = noise
                    elif pred_type == 'sample':
                        target = target_actions

                
                elif self.action_head_type == 'diffusion_openvla':
                    return_dict = self.action_head.sample_noisy_actions(target_actions)
                    noise, noisy_actions, timesteps = return_dict['noise'], return_dict['noisy_actions'], return_dict['diffusion_timestep_embeddings']
                    noise_pred = self.action_head.predict_noise(action_hidden_states,noisy_actions,timesteps)
                    # Get diffusion noise prediction MSE loss
                    noise_pred = noise_pred.reshape(noise.shape)
                    target = noise
                    pred = noise_pred

                
                return {  
                    'predicted_actions': predicted_actions,  
                    'diffusion_target': target if self.action_head_type != 'l1_regression' else None,
                    'diffusion_pred': pred if self.action_head_type != 'l1_regression' else None,
                    # 'noisy_actions': noisy_actions if self.action_head_type != 'l1_regression' else None,
                    'diff_timesteps': timesteps if self.action_head_type != 'l1_regression' else None,
                }
            
            def generate_action_flow_matching(attn_key_values):
                llm_dtype = next(self.transformer.parameters()).dtype
                # 确保 action_head 参数与主 LLM 一致的 dtype（bf16/fp32）
                self.action_head.to(llm_dtype)
                
                # sample noisy path
                B = target_actions.shape[0]
                device = target_actions.device
                dtype = llm_dtype
                fm = self.action_head.sample_noisy_actions(target_actions)
                noise, x_t, t = fm['noise'], fm['x_t'], fm['t']
                timesteps = t.unsqueeze(1)

                assert self.config.use_proprio and action_proprio is not None, "flow_matching requires action_proprio"
                # 训练阶段：基于 input_ids 估计有效前缀长度（忽略 -1 的 padding）
                pos_offset = (input_ids != -1).to(torch.int64).sum(dim=1)
                pred = self.action_head.predict_vector_field(attn_key_values, action_proprio.to(dtype), x_t.to(dtype), t.to(dtype), pos_offset=pos_offset)
                target = (noise - target_actions).to(dtype)
                # 显式确保计算图中的张量 dtype 统一
                pred = pred.to(dtype)
                predicted_actions = None

                return {  
                    'predicted_actions': predicted_actions,  
                    'diffusion_target': target,
                    'diffusion_pred': pred,
                    'diff_timesteps': timesteps,
                }
            

            outputs_list = []
            # randomly sample a layer to exit, or supervise all layers

            # flow matching单独的处理逻辑
            if self.action_head_type == "flow_matching":
                assert attn_key_values is not None, "attn_key_values is required for flow matching"
                assert len(attn_key_values) > 0, "attn_key_values is required for flow matching"
                n = len(attn_key_values)
                for i in range(n):
                    if train_exit_random_layer is not None:
                        if i != train_exit_random_layer_id:
                            continue
                    # gen the attn_key_values up to i-th layer
                    attn_key_values_i = attn_key_values[:i+1]
                    outputs = generate_action_flow_matching(attn_key_values_i)
                    outputs_list.append(outputs)

            else:
                assert len(all_hidden_states) > 0, "all_hidden_states for early exit training"
                
                for i, hidden_state in enumerate(all_hidden_states):
                    if train_exit_random_layer is not None:
                        # if i != train_exit_random_layer or i!= self.config.n_layers - 1:
                        if i != train_exit_random_layer_id:
                            continue
                    action_hidden_states = self.extract_action_hidden_states(hidden_state, input_ids,proprio_token_idx)  
                    outputs = generate_action(action_hidden_states)
                    outputs_list.append(outputs)

            return outputs_list




    @classmethod
    def from_checkpoint(
        cls, checkpoint_dir: PathOrStr, device: str = "cpu",
        checkpoint_type: Optional[CheckpointType] = None
    ) -> Molmo:
        """
        Load an OLMo model from a checkpoint.
        """
        from a1.util import resource_path
        if checkpoint_dir.startswith("hf:"):
            from a1.hf_molmo import load_hf_model
            return load_hf_model(checkpoint_dir[3:])

        # Guess checkpoint type.
        if checkpoint_type is None:
            try:
                if resource_path(checkpoint_dir, "model.pt").is_file():
                    checkpoint_type = CheckpointType.unsharded
                else:
                    checkpoint_type = CheckpointType.sharded
            except FileNotFoundError:
                checkpoint_type = CheckpointType.sharded

        # Load config.
        if Path(join(checkpoint_dir, "model.yaml")).exists():
            model_config = ModelConfig.load(Path(join(checkpoint_dir, "model.yaml")))
        else:
            config_path = resource_path(checkpoint_dir, "config.yaml")
            model_config = ModelConfig.load(config_path, key="model", validate_paths=False)

        if checkpoint_type == CheckpointType.unsharded:
            # Initialize model (always on CPU to start with so we don't run out of GPU memory).
            model_config.init_device = "cpu"
            model = AffordVLAEarlyExit(model_config)

            # Load state dict directly to target device.
            state_dict_path = resource_path(checkpoint_dir, "model.pt")
            state_dict = torch.load(state_dict_path, map_location="cpu")
            dtype = state_dict[list(state_dict.keys())[0]].dtype
            log.info(f"Checkpoint weight dtype: {dtype}")
            model.load_state_dict(model._make_state_dict_compatible(state_dict)[0])
            model = model.to(torch.device(device))
        else:
            from a1.checkpoint import load_model_state

            # Initialize model on target device. In this case the state dict is loaded in-place
            # so it's not necessary to start on CPU if the target device is a GPU.
            model_config.init_device = device
            model = AffordVLAEarlyExit(model_config)

            # Load state dict in place.
            load_model_state(checkpoint_dir, model)

        return model.eval()