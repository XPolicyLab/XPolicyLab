from typing import Dict, Any, List

import numpy as np
import torch


# from a1.vla.constants import NUM_ACTIONS_CHUNK, ACTION_DIMS_MAPPING
from a1.tokenizer import (get_special_token_ids, 
                            DEFAULT_ACT_START_TOKEN, DEFAULT_ACT_END_TOKEN, 
                            DEFAULT_RIGHT_EEF_TOKEN, DEFAULT_LEFT_EEF_TOKEN, 
                            DEFAULT_MOBILE_BASE_TOKEN,DEFAULT_EMPTY_ACT_TOKEN,
                            DEFAULT_PROPRIO_TOKEN,DEFAULT_TIMESTEP_TOKEN)

from a1.tokenizer import (RIGHT_EEF_X_AXIS_TOKEN,RIGHT_EEF_Y_AXIS_TOKEN,RIGHT_EEF_Z_AXIS_TOKEN,
                            RIGHT_EEF_ROLL_TOKEN,RIGHT_EEF_PITCH_TOKEN,RIGHT_EEF_YAW_TOKEN,RIGHT_EEF_GRIPPER_TOKEN)
from a1.tokenizer import (LEFT_EEF_X_AXIS_TOKEN,LEFT_EEF_Y_AXIS_TOKEN,LEFT_EEF_Z_AXIS_TOKEN,
                            LEFT_EEF_ROLL_TOKEN,LEFT_EEF_PITCH_TOKEN,LEFT_EEF_YAW_TOKEN,LEFT_EEF_GRIPPER_TOKEN)

numpy_to_torch_dtype_dict = {
    np.dtype("bool"): torch.bool,
    np.dtype("uint8"): torch.uint8,
    np.dtype("int8"): torch.int8,
    np.dtype("int16"): torch.int16,
    np.dtype("int32"): torch.int32,
    np.dtype("int64"): torch.int64,
    np.dtype("float16"): torch.float16,
    np.dtype("float32"): torch.float32,
    np.dtype("float64"): torch.float64,
    np.dtype("complex64"): torch.complex64,
    np.dtype("complex128"): torch.complex128,
}


def _collate(tensors, max_sequence_length=None, dtype=None, pad=False, pad_value=-1):
    # 考虑允许动态长度，你也可以把 max_len 设为本 batch 的最大序列长度，从而仅做 pad、不做截断
    # actual_max_len = max((0 if x is None else x.shape[0]) for x in tensors)
    # print(f"_collate, actual_max_len: {actual_max_len}")
    if pad == "to_max":
        max_len = max_sequence_length
        tensor = [x for x in tensors if x is not None][0]
        arr = np.full([len(tensors), max_len] + list(tensor.shape[1:]), pad_value,
                      dtype=dtype or tensor.dtype)
    else:
        max_len = max((0 if x is None else x.shape[0]) for x in tensors)
        if max_sequence_length:
            max_len = min(max_len, max_sequence_length)
        elif pad is not None:
            raise NotImplementedError(pad)

        arr = np.full([len(tensors), max_len] + list(tensors[0].shape[1:]), pad_value,
                      dtype=dtype or tensors[0].dtype)

    for ix, tensor in enumerate(tensors):
        if tensor is not None:
            assert len(tensor) <= max_len, (
                f"Tensor at index {ix} (len={len(tensor)}) "
                f"exceeds max_sequence_length={max_len}. "
                "This code only supports padding, not truncation."
            )
            arr[ix, :len(tensor)] = tensor[:max_len]
    return torch.from_numpy(arr)


class MMCollator:
    """Converts list of examples from our datasets into a tensor batch"""

    TEXT_KEYS = ["input_tokens", "target_tokens", "loss_masks", "subsegment_ids", "position_ids"]
    IMAGE_KEYS = ["images", "image_masks", "image_input_idx",]

    def __init__(self, max_sequence_length=None, include_metadata=True, pad=None,
                 max_crops=None):
        """
        :param max_sequence_length: truncate examples longer than this length
        :param include_metadata: whether to include the metadata in the out batch
        :param pad: how to pad the tensors
        :param max_crops: max number of crops to use if padding to the max sequence length
        """
        if pad:
            assert max_sequence_length is not None and max_crops is not None
        self.max_sequence_length = max_sequence_length
        self.max_crops = max_crops
        self.include_metadata = include_metadata
        self.pad = pad

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        assert len(batch) > 0, "Given an empty batch"
        keys = batch[0].keys()
        out = {}
        for key in self.TEXT_KEYS:
            # If one examples has subsegment_ids, all examples need it so with ones
            # matching the input tokens
            if any(key in ex for ex in batch):
                if key == "subsegment_ids":
                    for ex in batch:
                        if "subsegment_ids" not in ex:
                            ex["subsegment_ids"] = np.ones_like(ex["input_tokens"])

                dtype = np.float32 if key == "loss_masks" else np.int64
                out[key] = _collate(
                    [ex.get(key) for ex in batch], self.max_sequence_length, dtype, pad=self.pad)

        for key in self.IMAGE_KEYS:
            if any(key in ex for ex in batch):
                out[key] = _collate([ex.get(key) for ex in batch], self.max_crops, pad=self.pad)
        out["input_ids"] = out.pop("input_tokens")
        if "target_tokens" in out:
            out["labels"] = out.pop("target_tokens")
        if self.include_metadata:
            out["metadata"] = [ex.get("metadata", {}) for ex in batch]
        return out

class MMCollatorForAction:
    """Converts list of examples from our datasets into a tensor batch"""

    TEXT_KEYS = ["input_tokens", "target_tokens", "loss_masks", "subsegment_ids", "position_ids"]
    IMAGE_KEYS = ["images", "image_masks", "image_input_idx",]

    def __init__(self,model_config,use_proprio=False, max_sequence_length=None, include_metadata=True, pad=None, max_crops=None):
        """
        :param max_sequence_length: truncate examples longer than this length
        :param include_metadata: whether to include the metadata in the out batch
        :param pad: how to pad the tensors
        :param max_crops: max number of crops to use if padding to the max sequence length
        """
        if pad:
            assert max_sequence_length is not None and max_crops is not None
        self.max_sequence_length = max_sequence_length
        self.max_crops = max_crops
        self.include_metadata = include_metadata
        self.pad = pad

        self.use_proprio = use_proprio
        self.model_config = model_config
        self.num_actions_chunk = model_config.num_actions_chunk
        self.use_left_eef = model_config.action_use_left_eef
        self.use_mobile_base = model_config.action_use_mobile_base

        self.action_dims_mapping = {
            "right_end_effector": model_config.right_end_effector_dim,
            "left_end_effector": model_config.left_end_effector_dim,
            "mobile_base": model_config.mobile_base_dim,
        }


        self.tokenizer = model_config.get_tokenizer() 
        special_tokens = get_special_token_ids(self.tokenizer)

        # action token ids
        
        self.action_start_token_id = special_tokens[DEFAULT_ACT_START_TOKEN]  
        self.action_end_token_id = special_tokens[DEFAULT_ACT_END_TOKEN]
        self.empty_action_token_id = special_tokens[DEFAULT_EMPTY_ACT_TOKEN]  
        self.right_eef_token_id = special_tokens[DEFAULT_RIGHT_EEF_TOKEN]  
        self.left_eef_token_id = special_tokens[DEFAULT_LEFT_EEF_TOKEN]  
        self.mobile_base_token_id = special_tokens[DEFAULT_MOBILE_BASE_TOKEN]
        self.proprio_token_id = special_tokens[DEFAULT_PROPRIO_TOKEN]
        self.timestep_token_id = special_tokens[DEFAULT_TIMESTEP_TOKEN]

        self.right_eef_x_axis_token_id = special_tokens[RIGHT_EEF_X_AXIS_TOKEN]
        self.right_eef_y_axis_token_id = special_tokens[RIGHT_EEF_Y_AXIS_TOKEN]
        self.right_eef_z_axis_token_id = special_tokens[RIGHT_EEF_Z_AXIS_TOKEN]
        self.right_eef_roll_token_id = special_tokens[RIGHT_EEF_ROLL_TOKEN]
        self.right_eef_pitch_token_id = special_tokens[RIGHT_EEF_PITCH_TOKEN]
        self.right_eef_yaw_token_id = special_tokens[RIGHT_EEF_YAW_TOKEN]
        self.right_eef_gripper_token_id = special_tokens[RIGHT_EEF_GRIPPER_TOKEN]

        self.left_eef_x_axis_token_id = special_tokens[LEFT_EEF_X_AXIS_TOKEN]
        self.left_eef_y_axis_token_id = special_tokens[LEFT_EEF_Y_AXIS_TOKEN]
        self.left_eef_z_axis_token_id = special_tokens[LEFT_EEF_Z_AXIS_TOKEN]
        self.left_eef_roll_token_id = special_tokens[LEFT_EEF_ROLL_TOKEN]
        self.left_eef_pitch_token_id = special_tokens[LEFT_EEF_PITCH_TOKEN]
        self.left_eef_yaw_token_id = special_tokens[LEFT_EEF_YAW_TOKEN]
        self.left_eef_gripper_token_id = special_tokens[LEFT_EEF_GRIPPER_TOKEN]

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        assert len(batch) > 0, "Given an empty batch"
        keys = batch[0].keys()
        out = {}

        # 如果需要添加 action tokens，先处理 input_tokens
        if any("input_tokens" in ex for ex in batch):
            # only l1 regression need to add action tokens, flow matching not need
            if self.model_config.action_head == 'l1_regression':
                self._add_action_tokens_to_batch(batch)
            elif self.model_config.action_head == 'flow_matching':
                self._add_proprio_tokens_to_batch(batch)

        for key in self.TEXT_KEYS:
            # If one examples has subsegment_ids, all examples need it so with ones
            # matching the input tokens
            if any(key in ex for ex in batch):
                if key == "subsegment_ids":
                    for ex in batch:
                        if "subsegment_ids" not in ex:
                            ex["subsegment_ids"] = np.ones_like(ex["input_tokens"])

                dtype = np.float32 if key == "loss_masks" else np.int64
                out[key] = _collate(
                    [ex.get(key) for ex in batch], self.max_sequence_length, dtype, pad=self.pad)

        for key in self.IMAGE_KEYS:
            if any(key in ex for ex in batch):
                out[key] = _collate([ex.get(key) for ex in batch], self.max_crops, pad=self.pad)
        out["input_ids"] = out.pop("input_tokens")
        if "target_tokens" in out:
            out["labels"] = out.pop("target_tokens")
        if self.include_metadata:
            out["metadata"] = [ex.get("metadata", {}) for ex in batch]

        action_list = [torch.from_numpy(ex["action"].copy()) for ex in batch] # 添加 action 字段
        # 将 action 列表转换为 tensor
        out['action']= torch.stack(action_list, dim=0)

        action_tokens_length_list = [ex["action_tokens_length"] for ex in batch]
        out['action_tokens_length'] = torch.tensor(action_tokens_length_list, dtype=torch.int32)

        # 如果使用了 proprio 数据，将其转换为 tensor 并添加到输出中
        if self.use_proprio:
            proprio_list = [torch.from_numpy(ex["proprio"].copy()) for ex in batch]
            out['proprio'] = torch.stack(proprio_list, dim=0)
            # 添加 proprio token 的索引
            proprio_token_idx_list = [ex["proprio_token_idx"] for ex in batch]
            first_element = proprio_token_idx_list[0]
            assert all(x == first_element for x in proprio_token_idx_list), "All proprio_token_idx should be the same in a batch."
            out['proprio_token_idx'] = torch.tensor(proprio_token_idx_list, dtype=torch.int32)
        
        # for action_pad_mask
        action_pad_mask_list = [torch.from_numpy(ex["action_pad_mask"]) for ex in batch]
        out['action_pad_mask'] = torch.stack(action_pad_mask_list, dim=0)

        # to get timestep and episode_index for the training data
        # assert "timestep" in batch[0], "timestep not in batch[0]"
        # timestep_list = [torch.tensor(ex["timestep"], dtype=torch.int64) for ex in batch]
        # out['timestep'] = torch.stack(timestep_list, dim=0)

        # assert "episode_index" in batch[0], "episode_index not in batch[0]"
        # if batch[0]["episode_index"] is not None:
            
        #     episode_index_list = [torch.tensor(ex["episode_index"], dtype=torch.int64) for ex in batch]
        #     out['episode_index'] = torch.stack(episode_index_list, dim=0)
        # else:
        #     pass
            # print("MMCollatorForAction call: episode_index is None")
            # out['episode_index'] = None


        if "text_attention_mask" in batch[0] and batch[0]["text_attention_mask"] is not None:
            attention_mask_list = [ex["text_attention_mask"] for ex in batch] 
            out["text_attention_mask"] = torch.cat(attention_mask_list, dim=0)


        return out
    
    def _add_proprio_tokens_to_batch(self, batch):
        """为批次中的每个样本添加 proprio tokens"""
        max_input_length = max(len(ex["input_tokens"]) for ex in batch)

        for ex in batch:
            input_tokens = ex["input_tokens"]
            original_length = len(input_tokens)
            if original_length < max_input_length:
                pad_length = max_input_length - original_length
                input_tokens = np.pad(input_tokens, (0, pad_length), constant_values=-1)
            proprio_token = np.array( [self.proprio_token_id],dtype=np.int32) # add proprio token
            ex["input_tokens"] = input_tokens
            ex['proprio_token_idx'] = max_input_length  # proprio token 的索引
            ex['action_tokens_length'] = 0
        
            # 处理相关的其他字段
            if "loss_masks" in ex:
                loss_masks = ex["loss_masks"]
                # pad loss_masks 到相同长度
                if len(loss_masks) < max_input_length:
                    pad_length = max_input_length - len(loss_masks)
                    loss_masks = np.pad(loss_masks, (0, pad_length), constant_values=0.0)

            if "subsegment_ids" in ex:
                subsegment_ids = ex["subsegment_ids"]
                # pad subsegment_ids 到相同长度
                if len(subsegment_ids) < max_input_length:
                    pad_length = max_input_length - len(subsegment_ids)
                    subsegment_ids = np.pad(subsegment_ids, (0, pad_length), constant_values=subsegment_ids[-1])

    def _add_action_tokens_to_batch(self, batch):
        """为批次中的每个样本添加 action tokens"""

        # 1. 找到批次中最长的 input_tokens 长度
        max_input_length = max(len(ex["input_tokens"]) for ex in batch)
        
        # 2. 生成 action tokens
        action_tokens = self._build_action_tokens()
        
        # 3. 为每个样本处理
        for ex in batch:
            input_tokens = ex["input_tokens"]
            original_length = len(input_tokens)
            
            # 先将 input_tokens pad 到批次最长长度
            if original_length < max_input_length:
                pad_length = max_input_length - original_length
                input_tokens = np.pad(input_tokens, (0, pad_length), constant_values=-1)
            
            # 然后拼接 action tokens
            proprio_token = np.array( [self.proprio_token_id],dtype=np.int32) # add proprio token
            timestep_token = np.array( [self.timestep_token_id],dtype=np.int32) # add timestep token
            combined_tokens = np.concatenate([input_tokens, proprio_token,timestep_token,action_tokens])
            ex["input_tokens"] = combined_tokens
            ex['proprio_token_idx'] = max_input_length  # proprio token 的索引
            ex['action_tokens_length'] = len(action_tokens)
            
            # 处理相关的其他字段
            if "loss_masks" in ex:
                loss_masks = ex["loss_masks"]
                # pad loss_masks 到相同长度
                if len(loss_masks) < max_input_length:
                    pad_length = max_input_length - len(loss_masks)
                    loss_masks = np.pad(loss_masks, (0, pad_length), constant_values=0.0)
                
                # 为 action tokens 添加 loss mask (通常为 0，不计算损失)
                action_loss_mask = np.zeros(len(action_tokens), dtype=np.float32)
                combined_loss_mask = np.concatenate([loss_masks, action_loss_mask])
                ex["loss_masks"] = combined_loss_mask
            
            if "target_tokens" in ex:
                target_tokens = ex["target_tokens"]
                # pad target_tokens 到相同长度
                if len(target_tokens) < max_input_length:
                    pad_length = max_input_length - len(target_tokens)
                    target_tokens = np.pad(target_tokens, (0, pad_length), constant_values=-1)
                
                # 拼接 action tokens
                combined_target = np.concatenate([target_tokens, action_tokens])
                ex["target_tokens"] = combined_target
            
            if "subsegment_ids" in ex:
                subsegment_ids = ex["subsegment_ids"]
                # pad subsegment_ids 到相同长度
                if len(subsegment_ids) < max_input_length:
                    pad_length = max_input_length - len(subsegment_ids)
                    subsegment_ids = np.pad(subsegment_ids, (0, pad_length), constant_values=subsegment_ids[-1])
                
                # 为 action tokens 分配特殊的 subsegment id
                action_subsegment_id = 99999
                action_subsegment = np.full(len(action_tokens), action_subsegment_id, dtype=np.int32)
                combined_subsegment = np.concatenate([subsegment_ids, action_subsegment])
                ex["subsegment_ids"] = combined_subsegment

                # 同步重算 position_ids（关键修复：为新增的动作段分配正确的段内位置索引）
                position_ids = np.zeros_like(combined_subsegment, dtype=np.int64)
                unique_ids = np.unique(combined_subsegment)
                for sid in unique_ids:
                    # 在每个子段内递增，确保动作段内的步长具有唯一位置
                    segment_mask = (combined_subsegment == sid)
                    position_ids = np.where(segment_mask, np.cumsum(segment_mask) - 1, position_ids)
                ex["position_ids"] = position_ids
            else:
                # 如果没有 subsegment_ids，就按绝对位置生成
                ex["position_ids"] = np.arange(len(combined_tokens),dtype=np.int64)

    def _build_action_tokens(self):
        """Construct the action token sequence of ACTION_DIMS*NUM_ACTIONS_CHUNK, 
            predict the next NUM_ACTIONS_CHUNK steps of actions, and insert them at the end of the sequence."""
        
        action_tokens = [self.action_start_token_id]
        
        for timestep in range(self.num_actions_chunk):
            # Right-hand end effector: 7 tokens
            # scheme one:
            # action_tokens.extend([self.right_eef_token_id] * ACTION_DIMS_MAPPING['right_end_effector'])
            # scheme two:
            right_eef_tokens = [self.right_eef_x_axis_token_id,self.right_eef_y_axis_token_id,self.right_eef_z_axis_token_id,
                                 self.right_eef_roll_token_id,self.right_eef_pitch_token_id,self.right_eef_yaw_token_id,
                                 self.right_eef_gripper_token_id]
            if len(right_eef_tokens) < self.action_dims_mapping['right_end_effector']:
                right_eef_tokens.append(self.right_eef_gripper_token_id)
                
            assert len(right_eef_tokens) == self.action_dims_mapping['right_end_effector']
            action_tokens.extend(right_eef_tokens)

            if self.use_left_eef:
                # Left-hand end effector: 7 tokens   
                left_eef_tokens = [self.left_eef_x_axis_token_id,self.left_eef_y_axis_token_id,self.left_eef_z_axis_token_id,
                                   self.left_eef_roll_token_id,self.left_eef_pitch_token_id,self.left_eef_yaw_token_id,
                                   self.left_eef_gripper_token_id]
                if len(left_eef_tokens) < self.action_dims_mapping['left_end_effector']:
                    left_eef_tokens.append(self.left_eef_gripper_token_id)
                assert len(left_eef_tokens) == self.action_dims_mapping['left_end_effector']
                action_tokens.extend(left_eef_tokens)
                # action_tokens.extend([self.left_eef_token_id] * self.action_dims_mapping['left_end_effector'])
            if self.use_mobile_base:
                # Mobile base: 3 tokens
                action_tokens.extend([self.mobile_base_token_id] * self.action_dims_mapping['mobile_base'])
        
        action_tokens.append(self.action_end_token_id)
        return np.array(action_tokens, dtype=np.int32)
    

class DiTActionCollator:
    """用于 DiffusionTransformerAction 模型的简洁 collator"""
    
    def __init__(self, include_metadata=True,use_proprio=False,pad=None,max_sequence_length=None,pad_value=-1):
        self.include_metadata = include_metadata
        self.use_proprio = use_proprio
        self.pad = pad
        self.max_sequence_length = max_sequence_length
        self.pad_value = pad_value
    
    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        处理经过 DiTActionRLDSBatchTransform 转换后的数据
        """
        assert len(batch) > 0, "Given an empty batch"
        
        out = {}
        
        # 处理图像数据 - pixel_values
        if "pixel_values" in batch[0]:
            pixel_values_list = [ex["pixel_values"] for ex in batch]
            # 如果已经是tensor，直接stack；否则转换后stack
            if isinstance(pixel_values_list[0], torch.Tensor):
                out["pixel_values"] = torch.cat(pixel_values_list, dim=0)
                # out["pixel_values"] = torch.stack(pixel_values_list, dim=0)  # 改为 stack，创建新维度
            else:
                out["pixel_values"] = torch.stack([torch.tensor(pv) for pv in pixel_values_list])
        
        # 处理文本数据 - input_ids 和 attention_mask
        # if "input_ids" in batch[0]:
        #     input_ids_list = [ex["input_ids"] for ex in batch]
        #     if isinstance(input_ids_list[0], torch.Tensor):
        #         out["input_ids"] = torch.cat(input_ids_list, dim=0)
        #     else:
        #         out["input_ids"] = torch.tensor(np.stack(input_ids_list),) # dtype=torch.int64
        
        out["input_ids" ] = _collate(
                [ex.get("input_ids") for ex in batch], self.max_sequence_length, np.int64, pad=self.pad,
                pad_value=self.pad_value)
            
        if "text_attention_mask" in batch[0] and batch[0]["text_attention_mask"] is not None:
            attention_mask_list = [ex["text_attention_mask"] for ex in batch] 
            out["text_attention_mask"] = torch.cat(attention_mask_list, dim=0)
        else:
        #     # out["text_attention_mask"] = torch.ones_like(out["input_ids"], ) #dtype=torch.int64
            out["text_attention_mask"] = None  
        
        # 处理action数据
        action_list = [torch.from_numpy(ex["action"]) for ex in batch] # 添加 action 字段
        # 将 action 列表转换为 numpy 数组
        out['action']= torch.stack(action_list, dim=0)

        assert "timestep" in batch[0], "timestep not in batch[0]"
        timestep_list = [torch.tensor(ex["timestep"], dtype=torch.int64) for ex in batch]
        out['timestep'] = torch.stack(timestep_list, dim=0)

        assert "episode_index" in batch[0], "episode_index not in batch[0]"
        episode_index_list = [torch.tensor(ex["episode_index"].copy(), dtype=torch.int64) for ex in batch]
        out['episode_index'] = torch.stack(episode_index_list, dim=0)
        
        # 处理proprio数据（可选）
        if self.use_proprio:
            proprio_list = [torch.from_numpy(ex["proprio"]) for ex in batch]
            out['proprio'] = torch.stack(proprio_list, dim=0)
        
        # 处理文本字段（用于调试）
        # text_fields = ["question", "answer", "style"]
        # for field in text_fields:
        #     if field in batch[0]:
        #         out[field] = [ex[field] for ex in batch]
        
        # 包含元数据
        if self.include_metadata:
            out["metadata"] = [ex.get("metadata", {}) for ex in batch]
        
        return out
