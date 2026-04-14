# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates.
# Copyright (c) 2026 BeingBeyond Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
#
# This file has been modified by BeindBeyond Ltd. and/or its affiliates. on 2026-01-10.


import torch
import random
from typing import Optional
from dataclasses import dataclass
from configs.dataset_info import DATASET_INFO, DATASET_REGISTRY


@dataclass
class RobotDatasetConfig:
    """Configuration for robot manipulation datasets."""
    max_view_num: int = 1
    """Maximum number of camera views"""

    use_fixed_view: bool = False
    """Whether to use only ego view"""

    is_relative: bool = False
    """Whether actions are relative"""

    is_abstract_action: bool = False
    """Whether to use abstract action space"""

    gen_action_type: str = "action_token"
    """Action generation method"""

    action_chunk_length: int = 16
    """Number of actions per chunk"""

    history_num: int = 1
    """Number of history observations"""

    prompt_template: str = "long"
    """Instruction prompt template"""

    vit_dropout_prob: float = 0
    state_dropout_prob: float = 0
    override_stats_path: Optional[str] = None


class PackedDataset(torch.utils.data.IterableDataset):
    """
    Iterable dataset that packs multiple samples into sequences.
    
    Features:
    - Dynamic sequence packing for efficient GPU utilization
    - Multi-dataset weighted sampling
    - Support for vision, language, state, and action modalities
    - Configurable attention modes and token limits
    """
    
    def __init__(
        self, 
        tokenizer,
        template_name,
        grouped_dataset_meta=None,
        robot_config: RobotDatasetConfig = None,
        special_tokens = {},
        force_image_size = 448,
        img_patch_size = 14,
        img_downsample_ratio = 0.5,
        expected_num_tokens=32768, 
        max_num_tokens_per_sample=16384,
        max_num_tokens=36864,
        max_buffer_size=50,
        prefer_buffer_before=16384,
        attn_mode="causal",
        is_train=True,
        logger=None,
        local_rank=0, world_size=0, num_workers=0, 
        **kwargs
    ):
        self.robot_config = robot_config or RobotDatasetConfig(**{k: v for k, v in kwargs.items() 
                                                  if k in RobotDatasetConfig.__annotations__})
        self.prefer_buffer_before = prefer_buffer_before
        
        for k, v in special_tokens.items():
            setattr(self, k, v)

        self.tokenizer = tokenizer
        self.template_name = template_name
        self.max_num_tokens = max_num_tokens
        self.max_num_tokens_per_sample = max_num_tokens_per_sample
        self.max_buffer_size = max_buffer_size
        self.expected_num_tokens = expected_num_tokens
        self.attn_mode = attn_mode
        
        # this must be the same for all datasets
        self.force_image_size = force_image_size
        self.img_patch_size = img_patch_size
        self.img_downsample_ratio = img_downsample_ratio

        self.local_rank = local_rank
        self.world_size = world_size
        self.num_workers = num_workers   
        self.is_train = is_train
        self.logger = logger
        
        (self.grouped_datasets, self.is_mandatory,
         self.grouped_weights) = self.build_datasets(grouped_dataset_meta)
        self.dataset_iters = [iter(dataset) for dataset in self.grouped_datasets]

    def build_datasets(self, datasets_metainfo):
        """
        Build individual datasets from metainfo.
        
        Args:
            datasets_metainfo: Dataset group metadata
            
        Returns:
            Tuple of (datasets, is_mandatory, weights)
        """
        datasets = []
        is_mandatory = []
        grouped_weights = []

        for grouped_dataset_name, dataset_args in datasets_metainfo.items():
            is_mandatory.append(dataset_args.pop('is_mandatory', False))
            grouped_weights.append(dataset_args.pop('weight', 0.0))
            
            dataset_names = dataset_args.pop('dataset_names')
            dataset_args['dataset_path_list'] = []
            dataset_args['dataset_list'] = []

            # collect dataset paths
            for item in dataset_names:
                if self.local_rank == 0:
                    print(f'Preparing Dataset {grouped_dataset_name}/{item}')
                    
                meta_info = DATASET_INFO[grouped_dataset_name][item]
                dataset_args['dataset_path_list'].append(meta_info['dataset_path'])
                dataset_args['dataset_list'].append(item)
                
                if 'jsonl_path' in meta_info.keys(): 
                    if 'jsonl_path_list' not in dataset_args.keys():
                        dataset_args['jsonl_path_list'] = []
                    dataset_args['jsonl_path_list'].append(meta_info['jsonl_path'])

            # num_image_tokens must be the same to different Datasets
            dataset_args['num_image_tokens'] = int((self.force_image_size // self.img_patch_size) ** 2 * (self.img_downsample_ratio ** 2))

            DatasetClass = DATASET_REGISTRY[grouped_dataset_name] 
            dataset = DatasetClass(
                dataset_name=grouped_dataset_name,
                template_name=self.template_name,
                force_image_size=self.force_image_size,
                tokenizer=self.tokenizer,
                local_rank=self.local_rank,
                world_size=self.world_size,
                num_workers=self.num_workers,
                is_train=self.is_train,
                logger=self.logger,
                **{**self.robot_config.__dict__, **dataset_args}
            )

            datasets.append(dataset)

        return datasets, is_mandatory, grouped_weights
    
    def set_sequence_status(self):
        sequence_status = dict(
            curr                        = 0,
            sample_lens                 = list(),
            packed_position_ids         = list(),
            nested_attention_masks      = list(),
            split_lens                  = list(),
            attn_modes                  = list(),
            packed_text_ids             = list(), 
            packed_text_indexes         = list(),
            packed_label_ids            = list(),
            ce_loss_indexes             = list(),
            packed_vit_tokens           = list(), 
            vit_token_seqlens           = list(),
            packed_vit_position_ids     = list(),
            packed_vit_token_indexes    = list(), 
            # addition for VLA
            state_tensors               = list(),
            action_tensors              = list(),
            embodiment_ids              = list(),
            action_masks                = list(),
            packed_state_indexes        = list(),
            packed_action_indexes       = list(),
            num_und_samples              = 0,
            num_gen_samples              = 0,
        )
        return sequence_status
    
    def __iter__(self):
        """Iterate over packed sequences."""

        sequence_status = self.set_sequence_status()
        buffer = []

        # calculate cumulative probabilities for sampling
        total_weights = sum(self.grouped_weights)
        assert total_weights > 0.0, "Total dataset weight must be positive"

        group_cumprobs = [sum(self.grouped_weights[:i + 1]) / total_weights 
                          for i in range(len(self.grouped_weights))]
        
        while True:
            if sequence_status['curr'] == 0:
                for group_index, group_iter in enumerate(self.dataset_iters):
                    if self.is_mandatory[group_index]:
                        while True:
                            sample = next(group_iter)
                            # if a sample is too long, skip it
                            num_tokens = sample['num_tokens'] 
                            if num_tokens < self.max_num_tokens_per_sample:
                                sequence_status = self.pack_sequence(sample, sequence_status)
                                break
            
            # decide whether to use buffer or sample new
            if sequence_status['curr'] < self.prefer_buffer_before and len(buffer) > 0:
                sample = buffer.pop(0)
                sample_from_buffer = True
            else:
                n = random.random()
                group_index = 0
                for i, cumprob in enumerate(group_cumprobs):
                    if n < cumprob:
                        group_index = i
                        break
                sample = next(self.dataset_iters[group_index])
                sample_from_buffer = False
     
            # skip oversized samples
            num_tokens = sample['num_tokens'] 
            if num_tokens > self.max_num_tokens_per_sample:
                continue

            if sequence_status['curr'] + num_tokens > self.max_num_tokens:
                # add to buffer or yield current sequence
                if len(buffer) < self.max_buffer_size and not sample_from_buffer:
                    buffer.append(sample)
                else:
                    data = self.to_tensor(sequence_status)
                    yield data
                    sequence_status = self.set_sequence_status()

                continue
            
            # pack sample into sequence
            sequence_status = self.pack_sequence(sample, sequence_status)

            # yield if sequence is long enough
            if sequence_status['curr'] >= self.expected_num_tokens:
                data = self.to_tensor(sequence_status)
                yield data
                sequence_status = self.set_sequence_status()

    def pack_sequence(self, sample, sequence_status):
        image_tensor_list = sample['image_tensor_list']
        text_ids_list = sample['text_ids_list']
        sequence_plan = sample['sequence_plan']

        split_lens, attn_modes = list(), list()
        curr = sequence_status['curr']
        curr_rope_id = 0
        sample_lens = 0

        for item in sequence_plan:
            split_start = item.get('split_start', True)
            if split_start:
                curr_split_len = 0
            
            # bos and eos are with text or a single sequence, never appear in vit, state, action, and etc
            if item['type']=='text':
                text_ids = text_ids_list.pop(0)
                
                if item.get("is_bos"):
                    shifted_text_ids = [self.bos_token_id] + text_ids
                else:
                    shifted_text_ids = text_ids

                sequence_status['packed_text_ids'].extend(shifted_text_ids)
                sequence_status['packed_text_indexes'].extend(range(curr, curr + len(shifted_text_ids)))    # left-closed, right-open
                if item['has_loss'] == 1:
                    sequence_status['num_und_samples'] += 1
                    sequence_status['ce_loss_indexes'].extend(range(curr, curr + len(shifted_text_ids)))
                    sequence_status['packed_label_ids'].extend(text_ids + [self.eos_token_id])
                
                curr += len(shifted_text_ids)
                curr_split_len += len(shifted_text_ids)

                if item.get("is_end"): # at the end of entire conversation
                    sequence_status['packed_text_ids'].append(self.eos_token_id)
                    sequence_status['packed_text_indexes'].append(curr)
                    if item['special_token_loss'] == 1: # <|im_end|> may have loss
                        sequence_status['ce_loss_indexes'].append(curr)
                        sequence_status['packed_label_ids'].append(self.newline_token_id)
                    curr += 1
                    curr_split_len += 1

                elif item.get("is_eos"): # eos + newline
                    assert "is_end" not in item
                    sequence_status['packed_text_ids'].extend([self.eos_token_id, self.newline_token_id]) # add \n after each eos token
                    sequence_status['packed_text_indexes'].extend([curr, curr+1])
                    if item['special_token_loss'] == 1: # <|im_end|> may have loss
                        sequence_status['ce_loss_indexes'].extend([curr, curr+1])
                        sequence_status['packed_label_ids'].extend([self.newline_token_id, self.bos_token_id])
                    curr += 2
                    curr_split_len += 2

                # update sequence status
                attn_modes.append("causal")
                sequence_status['packed_position_ids'].extend(range(curr_rope_id, curr_rope_id + curr_split_len))
                curr_rope_id += curr_split_len
     
            elif item['type'] == 'vit_image':
                image_tensor = image_tensor_list.pop(0)

                drop_vit_cond = item.get("drop_vit_cond", False)
                if drop_vit_cond:
                    # curr_rope_id += 1
                    continue
                
                # add a <|startofimage|> token
                sequence_status['packed_text_ids'].append(self.start_of_image)
                sequence_status['packed_text_indexes'].append(curr)
                curr += 1
                curr_split_len += 1

                sequence_status['packed_vit_token_indexes'].extend(range(curr, curr + item['num_image_tokens']))
                curr += item['num_image_tokens']
                curr_split_len += item['num_image_tokens']

                vit_tokens = image_tensor
                sequence_status['packed_vit_tokens'].append(vit_tokens)
                sequence_status['vit_token_seqlens'].append(item['num_image_tokens'])

                # add a <|endofimage|> token
                sequence_status['packed_text_ids'].append(self.end_of_image)
                sequence_status['packed_text_indexes'].append(curr)
                if item['special_token_loss'] == 1: # <|endofimage|> may have loss
                    sequence_status['ce_loss_indexes'].append(curr)
                    sequence_status['packed_label_ids'].append(item['special_token_label'])
                curr += 1
                curr_split_len += 1

                if "causal" in self.attn_mode:
                    attn_modes.append("causal")
                    sequence_status['packed_position_ids'].extend(range(curr_rope_id, curr_rope_id + curr_split_len))
                    curr_rope_id += curr_split_len
                else:
                    attn_modes.append("full")
                    sequence_status['packed_position_ids'].extend([curr_rope_id] * curr_split_len)
                    if 'frame_delta' in item.keys():
                        curr_rope_id += item['frame_delta']
                    else:
                        curr_rope_id += 1
           
            elif item['type'] == 'state':
                embodiment_id = sample.get('embodiment_id', 0)
                sequence_status['embodiment_ids'].append(embodiment_id)
                state_tensor = sample['state_tensor_list'].pop(0)
                
                # add a <|startofstate|> token
                sequence_status['packed_text_ids'].append(self.start_of_state)
                sequence_status['packed_text_indexes'].append(curr)
                curr += 1
                curr_split_len += 1

                sequence_status['state_tensors'].append(state_tensor)
                num_state_tokens = state_tensor.shape[0]
                sequence_status['packed_state_indexes'].append(curr)
                curr += num_state_tokens
                curr_split_len += num_state_tokens
             
                # add a <|endofstate|> token
                sequence_status['packed_text_ids'].append(self.end_of_state)
                sequence_status['packed_text_indexes'].append(curr)
                curr += 1
                curr_split_len += 1

                attn_modes.append("causal")
                sequence_status['packed_position_ids'].extend(range(curr_rope_id, curr_rope_id + curr_split_len))
                curr_rope_id += curr_split_len

            elif item['type'] == "action":
                action_tensor = sample['action_tensor_list'].pop(0)
                action_mask = sample.get('action_mask')
                if action_mask is None:
                    # action_mask = torch.ones(len(action_tensor), dtype=torch.bool)
                    action_mask = torch.ones_like(action_tensor, dtype=torch.bool)
                sequence_status['action_masks'].append(action_mask)

                num_action_tokens = len(action_tensor)
                sequence_status['action_tensors'].append(action_tensor)
                # sequence_status['action_mask_tensors'].append(action_tensor)
                
                sequence_status['packed_action_indexes'].extend(range(curr, curr + num_action_tokens))
                curr += num_action_tokens
                
                curr_split_len += num_action_tokens   
                sequence_status['num_gen_samples'] += 1
                
                if "actionfull" in self.attn_mode:
                    attn_modes.append("full")
                else:
                    attn_modes.append("causal")
                sequence_status['packed_position_ids'].extend(range(curr_rope_id, curr_rope_id + curr_split_len))
                curr_rope_id += curr_split_len
                
            if item.get('split_end', True):
                split_lens.append(curr_split_len)
                sample_lens += curr_split_len

        sequence_status['curr'] = curr
        sequence_status['sample_lens'].append(sample_lens)
        sequence_status['split_lens'].extend(split_lens)
        sequence_status['attn_modes'].extend(attn_modes)

        return sequence_status

    def to_tensor(self, sequence_status):
        """Convert sequence status to tensor dictionary."""
        data = dict(
            sequence_length=sum(sequence_status['sample_lens']),
            num_gen_samples=sequence_status['num_gen_samples'],
            num_und_samples=sequence_status['num_und_samples'],
            sample_lens=sequence_status['sample_lens'],
            packed_text_ids=torch.tensor(sequence_status['packed_text_ids']),
            packed_text_indexes=torch.tensor(sequence_status['packed_text_indexes']),
            packed_position_ids=torch.tensor(sequence_status['packed_position_ids']),
        )
 
        if len(sequence_status['embodiment_ids']) > 0:
            data['embodiment_ids'] = torch.as_tensor(sequence_status['embodiment_ids'], dtype=torch.long)

        sequence_len = data['sequence_length']
 
        pad_len = self.max_num_tokens - sequence_len
        assert pad_len >= 0, "Sequence length exceeds maximum"
        
        data['split_lens'] = sequence_status['split_lens'] + [pad_len]
        data['attn_modes'] = sequence_status['attn_modes'] + ['causal']
        data['sample_lens'] += [pad_len]

        if len(sequence_status['state_tensors']) > 0:
            state_tensors = sequence_status.pop('state_tensors')
            data['padded_state'] = torch.vstack(state_tensors)
            data['packed_state_indexes'] = torch.as_tensor(sequence_status['packed_state_indexes'])
            
        if len(sequence_status['action_tensors']) > 0:
            action_tensors = sequence_status.pop('action_tensors')
            data['padded_action'] = torch.cat(action_tensors)
            data['packed_action_indexes'] = torch.as_tensor(sequence_status['packed_action_indexes'])

            action_masks = sequence_status.pop('action_masks')
            if len(action_masks) > 0:
                data['padded_action_mask'] = torch.cat(action_masks)

        if len(sequence_status['packed_vit_tokens']) > 0:

            data['packed_vit_tokens'] = torch.cat(sequence_status['packed_vit_tokens'])
            data['packed_vit_token_indexes'] = torch.tensor(sequence_status['packed_vit_token_indexes'])
            data['vit_token_seqlens'] = torch.tensor(sequence_status['vit_token_seqlens'])

        if len(sequence_status['packed_label_ids']) > 0:
            data['packed_label_ids'] = torch.tensor(sequence_status['packed_label_ids'])
            data['ce_loss_indexes'] = torch.tensor(sequence_status['ce_loss_indexes'])

        return data
    

class SimpleCustomBatch:
    def __init__(self, batch):
        data = batch[0]
        self.sequence_length = data["sequence_length"]
        self.num_und_samples = data['num_und_samples']
        self.num_gen_samples = data['num_gen_samples']

        self.sample_lens = data["sample_lens"]
        self.packed_text_ids = data["packed_text_ids"]
        self.packed_text_indexes = data["packed_text_indexes"]
        self.packed_position_ids = data["packed_position_ids"]

        self.use_flex = "nested_attention_masks" not in data.keys()

        self.split_lens = data["split_lens"]
        self.attn_modes = data["attn_modes"]

        if "packed_vit_tokens" in data.keys():
            self.packed_vit_tokens = data["packed_vit_tokens"]
            self.packed_vit_token_indexes = data["packed_vit_token_indexes"]
            self.vit_token_seqlens = data["vit_token_seqlens"]

        if "packed_label_ids" in data.keys():
            self.packed_label_ids = data["packed_label_ids"]
            self.ce_loss_indexes = data["ce_loss_indexes"]

        self.embodiment_ids = data.get("embodiment_ids")
        
        if "padded_action" in data.keys():
            self.padded_action = data["padded_action"]
            # self.padded_action_masks = data["padded_action_masks"]
            self.packed_action_indexes = data["packed_action_indexes"]
            self.padded_state = data["padded_state"]
            # self.padded_state_masks = data["padded_state_masks"]
            self.packed_state_indexes = data["packed_state_indexes"]
            self.padded_action_mask = data.get("padded_action_mask")

    def pin_memory(self):
        self.packed_text_ids = self.packed_text_ids.pin_memory()
        self.packed_text_indexes = self.packed_text_indexes.pin_memory()
        self.packed_position_ids = self.packed_position_ids.pin_memory()

        if hasattr(self, 'packed_vit_tokens'):
            self.packed_vit_tokens = self.packed_vit_tokens.pin_memory()
            self.packed_vit_token_indexes = self.packed_vit_token_indexes.pin_memory()
            self.vit_token_seqlens = self.vit_token_seqlens.pin_memory()

        if hasattr(self, 'packed_label_ids'):
            self.packed_label_ids = self.packed_label_ids.pin_memory()
            self.ce_loss_indexes = self.ce_loss_indexes.pin_memory()
   
        if hasattr(self, 'padded_action'):
            self.padded_action = self.padded_action.pin_memory()
            self.packed_action_indexes = self.packed_action_indexes.pin_memory()
            self.padded_state = self.padded_state.pin_memory()
            self.packed_state_indexes = self.packed_state_indexes.pin_memory()

        if hasattr(self, 'padded_action_mask') and self.padded_action_mask is not None:
            self.padded_action_mask = self.padded_action_mask.pin_memory()
        
        if hasattr(self, 'embodiment_ids') and self.embodiment_ids is not None:
            self.embodiment_ids = self.embodiment_ids.pin_memory()

        return self

    def cuda(self, device):
        self.packed_text_ids = self.packed_text_ids.to(device)
        self.packed_text_indexes = self.packed_text_indexes.to(device)
        self.packed_position_ids = self.packed_position_ids.to(device)

        if hasattr(self, 'packed_vit_tokens'):
            self.packed_vit_tokens = self.packed_vit_tokens.to(device)
            self.packed_vit_token_indexes = self.packed_vit_token_indexes.to(device)
            self.vit_token_seqlens = self.vit_token_seqlens.to(device)

        if hasattr(self, 'packed_label_ids'):
            self.packed_label_ids = self.packed_label_ids.to(device)
            self.ce_loss_indexes = self.ce_loss_indexes.to(device)

        if hasattr(self, 'padded_action'):
            self.padded_action = self.padded_action.to(device)
            self.packed_action_indexes = self.packed_action_indexes.to(device)
            self.padded_state = self.padded_state.to(device)
            self.packed_state_indexes = self.packed_state_indexes.to(device)

        if hasattr(self, 'padded_action_mask') and self.padded_action_mask is not None:
            self.padded_action_mask = self.padded_action_mask.to(device)
            
        if hasattr(self, 'embodiment_ids') and self.embodiment_ids is not None:
            self.embodiment_ids = self.embodiment_ids.to(device)

        return self

    def to_dict(self):
        data = dict(
            sequence_length = self.sequence_length,
            num_gen_samples = self.num_gen_samples,
            num_und_samples = self.num_und_samples,
            sample_lens = self.sample_lens,
            packed_text_ids = self.packed_text_ids,
            packed_text_indexes = self.packed_text_indexes,
            packed_position_ids = self.packed_position_ids,
        )

        data['split_lens'] = self.split_lens
        data['attn_modes'] = self.attn_modes

        if hasattr(self, 'packed_vit_tokens'):
            data['packed_vit_tokens'] = self.packed_vit_tokens
            data['packed_vit_token_indexes'] = self.packed_vit_token_indexes
            data['vit_token_seqlens'] = self.vit_token_seqlens

        if hasattr(self, 'packed_label_ids'):
            data['packed_label_ids'] = self.packed_label_ids
            data['ce_loss_indexes'] = self.ce_loss_indexes

        if hasattr(self, 'embodiment_ids') and self.embodiment_ids is not None:
            data['embodiment_ids'] = self.embodiment_ids

        if hasattr(self, 'padded_action'):
            data['padded_action'] = self.padded_action
            data['packed_action_indexes'] = self.packed_action_indexes
            data['padded_state'] = self.padded_state
            data['packed_state_indexes'] = self.packed_state_indexes

            if hasattr(self, 'padded_action_mask') and self.padded_action_mask is not None:
                data['padded_action_mask'] = self.padded_action_mask

        return data
    

def collate_wrapper():
    def collate_fn(batch):
        return SimpleCustomBatch(batch)
    return collate_fn

