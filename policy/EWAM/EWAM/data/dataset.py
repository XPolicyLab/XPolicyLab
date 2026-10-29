# Dataset Factory
# Simple factory to create different types of datasets

from typing import Dict, Any, List, Optional
from omegaconf import OmegaConf
import torch


def create_dataset(config: OmegaConf, val: bool = False):
    """
    Create dataset based on config.

    Args:
        config: Configuration object
        val: Whether to create validation dataset

    Returns:
        Dataset instance
    """
    dataset_type = config.dataset.get('type', 'robotwin_dim16')

    if dataset_type == 'robotwin_dim16':
        from .robotwin2.robotwin_agilex_dataset_dim16 import RobotWinTaskDataset as RobotWinTaskDatasetDim16

        params = {}

        if hasattr(config, 'common'):
            params.update({
                'global_downsample_rate': config.common.global_downsample_rate,
                'video_action_freq_ratio': config.common.video_action_freq_ratio,
                'num_video_frames': config.common.num_video_frames,
                'video_size': (config.common.video_height, config.common.video_width),
            })

        if hasattr(config.dataset, 'dataset_dir'):
            params['dataset_dir'] = config.dataset.dataset_dir
        if hasattr(config.dataset, 'data_mode'):
            params['data_mode'] = config.dataset.data_mode
        if hasattr(config.dataset, 'task_mode'):
            params['task_mode'] = config.dataset.task_mode
        if hasattr(config.dataset, 'task_name'):
            params['task_name'] = config.dataset.task_name
        if hasattr(config.dataset, 'max_episodes'):
            params['max_episodes'] = config.dataset.max_episodes
        if hasattr(config.dataset, 'image_aug'):
            params['image_aug'] = config.dataset.image_aug and not val
        if hasattr(config.dataset, 'randomized_limit_per_task'):
            params['randomized_limit_per_task'] = config.dataset.randomized_limit_per_task

        if hasattr(config.model, 'vlm') and hasattr(config.model.vlm, 'checkpoint_path'):
            params['vlm_checkpoint_path'] = config.model.vlm.checkpoint_path

        if hasattr(config.dataset, 'params'):
            additional_params = OmegaConf.to_object(config.dataset.params)
            params.update(additional_params)

        params['val'] = val

        return RobotWinTaskDatasetDim16(**params)

    elif dataset_type == "robodojo":
        from .robodojo.robodojo_dataset import RoboDojoDataset

        params = {}

        if hasattr(config, 'common'):
            params.update({
                'global_downsample_rate': config.common.global_downsample_rate,
                'video_action_freq_ratio': config.common.video_action_freq_ratio,
                'num_video_frames': config.common.num_video_frames,
                'video_size': (config.common.video_height, config.common.video_width),
            })

        params['dataset_dir'] = config.dataset.dataset_dir
        if hasattr(config.dataset, 'max_episodes'):
            params['max_episodes'] = config.dataset.max_episodes
        if hasattr(config.dataset, 'image_aug'):
            params['image_aug'] = config.dataset.image_aug and not val

        if hasattr(config.model, 'vlm') and hasattr(config.model.vlm, 'checkpoint_path'):
            params['vlm_checkpoint_path'] = config.model.vlm.checkpoint_path

        if hasattr(config.dataset, 'params'):
            additional_params = OmegaConf.to_object(config.dataset.params)
            params.update(additional_params)

        params['val'] = val

        return RoboDojoDataset(**params)

    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}. Available types: robotwin_dim16, robodojo")


def _process_vlm_inputs_batch(vlm_inputs: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
    """Process and batch VLM inputs with padding."""
    # Extract components
    input_ids_list = [vlm_input['input_ids'] for vlm_input in vlm_inputs]
    pixel_values_list = [vlm_input.get('pixel_values') for vlm_input in vlm_inputs]
    image_grid_thw_list = [vlm_input.get('image_grid_thw') for vlm_input in vlm_inputs]
    attention_mask_list = [vlm_input.get('attention_mask') for vlm_input in vlm_inputs]

    # Pad input_ids to same length (simplified like model implementation)
    max_seq_len = max(ids.shape[1] for ids in input_ids_list)
    padded_input_ids = []
    padded_attention_masks = []

    for ids, mask in zip(input_ids_list, attention_mask_list):
        if ids.shape[1] < max_seq_len:
            padding_size = max_seq_len - ids.shape[1]
            # Pad input_ids
            padding = torch.zeros(ids.shape[0], padding_size, dtype=ids.dtype, device=ids.device)
            padded_ids = torch.cat([ids, padding], dim=1)
            # Pad attention_mask
            if mask is not None:
                mask_padding = torch.zeros(mask.shape[0], padding_size, dtype=mask.dtype, device=mask.device)
                padded_mask = torch.cat([mask, mask_padding], dim=1)
            else:
                padded_mask = None
        else:
            padded_ids = ids
            padded_mask = mask

        padded_input_ids.append(padded_ids)
        padded_attention_masks.append(padded_mask)

    # Batch everything
    return {
        'input_ids': torch.cat(padded_input_ids, dim=0),
        'pixel_values': torch.cat([pv for pv in pixel_values_list if pv is not None], dim=0) if pixel_values_list and any(pv is not None for pv in pixel_values_list) else None,
        'image_grid_thw': torch.cat([igt for igt in image_grid_thw_list if igt is not None], dim=0) if image_grid_thw_list and any(igt is not None for igt in image_grid_thw_list) else None,
        'attention_mask': torch.cat([mask for mask in padded_attention_masks if mask is not None], dim=0) if any(mask is not None for mask in padded_attention_masks) else None,
    }


def _process_language_embeddings_batch(language_embeddings: List[torch.Tensor], text_len: int = 512) -> torch.Tensor:
    """Process and batch language embeddings with padding."""
    if not language_embeddings or all(emb is None for emb in language_embeddings):
        return None

    padded_embeddings = []

    for emb in language_embeddings:
        if emb is None:
            # Determine expected dimension from first valid embedding
            valid_emb = next(e for e in language_embeddings if e is not None)
            padded = torch.zeros(text_len, valid_emb.shape[-1], dtype=valid_emb.dtype)
        elif emb.shape[0] < text_len:
            padded = torch.cat([emb, emb.new_zeros(text_len - emb.shape[0], emb.shape[1])])
        else:
            padded = emb[:text_len]
        padded_embeddings.append(padded)

    # Stack to [B, seq_len, dim]
    return torch.stack(padded_embeddings, dim=0)


def collate_fn(batch: List[Optional[Dict[str, Any]]]) -> Optional[Dict[str, Any]]:
    """
    Universal collate function for all datasets.

    Supports two text embedding fields:
    - text_embedding: Pre-computed embeddings with any dimension
    - language_embedding: UMT5/T5 embeddings with 1024 dimension

    Args:
        batch: List of sample dictionaries (may contain None)

    Returns:
        Batched dictionary or None if all samples are None
    """
    # Filter out None samples
    batch = [sample for sample in batch if sample is not None]

    if len(batch) == 0:
        return None

    # Stack tensors (samples may omit initial_state)
    first_frames = torch.stack([sample['first_frame'] for sample in batch])
    video_frames = torch.stack([sample['video_frames'] for sample in batch])
    action_sequences = torch.stack([sample['action_sequence'] for sample in batch])
    has_initial_state = all(('initial_state' in sample and sample['initial_state'] is not None) for sample in batch)
    initial_states = torch.stack([sample['initial_state'] for sample in batch]) if has_initial_state else None

    # Process VLM inputs with padding in collate_fn
    vlm_inputs = [sample.get('vlm_inputs') for sample in batch]
    processed_vlm_inputs = None
    if vlm_inputs and all(vlm_input is not None for vlm_input in vlm_inputs):
        processed_vlm_inputs = _process_vlm_inputs_batch(vlm_inputs)

    # Process text_embedding (pre-computed, any dimension)
    text_embeddings = [sample.get('text_embedding') for sample in batch]
    processed_text_embeddings = _process_language_embeddings_batch(text_embeddings)

    # Process language_embeddings (legacy, 1024 dimension for UMT5/T5)
    language_embeddings = [sample.get('language_embedding') for sample in batch]
    processed_language_embeddings = _process_language_embeddings_batch(language_embeddings)

    # Collect text instructions (for online encoding)
    text_instructions = [sample.get('text_instruction') for sample in batch]
    processed_text_instructions = text_instructions if any(t is not None for t in text_instructions) else None

    result = {
        'first_frame': first_frames,             # [B, C, H, W]
        'video_frames': video_frames,            # [B, F, C, H, W]
        'action_sequence': action_sequences,     # [B, F, D]
        'vlm_inputs': processed_vlm_inputs,
        'text_embedding': processed_text_embeddings,  # [B, seq_len, dim] or None
        'language_embedding': processed_language_embeddings,  # [B, seq_len, 1024] or None
        'text_prompts': processed_text_instructions,  # List[str] or None
    }

    if initial_states is not None:
        result['initial_state'] = initial_states

    # Optional action_dim_is_pad mask (only when datasets produce dim-padding, e.g. unify_to_16)
    action_dim_is_pads = [s.get('action_dim_is_pad') for s in batch]
    if all(p is not None for p in action_dim_is_pads):
        result['action_dim_is_pad'] = torch.stack(action_dim_is_pads)  # [B, D]

    return result