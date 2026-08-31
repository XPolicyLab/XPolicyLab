"""RoboTwin dataset factory and batch collation for OLA-SEM LAP training."""

from typing import Any, Dict, List, Optional

from omegaconf import OmegaConf
import torch


def create_dataset(config: OmegaConf, val: bool = False):
    """Create the only dataset supported by this open-source subset: RoboTwin."""
    dataset_type = config.dataset.get("type", "robotwin")
    if dataset_type != "robotwin":
        raise ValueError(
            f"Unsupported dataset type {dataset_type!r}; this release supports only 'robotwin'."
        )

    from .robotwin2.robotwin_agilex_dataset import RobotWinTaskDataset

    params = {
        "global_downsample_rate": config.common.global_downsample_rate,
        "video_action_freq_ratio": config.common.video_action_freq_ratio,
        "num_video_frames": config.common.num_video_frames,
        "video_size": (config.common.video_height, config.common.video_width),
        "dataset_dir": config.dataset.dataset_dir,
        "val": val,
    }
    optional_dataset_fields = (
        "data_mode",
        "task_mode",
        "task_name",
        "max_episodes",
        "randomized_limit_per_task",
        "use_language_action",
        "enable_ik_language_action_sampling",
        "ik_language_action_sampling_rate",
    )
    for field in optional_dataset_fields:
        if hasattr(config.dataset, field):
            params[field] = getattr(config.dataset, field)

    if hasattr(config.dataset, "image_aug"):
        params["image_aug"] = bool(config.dataset.image_aug) and not val
    if hasattr(config.model, "vlm") and hasattr(config.model.vlm, "checkpoint_path"):
        params["vlm_checkpoint_path"] = config.model.vlm.checkpoint_path
    if hasattr(config.dataset, "params"):
        params.update(OmegaConf.to_object(config.dataset.params))

    flow_source = config.model.get("flow_source", {})
    params["include_history_actions"] = flow_source.get("mode", "gaussian") == "history"
    params["history_action_length"] = int(
        flow_source.get(
            "history_length",
            config.common.num_video_frames * config.common.video_action_freq_ratio,
        )
    )
    return RobotWinTaskDataset(**params)


def _process_vlm_inputs_batch(vlm_inputs: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
    """Process and batch VLM inputs with padding."""
    # Extract components
    input_ids_list = [vlm_input['input_ids'] for vlm_input in vlm_inputs]
    pixel_values_list = [vlm_input.get('pixel_values') for vlm_input in vlm_inputs]
    image_grid_thw_list = [vlm_input.get('image_grid_thw') for vlm_input in vlm_inputs]
    attention_mask_list = [vlm_input.get('attention_mask') for vlm_input in vlm_inputs]
    # if any(vlm_input.get('labels') is not None for vlm_input in vlm_inputs):
    #     labels_list = [vlm_input.get('labels') for vlm_input in vlm_inputs]
    # else:
    #     labels_list = None

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
            # padded_labels = labels
        padded_input_ids.append(padded_ids)
        padded_attention_masks.append(padded_mask)
        # padded_labels.append(padded_labels)
    
    # Batch everything
    return {
        'input_ids': torch.cat(padded_input_ids, dim=0),
        'pixel_values': torch.cat([pv for pv in pixel_values_list if pv is not None], dim=0) if pixel_values_list and any(pv is not None for pv in pixel_values_list) else None,
        'image_grid_thw': torch.cat([igt for igt in image_grid_thw_list if igt is not None], dim=0) if image_grid_thw_list and any(igt is not None for igt in image_grid_thw_list) else None,
        'attention_mask': torch.cat([mask for mask in padded_attention_masks if mask is not None], dim=0) if any(mask is not None for mask in padded_attention_masks) else None,
    }
def _process_vlm_inputs_batch_lap(vlm_inputs: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
    """Process and batch VLM inputs with padding."""
    # Extract components
    input_ids_list = [vlm_input['input_ids'] for vlm_input in vlm_inputs]
    pixel_values_list = [vlm_input.get('pixel_values') for vlm_input in vlm_inputs]
    image_grid_thw_list = [vlm_input.get('image_grid_thw') for vlm_input in vlm_inputs]
    attention_mask_list = [vlm_input.get('attention_mask') for vlm_input in vlm_inputs]
    if any(vlm_input.get('labels') is not None for vlm_input in vlm_inputs):
        labels_list = [vlm_input.get('labels') for vlm_input in vlm_inputs]
    else:
        labels_list = None
    if any(vlm_input.get('answer_start') is not None for vlm_input in vlm_inputs):
        answer_start_list = [vlm_input.get('answer_start') for vlm_input in vlm_inputs]
    else:
        answer_start_list = None
    # if any(vlm_input.get('language_action') is not None for vlm_input in vlm_inputs):
    #     language_action_list = [vlm_input.get('language_action') for vlm_input in vlm_inputs]
    # else:
    #     language_action_list = None
    
    # Pad input_ids to same length (simplified like model implementation)
    max_seq_len = max(ids.shape[1] for ids in input_ids_list)
    padded_input_ids = []
    padded_attention_masks = []
    padded_labels_list = []

    for ids, mask, labels in zip(input_ids_list, attention_mask_list,labels_list):
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
            if labels is not None:
                labels_padding = torch.full((mask.shape[0], padding_size), -100, dtype=labels.dtype, device=labels.device)
                padded_labels = torch.cat([labels, labels_padding], dim=1)
            else:
                padded_labels = None
        else:
            padded_ids = ids
            padded_mask = mask
            padded_labels = labels
        padded_input_ids.append(padded_ids)
        padded_attention_masks.append(padded_mask)
        padded_labels_list.append(padded_labels)
    
    # Batch everything
    return {
        'input_ids': torch.cat(padded_input_ids, dim=0),
        'pixel_values': torch.cat([pv for pv in pixel_values_list if pv is not None], dim=0) if pixel_values_list and any(pv is not None for pv in pixel_values_list) else None,
        'image_grid_thw': torch.cat([igt for igt in image_grid_thw_list if igt is not None], dim=0) if image_grid_thw_list and any(igt is not None for igt in image_grid_thw_list) else None,
        'attention_mask': torch.cat([mask for mask in padded_attention_masks if mask is not None], dim=0) if any(mask is not None for mask in padded_attention_masks) else None,
        'labels': torch.cat(padded_labels_list, dim=0),
        'answer_start': torch.cat([answer_start for answer_start in answer_start_list if answer_start is not None], dim=0) if any(answer_start is not None for answer_start in answer_start_list) else None,
        # 'language_action': torch.cat([language_action for language_action in language_action_list if language_action is not None], dim=0) if any(language_action is not None for language_action in language_action_list) else None,
    }


def _process_language_embeddings_batch(language_embeddings: List[torch.Tensor], text_len: int = 512) -> torch.Tensor:
    """Process and batch language embeddings with padding."""
    padded_embeddings = []
    
    for emb in language_embeddings:
        if emb.shape[0] <= text_len:
            padded = torch.cat([emb, emb.new_zeros(text_len - emb.shape[0], emb.shape[1])])
        else:
            padded = emb[:text_len]
        padded_embeddings.append(padded)
    
    # Stack to [B, seq_len, dim]
    return torch.stack(padded_embeddings, dim=0)


def collate_fn(batch: List[Optional[Dict[str, Any]]]) -> Optional[Dict[str, Any]]:
    """
    Universal collate function for all datasets.
    
    Args:
        batch: List of sample dictionaries (may contain None)
        
    Returns:
        Batched dictionary or None if all samples are None
    """
    # Filter out None samples
    batch = [sample for sample in batch if sample is not None]
    
    if len(batch) == 0:
        return None
    
    # Stack tensors（支持无 initial_state 的样本）
    first_frames = torch.stack([sample['first_frame'] for sample in batch])
    video_frames = torch.stack([sample['video_frames'] for sample in batch])
    action_sequences = torch.stack([sample['action_sequence'] for sample in batch])
    has_history_actions = all(
        ('history_action_sequence' in sample and sample['history_action_sequence'] is not None)
        for sample in batch
    )
    history_action_sequences = (
        torch.stack([sample['history_action_sequence'] for sample in batch])
        if has_history_actions
        else None
    )
    has_action_mask = all(('action_mask' in sample and sample['action_mask'] is not None) for sample in batch)
    action_masks = torch.stack([sample['action_mask'] for sample in batch]) if has_action_mask else None
    has_initial_state = all(('initial_state' in sample and sample['initial_state'] is not None) for sample in batch)
    initial_states = torch.stack([sample['initial_state'] for sample in batch]) if has_initial_state else None
    dataset_names = [sample.get('dataset_name') for sample in batch]
    
    # Process VLM inputs with padding in collate_fn
    vlm_inputs = [sample.get('vlm_inputs') for sample in batch]
    processed_vlm_inputs = None
    if vlm_inputs and all(vlm_input is not None for vlm_input in vlm_inputs):
        if any(vlm_input.get('labels') is not None for vlm_input in vlm_inputs):
            processed_vlm_inputs = _process_vlm_inputs_batch_lap(vlm_inputs)
        else:
            processed_vlm_inputs = _process_vlm_inputs_batch(vlm_inputs)
    
    # Process language embeddings with padding in collate_fn  
    language_embeddings = [sample.get('language_embedding') for sample in batch if 'language_embedding' in sample]
    processed_language_embeddings = None
    if language_embeddings and any(emb is not None for emb in language_embeddings):
        processed_language_embeddings = _process_language_embeddings_batch(language_embeddings)
    # print("labels:",processed_vlm_inputs['labels'].shape)
    # print("answer_start:",processed_vlm_inputs['answer_start'].shape)
    result = {
        'first_frame': first_frames,             # [B, C, H, W]
        'video_frames': video_frames,            # [B, F, C, H, W]
        'action_sequence': action_sequences,     # [B, F, D]
        'vlm_inputs': processed_vlm_inputs,
        'language_embedding': processed_language_embeddings,
    }

    if action_masks is not None:
        result['action_mask'] = action_masks
    if history_action_sequences is not None:
        result['history_action_sequence'] = history_action_sequences
    if initial_states is not None:
        result['initial_state'] = initial_states
    if any(name is not None for name in dataset_names):
        result['dataset_name'] = dataset_names
    
    return result

