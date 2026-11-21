import torch

_IMAGE_TOKEN_IDS = frozenset({151655, 248056})


def slice_vlm_text_part(
    hidden_states_list, attention_mask, input_ids, vision_end_id,
    fixed_text_length=None,
):
    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids)
    seq_len = input_ids.shape[1]
    pos = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
    is_vision_end = input_ids == vision_end_id
    last_vision_end = torch.where(is_vision_end, pos, pos.new_full(pos.shape, -1)).max(dim=1).values
    text_mask = (pos > last_vision_end.unsqueeze(1)) & attention_mask.bool()

    if fixed_text_length is not None:
        keep = fixed_text_length
    else:
        keep = max(int(text_mask.sum(dim=1).max().item()), 1)
    sliced = [h[:, seq_len - keep:, :] for h in hidden_states_list]
    return sliced, text_mask[:, seq_len - keep:].to(attention_mask.dtype)


def compute_fixed_text_length(input_ids, vlm_fixed_sequence_length):
    batch_size = input_ids.shape[0]
    n_img_total = sum(
        (input_ids == tid).sum().item() for tid in _IMAGE_TOKEN_IDS
    )
    n_img_per_sample = n_img_total // max(batch_size, 1)
    text_length = vlm_fixed_sequence_length - n_img_per_sample
    if text_length <= 0:
        raise ValueError(
            f"Vision tokens ({n_img_per_sample} per sample) consume the "
            f"entire vlm_fixed_sequence_length ({vlm_fixed_sequence_length}). "
            f"Multi-camera VLM input exceeds the sequence budget — reduce "
            f"the number of input images or increase vlm_fixed_sequence_length."
        )
    return text_length
