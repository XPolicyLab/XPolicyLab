import torch
def build_batched_joint_mask(text_valid: torch.Tensor, target_len: int, ref_len: int, action_len: int, dtype: torch.dtype, joint_layout: str='legacy') -> torch.Tensor:
    if joint_layout not in {'legacy', 'native'}:
        raise ValueError(f'Unsupported Liber_0_lite joint_layout: {joint_layout!r}')
    (batch_size, text_len) = text_valid.shape
    device = text_valid.device
    tms = text_len - 1
    proprio = text_len
    video_start = text_len + 1
    video_end = video_start + target_len + ref_len
    action_start = video_end
    total = action_start + action_len
    allowed = torch.zeros(batch_size, total, total, device=device, dtype=torch.bool)
    if tms:
        causal = torch.tril(torch.ones(tms, tms, device=device, dtype=torch.bool))
        valid_ar = text_valid[:, :tms]
        allowed[:, :tms, :tms] = causal.unsqueeze(0) & valid_ar.unsqueeze(1) & valid_ar.unsqueeze(2)
    (batch_indices, padding_indices) = torch.where(~text_valid)
    allowed[batch_indices, padding_indices, padding_indices] = True
    allowed[:, tms, :text_len] = text_valid
    allowed[:, tms, proprio:video_end] = True
    allowed[:, proprio, :text_len] = text_valid
    allowed[:, proprio, proprio] = True
    allowed[:, video_start:video_end, :text_len] = text_valid.unsqueeze(1)
    allowed[:, video_start:video_end, proprio:video_end] = True
    allowed[:, action_start:total, :text_len] = text_valid.unsqueeze(1)
    allowed[:, action_start:total, proprio:total] = True
    if joint_layout == 'native':
        allowed[:, tms, proprio] = False
        allowed[:, video_start:video_end, proprio] = False
    mask = torch.full((batch_size, 1, total, total), torch.finfo(dtype).min, device=device, dtype=dtype)
    return mask.masked_fill(allowed.unsqueeze(1), 0.0)
