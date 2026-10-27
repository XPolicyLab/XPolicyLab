import torch
from policy_core.utils.common_utils import get_device

def adaptive_weighted_loss(loss_per_sample: torch.Tensor, norm_p: float=1.0, norm_eps: float=0.01):
    """
    Adaptive weighting: loss / stopgrad((loss + eps)^p)
    """
    weight = (loss_per_sample + norm_eps).pow(norm_p)
    return loss_per_sample / weight.detach()

def compute_video_loss(cfg, *, batch, device=None, batch_size, n_view, n_sparse_frames, n_dense_frames, mem_size, video_frame_indices, data_config, conditioning_mask, V, target, v_tangent, skip_v_head, future_token_count, future_token_start, sparse_tok_start, u, h_du_dt_raw, h_du_dt_capped, dudt_limit_on):
    if device is None:
        device = get_device()
    cond_weight = 1 - conditioning_mask.unsqueeze(-1)
    err_u = (V.float() - target.float()).pow(2)
    err_v = None if skip_v_head else (v_tangent.float() - target.float()).pow(2)
    n_channels = err_u.shape[2]
    if bool(cfg.cotrain.loss_normalize_by_total_tokens):
        loss_token_denom = err_u.shape[1] * n_channels
    else:
        loss_token_denom = future_token_count * n_channels

    def region_loss(err, start, end):
        end = err.shape[1] if end is None else end
        if end <= start:
            return err.new_zeros(err.shape[0])
        return (err[:, start:end] * cond_weight[:, start:end]).sum(dim=(1, 2)) / loss_token_denom
    loss_u_dense = region_loss(err_u, future_token_start, sparse_tok_start)
    loss_u_sparse = region_loss(err_u, sparse_tok_start, None)
    loss_u_uncapped_dense = loss_u_uncapped_sparse = None
    loss_u_capped_dense = loss_u_capped_sparse = None
    if h_du_dt_capped is not None:
        V_alt = u + (h_du_dt_raw if dudt_limit_on else h_du_dt_capped)
        err_alt = (V_alt.float() - target.float()).pow(2)
        alt_dense = region_loss(err_alt, future_token_start, sparse_tok_start)
        alt_sparse = region_loss(err_alt, sparse_tok_start, None)
        if dudt_limit_on:
            (loss_u_capped_dense, loss_u_capped_sparse) = (loss_u_dense, loss_u_sparse)
            (loss_u_uncapped_dense, loss_u_uncapped_sparse) = (alt_dense, alt_sparse)
        else:
            (loss_u_uncapped_dense, loss_u_uncapped_sparse) = (loss_u_dense, loss_u_sparse)
            (loss_u_capped_dense, loss_u_capped_sparse) = (alt_dense, alt_sparse)
    if skip_v_head:
        loss_v_dense = torch.zeros_like(loss_u_dense)
        loss_v_sparse = torch.zeros_like(loss_u_sparse)
    else:
        loss_v_dense = region_loss(err_v, future_token_start, sparse_tok_start)
        loss_v_sparse = region_loss(err_v, sparse_tok_start, None)
    fps = batch['fps'].to(device).float()
    fps_scale = (fps / 30.0).repeat_interleave(n_view)
    if n_sparse_frames > 1 and (not cfg.cotrain.uniform_future_loss_weight):
        slices = batch['video_slices'].to(device)[:, video_frame_indices]
        sparse_slices = slices[:, mem_size:][:, n_dense_frames:]
        ref_stride = data_config['continious_action_length'] / max(n_dense_frames, 1)
        sparse_stride = (sparse_slices[:, 1:].float() - sparse_slices[:, :-1].float()).mean(dim=1).clamp_min(ref_stride)
        w_sparse = (ref_stride / sparse_stride).clamp(0.1, 0.5)
        w_sparse_bv = w_sparse.repeat_interleave(n_view)
    else:
        sparse_stride = torch.ones(batch_size, device=device, dtype=fps.dtype)
        w_sparse_bv = torch.ones_like(fps_scale)
    loss_u_weighted = fps_scale * (loss_u_dense + w_sparse_bv * loss_u_sparse)
    loss_v_weighted = fps_scale * (loss_v_dense + w_sparse_bv * loss_v_sparse)
    norm_p = cfg.cotrain.imf_norm_p
    norm_eps = cfg.cotrain.imf_norm_eps
    if norm_p > 0:
        loss_video = adaptive_weighted_loss(loss_u_weighted, norm_p, norm_eps).mean() + adaptive_weighted_loss(loss_v_weighted, norm_p, norm_eps).mean()
    else:
        loss_video = (loss_u_weighted + loss_v_weighted).mean()
    return {'loss_video': loss_video, 'loss_u_dense': loss_u_dense, 'loss_u_sparse': loss_u_sparse, 'loss_v_dense': loss_v_dense, 'loss_v_sparse': loss_v_sparse, 'fps': fps, 'sparse_stride': sparse_stride, 'w_sparse_bv': w_sparse_bv, 'loss_u_uncapped_dense': loss_u_uncapped_dense, 'loss_u_uncapped_sparse': loss_u_uncapped_sparse, 'loss_u_capped_dense': loss_u_capped_dense, 'loss_u_capped_sparse': loss_u_capped_sparse}
