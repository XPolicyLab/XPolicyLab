"""Remove dead clean-future tokens from one-chunk teacher-forced training.

Layout before: [history][clean anchor/future][noisy anchor/future][action][state].
Only the clean anchor is read by the sole noisy/action block. History and clean
anchor never read the clean future, and clean outputs are not supervised. Keep
the noisy anchor so even its unsupervised output and gradients remain available.
No parameters, positions, attention visibility, or model output shapes change.
"""
import torch


def compact_clean_future(core, x, kwargs, clean_image_length, fpa_request):
    frame = core.frame_seqlen
    registers = core.num_action_per_block + core.num_state_per_block
    image_length = (1 + core.num_frame_per_block) * frame
    prefix = kwargs['clean_ctx_len']
    eligible = (
        core.prune_single_chunk_clean_future
        and kwargs['is_tf']
        and clean_image_length == image_length
        and kwargs['action_register_length'] == registers
        and x.shape[1] == prefix + 2 * image_length + registers
        and fpa_request is None
        and not any(b.self_attn.gather_teacher_forcing_qkv for b in core.blocks)
    )
    if not eligible:
        return x, kwargs, clean_image_length
    # Every block's attention, normalization, cross-attention and FFN sees only
    # the retained tokens. Keep each token's original timestep and RoPE index.
    def compact(value):
        return torch.cat((value[:, :prefix + frame], value[:, prefix + image_length:]), dim=1)

    kwargs = dict(kwargs, e=compact(kwargs['e']), teacher_forcing_clean_length=frame)
    return compact(x), kwargs, frame
