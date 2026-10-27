import torch

def _expand_batch_dict(batch: dict, N: int) -> dict:
    """Return a new batch dict with every leading-B dim expanded by N via
    repeat_interleave. Tensors get repeat_interleave(N, dim=0); lists/tuples
    of per-sample items get each element repeated N times in place; everything
    else passes through unchanged.

    repeat_interleave order: [a, b] with N=2 -> [a, a, b, b]. This matches the
    (B_raw, N) reshape used in reward aggregation: v_gen_flat[b*N + n] is the
    n-th member of group b.
    """
    out = {}
    for (k, v) in batch.items():
        if isinstance(v, torch.Tensor):
            out[k] = v.repeat_interleave(N, dim=0)
        elif isinstance(v, (list, tuple)):
            expanded = [x for x in v for _ in range(N)]
            out[k] = type(v)(expanded)
        else:
            out[k] = v
    return out
