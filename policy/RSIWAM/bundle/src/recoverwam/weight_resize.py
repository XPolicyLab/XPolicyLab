import torch
import torch.nn.functional as F


def _interpolate_last_dim(tensor: torch.Tensor, new_size: int) -> torch.Tensor:
    if tensor.shape[-1] == new_size:
        return tensor
    flat = tensor.reshape(-1, 1, tensor.shape[-1]).to(torch.float32)
    flat = F.interpolate(flat, size=new_size, mode="linear", align_corners=True)
    return flat.reshape(*tensor.shape[:-1], new_size)


def _resize_tensor_to_shape(src: torch.Tensor, target_shape: tuple[int, ...]) -> torch.Tensor:
    if tuple(src.shape) == tuple(target_shape):
        return src

    out = src.to(torch.float32)
    while out.ndim < len(target_shape):
        out = out.unsqueeze(0)
    while out.ndim > len(target_shape):
        if out.shape[0] != 1:
            raise ValueError(
                f"Cannot reduce tensor rank for resize: src shape={tuple(src.shape)}, target={target_shape}"
            )
        out = out.squeeze(0)

    for dim, new_size in enumerate(target_shape):
        current_size = out.shape[dim]
        if current_size == new_size:
            continue
        # Permute the target dimension to the end for interpolation
        perm = [i for i in range(out.ndim) if i != dim] + [dim]
        # Construct inverse permutation to restore original order
        inv_perm = [0] * out.ndim
        for i, p in enumerate(perm):
            inv_perm[p] = i
        # Permute, interpolate, and restore original order
        out_perm = out.permute(*perm).contiguous()
        prefix_shape = out_perm.shape[:-1]
        out_perm = _interpolate_last_dim(out_perm, new_size)
        out_perm = out_perm.reshape(*prefix_shape, new_size)
        out = out_perm.permute(*inv_perm).contiguous()

    if tuple(out.shape) != tuple(target_shape):
        raise ValueError(
            f"Resize produced wrong shape for tensor. src={tuple(src.shape)}, target={target_shape}, got={tuple(out.shape)}"
        )
    return out.to(dtype=src.dtype)


def _materialize_compact_tensor(
    value: torch.Tensor,
    *,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Detach a tensor from a large mmap/shared checkpoint storage.

    A torch zip checkpoint loaded with ``mmap=True`` can expose every state
    tensor as a view into one file-sized storage.  ``contiguous()`` is a no-op
    for an already contiguous view, so serializing even a 10 KiB bias may copy
    the entire 30+ GiB source storage.  Clone only when the logical tensor does
    not own an exact compact storage; resized/cast tensors already do.
    """

    compact = value.detach().to(dtype=dtype, device="cpu").contiguous()
    logical_bytes = compact.numel() * compact.element_size()
    if compact.storage_offset() != 0 or compact.untyped_storage().nbytes() != logical_bytes:
        compact = compact.clone(memory_format=torch.contiguous_format)
    return compact
