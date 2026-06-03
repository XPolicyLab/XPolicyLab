
FIXED_ACTION_DIM = 7
import torch


def make_att_2d_masks(pad_masks: torch.Tensor, att_masks: torch.Tensor) -> torch.Tensor:
    """构造块因果二维注意力掩码，语义与 openpi 一致。

    约定：
    - pad_masks: bool[B, N]，True 表示有效 token
    - att_masks: bool/int[B, N]，1 表示“开启新块”，0 表示与前一个 token 属于同一块
    规则：
    - 同一块内 token 可互相全连。
    - 块 k 可看见所有 ≤k 的块（块级因果）。
    - padding 始终不可见。
    返回：bool[B, N, N]，True 表示允许注意力。
    """
    if att_masks.ndim != 2:
        raise ValueError(att_masks.ndim)
    if pad_masks.ndim != 2:
        raise ValueError(pad_masks.ndim)

    att_masks_int = att_masks.to(dtype=torch.int32)
    cumsum = torch.cumsum(att_masks_int, dim=1)  # [B, N]
    att_2d_masks = cumsum[:, None, :] <= cumsum[:, :, None]  # [B, N, N]
    pad_2d_masks = pad_masks[:, None, :] & pad_masks[:, :, None]  # [B, N, N]
    return att_2d_masks & pad_2d_masks


def prepare_attention_bias_4d(att_2d_masks: torch.Tensor) -> torch.Tensor:
    """将二维可见性掩码转为 4D attention bias，匹配 openpi 数值语义。

    输入：bool[B, N, N]，True 代表允许注意力。
    输出：float[B, 1, N, N]，允许处为 0.0，禁止处为一个很小的负值（近似 -inf）。
    """
    if att_2d_masks.ndim != 3:
        raise ValueError(att_2d_masks.ndim)
    bad = -2.3819763e38
    zero = torch.zeros(1, dtype=torch.float32, device=att_2d_masks.device)
    neg = torch.tensor(bad, dtype=torch.float32, device=att_2d_masks.device)
    bias = torch.where(att_2d_masks, zero, neg)
    return bias[:, None, :, :]