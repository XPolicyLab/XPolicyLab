"""
Transformer for Action Prediction - 标准版本

本模块实现了基于Transformer的动作预测模型，采用RDT（Recurrent Diffusion Transformer）架构。
主要用于标准PSI Policy的动作序列预测，不包含扩散过程的时间步嵌入。

==================================================================================
核心组件：
==================================================================================

1. **TimestepEmbedder**
   - 时间步嵌入器（虽然定义但在此模块中未使用）
   - 用于将标量时间步转换为向量表示
   - 采用正弦位置编码 + MLP投影

2. **CrossAttention**
   - 交叉注意力层，用于融合条件信息（观察特征）到主序列（动作序列）
   - 支持Flash Attention加速
   - 支持可选的注意力掩码（用于训练时的数据增强）

3. **RDTBlock**
   - RDT Transformer块，包含：
     * 自注意力（Self-Attention）
     * 交叉注意力（Cross-Attention）
     * 前馈网络（FFN）
   - 使用RMSNorm而非LayerNorm（保持时间序列的对称性）

4. **TransformerForAction**
   - 主模型类，用于动作序列预测
   - 输入：动作序列 + 观察特征
   - 输出：预测的动作序列

==================================================================================
输入输出格式：
==================================================================================

**输入**：
- `sample`: (B, T, input_dim) - 动作序列
- `cond`: (B, N, n_emb) - 观察特征（已编码）

**输出**：
- `output`: (B, T, output_dim) - 预测的动作序列
- `attention_weights`: List - 可选的注意力权重（用于可视化）

==================================================================================
使用场景：
==================================================================================

✅ **推荐使用**：
- 标准PSI Policy训练
- RS-IMLE损失函数
- 不需要扩散过程的时间步信息

❌ **不推荐使用**：
- 扩散模型训练（使用TransformerForActionDiffusion）
- World Model训练（使用ST-Transformer）

作者：PSI Policy Team
创建日期：2024-12-29
最后更新：2024-12-29
"""

from typing import Union, Optional, Tuple, List
import logging
import torch
import torch.nn as nn
import numpy as np
from psi_policy.model.diffusion.positional_embedding import SinusoidalPosEmb
from psi_policy.model.common.module_attr_mixin import ModuleAttrMixin
import math

from torch.jit import Final
from timm.models.vision_transformer import Attention, Mlp, RmsNorm, use_fused_attn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

# ================================================================================
# 时间步嵌入器（Timestep Embedder）
# ================================================================================
# 注意：此模块在此文件中定义但未使用，主要用于扩散模型版本
# ================================================================================
class TimestepEmbedder(nn.Module):
    """
    时间步嵌入器：将标量时间步转换为向量表示

    此模块采用正弦位置编码 + MLP投影的方式，将扩散过程中的时间步t转换为
    高维向量表示，用于条件化扩散模型。

    参考：RDT (Recurrent Diffusion Transformer), DiT (Diffusion Transformer)

    Args:
        hidden_size (int): 输出隐藏层维度
        frequency_embedding_size (int): 频率嵌入维度，默认256
        dtype (torch.dtype): 数据类型，默认float32

    Notes:
        - 使用正弦/余弦位置编码生成频率嵌入
        - 通过MLP将频率嵌入投影到目标维度
        - 支持分数时间步（fractional timesteps）
    """
    def __init__(self, hidden_size, frequency_embedding_size=256, dtype=torch.float32):
        super().__init__()
        # MLP投影网络：频率嵌入 -> 隐藏层维度
        # 结构：Linear -> SiLU -> Linear
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),  # Swish激活函数
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size
        self.dtype = dtype

    def timestep_embedding(self, t, dim, max_period=10000):
        """
        创建正弦时间步嵌入

        使用正弦/余弦函数生成位置编码，类似于Transformer的位置编码，
        但用于时间步而非序列位置。

        Args:
            t (torch.Tensor): 时间步张量，shape=(N,)，每个batch元素一个时间步
                            可以是分数值（fractional）
            dim (int): 输出维度
            max_period (int): 控制最小频率，默认10000

        Returns:
            torch.Tensor: 位置嵌入，shape=(N, dim)

        Notes:
            - 参考OpenAI GLIDE的实现
            - 使用指数衰减的频率：freqs = exp(-log(max_period) * i / half)
            - 对每个频率计算cos和sin，然后拼接
            - 如果dim为奇数，最后一个维度补零
        """
        # 参考：https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2

        # 计算频率：指数衰减，从max_period到1
        # 频率越高，对应的时间步变化越敏感
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(
                start=0, end=half, dtype=torch.float32, device=t.device) / half
        )

        # 计算相位：t * freqs
        args = t[:, None].float() * freqs[None]

        # 生成正弦/余弦嵌入并拼接
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)

        # 如果维度为奇数，补零
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)

        return embedding.to(self.dtype)

    def forward(self, t):
        """
        前向传播：将时间步转换为嵌入向量

        Args:
            t (torch.Tensor): 时间步，shape=(N,)

        Returns:
            torch.Tensor: 时间步嵌入，shape=(N, hidden_size)
        """
        # 1. 生成频率嵌入
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)

        # 2. 通过MLP投影到目标维度
        t_emb = self.mlp(t_freq)

        return t_emb

# ================================================================================
# 交叉注意力层（Cross Attention）
# ================================================================================
# 用于将条件信息（观察特征）融合到主序列（动作序列）中
# ================================================================================
class CrossAttention(nn.Module):
    """
    交叉注意力层：将条件信息融合到主序列

    此模块实现交叉注意力机制，允许动作序列（query）关注观察特征（key/value）。
    支持Flash Attention加速和可选的注意力掩码（用于训练时的数据增强）。

    参考：RDT (Recurrent Diffusion Transformer)

    Args:
        dim (int): 特征维度
        num_heads (int): 注意力头数，默认8
        qkv_bias (bool): 是否使用QKV偏置，默认False
        qk_norm (bool): 是否对Q和K进行归一化，默认False
        attn_drop (float): 注意力dropout率，默认0
        proj_drop (float): 投影dropout率，默认0
        norm_layer (nn.Module): 归一化层类型，默认LayerNorm
        attn_mask_kwargs (dict): 注意力掩码配置，包含：
            - use_attn_mask: 是否使用掩码
            - obs_part_length: 观察部分的长度列表
            - max_cond_tokens: 最大条件token数
            - action_horizon: 动作序列长度

    Notes:
        - 使用分离的Q和KV投影（不同于自注意力的QKV）
        - 支持Flash Attention（如果可用）
        - 训练时随机选择掩码模式（数据增强）
    """
    fused_attn: Final[bool]  # 是否使用融合注意力（Flash Attention）
    def __init__(
            self,
            dim: int,
            num_heads: int = 8,
            qkv_bias: bool = False,
            qk_norm: bool = False,
            attn_drop: float = 0,
            proj_drop: float = 0,
            norm_layer: nn.Module = nn.LayerNorm,
            attn_mask_kwargs: dict = None,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.fused_attn = use_fused_attn()

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        # ================================================================
        # 注意力掩码配置（用于训练时的数据增强）
        # ================================================================
        self.use_attn_mask = attn_mask_kwargs['use_attn_mask']
        if self.use_attn_mask:
            raw_part_length = attn_mask_kwargs['obs_part_length']
            max_cond_tokens = int(attn_mask_kwargs['max_cond_tokens'])
            action_horizon = int(attn_mask_kwargs['action_horizon'])

            part_lengths = [int(v) for v in raw_part_length]
            if len(part_lengths) < 2:
                raise ValueError(
                    "use_attn_mask=True 时，obs_part_length 至少需要 2 个分段。"
                )
            if any(v <= 0 for v in part_lengths):
                raise ValueError(f"obs_part_length 必须全部为正整数，实际 {part_lengths}")
            if sum(part_lengths) != max_cond_tokens:
                raise ValueError(
                    "obs_part_length 与 max_cond_tokens 不一致: "
                    f"sum(obs_part_length)={sum(part_lengths)}, max_cond_tokens={max_cond_tokens}"
                )

            # 生成每段 token 的 [start, end) 区间。
            segment_ranges: list[tuple[int, int]] = []
            start = 0
            for length in part_lengths:
                end = start + length
                segment_ranges.append((start, end))
                start = end

            mask_candidates: list[torch.Tensor] = []
            for start, end in segment_ranges:
                # 注意：当前实现中 False 位置会被屏蔽（见 forward 的 logical_not 逻辑）。
                mask = torch.ones((action_horizon, max_cond_tokens), dtype=torch.bool)
                mask[:, start:end] = False
                mask_candidates.append(mask)

            # 最后一种模式：不做任何分段屏蔽。
            mask_candidates.append(torch.ones((action_horizon, max_cond_tokens), dtype=torch.bool))

            masks = torch.stack(mask_candidates, dim=0)
            self.register_buffer('masks', masks)

            # 约定：70% 概率使用全可见；其余概率均分到各个“单段屏蔽”模式。
            masked_mode_count = len(mask_candidates) - 1
            probs = torch.full((len(mask_candidates),), 0.0, dtype=torch.float32)
            probs[:-1] = 0.3 / float(masked_mode_count)
            probs[-1] = 0.7
            self.register_buffer('probs', probs)

    def _sample_attn_mask(self, *, training: bool) -> Optional[torch.Tensor]:
        if not (self.use_attn_mask and training):
            return None
        idx = torch.multinomial(self.probs, num_samples=1)
        return self.masks[idx.squeeze()]

    def _forward_impl(
        self,
        x: torch.Tensor,
        c: torch.Tensor,
        *,
        training: bool,
        capture_attn_map: bool,
    ) -> Tuple[torch.Tensor, Optional[np.ndarray]]:
        """
        前向传播：交叉注意力计算

        Args:
            x (torch.Tensor): 主序列（动作序列），shape=(B, T, n_emb)
                - B: batch size
                - T: 动作序列长度
                - n_emb: 特征维度
            c (torch.Tensor): 条件信息（观察特征），shape=(B, N, n_emb)
                - N: 条件token数量（观察特征 + 可选的时间嵌入）
            training (bool): 是否为训练模式，默认True
            capture_attn_map (bool): 是否生成注意力权重图

        Returns:
            torch.Tensor: 输出特征，shape=(B, T, n_emb)
            torch.Tensor: 注意力权重（如果gen_attn_map=True），否则None
        """
        B, N, C = x.shape  # 主序列形状
        _, L, _ = c.shape   # 条件序列形状

        # ================================================================
        # 1. 计算Query（来自主序列x）
        # ================================================================
        # Q: (B, T, n_emb) -> (B, num_heads, T, head_dim)
        q = self.q(x).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        # ================================================================
        # 2. 计算Key和Value（来自条件序列c）
        # ================================================================
        # KV: (B, L, n_emb) -> (B, L, 2, num_heads, head_dim) -> 分离K和V
        kv = self.kv(c).reshape(B, L, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        k, v = kv.unbind(0)  # K和V: (B, num_heads, L, head_dim)

        # ================================================================
        # 3. 归一化Q和K（如果启用）
        # ================================================================
        q, k = self.q_norm(q), self.k_norm(k)

        # ================================================================
        # 4. 生成注意力掩码（训练时的数据增强）
        # ================================================================
        attn_mask = self._sample_attn_mask(training=training)

        # ================================================================
        # 5. 计算注意力（使用Flash Attention或标准实现）
        # ================================================================
        attn_weights = None
        if self.fused_attn:
            # 使用Flash Attention（如果可用，更快）
            x = F.scaled_dot_product_attention(
                q, k, v,
                dropout_p=self.attn_drop.p if self.training else 0.,
                attn_mask=attn_mask
            )
            # 如果需要生成注意力权重图，手动计算
            if capture_attn_map:
                with torch.no_grad():
                    attn_weights = (q @ k.transpose(-2, -1)) * self.scale
                    if attn_mask is not None:
                        # 将mask的位置设为-inf（softmax后为0）
                        attn_weights = attn_weights.masked_fill_(attn_mask.logical_not(), float('-inf'))
                    attn_weights = attn_weights.softmax(dim=-1)  # (B, num_heads, T, L)
                    # 转换为numpy并只保留前2个batch的前几个头
                    attn_weights = attn_weights.detach().cpu().to(torch.float16).numpy()
                    attn_weights = attn_weights[:2, :, 0, :]
        else:
            # 标准注意力实现
            q = q * self.scale  # 缩放
            attn = q @ k.transpose(-2, -1)  # (B, num_heads, T, L)

            # 应用掩码
            if attn_mask is not None:
                attn = attn.masked_fill_(attn_mask.logical_not(), float('-inf'))

            attn = attn.softmax(dim=-1)  # Softmax归一化

            # 生成注意力权重图（如果需要）
            if capture_attn_map:
                attn_weights = attn.detach().cpu().to(torch.float16).numpy()  # (B, num_heads, T, L)
                attn_weights = attn_weights[:2, :, 0, :]

            # Dropout
            if self.attn_drop.p > 0:
                attn = self.attn_drop(attn)

            # 加权求和
            x = attn @ v  # (B, num_heads, T, head_dim)

        # ================================================================
        # 6. 重塑并投影输出
        # ================================================================
        # 重塑：(B, num_heads, T, head_dim) -> (B, T, n_emb)
        x = x.permute(0, 2, 1, 3).reshape(B, N, C)

        # 输出投影
        x = self.proj(x)

        # 输出dropout
        if self.proj_drop.p > 0:
            x = self.proj_drop(x)

        return x, attn_weights

    def forward(self, x: torch.Tensor, c: torch.Tensor, training=True) -> torch.Tensor:
        x, _ = self._forward_impl(x, c, training=training, capture_attn_map=False)
        return x

    def forward_with_attn_map(self, x: torch.Tensor, c: torch.Tensor, training=True) -> Tuple[torch.Tensor, Optional[np.ndarray]]:
        return self._forward_impl(x, c, training=training, capture_attn_map=True)


# ================================================================================
# RDT Transformer块（RDTBlock）
# ================================================================================
# RDT (Recurrent Diffusion Transformer) 的核心构建块
# 包含：自注意力 + 交叉注意力 + 前馈网络
# ================================================================================
class RDTBlock(nn.Module):
    """
    RDT Transformer块：标准Transformer块的变体

    此块包含三个主要组件：
    1. 自注意力（Self-Attention）：动作序列内部交互
    2. 交叉注意力（Cross-Attention）：动作序列与观察特征交互
    3. 前馈网络（FFN）：非线性变换

    使用RMSNorm而非LayerNorm，以保持时间序列的对称性。

    Args:
        hidden_size (int): 隐藏层维度
        num_heads (int): 注意力头数
        attn_mask_kwargs (dict): 注意力掩码配置
        **block_kwargs: 其他块参数（dropout等）
    """
    def __init__(self, hidden_size, num_heads, attn_mask_kwargs, **block_kwargs):
        super().__init__()

        # ================================================================
        # 1. 自注意力分支
        # ================================================================
        # 输入：x (B, T, n_emb) - 动作序列
        self.norm1 = RmsNorm(hidden_size, eps=1e-6)
        self.attn = Attention(
            dim=hidden_size, num_heads=num_heads,
            qkv_bias=True, qk_norm=True,  # 启用QKV偏置和QK归一化
            norm_layer=RmsNorm, **block_kwargs)

        # ================================================================
        # 2. 交叉注意力分支
        # ================================================================
        # 输入：x (B, T, n_emb) - 动作序列
        #       c (B, N, n_emb) - 观察特征（条件）
        self.cross_attn = CrossAttention(
            hidden_size, num_heads=num_heads,
            qkv_bias=True, qk_norm=True,
            norm_layer=RmsNorm,
            attn_mask_kwargs=attn_mask_kwargs, **block_kwargs)
        self.norm2 = RmsNorm(hidden_size, eps=1e-6)

        # ================================================================
        # 3. 前馈网络（FFN）
        # ================================================================
        # 使用近似GELU激活函数（tanh近似，更快）
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.ffn = Mlp(
            in_features=hidden_size,
            hidden_features=hidden_size,  # 隐藏层维度等于输入维度
            act_layer=approx_gelu,
            drop=0  # 不使用dropout
        )
        self.norm3 = RmsNorm(hidden_size, eps=1e-6)

    def forward(self, x, c, training=True):
        """
        前向传播：RDT块计算

        Args:
            x (torch.Tensor): 动作序列，shape=(B, T, n_emb)
            c (torch.Tensor): 观察特征（条件），shape=(B, N, n_emb)
            training (bool): 是否为训练模式，默认True
        Returns:
            torch.Tensor: 输出特征，shape=(B, T, n_emb)
        """
        # ================================================================
        # 1. 自注意力（残差连接）
        # ================================================================
        origin_x = x
        x = self.norm1(x)  # RMS归一化
        x = self.attn(x)   # 自注意力
        x = x + origin_x   # 残差连接

        # ================================================================
        # 2. 交叉注意力（残差连接）
        # ================================================================
        origin_x = x
        x = self.norm2(x)  # RMS归一化
        x = self.cross_attn(x, c, training=training)
        x = x + origin_x   # 残差连接

        # ================================================================
        # 3. 前馈网络（残差连接）
        # ================================================================
        origin_x = x
        x = self.norm3(x)  # RMS归一化
        x = self.ffn(x)    # 前馈网络
        x = x + origin_x   # 残差连接

        return x

    def forward_with_attn_map(self, x, c, training=True):
        origin_x = x
        x = self.norm1(x)
        x = self.attn(x)
        x = x + origin_x

        origin_x = x
        x = self.norm2(x)
        x, attn_weights = self.cross_attn.forward_with_attn_map(x, c, training=training)
        x = x + origin_x

        origin_x = x
        x = self.norm3(x)
        x = self.ffn(x)
        x = x + origin_x
        return x, attn_weights



# ================================================================================
# Transformer for Action - 主模型类
# ================================================================================
# 用于动作序列预测的Transformer模型（标准版本，不含扩散时间步）
# ================================================================================
class TransformerForAction(ModuleAttrMixin):
    """
    Transformer for Action：动作序列预测模型

    基于RDT架构的Transformer模型，用于预测机器人动作序列。
    输入动作序列和观察特征，输出预测的动作序列。

    特点：
    - 使用RMSNorm而非LayerNorm（保持时间序列对称性）
    - 支持可选的注意力掩码（训练时数据增强）
    - 可学习的位置嵌入

    Args:
        input_dim (int): 输入动作维度
        output_dim (int): 输出动作维度（通常等于input_dim）
        action_horizon (int): 动作序列长度（时间步数）
        n_layer (int): Transformer层数，默认7
        n_head (int): 注意力头数，默认8
        n_emb (int): 嵌入维度，默认768
        max_cond_tokens (int): 最大条件token数，默认800
        p_drop_attn (float): 注意力dropout率，默认0.1
        obs_part_length (List[int]): 观察部分的长度列表，用于掩码
        use_attn_mask (bool): 是否使用注意力掩码，默认True
    """
    def __init__(self,
        input_dim: int,
        output_dim: int,
        action_horizon: int,
        n_layer: int = 7,
        n_head: int = 8,
        n_emb: int = 768,
        max_cond_tokens: int=800,
        p_drop_attn: float = 0.1,
        obs_part_length: List[int] = None,
        use_attn_mask: bool = True,
        ) -> None:
        super().__init__()

        # ================================================================
        # 输入嵌入层
        # ================================================================
        # 将动作维度投影到嵌入维度
        self.input_emb = nn.Linear(input_dim, n_emb)

        # ================================================================
        # 位置嵌入（可学习）
        # ================================================================
        # 动作序列的位置嵌入：(1, action_horizon, n_emb)
        self.pos_emb = nn.Parameter(torch.randn((1, action_horizon, n_emb)))

        # 条件序列的位置嵌入：(1, max_cond_tokens, n_emb)
        self.cond_pos_emb = nn.Parameter(torch.randn((1, max_cond_tokens, n_emb)))

        attn_mask_kwargs = {
            'use_attn_mask': use_attn_mask,
            'obs_part_length': obs_part_length,
            'max_cond_tokens': max_cond_tokens,
            'action_horizon': action_horizon
        }

        # RDT blocks
        self.blocks = nn.ModuleList([
            RDTBlock(
                hidden_size=n_emb,
                num_heads=n_head,
                attn_drop=p_drop_attn,
                proj_drop=p_drop_attn,
                attn_mask_kwargs=attn_mask_kwargs
            )
            for _ in range(n_layer)
        ])

        # ================================================================
        # 输出层
        # ================================================================
        # 使用RMSNorm而非LayerNorm
        # 原因：时间序列预测任务中，LayerNorm的中心化操作可能导致
        # token偏移和注意力偏移，破坏时间序列的对称性
        # 参考：RDT论文
        self.ln_f = RmsNorm(n_emb, eps=1e-6)
        self.head = nn.Linear(n_emb, output_dim)  # 输出投影到动作维度

        self.action_horizon = action_horizon

        # init
        self.apply(self._init_weights)
        logger.info(
            "Number of parameters in DiT: %e", sum(p.numel() for p in self.parameters())
        )

    def _init_weights(self, module):
        ignore_types = (nn.Dropout,
            ## SinusoidalPosEmb,
            TimestepEmbedder,
            nn.TransformerEncoderLayer,
            nn.TransformerDecoderLayer,
            nn.TransformerEncoder,
            nn.TransformerDecoder,
            nn.ModuleList,
            nn.Mish,
            nn.Sequential,
            nn.Embedding,
            nn.SiLU,
            nn.GELU,
            nn.Identity,
            RDTBlock)
        if isinstance(module, (nn.Linear)):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, (Attention, CrossAttention)):
        # Attention模块中的权重初始化
            if hasattr(module, 'q'):
                torch.nn.init.normal_(module.q.weight, mean=0.0, std=0.02)
                if module.q.bias is not None:
                    torch.nn.init.zeros_(module.q.bias)
            if hasattr(module, 'k'):
                torch.nn.init.normal_(module.k.weight, mean=0.0, std=0.02)
                if module.k.bias is not None:
                    torch.nn.init.zeros_(module.k.bias)
            if hasattr(module, 'v'):
                torch.nn.init.normal_(module.v.weight, mean=0.0, std=0.02)
                if module.v.bias is not None:
                    torch.nn.init.zeros_(module.v.bias)
            if hasattr(module, 'proj'):
                torch.nn.init.normal_(module.proj.weight, mean=0.0, std=0.02)
                if module.proj.bias is not None:
                    torch.nn.init.zeros_(module.proj.bias)
        elif isinstance(module, Mlp):
            # Mlp模块中的权重初始化
            if hasattr(module, 'fc1'):
                torch.nn.init.normal_(module.fc1.weight, mean=0.0, std=0.02)
                if module.fc1.bias is not None:
                    torch.nn.init.zeros_(module.fc1.bias)
            if hasattr(module, 'fc2'):
                torch.nn.init.normal_(module.fc2.weight, mean=0.0, std=0.02)
                if module.fc2.bias is not None:
                    torch.nn.init.zeros_(module.fc2.bias)
        elif isinstance(module, nn.MultiheadAttention):
            weight_names = [
                'in_proj_weight', 'q_proj_weight', 'k_proj_weight', 'v_proj_weight']
            for name in weight_names:
                weight = getattr(module, name)
                if weight is not None:
                    torch.nn.init.normal_(weight, mean=0.0, std=0.02)

            bias_names = ['in_proj_bias', 'bias_k', 'bias_v']
            for name in bias_names:
                bias = getattr(module, name)
                if bias is not None:
                    torch.nn.init.zeros_(bias)
        elif isinstance(module, (nn.LayerNorm)):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)
        elif isinstance(module, RmsNorm):
            torch.nn.init.ones_(module.weight)
        elif isinstance(module, TransformerForAction):
            torch.nn.init.normal_(module.pos_emb, mean=0.0, std=0.02)
            if module.cond_pos_emb is not None:
                torch.nn.init.normal_(module.cond_pos_emb, mean=0.0, std=0.02)
        elif isinstance(module, ignore_types):
            # no param
            pass
        else:
            raise RuntimeError("Unaccounted module {}".format(module))

    def get_optim_groups(self, weight_decay: float=1e-3):
        """
        This long function is unfortunately doing something very simple and is being very defensive:
        We are separating out all parameters of the model into two buckets: those that will experience
        weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
        We are then returning the PyTorch optimizer object.
        """

        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear, torch.nn.MultiheadAttention)
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding, RmsNorm)
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = "%s.%s" % (mn, pn) if mn else pn  # full param name

                if pn.endswith("bias"):
                    # all biases will not be decayed
                    no_decay.add(fpn)
                elif pn.startswith("bias"):
                    # MultiheadAttention bias starts with "bias"
                    no_decay.add(fpn)
                elif "q_norm" in fpn or "k_norm" in fpn:
                    no_decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)

        # special case the position embedding parameter in the root GPT module as not decayed
        no_decay.add("pos_emb")
        no_decay.add("_dummy_variable")
        if hasattr(self, "cond_pos_emb"):
            no_decay.add("cond_pos_emb")

        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert (
            len(inter_params) == 0
        ), "parameters %s made it into both decay/no_decay sets!" % (str(inter_params),)
        assert (
            len(param_dict.keys() - union_params) == 0
        ), "parameters %s were not separated into either decay/no_decay set!" % (
            str(param_dict.keys() - union_params),
        )

        # create the pytorch optimizer object
        optim_groups = [
            {
                "params": [param_dict[pn] for pn in sorted(list(decay))],
                "weight_decay": weight_decay,
            },
            {
                "params": [param_dict[pn] for pn in sorted(list(no_decay))],
                "weight_decay": 0.0,
            },
        ]
        return optim_groups

    def configure_optimizers(self,
            learning_rate: float=1e-4,
            weight_decay: float=1e-3,
            betas: Tuple[float, float]=(0.9,0.95)):
        optim_groups = self.get_optim_groups(weight_decay=weight_decay)
        optimizer = torch.optim.AdamW(
            optim_groups, lr=learning_rate, betas=betas
        )
        return optimizer

    def _forward_impl(
        self,
        sample: torch.Tensor,
        *,
        cond: Optional[torch.Tensor] = None,
        training: bool = True,
    ):
        cond_emb = cond
        tc = cond_emb.shape[1]
        cond_pos_emb = self.cond_pos_emb[:, :tc, :]
        cond_emb = cond_emb + cond_pos_emb

        input_emb = self.input_emb(sample)
        t = input_emb.shape[1]
        pos_emb = self.pos_emb[:, :t, :]
        x = input_emb + pos_emb

        for block in self.blocks:
            x = block(x, cond_emb, training=training)

        x = self.ln_f(x)
        x = self.head(x)
        return x, None

    def forward_with_attn_map(
        self,
        sample: torch.Tensor,
        *,
        cond: Optional[torch.Tensor] = None,
        training: bool = True,
    ):
        cond_emb = cond
        tc = cond_emb.shape[1]
        cond_pos_emb = self.cond_pos_emb[:, :tc, :]
        cond_emb = cond_emb + cond_pos_emb

        input_emb = self.input_emb(sample)
        t = input_emb.shape[1]
        pos_emb = self.pos_emb[:, :t, :]
        x = input_emb + pos_emb

        attention_weights = []
        for block in self.blocks:
            x, attn_weights = block.forward_with_attn_map(x, cond_emb, training=training)
            attention_weights.append(attn_weights)

        x = self.ln_f(x)
        x = self.head(x)
        return x, attention_weights

    def forward(self,
        sample: torch.Tensor,
        cond: Optional[torch.Tensor]=None,
        training=True,
        gen_attn_map=False,
        **kwargs):
        """
        sample: (B,T,input_dim)
        timestep: (B,) or int, diffusion step
        cond: (B,N,n_emb)
        output: (B,T,input_dim)
        return:
            if gen_attn_map:
                output tensor (B,T,input_dim), attention_maps List[(B,num_heads,T,L)]
            else:
                output tensor (B,T,input_dim), None
        """

        if gen_attn_map:
            return self.forward_with_attn_map(sample, cond=cond, training=training)
        return self._forward_impl(sample, cond=cond, training=training)
