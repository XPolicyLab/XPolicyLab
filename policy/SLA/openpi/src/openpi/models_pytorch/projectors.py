"""Implementation of additional projectors for additional inputs to the VLA models."""
import torch
import torch.nn as nn
import openpi.models.gemma as _gemma

class AlignProjector(nn.Module):
    """
    calculate the alignment between LLM and VGGT embeddings.
    """
    def __init__(
            self,
            llm_dim: int, 
            config,
        ) -> None:
        super().__init__()

        self.llm_dim = llm_dim
        self.vggt_dim = config.vggt_dim
        self.vla_layers_align = config.vla_layers_align
        self.vla_adapt_layers_align = config.vla_adapt_layers_align
        self.vggt_layers_align = config.vggt_layers_align
        self.vggt_adapt_layers_align = config.vggt_adapt_layers_align
        self.iSF_conv = config.iSF_conv
        self.iSF_zscore = config.iSF_zscore

        if self.iSF_conv:
            kernal_size = 3
            self.conv = nn.Conv2d(self.llm_dim, 2 * self.vggt_dim, 
                                  kernel_size=kernal_size, stride=1, padding=kernal_size//2
                                  )
        else:
            self.fc1 = nn.Linear(self.llm_dim, 2 * self.vggt_dim, bias=True)
            self.fc2 = nn.Linear(2 * self.vggt_dim, 2 * self.vggt_dim, bias=True)
            self.act_fn1 = nn.GELU()
        
        self.vlm_norm = nn.LayerNorm(llm_dim) if config.use_vlm_norm else None

        if self.vla_adapt_layers_align:
            self.vla_mix_mat = nn.Embedding(19, 2 * self.vggt_dim)
        if self.vggt_adapt_layers_align:
            self.vggt_mix_mat = nn.Embedding(24, 2 * self.vggt_dim)

        self.initialize_weights()
    
    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            if isinstance(module, nn.Embedding):
                torch.nn.init.xavier_uniform_(module.weight)
        self.apply(_basic_init)

    def spatial_zscore(self, feat: torch.Tensor, alpha: float = 1.0, eps: float = 1e-6) -> torch.Tensor:
        """
        # Reference: iREPA (https://github.com/end2end-diffusion/irepa)
        Z-score normalization along spatial dimension.

        Args:
            feat: (B, S, D) patch tokens
            alpha: scaling factor for mean subtraction (default 1.0)
            eps: small constant for numerical stability

        Returns:
            Normalized features (B, S, D)
        """
        mean = feat.mean(dim=1, keepdim=True)
        std = feat.std(dim=1, keepdim=True)
        return (feat - alpha * mean) / (std + eps)

    def align_dimension(self, LLM_embedding: torch.Tensor = None, nhw = None) -> torch.Tensor:
        if self.vlm_norm is not None:
            LLM_embedding = self.vlm_norm(LLM_embedding)
        if self.iSF_conv:
            N, H, W = nhw
            bs, S, D = LLM_embedding.shape
            img_ratio = (H * W * N / S)**0.5
            H_feat, W_feat = int(H / img_ratio), int(W / img_ratio)
            # [bs, S, D] -> [bs*N, D, H, W]
            _features = LLM_embedding.reshape(bs*N, H_feat, W_feat, D).permute(0, 3, 1, 2).contiguous()
            projected_features = self.conv(_features)
            projected_features = projected_features.permute(0, 2, 3, 1).contiguous().reshape(bs, S, -1)
        else:
            projected_features = self.fc1(LLM_embedding)
            projected_features = self.act_fn1(projected_features)
            projected_features = self.fc2(projected_features)
        return projected_features
    
    def compute_align_loss_cosine(self, vision_hidden, vggt_hidden, align_mask):
        # vision_hidden has a shape of (bs, N, D)
        def mean_flat(x):
            return torch.mean(x, dim=list(range(1, len(x.size()))))
        align_loss = 0
        bsz = vision_hidden.shape[0]
        for _vision, _vggt, _mask in zip(vision_hidden, vggt_hidden, align_mask):
            _vision = torch.nn.functional.normalize(_vision, dim=-1)
            _vggt = torch.nn.functional.normalize(_vggt, dim=-1)
            # align_loss += 1 - torch.mean(vision_hidden * vggt_hidden).sum(dim=-1).mean()
            align_loss += 1 - mean_flat((_vision * _vggt)[_mask].sum(dim=-1))  # Cosine similarity loss
        align_loss /= bsz  # Average over batch size
        return align_loss
    
    def forward(self, LLM_embs, target_embs, align_mask=None, nhw=None):
        # get LLM_emb at specified layer
        if self.vla_adapt_layers_align:
            LLM_embs = [self.align_dimension(LLM_layer, nhw) for LLM_layer in LLM_embs]
            LLM_embs_pool = torch.stack([LLM_layer.mean(dim=1) for LLM_layer in LLM_embs], dim=1)
            LLM_mix_weight = torch.einsum('nd,bnd->bn', self.vla_mix_mat.weight, LLM_embs_pool)
            LLM_emb = torch.einsum('bn,bnsd->bsd', torch.softmax(LLM_mix_weight, dim=-1), torch.stack(LLM_embs, dim=1))
        else:
            LLM_emb = self.align_dimension(LLM_embs[self.vla_layers_align], nhw)

        # get target_emb at specified layer
        if self.vggt_adapt_layers_align:
            target_embs_pool = torch.stack([target_layer.mean(dim=1) for target_layer in target_embs], dim=1)
            target_mix_weight = torch.einsum('nd,bnd->bn', self.vggt_mix_mat.weight, target_embs_pool)
            target_emb = torch.einsum('bn,bnsd->bsd', torch.softmax(target_mix_weight, dim=-1), torch.stack(target_embs, dim=1))
        else:
            target_emb = target_embs[self.vggt_layers_align]
        if self.iSF_zscore:
            target_emb = self.spatial_zscore(target_emb)

        # project vla dimension and calculate align loss
        align_loss = self.compute_align_loss_cosine(LLM_emb, target_emb, align_mask).mean()  # mean for sequence length
        return align_loss
