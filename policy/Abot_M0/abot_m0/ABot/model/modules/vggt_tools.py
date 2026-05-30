from vggt.models.vggt import VGGT
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms as TF

class CrossAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_hidden: int,
        nhead: int = 8,
        dropout: float = 0.0,
        kv_dim: int = 2048
    ):
        super().__init__()
        self.d_model = d_model
        self.d_hidden = d_hidden if d_hidden is not None else d_model
        self.nhead = nhead
        self.head_dim = self.d_hidden // nhead
        assert self.d_hidden % nhead == 0, "d_hidden must be divisible by nhead"

        # Projections
        self.q_proj = nn.Linear(d_model, self.d_hidden)
        self.k_proj = nn.Linear(kv_dim, self.d_hidden)
        self.v_proj = nn.Linear(kv_dim, self.d_hidden)
        self.out_proj = nn.Linear(self.d_hidden, d_model)

        self.dropout_attn = nn.Dropout(dropout)
        self.dropout_out = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, image_feature: torch.Tensor, spatial_feature: torch.Tensor):
        """
        Args:
            image_feature: (B, N_img, d_model) — Query
            vggt_feature:   (B, N_vggt, kv_dim) — Key and Value

        Returns:
            fused_image_feature: (B, N_img, d_model)
        """
        B, N_img, _ = image_feature.shape
        _, N_spatial, _ = spatial_feature.shape

        # Project to d_hidden
        q = self.q_proj(image_feature)   # (B, N_img, d_hidden)
        k = self.k_proj(spatial_feature)     # (B, N_vggt, d_hidden)
        v = self.v_proj(spatial_feature)     # (B, N_vggt, d_hidden)

        # Reshape for multi-head: (B, N, d_hidden) -> (B, N, nhead, head_dim) -> (B, nhead, N, head_dim)
        q = q.view(B, N_img, self.nhead, self.head_dim).transpose(1, 2)  # (B, nhead, N_img, head_dim)
        k = k.view(B, N_spatial, self.nhead, self.head_dim).transpose(1, 2)  # (B, nhead, N_vggt, head_dim)
        v = v.view(B, N_spatial, self.nhead, self.head_dim).transpose(1, 2)  # (B, nhead, N_vggt, head_dim)

        # Scaled Dot-Product Attention
        scale = self.head_dim ** -0.5
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * scale  # (B, nhead, N_img, N_vggt)
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout_attn(attn_weights)

        # Weighted sum over values
        attn_output = torch.matmul(attn_weights, v)  # (B, nhead, N_img, head_dim)

        # Concatenate heads and project back
        attn_output = attn_output.transpose(1, 2).contiguous()  # (B, N_img, nhead, head_dim)
        attn_output = attn_output.view(B, N_img, self.d_hidden)  # (B, N_img, d_hidden)

        # Final projection to d_model
        output = self.out_proj(attn_output)  # (B, N_img, d_model)
        output = self.dropout_out(output)

        # Residual connection + LayerNorm
        output = self.norm(image_feature + output)

        return output


def preprocess_images(image_list, target_size, mode='crop'): #  [B，[PLT]]
    batch_images = []
    shapes = set()
    to_tensor = TF.ToTensor()
    # target_size = 518

    # First process all images and collect their shapes
    for imgs in image_list:
        epi_images = []
        for img in imgs:
            width, height = img.size

            if mode == "pad":
                # Make the largest dimension 518px while maintaining aspect ratio
                if width >= height:
                    new_width = target_size
                    new_height = round(height * (new_width / width) / 14) * 14  # Make divisible by 14
                else:
                    new_height = target_size
                    new_width = round(width * (new_height / height) / 14) * 14  # Make divisible by 14
            else:  # mode == "crop"
                # Original behavior: set width to 518px
                new_width = target_size
                # Calculate height maintaining aspect ratio, divisible by 14
                new_height = round(height * (new_width / width) / 14) * 14

            # Resize with new dimensions (width, height)
            # img = img.resize((new_width, new_height), Image.Resampling.BICUBIC)
            img = img.resize((new_width, new_height), Image.Resampling.BICUBIC)
            img = to_tensor(img)  # Convert to tensor (0, 1)

            # Center crop height if it's larger than 518 (only in crop mode)
            if mode == "crop" and new_height > target_size:
                start_y = (new_height - target_size) // 2
                img = img[:, start_y : start_y + target_size, :]

            # For pad mode, pad to make a square of target_size x target_size
            if mode == "pad":
                h_padding = target_size - img.shape[1]
                w_padding = target_size - img.shape[2]

                if h_padding > 0 or w_padding > 0:
                    pad_top = h_padding // 2
                    pad_bottom = h_padding - pad_top
                    pad_left = w_padding // 2
                    pad_right = w_padding - pad_left

                    # Pad with white (value=1.0)
                    img = torch.nn.functional.pad(
                        img, (pad_left, pad_right, pad_top, pad_bottom), mode="constant", value=1.0
                    )

            shapes.add((img.shape[1], img.shape[2]))
            epi_images.append(img)
        batch_images.append(torch.stack(epi_images))

    # Check if we have different shapes
    # In theory our model can also work well with different shapes
    if len(shapes) > 1:
        print(f"Warning: Found images with different shapes: {shapes}")
        # Find maximum dimensions
        max_height = max(shape[0] for shape in shapes)
        max_width = max(shape[1] for shape in shapes)

        # Pad images if necessary
        padded_images = []
        for img in batch_images:
            h_padding = max_height - img.shape[1]
            w_padding = max_width - img.shape[2]

            if h_padding > 0 or w_padding > 0:
                pad_top = h_padding // 2
                pad_bottom = h_padding - pad_top
                pad_left = w_padding // 2
                pad_right = w_padding - pad_left

                img = torch.nn.functional.pad(
                    img, (pad_left, pad_right, pad_top, pad_bottom), mode="constant", value=1.0
                )
            padded_images.append(img)
        batch_images = padded_images

    batch_images = torch.stack(batch_images)  # concatenate images

    # Ensure correct shape when single image
    if len(image_list) == 1:
        # Verify shape is (1, C, H, W)
        if batch_images.dim() == 3:
            batch_images = batch_images.unsqueeze(0)
    return batch_images