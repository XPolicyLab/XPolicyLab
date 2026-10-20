from einops import rearrange, repeat
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
CACHE_T = 2

def check_is_instance(model, module_class):
    if isinstance(model, module_class):
        return True
    if hasattr(model, 'module') and isinstance(model.module, module_class):
        return True
    return False

def block_causal_mask(x, block_size):
    b, n, s, _, device = (*x.size(), x.device)
    assert s % block_size == 0
    num_blocks = s // block_size
    mask = torch.zeros(b, n, s, s, dtype=torch.bool, device=device)
    for i in range(num_blocks):
        mask[:, :, i * block_size:(i + 1) * block_size, :(i + 1) * block_size] = 1
    return mask

class CausalConv3d(nn.Conv3d):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._padding = (self.padding[2], self.padding[2], self.padding[1], self.padding[1], 2 * self.padding[0], 0)
        self.padding = (0, 0, 0)

    def forward(self, x, cache_x=None):
        padding = list(self._padding)
        if cache_x is not None and self._padding[4] > 0:
            cache_x = cache_x.to(x.device)
            x = torch.cat([cache_x, x], dim=2)
            padding[4] -= cache_x.shape[2]
        x = F.pad(x, padding)
        return super().forward(x)

class RMS_norm(nn.Module):

    def __init__(self, dim, channel_first=True, images=True, bias=False):
        super().__init__()
        broadcastable_dims = (1, 1, 1) if not images else (1, 1)
        shape = (dim, *broadcastable_dims) if channel_first else (dim,)
        self.channel_first = channel_first
        self.scale = dim ** 0.5
        self.gamma = nn.Parameter(torch.ones(shape))
        self.bias = nn.Parameter(torch.zeros(shape)) if bias else 0.0

    def forward(self, x):
        return F.normalize(x, dim=1 if self.channel_first else -1) * self.scale * self.gamma + self.bias

class Upsample(nn.Upsample):

    def forward(self, x):
        return super().forward(x.float()).type_as(x)

class Resample(nn.Module):

    def __init__(self, dim, mode):
        assert mode in ('none', 'upsample2d', 'upsample3d', 'downsample2d', 'downsample3d')
        super().__init__()
        self.dim = dim
        self.mode = mode
        if mode == 'upsample2d':
            self.resample = nn.Sequential(Upsample(scale_factor=(2.0, 2.0), mode='nearest-exact'), nn.Conv2d(dim, dim // 2, 3, padding=1))
        elif mode == 'upsample3d':
            self.resample = nn.Sequential(Upsample(scale_factor=(2.0, 2.0), mode='nearest-exact'), nn.Conv2d(dim, dim // 2, 3, padding=1))
            self.time_conv = CausalConv3d(dim, dim * 2, (3, 1, 1), padding=(1, 0, 0))
        elif mode == 'downsample2d':
            self.resample = nn.Sequential(nn.ZeroPad2d((0, 1, 0, 1)), nn.Conv2d(dim, dim, 3, stride=(2, 2)))
        elif mode == 'downsample3d':
            self.resample = nn.Sequential(nn.ZeroPad2d((0, 1, 0, 1)), nn.Conv2d(dim, dim, 3, stride=(2, 2)))
            self.time_conv = CausalConv3d(dim, dim, (3, 1, 1), stride=(2, 1, 1), padding=(0, 0, 0))
        else:
            self.resample = nn.Identity()

    def forward(self, x, feat_cache=None, feat_idx=[0]):
        b, c, t, h, w = x.size()
        if self.mode == 'upsample3d':
            if feat_cache is not None:
                idx = feat_idx[0]
                if feat_cache[idx] is None:
                    feat_cache[idx] = 'Rep'
                    feat_idx[0] += 1
                else:
                    cache_x = x[:, :, -CACHE_T:, :, :].clone()
                    if cache_x.shape[2] < 2 and feat_cache[idx] is not None and (feat_cache[idx] != 'Rep'):
                        cache_x = torch.cat([feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(cache_x.device), cache_x], dim=2)
                    if cache_x.shape[2] < 2 and feat_cache[idx] is not None and (feat_cache[idx] == 'Rep'):
                        cache_x = torch.cat([torch.zeros_like(cache_x).to(cache_x.device), cache_x], dim=2)
                    if feat_cache[idx] == 'Rep':
                        x = self.time_conv(x)
                    else:
                        x = self.time_conv(x, feat_cache[idx])
                    feat_cache[idx] = cache_x
                    feat_idx[0] += 1
                    x = x.reshape(b, 2, c, t, h, w)
                    x = torch.stack((x[:, 0, :, :, :, :], x[:, 1, :, :, :, :]), 3)
                    x = x.reshape(b, c, t * 2, h, w)
        t = x.shape[2]
        x = rearrange(x, 'b c t h w -> (b t) c h w')
        x = self.resample(x)
        x = rearrange(x, '(b t) c h w -> b c t h w', t=t)
        if self.mode == 'downsample3d':
            if feat_cache is not None:
                idx = feat_idx[0]
                if feat_cache[idx] is None:
                    feat_cache[idx] = x.clone()
                    feat_idx[0] += 1
                else:
                    cache_x = x[:, :, -1:, :, :].clone()
                    x = self.time_conv(torch.cat([feat_cache[idx][:, :, -1:, :, :], x], 2))
                    feat_cache[idx] = cache_x
                    feat_idx[0] += 1
        return (x, feat_cache, feat_idx)

    def init_weight(self, conv):
        conv_weight = conv.weight
        nn.init.zeros_(conv_weight)
        c1, c2, t, h, w = conv_weight.size()
        one_matrix = torch.eye(c1, c2)
        init_matrix = one_matrix
        nn.init.zeros_(conv_weight)
        conv_weight.data[:, :, 1, 0, 0] = init_matrix
        conv.weight.data.copy_(conv_weight)
        nn.init.zeros_(conv.bias.data)

    def init_weight2(self, conv):
        conv_weight = conv.weight.data
        nn.init.zeros_(conv_weight)
        c1, c2, t, h, w = conv_weight.size()
        init_matrix = torch.eye(c1 // 2, c2)
        conv_weight[:c1 // 2, :, -1, 0, 0] = init_matrix
        conv_weight[c1 // 2:, :, -1, 0, 0] = init_matrix
        conv.weight.data.copy_(conv_weight)
        nn.init.zeros_(conv.bias.data)

def patchify(x, patch_size):
    if patch_size == 1:
        return x
    if x.dim() == 4:
        x = rearrange(x, 'b c (h q) (w r) -> b (c r q) h w', q=patch_size, r=patch_size)
    elif x.dim() == 5:
        x = rearrange(x, 'b c f (h q) (w r) -> b (c r q) f h w', q=patch_size, r=patch_size)
    else:
        raise ValueError(f'Invalid input shape: {x.shape}')
    return x

def unpatchify(x, patch_size):
    if patch_size == 1:
        return x
    if x.dim() == 4:
        x = rearrange(x, 'b (c r q) h w -> b c (h q) (w r)', q=patch_size, r=patch_size)
    elif x.dim() == 5:
        x = rearrange(x, 'b (c r q) f h w -> b c f (h q) (w r)', q=patch_size, r=patch_size)
    return x

class Resample38(Resample):

    def __init__(self, dim, mode):
        assert mode in ('none', 'upsample2d', 'upsample3d', 'downsample2d', 'downsample3d')
        super(Resample, self).__init__()
        self.dim = dim
        self.mode = mode
        if mode == 'upsample2d':
            self.resample = nn.Sequential(Upsample(scale_factor=(2.0, 2.0), mode='nearest-exact'), nn.Conv2d(dim, dim, 3, padding=1))
        elif mode == 'upsample3d':
            self.resample = nn.Sequential(Upsample(scale_factor=(2.0, 2.0), mode='nearest-exact'), nn.Conv2d(dim, dim, 3, padding=1))
            self.time_conv = CausalConv3d(dim, dim * 2, (3, 1, 1), padding=(1, 0, 0))
        elif mode == 'downsample2d':
            self.resample = nn.Sequential(nn.ZeroPad2d((0, 1, 0, 1)), nn.Conv2d(dim, dim, 3, stride=(2, 2)))
        elif mode == 'downsample3d':
            self.resample = nn.Sequential(nn.ZeroPad2d((0, 1, 0, 1)), nn.Conv2d(dim, dim, 3, stride=(2, 2)))
            self.time_conv = CausalConv3d(dim, dim, (3, 1, 1), stride=(2, 1, 1), padding=(0, 0, 0))
        else:
            self.resample = nn.Identity()

class ResidualBlock(nn.Module):

    def __init__(self, in_dim, out_dim, dropout=0.0):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.residual = nn.Sequential(RMS_norm(in_dim, images=False), nn.SiLU(), CausalConv3d(in_dim, out_dim, 3, padding=1), RMS_norm(out_dim, images=False), nn.SiLU(), nn.Dropout(dropout), CausalConv3d(out_dim, out_dim, 3, padding=1))
        self.shortcut = CausalConv3d(in_dim, out_dim, 1) if in_dim != out_dim else nn.Identity()

    def forward(self, x, feat_cache=None, feat_idx=[0]):
        h = self.shortcut(x)
        for layer in self.residual:
            if check_is_instance(layer, CausalConv3d) and feat_cache is not None:
                idx = feat_idx[0]
                cache_x = x[:, :, -CACHE_T:, :, :].clone()
                if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
                    cache_x = torch.cat([feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(cache_x.device), cache_x], dim=2)
                x = layer(x, feat_cache[idx])
                feat_cache[idx] = cache_x
                feat_idx[0] += 1
            else:
                x = layer(x)
        return (x + h, feat_cache, feat_idx)

class AttentionBlock(nn.Module):

    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.norm = RMS_norm(dim)
        self.to_qkv = nn.Conv2d(dim, dim * 3, 1)
        self.proj = nn.Conv2d(dim, dim, 1)
        nn.init.zeros_(self.proj.weight)

    def forward(self, x):
        identity = x
        b, c, t, h, w = x.size()
        x = rearrange(x, 'b c t h w -> (b t) c h w')
        x = self.norm(x)
        q, k, v = self.to_qkv(x).reshape(b * t, 1, c * 3, -1).permute(0, 1, 3, 2).contiguous().chunk(3, dim=-1)
        x = F.scaled_dot_product_attention(q, k, v)
        x = x.squeeze(1).permute(0, 2, 1).reshape(b * t, c, h, w)
        x = self.proj(x)
        x = rearrange(x, '(b t) c h w-> b c t h w', t=t)
        return x + identity

class AvgDown3D(nn.Module):

    def __init__(self, in_channels, out_channels, factor_t, factor_s=1):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.factor_t = factor_t
        self.factor_s = factor_s
        self.factor = self.factor_t * self.factor_s * self.factor_s
        assert in_channels * self.factor % out_channels == 0
        self.group_size = in_channels * self.factor // out_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pad_t = (self.factor_t - x.shape[2] % self.factor_t) % self.factor_t
        pad = (0, 0, 0, 0, pad_t, 0)
        x = F.pad(x, pad)
        B, C, T, H, W = x.shape
        x = x.view(B, C, T // self.factor_t, self.factor_t, H // self.factor_s, self.factor_s, W // self.factor_s, self.factor_s)
        x = x.permute(0, 1, 3, 5, 7, 2, 4, 6).contiguous()
        x = x.view(B, C * self.factor, T // self.factor_t, H // self.factor_s, W // self.factor_s)
        x = x.view(B, self.out_channels, self.group_size, T // self.factor_t, H // self.factor_s, W // self.factor_s)
        x = x.mean(dim=2)
        return x

class Down_ResidualBlock(nn.Module):

    def __init__(self, in_dim, out_dim, dropout, mult, temperal_downsample=False, down_flag=False):
        super().__init__()
        self.avg_shortcut = AvgDown3D(in_dim, out_dim, factor_t=2 if temperal_downsample else 1, factor_s=2 if down_flag else 1)
        downsamples = []
        for _ in range(mult):
            downsamples.append(ResidualBlock(in_dim, out_dim, dropout))
            in_dim = out_dim
        if down_flag:
            mode = 'downsample3d' if temperal_downsample else 'downsample2d'
            downsamples.append(Resample38(out_dim, mode=mode))
        self.downsamples = nn.Sequential(*downsamples)

    def forward(self, x, feat_cache=None, feat_idx=[0]):
        x_copy = x.clone()
        for module in self.downsamples:
            x, feat_cache, feat_idx = module(x, feat_cache, feat_idx)
        return (x + self.avg_shortcut(x_copy), feat_cache, feat_idx)

class Encoder3d_38(nn.Module):

    def __init__(self, dim=128, z_dim=4, dim_mult=[1, 2, 4, 4], num_res_blocks=2, attn_scales=[], temperal_downsample=[False, True, True], dropout=0.0):
        super().__init__()
        self.dim = dim
        self.z_dim = z_dim
        self.dim_mult = dim_mult
        self.num_res_blocks = num_res_blocks
        self.attn_scales = attn_scales
        self.temperal_downsample = temperal_downsample
        dims = [dim * u for u in [1] + dim_mult]
        scale = 1.0
        self.conv1 = CausalConv3d(12, dims[0], 3, padding=1)
        downsamples = []
        for i, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:])):
            t_down_flag = temperal_downsample[i] if i < len(temperal_downsample) else False
            downsamples.append(Down_ResidualBlock(in_dim=in_dim, out_dim=out_dim, dropout=dropout, mult=num_res_blocks, temperal_downsample=t_down_flag, down_flag=i != len(dim_mult) - 1))
            scale /= 2.0
        self.downsamples = nn.Sequential(*downsamples)
        self.middle = nn.Sequential(ResidualBlock(out_dim, out_dim, dropout), AttentionBlock(out_dim), ResidualBlock(out_dim, out_dim, dropout))
        self.head = nn.Sequential(RMS_norm(out_dim, images=False), nn.SiLU(), CausalConv3d(out_dim, z_dim, 3, padding=1))

    def forward(self, x, feat_cache=None, feat_idx=[0]):
        if feat_cache is not None:
            idx = feat_idx[0]
            cache_x = x[:, :, -CACHE_T:, :, :].clone()
            if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
                cache_x = torch.cat([feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(cache_x.device), cache_x], dim=2)
            x = self.conv1(x, feat_cache[idx])
            feat_cache[idx] = cache_x
            feat_idx[0] += 1
        else:
            x = self.conv1(x)
        for layer in self.downsamples:
            if feat_cache is not None:
                x, feat_cache, feat_idx = layer(x, feat_cache, feat_idx)
            else:
                x = layer(x)
        for layer in self.middle:
            if isinstance(layer, ResidualBlock) and feat_cache is not None:
                x, feat_cache, feat_idx = layer(x, feat_cache, feat_idx)
            else:
                x = layer(x)
        for layer in self.head:
            if isinstance(layer, CausalConv3d) and feat_cache is not None:
                idx = feat_idx[0]
                cache_x = x[:, :, -CACHE_T:, :, :].clone()
                if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
                    cache_x = torch.cat([feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(cache_x.device), cache_x], dim=2)
                x = layer(x, feat_cache[idx])
                feat_cache[idx] = cache_x
                feat_idx[0] += 1
            else:
                x = layer(x)
        return (x, feat_cache, feat_idx)

def count_conv3d(model):
    count = 0
    for m in model.modules():
        if isinstance(m, CausalConv3d):
            count += 1
    return count

class VideoVAE_(nn.Module):

    def clear_cache(self):
        self._enc_conv_num = count_conv3d(self.encoder)
        self._enc_conv_idx = [0]
        self._enc_feat_map = [None] * self._enc_conv_num

class WanVideoVAE(nn.Module):

    def single_encode(self, video, device):
        return self.model.encode(video.to(device), self.scale)

    def encode(self, videos, device, tiled=False, tile_size=(34, 34), tile_stride=(18, 16)):
        hidden_states = []
        for video in videos:
            video = video.unsqueeze(0)
            if tiled:
                raise NotImplementedError('Tiled encoding is not allowed yet.')
                tile_size = (tile_size[0] * self.upsampling_factor, tile_size[1] * self.upsampling_factor)
                tile_stride = (tile_stride[0] * self.upsampling_factor, tile_stride[1] * self.upsampling_factor)
                hidden_state = self.tiled_encode(video, device, tile_size, tile_stride)
            else:
                hidden_state = self.single_encode(video, device)
            hidden_state = hidden_state.squeeze(0)
            hidden_states.append(hidden_state)
        hidden_states = torch.stack(hidden_states)
        return hidden_states

class VideoVAE38_(VideoVAE_):

    def __init__(self, dim=160, z_dim=48, dec_dim=256, dim_mult=[1, 2, 4, 4], num_res_blocks=2, attn_scales=[], temperal_downsample=[False, True, True], dropout=0.0):
        super(VideoVAE_, self).__init__()
        self.dim = dim
        self.z_dim = z_dim
        self.dim_mult = dim_mult
        self.num_res_blocks = num_res_blocks
        self.attn_scales = attn_scales
        self.temperal_downsample = temperal_downsample
        self.temperal_upsample = temperal_downsample[::-1]
        self.encoder = Encoder3d_38(dim, z_dim * 2, dim_mult, num_res_blocks, attn_scales, self.temperal_downsample, dropout)
        self.conv1 = CausalConv3d(z_dim * 2, z_dim * 2, 1)

    def encode(self, x, scale):
        self.clear_cache()
        x = patchify(x, patch_size=2)
        t = x.shape[2]
        iter_ = 1 + (t - 1) // 4
        for i in range(iter_):
            self._enc_conv_idx = [0]
            if i == 0:
                out, self._enc_feat_map, self._enc_conv_idx = self.encoder(x[:, :, :1, :, :], feat_cache=self._enc_feat_map, feat_idx=self._enc_conv_idx)
            else:
                out_, self._enc_feat_map, self._enc_conv_idx = self.encoder(x[:, :, 1 + 4 * (i - 1):1 + 4 * i, :, :], feat_cache=self._enc_feat_map, feat_idx=self._enc_conv_idx)
                out = torch.cat([out, out_], 2)
        mu, log_var = self.conv1(out).chunk(2, dim=1)
        if isinstance(scale[0], torch.Tensor):
            scale = [s.to(dtype=mu.dtype, device=mu.device) for s in scale]
            mu = (mu - scale[0].view(1, self.z_dim, 1, 1, 1)) * scale[1].view(1, self.z_dim, 1, 1, 1)
        else:
            scale = scale.to(dtype=mu.dtype, device=mu.device)
            mu = (mu - scale[0]) * scale[1]
        self.clear_cache()
        return mu

class WanVideoVAE38(WanVideoVAE):

    def __init__(self, z_dim=48, dim=160):
        super(WanVideoVAE, self).__init__()
        mean = [-0.2289, -0.0052, -0.1323, -0.2339, -0.2799, 0.0174, 0.1838, 0.1557, -0.1382, 0.0542, 0.2813, 0.0891, 0.157, -0.0098, 0.0375, -0.1825, -0.2246, -0.1207, -0.0698, 0.5109, 0.2665, -0.2108, -0.2158, 0.2502, -0.2055, -0.0322, 0.1109, 0.1567, -0.0729, 0.0899, -0.2799, -0.123, -0.0313, -0.1649, 0.0117, 0.0723, -0.2839, -0.2083, -0.052, 0.3748, 0.0152, 0.1957, 0.1433, -0.2944, 0.3573, -0.0548, -0.1681, -0.0667]
        std = [0.4765, 1.0364, 0.4514, 1.1677, 0.5313, 0.499, 0.4818, 0.5013, 0.8158, 1.0344, 0.5894, 1.0901, 0.6885, 0.6165, 0.8454, 0.4978, 0.5759, 0.3523, 0.7135, 0.6804, 0.5833, 1.4146, 0.8986, 0.5659, 0.7069, 0.5338, 0.4889, 0.4917, 0.4069, 0.4999, 0.6866, 0.4093, 0.5709, 0.6065, 0.6415, 0.4944, 0.5726, 1.2042, 0.5458, 1.6887, 0.3971, 1.06, 0.3943, 0.5537, 0.5444, 0.4089, 0.7468, 0.7744]
        self.mean = torch.tensor(mean)
        self.std = torch.tensor(std)
        self.scale = [self.mean, 1.0 / self.std]
        self.model = VideoVAE38_(z_dim=z_dim, dim=dim).eval().requires_grad_(False)
        self.upsampling_factor = 16
        self.temporal_downsample_factor = 4
        self.z_dim = z_dim
