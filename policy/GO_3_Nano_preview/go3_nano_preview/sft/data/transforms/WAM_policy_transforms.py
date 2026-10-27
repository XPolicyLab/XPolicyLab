from einops import rearrange
import torch.nn.functional as F
from go3_nano_preview.sft.data.unified_dataset.data_utils import gen_crop_config, intrin_crop_transform, intrin_pad_transform, intrinsic_transform, random_crop_tensor

class Compose:

    def __init__(self, transforms):
        self.transforms = list(transforms)

    def __call__(self, sample, *, rng, **kwargs):
        for transform in self.transforms:
            sample = transform(sample, rng=rng, **kwargs)
        return sample

class RandomVideoPadding:

    def __call__(self, sample, *, rng, **kwargs):
        if rng.randint(0, 1) != 0:
            return sample
        video = sample['video']
        (height, width) = video.shape[-2:]
        pad_top = rng.randint(0, height // 3)
        pad_bottom = rng.randint(0, height // 3)
        pad_left = rng.randint(0, width // 3)
        pad_right = rng.randint(0, width // 3)
        fill = rng.uniform(0, 1)
        sample['video'] = F.pad(video, (pad_left, pad_right, pad_top, pad_bottom), value=fill)
        if sample.get('intrinsic') is not None:
            sample['intrinsic'] = intrin_pad_transform(sample['intrinsic'], pad_top, pad_left)
        return sample

class RandomVideoCrop:

    def __call__(self, sample, *, rng, **kwargs):
        if rng.randint(0, 1) != 0:
            return sample
        (h_start, w_start, crop_height, crop_width) = gen_crop_config(sample['video'], rng=rng)
        sample['video'] = random_crop_tensor(sample['video'], h_start, w_start, crop_height, crop_width)
        sample['resize_source_size'] = (crop_height, crop_width)
        if sample.get('intrinsic') is not None:
            sample['intrinsic'] = intrin_crop_transform(sample['intrinsic'], h_start, w_start)
        return sample

class ResizeVideo:

    def __init__(self, resize_transform, sample_size, preprocess):
        self.resize_transform = resize_transform
        self.sample_size = sample_size
        self.preprocess = preprocess

    def __call__(self, sample, *, rng, skip_resize=False, **kwargs):
        if sample.get('intrinsic') is not None:
            sample['intrinsic'] = intrinsic_transform(sample['intrinsic'], sample['resize_source_size'], self.sample_size, self.preprocess)
        if not skip_resize:
            sample['video'] = self.resize_transform(sample['video'])
        return sample

class NormalizeVideo:

    def __init__(self, normalize_transform):
        self.normalize_transform = normalize_transform

    def __call__(self, videos):
        (_, views, _, _, _) = videos.shape
        videos = rearrange(videos, 'c v t h w -> (v t) c h w')
        videos = self.normalize_transform(videos)
        return rearrange(videos, '(v t) c h w -> c v t h w', v=views)
