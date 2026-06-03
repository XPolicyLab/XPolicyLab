from collections.abc import Sequence
import logging

import torch

from openpi_value.shared import image_tools
import openpi_value.transforms as _transforms
import kornia.augmentation as K

logger = logging.getLogger("openpi")

# Constants moved from model.py


IMAGE_RESOLUTION = (224, 224)


def preprocess_observation_pytorch(
    observation,
    *,
    train: bool = False,
    # image_keys: Sequence[str] = IMAGE_KEYS,
    image_keys: Sequence[str] = None,
    image_resolution: tuple[int, int] = IMAGE_RESOLUTION,
    return_full_obs: bool = False,
    apply_shape_visual_aug: bool = False,
    apply_blur_visual_aug: bool = False,
    p_mask_base: float = 0.0,
    state_noise_snr: float | None = None,
):
    """Torch.compile-compatible version of preprocess_observation_pytorch with simplified type annotations.

    This function avoids complex type annotations that can cause torch.compile issues.
    """

    # if not set(image_keys).issubset(observation.images):
    #     raise ValueError(f"images dict missing keys: expected {image_keys}, got {list(observation.images)}")
    assert image_keys is None, "Deprecated: cannot use image_key anymore"
    
    # assert not (apply_blur_visual_aug and apply_shape_visual_aug), "Cannot apply both custom and official visual augmentations"

    batch_shape = observation.state.shape[:-1]

    image_keys = list(observation.images.keys())

    part_order = {'base': 0, 'left_wrist': 1, 'right_wrist': 2}
    def simple_sort_key(k):
        part, timestep_str, _ = k.rsplit('_', 2)
        timestep = int(timestep_str)
        return (timestep, part_order[part])
    image_keys = sorted(image_keys, key=simple_sort_key)

    out_images = {}

    for key in image_keys:
        image = observation.images[key]

        # Handle both [B, C, H, W] and [B, H, W, C] formats
        is_channels_first = image.shape[1] == 3  # Check if channels are in dimension 1

        if is_channels_first:
            # Convert [B, C, H, W] to [B, H, W, C] for processing
            image = image.permute(0, 2, 3, 1)

        if image.shape[1:3] != image_resolution:
            logger.info(f"Resizing image {key} from {image.shape[1:3]} to {image_resolution}")
            image = image_tools.resize_with_pad_torch(image, *image_resolution)

        if train:
            # Convert from [-1, 1] to [0, 1] for PyTorch augmentations
            image = image / 2.0 + 0.5
            
            # Apply PyTorch-based augmentations
            if "wrist" not in key and apply_shape_visual_aug:
                # Geometric augmentations for non-wrist cameras
                height, width = image.shape[1:3]

                # Random crop and resize
                crop_height = int(height * 0.95)
                crop_width = int(width * 0.95)

                # Random crop
                max_h = height - crop_height
                max_w = width - crop_width
                if max_h > 0 and max_w > 0:
                    # Use tensor operations instead of .item() for torch.compile compatibility
                    start_h = torch.randint(0, max_h + 1, (1,), device=image.device)
                    start_w = torch.randint(0, max_w + 1, (1,), device=image.device)
                    image = image[:, start_h : start_h + crop_height, start_w : start_w + crop_width, :]

                # Resize back to original size
                image = torch.nn.functional.interpolate(
                    image.permute(0, 3, 1, 2),  # [b, h, w, c] -> [b, c, h, w]
                    size=(height, width),
                    mode="bilinear",
                    align_corners=False,
                ).permute(0, 2, 3, 1)  # [b, c, h, w] -> [b, h, w, c]

                # Random rotation (small angles)
                # Use tensor operations instead of .item() for torch.compile compatibility
                angle = torch.rand(1, device=image.device) * 10 - 5  # Random angle between -5 and 5 degrees
                if torch.abs(angle) > 0.1:  # Only rotate if angle is significant
                    # Convert to radians
                    angle_rad = angle * torch.pi / 180.0

                    # Create rotation matrix
                    cos_a = torch.cos(angle_rad)
                    sin_a = torch.sin(angle_rad)

                    # Apply rotation using grid_sample
                    grid_x = torch.linspace(-1, 1, width, device=image.device)
                    grid_y = torch.linspace(-1, 1, height, device=image.device)

                    # Create meshgrid
                    grid_y, grid_x = torch.meshgrid(grid_y, grid_x, indexing="ij")

                    # Expand to batch dimension
                    grid_x = grid_x.unsqueeze(0).expand(image.shape[0], -1, -1)
                    grid_y = grid_y.unsqueeze(0).expand(image.shape[0], -1, -1)

                    # Apply rotation transformation
                    grid_x_rot = grid_x * cos_a - grid_y * sin_a
                    grid_y_rot = grid_x * sin_a + grid_y * cos_a

                    # Stack and reshape for grid_sample
                    grid = torch.stack([grid_x_rot, grid_y_rot], dim=-1)

                    image = torch.nn.functional.grid_sample(
                        image.permute(0, 3, 1, 2),  # [b, h, w, c] -> [b, c, h, w]
                        grid,
                        mode="bilinear",
                        padding_mode="zeros",
                        align_corners=False,
                    ).permute(0, 2, 3, 1)  # [b, c, h, w] -> [b, h, w, c]


            
            # * No color aug.
            # if apply_shape_visual_aug:
            #     # Color augmentations for all cameras
            #     # Random brightness
            #     # Use tensor operations instead of .item() for torch.compile compatibility
            #     brightness_factor = 0.7 + torch.rand(1, device=image.device) * 0.6  # Random factor between 0.7 and 1.3
            #     image = image * brightness_factor

            #     # Random contrast
            #     # Use tensor operations instead of .item() for torch.compile compatibility
            #     contrast_factor = 0.6 + torch.rand(1, device=image.device) * 0.8  # Random factor between 0.6 and 1.4
            #     mean = image.mean(dim=[1, 2, 3], keepdim=True)
            #     image = (image - mean) * contrast_factor + mean

            #     # Random saturation (convert to HSV, modify S, convert back)
            #     # For simplicity, we'll just apply a random scaling to the color channels
            #     # Use tensor operations instead of .item() for torch.compile compatibility
            #     saturation_factor = 0.5 + torch.rand(1, device=image.device) * 1.0  # Random factor between 0.5 and 1.5
            #     gray = image.mean(dim=-1, keepdim=True)
            #     image = gray + (image - gray) * saturation_factor



            # * add motionblur and gaussian blur
            if apply_blur_visual_aug:
                
                image_nchw = image.permute(0, 3, 1, 2).contiguous()

                aug = K.AugmentationSequential(
                    K.RandomMedianBlur(kernel_size=(3, 5), p=0.1),  # * prob too high
                    K.RandomMotionBlur(kernel_size=(3, 5), angle=35., direction=0.5, p=0.1),  # * smaller aug. Since the sensor is already blurry.
                    keepdim=True,
                )
                
                # Apply
                image_nchw = aug(image_nchw)
                
                # Permute back to [B, H, W, C]
                image = image_nchw.permute(0, 2, 3, 1).contiguous()
                
                
            # Clamp values to [0, 1]
            image = torch.clamp(image, 0, 1)

            # Back to [-1, 1]
            image = image * 2.0 - 1.0


        # Convert back to [B, C, H, W] format if it was originally channels-first
        if is_channels_first:
            image = image.permute(0, 3, 1, 2)  # [B, H, W, C] -> [B, C, H, W]

        out_images[key] = image

    out_masks = {}
    for key in out_images:
        if key not in observation.image_masks:
            # do not mask by default
            out_masks[key] = torch.ones(batch_shape, dtype=torch.bool, device=observation.state.device)
        else:
            out_masks[key] = observation.image_masks[key]
            
        if 'base' in key and train and p_mask_base > 0.0:
            # Randomly mask base images
            random_tensor = torch.rand(batch_shape, device=out_masks[key].device)
            base_mask = random_tensor > p_mask_base
            out_masks[key] = out_masks[key] & base_mask  # Combine with existing mask
 
    # * State augmentation
    
    
    # * Only for conveyor? using norm04
    state_std = [
        0.2079681158065796,
        0.7834290266036987,
        0.5441722273826599,
        0.14168238639831543,
        0.1750941127538681,
        0.15182428061962128,
        0.024107031524181366,
        0.19041913747787476,
        0.6899408102035522,
        0.4627247452735901,
        0.10430814325809479,
        0.1795605719089508,
        0.11770003288984299,
        0.03210258111357689,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0
    ],
    
    states = observation.state

    
    if state_noise_snr is not None:
        # 1. Calculate the noise standard deviation (sigma)
        # math.sqrt or **0.5 works fine for scalar operations here
        
        state_std = torch.tensor(state_std).to(states).reshape(1, -1)  # [1, state_dim]
       
        # noise_scale = state_std / (10 ** (state_noise_snr / 10))**0.5
        epsilon = 1e-6
        # noise_scale = state_std / torch.sqrt(10 ** (state_noise_snr / 10))
        noise_scale = state_std / torch.sqrt(torch.tensor(10) ** (state_noise_snr / 10) + epsilon)
        noise_scale = noise_scale.expand(states.shape)  # Now noise_scale has shape [4, 32]
        
        
        # 2. Add Gaussian noise
        # torch.randn_like(states) creates N(0,1) noise on the correct device (GPU/CPU)
        # We then multiply by noise_scale to adjust the spread
        states += torch.randn_like(states) * noise_scale
        

        
 
    # Create a simple object with the required attributes instead of using the complex Observation class
    class SimpleProcessedObservation:
        def __init__(self, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)

    if return_full_obs:
        return SimpleProcessedObservation(
            images=out_images,
            image_masks=out_masks,
            
            state=states,
            tokenized_prompt=observation.tokenized_prompt,
            tokenized_prompt_mask=observation.tokenized_prompt_mask,

            token_ar_mask=observation.token_ar_mask,
            token_loss_mask=observation.token_loss_mask,

            action_advantage=observation.action_advantage,
            action_advantage_original=observation.action_advantage_original,
            
            frame_index=observation.frame_index,
            frame_index_progress=observation.frame_index_progress,
            is_failure_data=observation.is_failure_data,
            is_infer_data=observation.is_infer_data,
            episode_length=observation.episode_length,

            image_original=observation.image_original,
            episode_index=observation.episode_index,
            
            inferred_action=observation.inferred_action,
            noise=observation.noise,
            
        )
    else:
        # * Simplified for sampling value
        return SimpleProcessedObservation(
            images=out_images,
            image_masks=out_masks,
            # state=observation.state,
            state=states,
            tokenized_prompt=observation.tokenized_prompt,
            tokenized_prompt_mask=observation.tokenized_prompt_mask,
        )
