import torch
import cv2
import numpy as np


def resize_images(images: torch.Tensor, target_size: tuple = (192, 256), normalize = True) -> torch.Tensor:
    """
        Resize images using cv2.resize

        Args:

        images: Input image tensor, shape=(bs, c, h, w)

        target_size: Target size, formatted as (height, width)

        Returns:

        resized_images: Resized image tensor, shape=(bs, c, target_h, target_w)
    """
    bs, c, h, w = images.shape
    target_h, target_w = target_size
    
    # If the target size has already been reached, return directly.
    if (h, w) == target_size:
        return images.clone()
    
    # Convert to NumPy array and resize
    resized_list = []
    for i in range(bs):
        # Convert to a NumPy array, with the shape (h, w, c)
        img_np = images[i].permute(1, 2, 0).cpu().numpy()
        
        
        if img_np.min() >= 0 and normalize:
            img_np = img_np/ 255.0 * 2.0 - 1.0  # Normalize to [-1, 1]
        elif not normalize and img_np.min() < 0:
            img_np = (img_np + 1) / 2 * 255 # reserve to [0,255]
            img_np = img_np.astype(np.uint8)
        

        resized_img = cv2.resize(img_np, (target_w, target_h))
        

        resized_tensor = torch.from_numpy(resized_img).permute(2, 0, 1).float()
        
        
        resized_list.append(resized_tensor)
    
    resized_images = torch.stack(resized_list, dim=0).to(images.device)
    
    return resized_images



def concat_obs_and_original(env_obs, image_original, wm_action_interval, device):
    """
        Concatenates historical observations and original images by key.

        Parameters:

            env_obs: A dictionary containing the key 'lerobot_history_obs', with values ​​being a list of length 3, each element being a dictionary containing an image tensor.

            image_original: A dictionary containing images with the same key, each value being a tensor with shape [8, 3, 224, 299].

            wm_action_interval: The interval between input frames of the world model.

        Returns:

            concat_result: The concatenated dictionary, where each key corresponds to a tensor with shape [4, 8, 3, 224, 299].

    """
    # Basic checks (to avoid data format errors)
    history_obs = list(env_obs['lerobot_history_obs'])[::wm_action_interval]
    assert len(history_obs) == 3, f"The historical observation length should be 3, currently it is {len(history_obs)}"
    
    # Retrieve all image keys (ensure the history and original keys are consistent)
    history_keys = set(history_obs[0].keys())
    original_keys = set(image_original.keys())
    assert history_keys == original_keys, \
        f"The historical observation and the original image key do not match! Historical key:{history_keys}, Original key:{original_keys}"
    image_keys = history_keys
    
    # Initialize the concatenation result dictionary
    concat_result = {}
    
    # Concatenate data key by key.
    for key in image_keys:
        # Extract three tensors from historical observations (each with shape: [8, 3, 224, 299])
        history_tensors = [obs[key].unsqueeze(0) for obs in history_obs]
        
        # Historical observation of splicing: 3 tensors are spliced ​​in the 0th dimension → shape: [3, 8, 3, 224, 299]
        history_concat = torch.cat(history_tensors, dim=0).to(device=device)
        
        # Extract the original image tensor and expand its dimensions (while maintaining dimensional consistency)
        original_tensor = image_original[key]  # shape: [8, 3, 224, 299]
        
        if original_tensor.shape[2] != history_concat.shape[3] or original_tensor.shape[3] != history_concat.shape[4]:
            original_tensor = resize_images(original_tensor, (history_concat.shape[3], history_concat.shape[4]))
        
        # Stitching History + Original: Stitching in 0th dimension → shape: [4, 8, 3, 224, 299]
        final_concat = torch.cat([history_concat, original_tensor.unsqueeze(0)], dim=0)
        
        concat_result[key] = final_concat
    
    return concat_result

def process_observations_dual_arm(forward_inputs: dict, target_size: tuple = (192, 256), use_his_obs = False) -> torch.Tensor:
    """
        Processing offline observation data from the double-arm observation database

        Args:

        forward_inputs: A dictionary containing the observation data

        target_size: Target size, formatted as (height, width)

        Returns:

        obs: Processed observation data, shape=(bs, 3, 3, 4, target_h, target_w)

    """

    if use_his_obs:
        t, b, c, h, w= forward_inputs['base_0_rgb'].shape
        head_image = forward_inputs['base_0_rgb'].view(t*b, c, h, w)
        left_wrist_image = forward_inputs['left_wrist_0_rgb'].view(t*b, c, h, w)
        right_wrist_image = forward_inputs['right_wrist_0_rgb'].view(t*b, c, h, w)
    else:
        head_image = forward_inputs['base_0_rgb'].view
        left_wrist_image = forward_inputs['left_wrist_0_rgb']
        right_wrist_image = forward_inputs['right_wrist_0_rgb']

    bs = head_image.shape[0]
    
    head_resized = resize_images(head_image, target_size)
    left_wrist_resized = resize_images(left_wrist_image, target_size)
    right_wrist_resized = resize_images(right_wrist_image, target_size)
    
    if use_his_obs:
        # (4*bs, 3, H, W) -> # (4, bs, 3, H, W)
        head_resized = head_resized.view(t, b, 3, target_size[0], target_size[1])
        left_wrist_resized = left_wrist_resized.view(t, b, 3, target_size[0], target_size[1])
        right_wrist_resized = right_wrist_resized.view(t, b, 3, target_size[0], target_size[1])
        # (4, bs, 3, H, W) * 3 -> (4, bs, 3*3, H, W)
        concatenated = torch.cat([head_resized, left_wrist_resized, right_wrist_resized], dim=2)
        # (4, bs, 3*3, H, W) -> (bs, 3*3, 4, H, W) -> (bs*3, 3, 4, H, W) 
        final_obs = concatenated.permute(1, 2, 0, 3, 4)
        final_obs = final_obs.view(b*3, 3, 4, target_size[0], target_size[1])
    else:   
        # Concatenating observation data:(bs, 3, H, W) + (bs, 6, H, W) -> (bs, 9, H, W)
        concatenated = torch.cat([head_resized, left_wrist_resized, right_wrist_resized], dim=1)
        
        # Repeated 4 times in the time dimension: (bs, 9, H, W) -> (bs, 9, 4, H, W)

        repeated = concatenated.unsqueeze(2).repeat(1, 1, 4, 1, 1)
        
        # Split channel dimensions: (bs, 9, 4, H, W) -> (bs, 3, 3, 4, H, W) -> (bs*3, 3, 4, H, W)
        final_obs = repeated.view(bs*3, 3, 4, target_size[0], target_size[1])
    
    return final_obs

def process_actions(actions: torch.Tensor, 
                    target_length: int = 25, 
                    feature_dim: int = 30, 
                    # inter_sample: bool=False
                    action_interval: int=1,
                    min_val: list=[],
                    max_val: list=[],
                ) -> torch.Tensor:
    """
        Processing Action Data

        Args:

            actions: Input action data, shape=(bs, seq_len, feat_dim)

            target_length: Target sequence length, default is 25

            feature_dim: Target feature dimension, default is 30

        Returns:

            act_tokens: Processed action tokens, shape=(bs, target_length, feature_dim)
    """
    bs, seq_len, feat_dim = actions.shape
    
    if action_interval > 1:
        actions = actions[:,::action_interval]
    
    act_tokens = torch.zeros(bs, target_length, feature_dim, device=actions.device)
    
    actual_len = min(seq_len, target_length)

    actions = (actions - torch.FloatTensor(min_val)) / (torch.FloatTensor(max_val) - torch.FloatTensor(min_val))
    actions = actions * 2.0 - 1.0
    
    # Processing feature dimensions
    if feat_dim == feature_dim:
        # Feature dimensions have been matched
        act_tokens[:, :actual_len, :] = actions[:, :actual_len, :]
    elif feat_dim < feature_dim:
        # If the feature dimension is smaller than the target dimension, it needs to be padded with 0s.
        act_tokens[:, :actual_len, :feat_dim] = actions[:, :actual_len, :feat_dim]
        # Fill the remaining dimensions with 0
        act_tokens[:, :actual_len, feat_dim:feature_dim] = 0
        
    return act_tokens
