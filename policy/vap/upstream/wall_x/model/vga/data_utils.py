"""Data utility functions for WAN video generation.

Extracted from wall_x.model.vga.utils — contains tensor/PIL conversion helpers
and prompt-building logic used by wan_video training and inference.
"""

import numpy as np
from PIL import Image
import torch


def frame_list_to_pil_list(data_list):
    pil_list = []
    for each_view in data_list:
        batch_size, frames, height, width, channels = each_view.shape
        pil_images = []
        for t in range(frames):
            frame = each_view[0, t]  # shape: [height, width, channels]
            frame_np = frame.cpu().numpy()
            if frame_np.dtype != np.uint8:
                if frame_np.max() <= 1.0:
                    frame_np = (frame_np * 255).astype(np.uint8)
                else:
                    frame_np = frame_np.astype(np.uint8)
            pil_image = Image.fromarray(frame_np)
            pil_images.append(pil_image)
        pil_list.append(pil_images)

    return pil_list


def tensor_to_pil_list(tensor_data):
    """
    Convert a PyTorch tensor to a list of PIL.Image objects.

    Args:
        tensor_data: tensor of shape [batch, time, height, width, channels]
                    e.g. torch.Size([4, 4, 480, 640, 3])

    Returns:
        list: list of PIL.Image objects
    """
    batch_size, frames, height, width, channels = tensor_data.shape

    pil_images = []

    for t in range(frames):
        frame = tensor_data[0, t]  # shape: [height, width, channels]
        frame_np = frame.cpu().numpy()
        if frame_np.dtype != np.uint8:
            if frame_np.max() <= 1.0:
                frame_np = (frame_np * 255).astype(np.uint8)
            else:
                frame_np = frame_np.astype(np.uint8)
        pil_image = Image.fromarray(frame_np)
        pil_images.append(pil_image)

    return pil_images


def tensor_video_to_vae_input(frames_list, device, dtype=torch.bfloat16):
    """Convert a list of multi-view training tensors directly into the format the VAE expects.

    Args:
        frames_list: List[Tensor], each of shape [B, T, H, W, C],
                     range [0,1] float or [0,255] uint8/float
        device: target device
        dtype: target dtype

    Returns:
        List[Tensor], each of shape [B, C, T, H, W], range [-1,1], on device
    """
    results = []
    for view_tensor in frames_list:
        x = view_tensor.to(device=device, dtype=torch.float32)
        if x.max() <= 1.0:
            x = x * 2.0 - 1.0  # [0,1] → [-1,1]
        else:
            x = x * (2.0 / 255.0) - 1.0  # [0,255] → [-1,1]
        x = x.permute(0, 4, 1, 2, 3).to(dtype=dtype)  # [B,C,T,H,W]
        results.append(x)
    return results


def tensor_frame_to_image_input(frame_tensor, device, dtype=torch.bfloat16):
    """Convert a single- or multi-frame tensor into the format the image encoder expects.

    Args:
        frame_tensor: Tensor of shape [H, W, C] or [B, H, W, C], range [0,1] or [0,255]

    Returns:
        Tensor of shape [1, C, H, W] or [B, C, H, W], range [-1,1]
    """
    x = frame_tensor.to(device=device, dtype=torch.float32)
    if x.max() <= 1.0:
        x = x * 2.0 - 1.0
    else:
        x = x * (2.0 / 255.0) - 1.0
    if x.dim() == 3:
        # [H, W, C] -> [1, C, H, W]
        x = x.permute(2, 0, 1).unsqueeze(0).to(dtype=dtype)
    else:
        # [B, H, W, C] -> [B, C, H, W]
        x = x.permute(0, 3, 1, 2).to(dtype=dtype)
    return x


def build_prompt_from_data(
    data,
    use_qwen25=False,
    prompt_template=None,
    qwen25_prompt_template=None,
    prompt_priority_order=None,
):
    """Build a prompt from the data.

    Supports three modes (in priority order):
    1) prompt_priority_order: take the first non-empty field by priority, e.g. ["distribute", "instruction"]
    2) prompt_template: template string with placeholders {instruction}, {subtask_generation}, {distribute}
    3) default: concatenate instruction + subtask_generation + distribute in order

    When batch_size > 1, the string fields in data are collated into lists of strings,
    in which case a list of prompts is returned (one per sample).
    """

    def _get_field(data, key, default=""):
        """Get a field value, returning list/scalar as-is."""
        v = data.get(key, default)
        if v is None:
            return default
        return v

    instruction = _get_field(data, "instruction", "")
    subtask_generation = _get_field(data, "subtask_generation", "")
    distribute = _get_field(data, "distribute", "")

    # Detect batch mode (fields are lists of strings)
    batch_fields = [instruction, subtask_generation, distribute]
    is_batch = any(isinstance(f, list) for f in batch_fields)

    if is_batch:
        # Normalize to lists, broadcasting scalar fields
        def _to_list(v, length):
            if isinstance(v, list):
                return [x or "" for x in v]
            return [v or ""] * length

        batch_size = max(len(f) for f in batch_fields if isinstance(f, list))
        instructions = _to_list(instruction, batch_size)
        subtask_generations = _to_list(subtask_generation, batch_size)
        distributes = _to_list(distribute, batch_size)

        prompts = []
        for i in range(batch_size):
            kwargs_i = {
                "instruction": instructions[i],
                "subtask_generation": subtask_generations[i],
                "distribute": distributes[i],
            }
            prompts.append(
                _build_single_prompt(
                    data,
                    kwargs_i,
                    use_qwen25,
                    prompt_template,
                    qwen25_prompt_template,
                    prompt_priority_order,
                )
            )
        return prompts
    else:
        kwargs_single = {
            "instruction": instruction or "",
            "subtask_generation": subtask_generation or "",
            "distribute": distribute or "",
        }
        return _build_single_prompt(
            data,
            kwargs_single,
            use_qwen25,
            prompt_template,
            qwen25_prompt_template,
            prompt_priority_order,
        )


def _build_single_prompt(
    data,
    template_kwargs,
    use_qwen25,
    prompt_template,
    qwen25_prompt_template,
    prompt_priority_order,
):
    """Build the prompt string for a single sample."""
    instruction = template_kwargs["instruction"]

    # Priority mode: take the first non-empty field in the given order
    if prompt_priority_order:
        if isinstance(prompt_priority_order, str):
            prompt_priority_order = [
                k.strip() for k in prompt_priority_order.split(",")
            ]
        for key in prompt_priority_order:
            val = template_kwargs.get(key, "")
            if not val:
                raw = data.get(key, "")
                val = raw[0] if isinstance(raw, list) else (raw or "")
            if val:
                return val
        return ""

    if use_qwen25:
        if qwen25_prompt_template:
            return qwen25_prompt_template.format(**template_kwargs)
        return (
            "图像的左中右三部分别表示机械臂的左臂视角，主视角和右臂视角，都是第一视角，左右臂视角的底部是机械臂的夹爪。图像中的机械臂将要遵循'"
            + instruction
            + "'指令对桌面上的物体进行操作。请你根据目前图像推测其相对整段指令的进度，直接给出未来两秒的视频的描述，动作符合物理规律，画面保持真实感。只需要未来视频的描述不要其他内容，也不要分条列举，用一句话说完，重点说明机械臂接下来的精细动作，如：机械臂伸向什么物体，要抓住什么物体的什么位置，保持抓紧物体移动到什么位置等。"
        )

    if prompt_template:
        return prompt_template.format(**template_kwargs)

    # Default: concatenate in order
    out = ""
    if template_kwargs["instruction"]:
        out += template_kwargs["instruction"]
    if template_kwargs["subtask_generation"]:
        out += template_kwargs["subtask_generation"]
    if template_kwargs["distribute"]:
        out += template_kwargs["distribute"]
    return out
