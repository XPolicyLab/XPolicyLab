import torch.distributed as dist
from contextlib import contextmanager
import sys, os
import torch
from einops import rearrange
import random
import torchvision.transforms as transforms
import av

def get_video_info(video_path):
    with av.open(video_path) as container:
        container = av.open(video_path)
        video_stream = next((s for s in container.streams if s.type == 'video'), None)
        if video_stream is None:
            raise ValueError('Video Stream Nout Found')
        codec_name = video_stream.codec_context.codec.name
        frames = video_stream.frames
    return (codec_name, frames)

def zero_rank_print(s):
    if not dist.is_initialized() or (dist.is_initialized() and dist.get_rank() == 0):
        print('### ' + s)

@contextmanager
def suppress_output():
    sys.stdout.flush()
    sys.stderr.flush()
    original_stdout_fd = sys.stdout.fileno()
    original_stderr_fd = sys.stderr.fileno()
    saved_stdout_fd = os.dup(original_stdout_fd)
    saved_stderr_fd = os.dup(original_stderr_fd)
    devnull = os.open(os.devnull, os.O_WRONLY)
    os.dup2(devnull, original_stdout_fd)
    os.dup2(devnull, original_stderr_fd)
    try:
        yield
    finally:
        os.dup2(saved_stdout_fd, original_stdout_fd)
        os.dup2(saved_stderr_fd, original_stderr_fd)
        os.close(saved_stdout_fd)
        os.close(saved_stderr_fd)
        os.close(devnull)

def nv12_to_rgb_multiframe(nv12_tensor, width, height):
    try:
        nv12_tensor = nv12_tensor.to(dtype=torch.float32)
        y_plane = nv12_tensor[:height, :width]
        uv_plane = nv12_tensor[height:height + height // 2, :].view(height // 2, width // 2, 2, -1).repeat_interleave(2, dim=0).repeat_interleave(2, dim=1)
        u_plane = uv_plane[:, :, 0] - 128
        v_plane = uv_plane[:, :, 1] - 128
        r = y_plane + 1.402 * v_plane
        g = y_plane - 0.344136 * u_plane - 0.714136 * v_plane
        b = y_plane + 1.772 * u_plane
        rgb_frame = torch.stack((r, g, b), dim=2).clamp(0, 255).byte()
        rgb_frame = rearrange(rgb_frame, 'h w c t -> c t h w') / 255.0
        rgb_frame_cpu = rgb_frame.detach().cpu()
        del rgb_frame
        return rgb_frame_cpu
    except Exception as e:
        print(f'Error converting NV12 to RGB: {e}')
        raise e

def intrin_crop_transform(intrinsic, h_start, w_start):
    (fx, fy, cx, cy) = (intrinsic[0][0], intrinsic[1][1], intrinsic[0][2], intrinsic[1][2])
    cx_new = cx - w_start
    cy_new = cy - h_start
    return torch.tensor([[fx, 0, cx_new], [0, fy, cy_new], [0, 0, 1]])

def gen_crop_config(tensor, rng=random):
    (_, _, h, w) = tensor.shape
    h_start = rng.randint(0, h // 3)
    w_start = rng.randint(0, w // 3)
    h_crop = rng.randint(2 * h // 3, h - h_start)
    w_crop = rng.randint(2 * w // 3, w - w_start)
    return (h_start, w_start, h_crop, w_crop)

def intrin_pad_transform(K, pad_top, pad_left):
    K_new = K.clone()
    K_new[0, 2] += pad_left
    K_new[1, 2] += pad_top
    return K_new

def mask_template(h, w):
    tensor = torch.zeros((1, h, w))
    center_h = random.randint(0, h - 1)
    center_w = random.randint(0, w - 1)
    rotation_transform = transforms.RandomRotation(degrees=(-20, 20), center=(center_h, center_w), fill=1)
    return rotation_transform(tensor)[0]

def intrinsic_transform(intrinsic, original_res, size, transform_mode):
    (fx, fy, cx, cy) = (intrinsic[0, 0], intrinsic[1, 1], intrinsic[0, 2], intrinsic[1, 2])
    original_height = original_res[0]
    original_width = original_res[1]
    if transform_mode == 'resize':
        resize_height = size[0]
        resize_width = size[1]
        scale_height = resize_height / original_height
        scale_width = resize_width / original_width
        fx_new = fx * scale_width
        fy_new = fy * scale_height
        cx_new = cx * scale_width
        cy_new = cy * scale_height
    elif transform_mode == 'center_crop_resize':
        if original_height <= original_width:
            scale_ratio = min(size) / original_height
        else:
            scale_ratio = min(size) / original_width
        resize_height = scale_ratio * original_height
        resize_width = scale_ratio * original_width
        fx_new = fx * scale_ratio
        fy_new = fy * scale_ratio
        cx_new = cx * scale_ratio
        cy_new = cy * scale_ratio
        crop_height = size[0]
        crop_width = size[1]
        cx_new = cx_new * (crop_width / resize_width)
        cy_new = cy_new * (crop_height / resize_height)
    else:
        raise NotImplementedError('No such transformation mode for image!')
    return torch.tensor([[fx_new, 0, cx_new], [0, fy_new, cy_new], [0, 0, 1]])

def intrinsic_transform_batch(intrinsic, original_res, size, transform_mode):
    b = intrinsic.shape[0]
    (fx, fy, cx, cy) = (intrinsic[:, 0, 0], intrinsic[:, 1, 1], intrinsic[:, 0, 2], intrinsic[:, 1, 2])
    original_height = original_res[0]
    original_width = original_res[1]
    if transform_mode == 'resize':
        resize_height = size[0]
        resize_width = size[1]
        scale_height = resize_height / original_height
        scale_width = resize_width / original_width
        fx_new = fx * scale_width
        fy_new = fy * scale_height
        cx_new = cx * scale_width
        cy_new = cy * scale_height
    elif transform_mode == 'center_crop_resize':
        if original_height <= original_width:
            scale_ratio = min(size) / original_height
        else:
            scale_ratio = min(size) / original_width
        resize_height = scale_ratio * original_height
        resize_width = scale_ratio * original_width
        fx_new = fx * scale_ratio
        fy_new = fy * scale_ratio
        cx_new = cx * scale_ratio
        cy_new = cy * scale_ratio
        crop_height = size[0]
        crop_width = size[1]
        cx_new = cx_new * (crop_width / resize_width)
        cy_new = cy_new * (crop_height / resize_height)
    else:
        raise NotImplementedError('No such transformation mode for image!')
    fx_expanded = fx_new
    fy_expanded = fy_new
    cx_expanded = cx_new
    cy_expanded = cy_new
    intrinsic_matrices = torch.zeros((b, 3, 3), dtype=fx.dtype, device=fx.device)
    intrinsic_matrices[:, 0, 0] = fx_expanded
    intrinsic_matrices[:, 1, 1] = fy_expanded
    intrinsic_matrices[:, 0, 2] = cx_expanded
    intrinsic_matrices[:, 1, 2] = cy_expanded
    intrinsic_matrices[:, 2, 2] = 1
    return intrinsic_matrices

def random_crop_tensor(tensor, h_start, w_start, h_crop, w_crop):
    cropped_tensor = tensor[:, :, h_start:h_start + h_crop, w_start:w_start + w_crop]
    return cropped_tensor
