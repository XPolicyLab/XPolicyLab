import torch


def check_resize_height_width(height, width, num_frames):
    if height % 16 != 0:
        height = (height + 15) // 16 * 16
    if width % 16 != 0:
        width = (width + 15) // 16 * 16
    if num_frames % 4 != 1:
        num_frames = (num_frames + 3) // 4 * 4 + 1
    return height, width, num_frames


def video_time_dim(video: torch.Tensor) -> int:
    if video.ndim == 5:
        return 2
    if video.ndim == 6:
        return 3
    raise ValueError(f"Expected video shape [B,C,T,H,W] or [B,N,C,T,H,W], got {tuple(video.shape)}")


def video_num_frames(video: torch.Tensor) -> int:
    return int(video.shape[video_time_dim(video)])


def video_batch_size(video: torch.Tensor) -> int:
    if video.ndim not in (5, 6):
        raise ValueError(f"Expected video shape [B,C,T,H,W] or [B,N,C,T,H,W], got {tuple(video.shape)}")
    return int(video.shape[0])


def video_spatial_shape(video: torch.Tensor) -> tuple[int, int]:
    if video.ndim not in (5, 6):
        raise ValueError(f"Expected video shape [B,C,T,H,W] or [B,N,C,T,H,W], got {tuple(video.shape)}")
    return int(video.shape[-2]), int(video.shape[-1])


def video_num_cameras(video: torch.Tensor) -> int:
    if video.ndim == 5:
        return 1
    if video.ndim == 6:
        return int(video.shape[1])
    raise ValueError(f"Expected video shape [B,C,T,H,W] or [B,N,C,T,H,W], got {tuple(video.shape)}")


def select_video_time(video: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    return video.index_select(video_time_dim(video), indices)


def slice_video_time(video: torch.Tensor, start: int, length: int) -> torch.Tensor:
    return video.narrow(video_time_dim(video), int(start), int(length))


def video_pad_mask_for(video: torch.Tensor, pad: torch.Tensor) -> torch.Tensor:
    if video.ndim == 5:
        return pad[:, None, :, None, None]
    if video.ndim == 6:
        return pad[:, None, None, :, None, None]
    raise ValueError(f"Expected video shape [B,C,T,H,W] or [B,N,C,T,H,W], got {tuple(video.shape)}")


def take_video_frame(video: torch.Tensor, frame_idx: int) -> torch.Tensor:
    frame = slice_video_time(video, int(frame_idx), 1)
    return frame.squeeze(video_time_dim(frame))


def flatten_camera_image_tensor(image: torch.Tensor) -> torch.Tensor:
    if image.ndim == 4:
        return image
    if image.ndim == 5:
        return torch.cat([image[:, cam_idx] for cam_idx in range(int(image.shape[1]))], dim=-1)
    raise ValueError(f"Expected image shape [B,C,H,W] or [B,N,C,H,W], got {tuple(image.shape)}")


def flatten_camera_video_tensor(video: torch.Tensor) -> torch.Tensor:
    if video.ndim == 5:
        return video
    if video.ndim == 6:
        return torch.cat([video[:, cam_idx] for cam_idx in range(int(video.shape[1]))], dim=-1)
    raise ValueError(f"Expected video shape [B,C,T,H,W] or [B,N,C,T,H,W], got {tuple(video.shape)}")


def normalize_input_image_tensor(input_image: torch.Tensor) -> torch.Tensor:
    if input_image.ndim == 3:
        return input_image.unsqueeze(0)
    if input_image.ndim == 4:
        if int(input_image.shape[0]) == 1 and int(input_image.shape[1]) == 3:
            return input_image
        if int(input_image.shape[1]) == 3:
            return input_image.unsqueeze(0)
    if input_image.ndim == 5:
        if int(input_image.shape[0]) == 1 and int(input_image.shape[2]) == 3:
            return input_image
    raise ValueError(
        "`input_image` must have shape [3,H,W], [1,3,H,W], [N,3,H,W], "
        f"or [1,N,3,H,W], got {tuple(input_image.shape)}"
    )
