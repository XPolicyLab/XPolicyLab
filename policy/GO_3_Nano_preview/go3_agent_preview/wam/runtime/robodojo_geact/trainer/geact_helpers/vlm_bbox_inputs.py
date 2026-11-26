from __future__ import annotations
import torch


def draw_bbox_outside_border(
    images: torch.Tensor,
    bboxes: torch.Tensor,
    *,
    border_pixels: int = 5,
) -> torch.Tensor:
    if images.ndim != 4 or images.shape[1] < 3:
        raise ValueError(f"images must be [B,C>=3,H,W], got {tuple(images.shape)}")
    if border_pixels <= 0:
        raise ValueError("border_pixels must be positive")
    boxes = torch.as_tensor(bboxes, device=images.device, dtype=torch.float32)
    if boxes.shape != (images.shape[0], 4):
        raise ValueError(
            f"bboxes must be [B,4] for batch {images.shape[0]}, got {tuple(boxes.shape)}"
        )
    valid = torch.isfinite(boxes).all(dim=1) & (boxes >= 0).all(dim=1) & (boxes <= 1000).all(dim=1)
    valid &= (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
    boxes = torch.where(valid[:, None], boxes, torch.zeros_like(boxes))
    _, _, height, width = images.shape
    x0 = torch.floor(boxes[:, 0] * width / 1000.0).long().clamp(0, width)
    y0 = torch.floor(boxes[:, 1] * height / 1000.0).long().clamp(0, height)
    x1 = torch.ceil(boxes[:, 2] * width / 1000.0).long().clamp(0, width)
    y1 = torch.ceil(boxes[:, 3] * height / 1000.0).long().clamp(0, height)
    result = images.clone()
    red = result.new_tensor([1.0, -1.0, -1.0])
    for index in range(result.shape[0]):
        if not bool(valid[index]):
            continue
        left, top, right, bottom = map(
            int, (x0[index], y0[index], x1[index], y1[index])
        )
        outer_left = max(0, left - border_pixels)
        outer_top = max(0, top - border_pixels)
        outer_right = min(width, right + border_pixels)
        outer_bottom = min(height, bottom + border_pixels)
        result[index, :3, outer_top:top, outer_left:outer_right] = red[:, None, None]
        result[index, :3, bottom:outer_bottom, outer_left:outer_right] = red[:, None, None]
        result[index, :3, top:bottom, outer_left:left] = red[:, None, None]
        result[index, :3, top:bottom, right:outer_right] = red[:, None, None]
    return result

def build_bbox_vlm_inputs(
    vlm_interface,
    *,
    video,
    bboxes,
    frame_index,
    head_view_index,
    instructions,
    crop_keep_mask=None,
    crop_images=None,
    bbox_reference_images=None,
    bbox_add_image_description=True,
):
    if video.ndim != 6:
        raise ValueError(f"video must be [B,C,V,T,H,W], got {tuple(video.shape)}")
    if not 0 <= int(head_view_index) < video.shape[2]:
        raise ValueError(f"head_view_index {head_view_index} is outside {video.shape[2]} views")
    if not 0 <= int(frame_index) < video.shape[3]:
        raise ValueError(f"frame_index {frame_index} is outside {video.shape[3]} frames")
    batch_size = video.shape[0]
    if len(instructions) != batch_size:
        raise ValueError("instructions must contain one task per batch element")
    if any(not isinstance(instruction, str) for instruction in instructions):
        raise TypeError("each task instruction must be a string")
    global_image = video[:, :, int(head_view_index), int(frame_index)]
    global_hwc = ((global_image.detach().cpu() + 1) / 2 * 255.0).permute(0, 2, 3, 1)
    if bboxes is None:
        if crop_keep_mask is not None:
            raise ValueError("crop_keep_mask must be omitted when bboxes is None")
        reference_kept = [False] * batch_size
    elif crop_keep_mask is None:
        reference_kept = [True] * batch_size
    else:
        keep_mask = torch.as_tensor(crop_keep_mask)
        if keep_mask.ndim != 1 or keep_mask.shape[0] != batch_size:
            raise ValueError(f"crop_keep_mask must be [B] for batch {batch_size}")
        reference_kept = keep_mask.detach().cpu().bool().tolist()

    if bboxes is not None:
        boxes = torch.as_tensor(bboxes)
        if boxes.shape != (batch_size, 4):
            raise ValueError(f"bboxes must be [B,4], got {tuple(boxes.shape)}")
        valid = torch.isfinite(boxes).all(dim=1) & (boxes >= 0).all(dim=1) & (boxes <= 1000).all(dim=1)
        valid &= (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
        reference_kept = [kept and usable for kept, usable in zip(reference_kept, valid.cpu().tolist())]

    if not any(reference_kept):
        return vlm_interface.build_qwenvl_inputs(
            images=[[image] for image in global_hwc],
            instructions=list(instructions),
        )
    if not hasattr(bboxes, "to"):
        bboxes = torch.as_tensor(bboxes)
    if crop_images is None:
        crop_source = global_image
        if bbox_reference_images is not None:
            crop_source = torch.as_tensor(
                bbox_reference_images,
                device=video.device,
                dtype=video.dtype,
            )
            expected = (video.shape[0], video.shape[1], video.shape[4], video.shape[5])
            if tuple(crop_source.shape) != expected:
                raise ValueError(
                    "bbox_reference_images must have shape "
                    f"{expected}, got {tuple(crop_source.shape)}"
                )
        crop = draw_bbox_outside_border(crop_source, bboxes)
    else:
        crop = torch.as_tensor(crop_images, device=video.device, dtype=video.dtype)
        expected = (video.shape[0], video.shape[1], video.shape[4], video.shape[5])
        if tuple(crop.shape) != expected:
            raise ValueError(f"crop_images must have shape {expected}, got {tuple(crop.shape)}")
    described_instructions = []
    for instruction, kept in zip(instructions, reference_kept):
        if not kept or not bbox_add_image_description:
            described_instructions.append(instruction)
            continue
        description = "Image 1: current view, Image 2: step-start reference, red box: target object or destination."
        described_instructions.append(f"{description}\n\nTask: {instruction}")
    crop_hwc = ((crop.detach().cpu() + 1) / 2 * 255.0).permute(0, 2, 3, 1)
    images = [
        [global_hwc[index], crop_hwc[index]] if kept else [global_hwc[index]]
        for index, kept in enumerate(reference_kept)
    ]
    return vlm_interface.build_qwenvl_inputs(
        images=images, instructions=described_instructions
    )
