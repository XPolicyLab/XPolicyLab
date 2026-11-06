---
name: policy_camera_composition
description: "Adapter-bound Model._stitch_image combines the three public RGB camera arrays using self.data_cfg.combine_views and the _CAM_ENV_TO_DS global, returning HWC uint8 RGB. It converts each camera to a normalized CHW tensor, combines/resizes to the configured video size, then converts back to HWC uint8."
---

# Policy camera composition

## Function

Adapter-bound Model._stitch_image combines the three public RGB camera arrays using self.data_cfg.combine_views and the _CAM_ENV_TO_DS global, returning HWC uint8 RGB. It converts each camera to a normalized CHW tensor, combines/resizes to the configured video size, then converts back to HWC uint8.

## Inputs

- obs: dict whose 'vision' mapping contains the env camera keys required by _CAM_ENV_TO_DS; each value may be an array or a dict with 'color'
- self.data_cfg: adapter config providing combine_views, video_height, video_width, and resize
- _CAM_ENV_TO_DS: adapter global mapping env camera keys to dataset keys

## Outputs

- np.ndarray of shape [video_height, video_width, 3] with dtype uint8 in RGB order

## Callable interface

Implementation: [implementation.py](implementation.py).

- `_stitch_image(self, obs: dict)`

## Constraints

- Requires every mapped camera; missing camera raises KeyError and non-HWC or fewer-than-3-channel input raises ValueError
- Non-uint8 arrays are scaled by 255 when max<=1.5, otherwise clipped to 0..255 before uint8 conversion
- Uses the same RoboDojo training view layout via combine_views so eval layout does not drift; the official server owns decoding
- The snippet depends on the calling adapter's self, globals, constants, and exception behavior
