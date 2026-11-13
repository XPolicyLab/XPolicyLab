"""Per-episode selected RGB observations, clocked by native control steps."""
import torch
import torch.nn.functional as F
import numpy as np
from .history_sampling import history_frame_indices, selected_capture_step, SPLIT
from .processing.observation import CAMERAS


def prepare_history_rgb(observation, *, device, dtype):
    per_cam = {}
    for slot, key, hw in CAMERAS:
        rgb = np.asarray(observation['observation'][key]['rgb'])
        if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[-1] != 3:
            raise ValueError('History requires decoded uint8 three-camera RGB')
        image = torch.as_tensor(rgb.copy(), device=device).permute(2, 0, 1).float()[None] / 127.5 - 1
        per_cam[slot] = F.interpolate(image, size=hw, mode='bilinear',
                                     align_corners=False, antialias=True).to(dtype)
    wrists = [F.interpolate(per_cam[slot], size=(128, 160), mode='bilinear',
                             align_corners=False, antialias=True)
              for slot in ('cam_left_wrist', 'cam_right_wrist')]
    return torch.cat([per_cam['cam_high'], torch.cat(wrists, dim=-1)], dim=-2)[0]


class SelectedHistory:
    """One instance per environment; observation updates follow executed actions."""
    def __init__(self):
        self.clear()

    def clear(self):
        self.step = None
        self.first = None
        self.selected = {}

    def observe(self, observation, *, device, dtype):
        step = 0 if self.step is None else self.step + 1
        if selected_capture_step(step):
            self.record(step, prepare_history_rgb(observation, device=device, dtype=dtype))
        else:
            self.record(step)

    def record(self, step, frame=None):
        expected = 0 if self.step is None else self.step + 1
        if isinstance(step, bool) or step != expected:
            raise ValueError('Record each native control step once, in order')
        if selected_capture_step(step):
            if frame is None or frame.shape != (3, 384, 320) or not torch.is_floating_point(frame):
                raise ValueError('A selected step needs composed RGB [3,384,320]')
            if not torch.isfinite(frame).all() or (frame.abs() > 1.00001).any():
                raise ValueError('Expected finite normalized history RGB')
            frame = frame.detach().clone()
            if step == 0:
                self.first = frame
            self.selected[step] = frame
            if step % 32 == 0:
                self.selected = {k: v for k, v in self.selected.items() if k >= step - 32}
        elif frame is not None:
            raise ValueError('An unselected native step does not consume an RGB frame')
        self.step = int(step)

    def inputs(self, current_image):
        if self.step is None or self.step % 32:
            raise ValueError('Request the next chunk at its observed 32-step boundary')
        current = current_image[0] if current_image.ndim == 4 else current_image
        if not torch.equal(current.to(self.selected[self.step]), self.selected[self.step]):
            raise ValueError('Current RGB differs from the latest observed control step')
        indices = history_frame_indices(self.step, SPLIT)
        frames = [self.first if k == 0 else self.selected[k] for k in indices]
        return dict(history_video=torch.stack(frames, 1)[None],
                    history_frame_offsets=torch.tensor([[k-self.step for k in indices]],
                        device=self.first.device, dtype=torch.float32))
