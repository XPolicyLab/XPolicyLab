# Copyright © 2026 Martingale. All Rights Reserved.
import copy
import numpy as np
from PIL import Image, ImageDraw, ImageOps
from martingale.inference.extensions.key.key_alignment_report import project

def remember_opening(robot, result):
    """Cache only the same accepted calibration already returned to policy."""
    if result.get('accepted') is True:
        center = np.asarray(result.get('entry_xyz'), float)
        normal = np.asarray(result.get('normal_xyz'), float)
        if center.shape == (3,) and normal.shape == (3,) and np.isfinite(center).all() and np.isfinite(normal).all() and (abs(np.linalg.norm(normal) - 1.0) < 1e-06):
            robot._key_feedback_opening = copy.deepcopy(result)
    return result

def panels(robot):
    """Use fresh camera calibration/RGB only; no key landmarks or motion."""
    step = robot.state().step
    opening = getattr(robot, '_key_feedback_opening', None)
    canvas = Image.new('RGB', (640, 854), '#161a20')
    draw = ImageDraw.Draw(canvas)
    draw.text((8, 8), f'Current views, observation {step}. Left: RAW. Right: projected axis.', fill='white')
    draw.text((8, 24), 'Green axis is calibrated; key pose/alignment is NOT measured.', fill='white')
    metadata = dict(observed_step=step, cameras={}, key_pose_measured=False, descent_certified=False, episode_actions=0)
    for row, camera in enumerate(('cam_head', 'cam_left_wrist', 'cam_right_wrist')):
        y = 48 + row * 266
        try:
            raw = Image.fromarray(robot.image(camera, step=step)).convert('RGB')
            meta = robot.camera_meta(camera, step=step)
            width, height = raw.size
            points = []
            if opening is not None:
                center = np.asarray(opening['entry_xyz'], float)
                normal = np.asarray(opening['normal_xyz'], float)
                points = [project(meta, center + distance * normal) for distance in np.linspace(0.0, 0.08, 17)]
            visible = [p for p in points if p is not None and np.isfinite(p).all() and (0 <= p[0] < width) and (0 <= p[1] < height)]
            anchors = visible + ([[width / 2, height / 2]] if camera != 'cam_head' else [])
            if anchors:
                lo = np.min(anchors, axis=0) - 56
                hi = np.max(anchors, axis=0) + 56
                box = (max(0, int(lo[0])), max(0, int(lo[1])), min(width, int(np.ceil(hi[0]))), min(height, int(np.ceil(hi[1]))))
            else:
                box = (0, 0, width, height)
            marked = raw.copy()
            overlay = ImageDraw.Draw(marked)
            for a, b in zip(points, points[1:]):
                if a is not None and b is not None and np.isfinite([a, b]).all():
                    if max((abs(v) for p in (a, b) for v in p)) < 1000000.0:
                        overlay.line([tuple(a), tuple(b)], fill=(0, 255, 80), width=2)
            if points and points[0] is not None:
                u, v = points[0]
                if 0 <= u < width and 0 <= v < height:
                    overlay.ellipse((u - 4, v - 4, u + 4, v + 4), outline=(0, 255, 80), width=2)
            for col, picture in enumerate((raw, marked)):
                crop = ImageOps.contain(picture.crop(box), (312, 238))
                canvas.paste(crop, (col * 320 + 4, y + 20))
            draw.text((8, y), camera + (' / axis visible' if visible else ' / axis unavailable here'), fill='white')
            metadata['cameras'][camera] = dict(crop_xyxy=list(box), axis_visible=bool(visible))
        except Exception as exc:
            metadata['cameras'][camera] = dict(error=f'{type(exc).__name__}: {exc}')
            draw.text((8, y), camera + ' unavailable; use other current views.', fill='white')
    return (canvas, metadata)
