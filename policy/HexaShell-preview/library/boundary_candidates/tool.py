"""Read-only RGB contour proposals; no metric geometry or semantic inference."""
import io
import json
import importlib.util
from pathlib import Path
import cv2
import numpy as np
from PIL import Image

from types import SimpleNamespace
# The source loaded pour_liquid_into_cup/tools/region_geometry only for this camera-name table.
region = SimpleNamespace(CAMERAS={"head": "cam_head", "wrist_l": "cam_left_wrist", "wrist_r": "cam_right_wrist"})
TOOL = {'name': 'boundary_candidates', 'commands': [{
    'name': 'boundary_candidates', 'budget': False,
    'help': 'propose measured near-elliptical RGB contours inside a pixel rectangle',
    'args': [{'name': 'camera', 'choices': list(region.CAMERAS), 'default': 'head'},
             {'name': 'roi', 'required': True, 'help': 'JSON [left,top,right,bottom] in original image pixels'}]}]}


def candidates(rgb, roi):
    bounds = np.asarray(json.loads(roi) if isinstance(roi, str) else roi, float)
    h, w = rgb.shape[:2]
    if bounds.shape != (4,) or not np.isfinite(bounds).all():
        raise ValueError('roi must contain four finite pixel coordinates')
    if not np.equal(bounds, np.round(bounds)).all():
        raise ValueError('roi coordinates must be integers')
    x0, y0, x1, y1 = bounds.astype(int)
    if not (0 <= x0 < x1 <= w and 0 <= y0 < y1 <= h and
            12 <= x1-x0 <= 320 and 12 <= y1-y0 <= 320):
        raise ValueError('roi must be inside RGB with width/height 12..320 pixels')
    crop = rgb[y0:y1, x0:x1]
    found = []
    for channel in range(3):
        gray = cv2.GaussianBlur(crop[:, :, channel], (3, 3), .7)
        contours, _ = cv2.findContours(cv2.Canny(gray, 30, 80), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        for contour in contours:
            p = contour[:, 0, :].astype(float)
            if len(p) < 20 or np.any(p < 2) or np.any(p[:, 0] >= x1-x0-2) or np.any(p[:, 1] >= y1-y0-2):
                continue
            (cx, cy), (a, b), degrees = cv2.fitEllipse(p.astype(np.float32))
            if min(a, b) < 8 or max(a, b)/min(a, b) > 6 or max(a, b) > 1.5*max(crop.shape[:2]):
                continue
            angle = np.radians(degrees)
            rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
            local = (p-[cx, cy]) @ rotation
            normalized = local/[a/2, b/2]
            theta = np.mod(np.arctan2(normalized[:, 1], normalized[:, 0]), 2*np.pi)
            # Approximate distance to ellipse, used only to reject contour clutter.
            radial = np.linalg.norm(normalized, axis=1)
            residual = np.abs(radial-1)*min(a, b)/2
            if np.sqrt(np.mean(residual**2)) > 1.25 or np.quantile(residual, .95) > 2.5:
                continue
            bins = np.floor(theta/(2*np.pi)*24).astype(int) % 24
            if len(np.unique(bins)) < 22:
                continue
            chosen = []
            for sector in range(24):
                indices = np.flatnonzero(bins == sector)
                if len(indices):
                    # Return actual observed edge pixels, never fitted ellipse samples.
                    target = (sector+.5)*2*np.pi/24
                    chosen.append(p[indices[np.argmin(np.abs(theta[indices]-target))]])
            measured = np.unique(np.asarray(chosen)+[x0, y0], axis=0)
            # Sort around the fit for readable outlines and downstream coverage.
            order = np.arctan2(measured[:, 1]-cy-y0, measured[:, 0]-cx-x0).argsort()
            measured = measured[order]
            center = np.array([cx+x0, cy+y0])
            duplicate = any(np.linalg.norm(center-np.array(f['ellipse_center_px'])) < 1.5 and
                            np.max(np.abs(np.sort([a, b])-np.sort(f['ellipse_diameters_px']))) < 2
                            for f in found)
            if duplicate:
                continue
            found.append(dict(points=measured.tolist(), ellipse_center_px=center.tolist(),
                              ellipse_diameters_px=[float(a), float(b)],
                              ellipse_angle_deg=float(degrees), channel=channel,
                              contour_rms_px=float(np.sqrt(np.mean(residual**2))),
                              angular_sectors=len(np.unique(bins))))
    found.sort(key=lambda f: (-f['angular_sectors'], f['contour_rms_px']))
    return found[:8]


def run(api, command, args):
    try:
        if command != 'boundary_candidates':
            raise ValueError('unknown command')
        name = args.get('camera', 'head')
        obs = api.observe()
        with Image.open(io.BytesIO(obs['png'][region.CAMERAS[name]])) as im:
            rgb = np.asarray(im.convert('RGB'))
        found = candidates(rgb, args['roi'])
        return dict(plan_ok=bool(found),
                    plan_fail_reason=None if found else 'no_closed_rgb_boundary',
                    camera=name, candidates=found,
                    note='Unverified RGB edge proposals, not world geometry. Different contours may be different physical boundaries. Ellipse centre is not the projection of a 3D circle centre. No cycle_geometry or motion.'), 0 if found else 2
    except Exception as exc:
        return dict(plan_ok=False, plan_fail_reason='boundary_candidates_unavailable', plan_detail=str(exc)), 2
