
from __future__ import annotations

import numpy as np

from ...lib.kinematics import chain as kin
from ...lib.geometry.transforms import TABLE_Z

CAM_BOX = ((0.045, -0.032, 0.022), (0.100, 0.032, 0.078))
LINKS = ("link4", "link5", "link6", "link7", "link8", "cam")
GRIPPER = ("link6", "cam")
_LO = {n: np.array(kin.LINK_BOXES[n][0]) for n in LINKS if n != "cam"}
_HI = {n: np.array(kin.LINK_BOXES[n][1]) for n in LINKS if n != "cam"}
_LO["cam"], _HI["cam"] = np.array(CAM_BOX[0]), np.array(CAM_BOX[1])

def axis_points(ob, step=0.015):
    if ob["kind"] == "standing":
        a = np.array([ob["xy"][0], ob["xy"][1], TABLE_Z])
        b = a + np.array([0.0, 0.0, ob["h"]])
    else:
        u = np.array([ob["u"][0], ob["u"][1], 0.0])
        c = np.array([ob["xy"][0], ob["xy"][1], TABLE_Z + ob["r"]])
        a, b = c - u * ob["len"] / 2, c + u * ob["len"] / 2
    n = int(np.ceil(np.linalg.norm(b - a) / step)) + 1
    t = np.linspace(0.0, 1.0, n)
    return a + t[:, None] * (b - a)

def _seg_dist(P, a, b):
    ab = b - a
    t = np.clip(((P - a) @ ab) / max(ab @ ab, 1e-12), 0.0, 1.0)
    return np.linalg.norm(P - (a + t[:, None] * ab), axis=1)

def held_segment(T_tcp, held):
    ax = np.array([0.0, 0.0, 1.0]) if held["axis"] == "world" else T_tcp[:3, 2]
    c = T_tcp[:3, 3]
    return c + held["lo"] * ax, c + held["hi"] * ax, held["r"]

def _box_frames(arm, q, grip, links=LINKS):
    frames = kin.link_frames(arm, q, grip)
    frames["cam"] = frames["link6"]
    return frames, [(frames[n], _LO[n], _HI[n]) for n in links]

def hits_config(frames_boxes, held_seg, obstacles, margin=0.012, axes=None):
    frames, boxes = frames_boxes
    out = []
    for k, ob in enumerate(obstacles):
        P = axis_points(ob) if axes is None else axes[k]
        lim = ob["r"] + margin
        hit = False
        for T, lo, hi in boxes:
            L = (P - T[:3, 3]) @ T[:3, :3]
            d = np.linalg.norm(L - np.clip(L, lo, hi), axis=1)
            if np.any(d < lim):
                hit = True
                break
        if not hit and held_seg is not None:
            a, b, r = held_seg
            if np.any(_seg_dist(P, a, b) < lim + r):
                hit = True
        if hit:
            out.append(k)
    return out

def config_hits(arm, q, obstacles, margin=0.012, grip=kin.GRIP_MAX, held=None, axes=None, links=LINKS):
    if not obstacles:
        return []
    fb = _box_frames(arm, q, grip, links)
    hs = None if held is None else held_segment(kin.tcp_from_link6(fb[0]["link6"]), held)
    return hits_config(fb, hs, obstacles, margin, axes)

def gripper_clearance_ok(arm, qs, target, margin=0.008):
    axes = [axis_points(target)]
    return all(not config_hits(arm, q, [target], margin, axes=axes, links=GRIPPER) for q in qs)

def path_hits(arm, qs, obstacles, margin=0.012, every=2, grip=kin.GRIP_MAX, held=None):
    if not obstacles or not qs:
        return []
    axes = [axis_points(ob) for ob in obstacles]
    found = set()
    idx = list(range(0, len(qs), every))
    if idx[-1] != len(qs) - 1:
        idx.append(len(qs) - 1)
    for i in idx:
        found.update(config_hits(arm, qs[i], obstacles, margin, grip, held, axes))
        if len(found) == len(obstacles):
            break
    return sorted(found)

def departing_hits(arm, qs, obstacles, start_idx, margin=0.012, every=2, grip=kin.GRIP_MAX, held=None):
    if not start_idx or not qs:
        return []
    sub = [obstacles[k] for k in start_idx]
    axes = [axis_points(ob) for ob in sub]
    free = [False] * len(sub)
    bad = set()
    idx = list(range(0, len(qs), every))
    if idx[-1] != len(qs) - 1:
        idx.append(len(qs) - 1)
    for i in idx:
        h = set(config_hits(arm, qs[i], sub, margin, grip, held, axes))
        for j in range(len(sub)):
            if j in h:
                if free[j]:
                    bad.add(start_idx[j])
            else:
                free[j] = True
    return sorted(bad)

def top(ob):
    return TABLE_Z + (ob["h"] if ob["kind"] == "standing" else 2 * ob["r"])
