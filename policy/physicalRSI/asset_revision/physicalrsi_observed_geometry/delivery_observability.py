"""Predict two-view centerline conditioning from live hand-eye calibration.

This rejects geometrically degenerate view pairs; it does not predict occlusion,
segmentation quality, or slip of the object inside the gripper.
"""
import cv2
import numpy as np
from . import Camera, GeometryUnavailable
from .multiview import projection
from .insert_obstacles import transform


def centerline_condition(cameras, root, axis, length):
    root, axis = np.asarray(root, float), np.asarray(axis, float)
    if (len(cameras) != 2 or root.shape != (3,) or axis.shape != (3,)
            or not np.isfinite(np.r_[root, axis, length]).all()
            or length <= 0 or np.linalg.norm(axis) < 1e-8):
        raise GeometryUnavailable('Two calibrated views and finite observed tube axis required')
    axis = axis / np.linalg.norm(axis)
    planes = []
    for camera in cameras:
        try:
            pixels, depth = camera.project([root, root + axis * length])
        except GeometryUnavailable:
            return 0.
        if (not np.isfinite(pixels).all() or np.any(depth <= 0)
                or np.max(abs(pixels)) > 1e6 or np.linalg.norm(pixels[1] - pixels[0]) < 3):
            return 0.
        if not cv2.clipLine((0, 0, camera.width, camera.height),
                tuple(np.rint(pixels[0]).astype(int)), tuple(np.rint(pixels[1]).astype(int)))[0]:
            return 0.
        line = np.cross(np.r_[pixels[0], 1.], np.r_[pixels[1], 1.])
        plane = projection(camera).T @ line
        norm = np.linalg.norm(plane[:3])
        if norm < 1e-10:
            return 0.
        planes.append(plane[:3] / norm)
    singular = np.linalg.svd(planes, compute_uv=False)
    return float(singular[1] / singular[0])


def observable_candidates(groups, observation, tube_poses, lengths, pickup_contacts,
                          delivery_poses, to_ee, minimum_condition=.075):
    """Retain pose variants whose nominal delivery permits centerline reconstruction."""
    head = Camera.from_observation(observation)
    wrists = {arm: Camera.from_observation(observation, f'cam_{arm}_wrist') for arm in ('left', 'right')}
    mounts = {arm: np.linalg.inv(transform(observation['state'][arm + '_ee_pose'])) @ camera.transform
              for arm, camera in wrists.items()}
    result = []
    for group in groups:
        keep = []
        for candidate in group:
            i = int(candidate.label.removeprefix('tube'))
            tube, length = tube_poses[i], lengths[i]
            pickup_pose = pickup_contacts(tube, length)[candidate.pickup_index]
            pickup = transform(pickup_pose)
            attachment = pickup[:3, :3].T @ (np.asarray(tube[:3]) - pickup[:3, 3])
            axis = transform(candidate.slot_pose)[:3, 2]
            poses = delivery_poses(tube, pickup_pose, candidate.slot_pose)
            valid = True
            for phase in ('preinsert', 'align', 'realign', 'guide'):
                semantic = transform(poses[phase])
                root = semantic[:3, 3] + semantic[:3, :3] @ attachment
                moved = transform(to_ee(poses[phase])) @ mounts[candidate.arm]
                wrist = wrists[candidate.arm]
                camera = Camera(wrist.image, wrist.k, moved)
                if centerline_condition([head, camera], root, axis, length) < minimum_condition:
                    valid = False
                    break
            if valid:
                keep.append(candidate)
        result.append(tuple(keep))
    return result
