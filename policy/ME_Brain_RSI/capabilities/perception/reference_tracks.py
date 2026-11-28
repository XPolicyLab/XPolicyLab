"""Re-register depleted tracks against immutable, same-instance RGB evidence.

This returns correspondences, never a grasp/attachment decision. The original
static-versus-attached and metric object-pose gates consume the correspondences.
"""
import cv2
import numpy as np


class ReferenceTracks:
    def __init__(self, rgb, mask):
        self.gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        self.mask = np.asarray(mask, np.uint8).copy()
        self.points = cv2.goodFeaturesToTrack(self.gray, 120, .005, 2, mask=self.mask)

    def match(self, rgb, foreground, hint=None):
        if self.points is None or len(self.points) < 6:
            raise ValueError('reference_features_insufficient')
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        guess = None
        if hint is not None and len(hint[0]) >= 3:
            matrix, _ = cv2.estimateAffinePartial2D(
                np.asarray(hint[0]).reshape(-1, 2), np.asarray(hint[1]).reshape(-1, 2),
                method=cv2.RANSAC, ransacReprojThreshold=1.5)
            if matrix is not None and np.isfinite(matrix).all():
                guess = cv2.transform(self.points, matrix).astype(np.float32)
        points, status, error = cv2.calcOpticalFlowPyrLK(
            self.gray, gray, self.points, guess, winSize=(21, 21), maxLevel=3,
            flags=cv2.OPTFLOW_USE_INITIAL_FLOW if guess is not None else 0)
        if points is None:
            raise ValueError('reference_flow_missing')
        back, backward_status, _ = cv2.calcOpticalFlowPyrLK(
            gray, self.gray, points, None, winSize=(21, 21), maxLevel=3)
        if back is None:
            raise ValueError('reference_backward_missing')
        finite = np.isfinite(points[:, 0]).all(axis=1)
        xy = np.nan_to_num(points[:, 0]).round().astype(int)
        h, w = gray.shape
        inside = finite & (xy[:, 0] >= 4) & (xy[:, 0] < w-4) & (xy[:, 1] >= 4) & (xy[:, 1] < h-4)
        color = np.zeros(len(xy), bool)
        color[inside] = foreground[xy[inside, 1], xy[inside, 0]]
        keep = (status[:, 0] > 0) & (backward_status[:, 0] > 0) & color
        keep &= (np.linalg.norm(back[:, 0]-self.points[:, 0], axis=1) < .8) & (error[:, 0] < 25)
        source = self.points[keep, 0]; target = points[keep, 0]
        if len(source) < 6:
            raise ValueError('reference_correspondences_insufficient')
        matrix, inliers = cv2.estimateAffinePartial2D(
            source, target, method=cv2.RANSAC, ransacReprojThreshold=1.5,
            maxIters=1000, confidence=.99, refineIters=10)
        if matrix is None:
            raise ValueError('reference_geometry_missing')
        valid = inliers[:, 0] > 0
        if valid.sum() < 6 or valid.mean() < .7:
            raise ValueError('reference_geometry_inconsistent')
        source = source[valid]; target = target[valid]
        scale = float(np.linalg.norm(matrix[0, :2]))
        angle = float(np.arctan2(matrix[1, 0], matrix[0, 0]))
        if not .8 <= scale <= 1.2 or abs(angle) > .4:
            raise ValueError('reference_transform_excessive')
        area = cv2.contourArea(cv2.convexHull(source.astype(np.float32)))
        if area < 60 or np.min(np.ptp(source, axis=0)) < 6:
            raise ValueError('reference_geometry_degenerate')
        warped = cv2.warpAffine(self.mask, matrix, (w, h), flags=cv2.INTER_NEAREST) > 0
        coverage = float(np.mean(foreground[warped])) if warped.any() else 0.
        if coverage < .65:
            raise ValueError('reference_foreground_missing')
        # The reference is never refreshed from an uncertain current frame.
        # Compare textured RGB patches in addition to hue/shape consistency.
        aligned = cv2.warpAffine(gray, matrix, (w, h),
                                flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP)
        correlations = []
        for src, dst in zip(source, target):
            a = cv2.getRectSubPix(self.gray, (7, 7), tuple(map(float, src))).astype(float)
            c = cv2.getRectSubPix(aligned, (7, 7), tuple(map(float, src))).astype(float)
            a -= a.mean(); c -= c.mean()
            correlations.append(float((a*c).sum()/max(np.linalg.norm(a)*np.linalg.norm(c), 1.)))
        correlations = np.asarray(correlations)
        reliable = correlations >= .65
        if reliable.sum() < 6 or np.median(correlations) < .8:
            raise ValueError('reference_texture_mismatch')
        source = source[reliable]; target = target[reliable]
        if cv2.contourArea(cv2.convexHull(source.astype(np.float32))) < 60:
            raise ValueError('reference_supported_geometry_degenerate')
        return source[:, None].astype(np.float32), target[:, None].astype(np.float32), dict(
            features=len(source), foreground_coverage=coverage,
            median_patch_correlation=float(np.median(correlations)),
            scale=scale, rotation_rad=angle, source='immutable_initial_RGB')
