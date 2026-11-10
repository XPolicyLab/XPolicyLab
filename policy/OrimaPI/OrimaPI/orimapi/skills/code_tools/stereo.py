"""Current-RGB epipolar diagnostics and experimental feature proposals.

Only current RGB and public camera calibration enter this function. Photometric
and epipolar agreement do not establish physical-point identity, contact or task
success. Both guides and feature candidates are read-only proposals; neither
automatically accepts correspondences or creates object geometry.
"""

import cv2
import numpy as np


def roi(image, bounds):
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Expected original uint8 RGB")
    box = np.asarray(bounds)
    if box.shape != (4,) or box.dtype.kind not in "iu":
        raise ValueError("ROI requires four integer original-image coordinates")
    x0, y0, x1, y1 = map(int, box)
    if not (0 <= x0 < x1 <= image.shape[1] and 0 <= y0 < y1 <= image.shape[0]):
        raise ValueError("ROI outside original image")
    if min(x1 - x0, y1 - y0) < 12:
        raise ValueError("ROI must span at least 12 pixels")
    mask = np.zeros(image.shape[:2], np.uint8)
    mask[y0:y1, x0:x1] = 255
    return mask


def fundamental(first, second):
    relative = np.linalg.inv(second.optical) @ first.optical
    translation = relative[:3, 3]
    if np.linalg.norm(translation) < 0.01:
        raise ValueError("Camera baseline must exceed 1 cm")
    x, y, z = translation
    skew = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
    return np.linalg.inv(second.k).T @ skew @ relative[:3, :3] @ np.linalg.inv(first.k)


def epipolar_error(matrix, first, second):
    a, b = np.r_[first, 1.0], np.r_[second, 1.0]
    in_second, in_first = matrix @ a, matrix.T @ b
    residual = abs(float(b @ matrix @ a))
    norms = min(np.linalg.norm(in_first[:2]), np.linalg.norm(in_second[:2]))
    return residual / max(float(norms), 1e-12)


def propose(
    first_rgb, second_rgb, first_camera, second_camera, first_roi, second_roi, maximum=8
):
    if type(maximum) is not int or not 1 <= maximum <= 12:
        raise ValueError("Use 1..12 correspondence proposals")
    for image, camera, box in (
        (first_rgb, first_camera, first_roi),
        (second_rgb, second_camera, second_roi),
    ):
        if image.shape != (camera.height, camera.width, 3):
            raise ValueError("Original RGB dimensions must match public calibration")
        roi(image, box)
        if box[2] - box[0] > 320 or box[3] - box[1] > 320:
            raise ValueError("Stereo candidate ROIs are bounded to 320 pixels per side")
    first_mask, second_mask = roi(first_rgb, first_roi), roi(second_rgb, second_roi)
    matrix = fundamental(first_camera, second_camera)
    detector = cv2.SIFT_create(nfeatures=600, contrastThreshold=0.025, edgeThreshold=8)
    first_keys, first_desc = detector.detectAndCompute(
        cv2.cvtColor(first_rgb, cv2.COLOR_RGB2GRAY), first_mask
    )
    second_keys, second_desc = detector.detectAndCompute(
        cv2.cvtColor(second_rgb, cv2.COLOR_RGB2GRAY), second_mask
    )
    output = {
        "status": "unknown",
        "proposals": [],
        "keypoints": [len(first_keys), len(second_keys)],
        "official_success": None,
        "scope": (
            "Current-RGB photometric/epipolar proposals only; inspect both views to "
            "confirm the SAME physical material feature before marking/localizing. "
            "A descriptor centered on a silhouette or uniform face may not identify "
            "a shared physical point. Ratios and residuals are not accuracy probabilities. No object "
            "identity, metric accuracy, grasp, contact or task-success guarantee."
        ),
    }
    if (
        first_desc is None
        or second_desc is None
        or min(len(first_desc), len(second_desc)) < 2
    ):
        output["reason"] = "Insufficient textured features inside the selected ROIs"
        return output
    matcher = cv2.BFMatcher(cv2.NORM_L2)
    forward = matcher.knnMatch(first_desc, second_desc, k=2)
    reverse = matcher.knnMatch(second_desc, first_desc, k=2)
    reverse_good = {
        pair[0].queryIdx: pair[0].trainIdx
        for pair in reverse
        if len(pair) == 2 and pair[0].distance < 0.72 * pair[1].distance
    }
    rows = []
    for pair in forward:
        if len(pair) != 2:
            continue
        match, alternative = pair
        if (
            match.distance >= 0.72 * alternative.distance
            or reverse_good.get(match.trainIdx) != match.queryIdx
        ):
            continue
        a = np.asarray(first_keys[match.queryIdx].pt, float)
        b = np.asarray(second_keys[match.trainIdx].pt, float)
        error = epipolar_error(matrix, a, b)
        if not np.isfinite(error) or error > 1.5:
            continue
        ratio = match.distance / max(alternative.distance, 1e-6)
        rows.append(
            {
                "first_pixel": a.tolist(),
                "second_pixel": b.tolist(),
                "descriptor_ratio": float(ratio),
                "epipolar_error_px": float(error),
            }
        )
    rows.sort(key=lambda row: (row["descriptor_ratio"], row["epipolar_error_px"]))
    selected = []
    for row in rows:
        if any(
            np.linalg.norm(np.asarray(row["first_pixel"]) - other["first_pixel"]) < 6
            or np.linalg.norm(np.asarray(row["second_pixel"]) - other["second_pixel"])
            < 6
            for other in selected
        ):
            continue
        selected.append(row)
        if len(selected) == maximum:
            break
    output.update(
        status="proposals" if selected else "unknown",
        proposals=selected,
        reason=(
            "Inspect proposed correspondences in original current views"
            if selected
            else "No unambiguous reciprocal epipolar matches; do not force a correspondence"
        ),
    )
    return output


def clipped_line(line, width, height):
    a, b, c = map(float, line)
    points = []
    for x in (0.0, float(width - 1)):
        if abs(b) > 1e-12:
            y = -(a * x + c) / b
            if 0 <= y <= height - 1:
                points.append(np.array([x, y]))
    for y in (0.0, float(height - 1)):
        if abs(a) > 1e-12:
            x = -(b * y + c) / a
            point = np.array([x, y])
            if 0 <= x <= width - 1 and not any(
                np.linalg.norm(point - p) < 1e-6 for p in points
            ):
                points.append(point)
    return [p.tolist() for p in points[:2]] if len(points) >= 2 else []


def guides(
    first_rgb,
    second_rgb,
    first_camera,
    second_camera,
    pixels,
    second_roi,
    other_pixels=None,
):
    """Bounded geometric/corner hints, not automatic accepted correspondences."""
    points = np.asarray(pixels, dtype=float)
    if (
        points.ndim != 2
        or points.shape[1] != 2
        or not 1 <= len(points) <= 8
        or not np.isfinite(points).all()
        or (points < 0).any()
        or (points[:, 0] >= first_rgb.shape[1]).any()
        or (points[:, 1] >= first_rgb.shape[0]).any()
    ):
        raise ValueError("Use 1..8 finite current first-image pixels")
    roi(first_rgb, [0, 0, first_rgb.shape[1], first_rgb.shape[0]])
    mask = roi(second_rgb, second_roi)
    for image, camera in ((first_rgb, first_camera), (second_rgb, second_camera)):
        if image.shape[:2] != (camera.height, camera.width):
            raise ValueError("RGB dimensions disagree with calibrated camera")
    x0, y0, x1, y1 = map(int, second_roi)
    if max(x1 - x0, y1 - y0) > 320:
        raise ValueError("Target ROI must not exceed 320 pixels per side")
    selected = None
    if other_pixels is not None:
        selected = np.asarray(other_pixels, dtype=float)
        if (
            selected.shape != points.shape
            or not np.isfinite(selected).all()
            or (selected < 0).any()
            or (selected[:, 0] >= second_rgb.shape[1]).any()
            or (selected[:, 1] >= second_rgb.shape[0]).any()
        ):
            raise ValueError(
                "Paired target pixels must be finite current-image coordinates"
            )
    matrix = fundamental(first_camera, second_camera)
    gray = cv2.cvtColor(second_rgb, cv2.COLOR_RGB2GRAY)
    corners = cv2.goodFeaturesToTrack(
        gray,
        maxCorners=128,
        qualityLevel=0.015,
        minDistance=4,
        mask=mask,
        blockSize=3,
        useHarrisDetector=False,
    )
    if corners is None:
        corners = np.empty((0, 2), dtype=float)
    else:
        corners = cv2.cornerSubPix(
            gray,
            corners,
            (3, 3),
            (-1, -1),
            (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 20, 0.01),
        ).reshape(-1, 2)
        corners = np.array(
            [p for p in corners if x0 <= p[0] < x1 and y0 <= p[1] < y1]
        ).reshape(-1, 2)
    rows = []
    for index, point in enumerate(points):
        line = matrix @ np.r_[point, 1.0]
        norm = np.linalg.norm(line[:2])
        if norm < 1e-12:
            raise ValueError("Source pixel is at the epipole; use a different view")
        line /= norm
        distances = np.abs(corners @ line[:2] + line[2])
        candidates = [
            {"pixel": corners[i].tolist(), "line_distance_px": float(distances[i])}
            for i in np.argsort(distances)
            if distances[i] <= 3
        ][:8]
        row = {
            "first_pixel": point.tolist(),
            "target_line_abc": line.tolist(),
            "target_image_segment": clipped_line(
                line, second_rgb.shape[1], second_rgb.shape[0]
            ),
            "corner_candidates": candidates,
        }
        if selected is not None:
            row["selected_target_line_distance_px"] = float(
                abs(selected[index] @ line[:2] + line[2])
            )
        rows.append(row)
    return {
        "status": "diagnostic",
        "guides": rows,
        "official_success": None,
        "scope": (
            "An epipolar line constrains possible correspondence but does not identify "
            "the object, exact feature or depth. Corners are unverified proposals. "
            "Inspect current original RGB and explicitly mark a matching feature; "
            "the original triangulation/reprojection checks remain required."
        ),
    }
