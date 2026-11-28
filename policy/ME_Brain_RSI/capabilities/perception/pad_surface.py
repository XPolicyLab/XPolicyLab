"""Measured pad support; exterior-connected background is outside the pad."""
import cv2
import numpy as np


def occupancy(rgb, circle_mask):
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    disk = circle_mask > 0
    pale = (hsv[..., 1] < 100) & (hsv[..., 2] > 130)
    # Hough supplies localization only. The measured pale contour supplies support.
    roi = cv2.dilate(disk.astype('uint8'), np.ones((11, 11), np.uint8)) > 0
    n, labels, stats, _ = cv2.connectedComponentsWithStats((pale & roi).astype('uint8'))
    ids = list(range(1, n))
    if not ids:
        return dict(status='unknown', reason='no_visible_support'), {}
    chosen = max(ids, key=lambda i: np.count_nonzero((labels == i) & disk))
    visible = (labels == chosen).astype('uint8')
    contours, _ = cv2.findContours(visible, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contour = max(contours, key=cv2.contourArea)
    support = np.zeros_like(visible)
    # Convex mat geometry fills holes/occluder notches, retaining missed foreground.
    cv2.drawContours(support, [cv2.convexHull(contour)], -1, 1, -1)
    support = support > 0
    coverage = float(np.count_nonzero(support & disk) / max(1, disk.sum()))
    # Every saturated hue contradicts pale support, including yellow digits.
    # registered's hue exclusion silently accepted missed yellow foreground.
    contrast = (hsv[..., 1] > 100) | (hsv[..., 2] < 75)
    _, background_labels = cv2.connectedComponents((~pale & roi).astype('uint8'))
    outside_ids = np.unique(background_labels[roi & ~disk & ~pale])
    outside_ids = outside_ids[outside_ids != 0]
    exterior_connected = np.isin(background_labels, outside_ids)
    # Hough localizes the mat; it must not turn real exterior background into
    # on-pad foreground. Interior contrast stays protected even if exterior-connected.
    exterior_background = disk & contrast & ~support & exterior_connected
    protected = disk & contrast & ~exterior_background
    # registered: the observed smooth mat boundary must not inherit hull corners.
    # Retain every interior contrast pixel and isolated exterior component.
    fit_report = dict(qualified=False)
    if len(contour) >= 5:
        ellipse = cv2.fitEllipse(contour)
        center, axes, angle_deg = ellipse
        angle = np.deg2rad(angle_deg)
        def elliptic_radius(x, y):
            dx, dy = x-center[0], y-center[1]
            u = (np.cos(angle)*dx+np.sin(angle)*dy)/(axes[0]/2)
            v = (-np.sin(angle)*dx+np.cos(angle)*dy)/(axes[1]/2)
            return np.hypot(u, v), np.arctan2(v, u)
        points = contour[:, 0, :]
        radius, theta = elliptic_radius(points[:, 0], points[:, 1])
        residual = np.abs(radius-1)*min(axes)/2
        sectors = np.unique(np.floor((theta+np.pi)/(2*np.pi)*12).astype(int)%12)
        y, x = np.indices(disk.shape)
        observed_radius, _ = elliptic_radius(x, y)
        fitted_support = observed_radius <= 1
        fit_coverage = float(np.count_nonzero(fitted_support & disk)/max(1,disk.sum()))
        qualified = bool(len(sectors)==12 and min(axes)/max(axes)>=.85
                         and np.quantile(residual,.9)<=2.5 and fit_coverage>=.85)
        fit_report = dict(qualified=qualified, center=list(center), axes=list(axes),
                          angle_degrees=angle_deg, observed_sectors=len(sectors),
                          residual_p90_pixels=float(np.quantile(residual,.9)),
                          visible_support_fraction=fit_coverage)
        if qualified:
            measured_exterior = disk & contrast & ~fitted_support & exterior_connected
            candidate = disk & contrast & ~measured_exterior
            fit_report['protected_contrast_pixels'] = int(candidate.sum())
            # Existing occupancy branch remains; this is separately measured support.
            if int(candidate.sum()) < int(protected.sum()):
                protected = candidate
                exterior_background = measured_exterior
                coverage = fit_coverage
                support = fitted_support
    count = int(protected.sum())
    report = dict(status='empty' if count <= 60 and coverage >= .85 else 'unknown',
                  reason='visible_support' if count <= 60 and coverage >= .85 else 'foreground_or_incomplete_support',
                  raw_contrast_pixels=int((disk & contrast).sum()),
                  protected_contrast_pixels=count, exterior_background_pixels=int(exterior_background.sum()),
                  visible_support_fraction=coverage, threshold=60, measured_ellipse=fit_report)
    return report, dict(support=support, exterior_background=exterior_background, protected=protected)
