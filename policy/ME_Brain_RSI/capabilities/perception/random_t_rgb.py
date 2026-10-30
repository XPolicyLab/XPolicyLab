"""Color-tolerant T candidates from RGB only; no layout identities or truth.

The target can be tinted by illumination. Stable extremal regions in image
channels propose silhouettes; the same directed full-template fit verifies them.
"""
import cv2
import numpy as np


def source_mask(image):
    red, green, blue = image.astype(float).transpose(2, 0, 1)
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    # A warm-lit neutral target can satisfy the old red difference test.
    # Retain genuinely chromatic red, not the brown printed target.
    old_red = ((red > 130) & (red-green > 90) & (red-blue > 90)
               & ((hsv[:, :, 0] <= 3) | (hsv[:, :, 0] >= 168))
               & (hsv[:, :, 1] > 170))
    # Green/yellow illumination raises G on the blue rigid T. Requiring
    # B > 2.1 G and S > 150 fragmented its visible silhouette. Require
    # blue dominance over BOTH channels instead; shape/scale/direction
    # and competing-object checks still apply to every proposed region.
    # A blue-lit gray target can also have B dominance. Require a bright
    # chromatic core as well; dark/uncertain regions must abstain.
    blue_t = ((blue > 195) & (blue-red > 45) & (blue-green > 45)
              & (blue > 1.35*np.maximum(red,green))
              & (hsv[:, :, 0] >= 90) & (hsv[:, :, 0] <= 165)
              & (hsv[:, :, 1] > 90))
    return old_red | blue_t


def target_region_masks(image, valid):
    """Bounded MSER candidates, filtered before expensive template fitting.

    Channel-relative segmentation needs neither fixed gray brightness nor
    neutral RGB. Full template/scale/direction checks remain mandatory.
    """
    detector = cv2.MSER_create(5, 280, 4200, 0.15, 0.2)
    channels = [image[:, :, i] for i in range(3)]
    channels.append(cv2.cvtColor(image, cv2.COLOR_RGB2GRAY))
    candidates = {}
    for channel in channels:
        regions, _ = detector.detectRegions(np.ascontiguousarray(channel))
        for region in regions:
            x, y, w, h = cv2.boundingRect(region)
            if not (24 < min(w, h) < 78 and 39 < max(w, h) < 96):
                continue
            points = region.reshape(-1, 2)
            if np.any(valid[points[:, 1], points[:, 0]] == 0):
                continue
            # Dedupe stable regions at adjacent thresholds/channels.
            key = (x//2, y//2, w//2, h//2, len(points)//50)
            if key not in candidates:
                mask = np.zeros(image.shape[:2], np.uint8)
                mask[points[:, 1], points[:, 0]] = 255
                contours,_ = cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
                if len(contours)!=1:
                    continue
                contour=contours[0]
                area=cv2.contourArea(contour)
                hull=cv2.contourArea(cv2.convexHull(contour))
                if not (432<area<4320 and .35<area/max(hull,1)<.94):
                    continue
                # Only cheap shape-ranked proposals receive the expensive fit.
                # No scene location or layout identity affects ranking.
                footprint=np.array([[-42,-34],[-42,34],[-22,34],[-22,10],
                                    [42,10],[42,-10],[-22,-10],[-22,-34]],np.float32)
                error=cv2.matchShapes(contour,footprint,cv2.CONTOURS_MATCH_I1,0)
                if error<.65:
                    candidates[key]=(error,mask)
    return [item[1] for item in sorted(candidates.values(),key=lambda x:x[0])[:64]]
