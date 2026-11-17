"""Standard-table RGB digits and circular pads; no scene IDs or hidden poses.

Adapt the press_by_number pattern of colour segmentation, connected components,
template ranking and ambiguity rejection to the public rigid 0..9 silhouettes.
"""
import cv2
import numpy as np
from .number_perception import NumberPerception


class StandardNumberPerception(NumberPerception):
    def __init__(self, calibration):
        super().__init__(calibration)
        self.last_diagnostics = {}

    def digits_from_rgb(self, rgb):
        image = self.rectify(rgb, .014)
        r, g, b = image.astype(np.float32).transpose(2, 0, 1)
        # Standard objects are saturated blue. Ratios tolerate exposure changes;
        # acceptance still requires a complete uniquely-labelled shape group.
        masks = [('blue', (b > 55) & (b > 1.35*r) & (b > 1.20*g)),
                 ('yellow', (r > 100) & (g > 100) & (r > 1.35*b) & (g > 1.35*b) & (g > .80*r)),
                 ('green', (g > 55) & (g > 1.30*r) & (g > 1.20*b)),
                 ('red', (r > 70) & (r > 1.5*g) & (r > 1.5*b)),
                 ('purple', (r > 55) & (b > 55) & (r > 1.3*g) & (b > 1.3*g)),
                 ('neutral', (np.minimum(np.minimum(r,g),b) > 140) & (np.maximum(np.maximum(r,g),b)-np.minimum(np.minimum(r,g),b) < 42))]
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB).astype(np.float32)
        background = np.median(lab.reshape(-1, 3), axis=0)
        # Orange digits and the brown table share hue. Separate their red-green
        # chroma relative to the dominant table, not by absolute brightness.
        masks += [('relative_red', (lab[:,:,1] > background[1]+16) & (r > 80))]
        found = self.match_families(masks, minimum=.58)
        self.last_diagnostics = dict(colour_pixels={name:int(mask.sum()) for name,mask in masks}, digit_count=len(found))
        return sorted(found, key=lambda item: item['xy'][0])

    def pads_from_rgb(self, rgb, count):
        image = self.rectify(rgb, .005)
        hue, saturation, value = cv2.cvtColor(image, cv2.COLOR_RGB2HSV).transpose(2, 0, 1)
        r, g, b = image.astype(np.float32).transpose(2, 0, 1)
        # Pale wood circles versus a dark warm wooden tabletop. Same segmentation
        # pattern as press_by_number cards, followed by metric circle geometry.
        raw = (hue > 10) & (hue < 48) & (saturation > 8) & (saturation < 116) & (value > 145) & (g > .82*r)
        mask = cv2.morphologyEx(np.uint8(raw)*255, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        circles = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if not 2800 < area < 7100:
                continue
            hull = cv2.convexHull(contour)
            hull_area = cv2.contourArea(hull)
            perimeter = cv2.arcLength(hull, True)
            circularity = 4*np.pi*hull_area / max(perimeter*perimeter, 1)
            solidity = area/max(hull_area, 1)
            (cx, cy), radius = cv2.minEnclosingCircle(hull)
            if not (33 < radius < 47 and circularity > .88 and solidity > .91):
                continue
            xy = np.array([self.bounds[0], self.bounds[2]]) + np.array([cx, cy])/self.scale
            if not (-.46 < xy[0] < .46 and -.30 < xy[1] < .015):
                continue
            circles.append(dict(xy=xy.tolist(), radius_m=float(radius/self.scale),
                                confidence=float(circularity*solidity), family='rgb_circle'))
        circles.sort(key=lambda item: item['xy'][0])
        self.last_diagnostics.update(pad_candidates=circles, pad_foreground_pixels=int(raw.sum()))
        if count in (4, 5) and len(circles) != count:
            # Colour is a proposal, not the definition of a pad. A circular
            # brightness boundary can recover slight white-balance shifts.
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            hypotheses = cv2.HoughCircles(cv2.GaussianBlur(gray,(5,5),1.2),
                cv2.HOUGH_GRADIENT, dp=1, minDist=70, param1=70, param2=32,
                minRadius=34, maxRadius=44)
            fallback=[]
            if hypotheses is not None:
                yy,xx=np.indices(gray.shape)
                for cx,cy,radius in hypotheses[0]:
                    distance=np.hypot(xx-cx,yy-cy)
                    inside=gray[distance < radius-5]
                    outside=gray[(distance > radius+4)&(distance < radius+10)]
                    contrast=float(np.median(inside)-float(np.median(outside)))
                    if contrast < 22: continue
                    xy=np.array([self.bounds[0],self.bounds[2]])+np.array([cx,cy])/self.scale
                    if not (-.46<xy[0]<.46 and -.30<xy[1]<.015): continue
                    fallback.append(dict(xy=xy.tolist(),radius_m=float(radius/self.scale),
                        confidence=min(1.,contrast/60),family='rgb_circle_edge',contrast=contrast))
            self.last_diagnostics['circle_edge_candidates']=fallback
            if len(fallback)==count:
                circles=sorted(fallback,key=lambda item:item['xy'][0])
        if count not in (4, 5) or len(circles) != count:
            self.last_diagnostics['pad_rejection'] = 'count_mismatch'
            return []
        points = np.array([c['xy'] for c in circles])
        spacing = np.linalg.norm(np.diff(points, axis=0), axis=1)
        # Validate a row but never substitute ideal centres for RGB measurements.
        if np.ptp(points[:, 1]) > .015 or np.any((spacing < .072) | (spacing > .098)):
            self.last_diagnostics['pad_rejection'] = 'not_regular_row'
            return []
        return circles
