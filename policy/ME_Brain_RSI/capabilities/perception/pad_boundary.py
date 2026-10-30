"""Additional pad evidence from visible boundaries, without guessed centers."""
import cv2
import numpy as np


def qualify(top, proposals):
    gray = cv2.cvtColor(top, cv2.COLOR_RGB2GRAY).astype(np.float32)
    # Only the original color-qualified observations establish scale and row.
    peers = []
    for p in proposals:
        if p['pad'] and 37 <= p['radius'] <= 48:
            if not any(np.linalg.norm(np.subtract(p['center'], q['center'])) < 25 for q in peers):
                peers.append(p)
    audit = []
    theta = np.arange(72) * (2 * np.pi / 72)
    direction = np.column_stack((np.cos(theta), np.sin(theta)))
    for p in proposals:
        if p['pad']:
            continue
        center = np.asarray(p['center'])
        radius = p['radius']
        row_peers = [q for q in peers if abs(q['center'][1] - center[1]) < 10]
        measured_radius = float(np.median([q['radius'] for q in row_peers])) if row_peers else None
        scale_ok = bool(len(row_peers) >= 2 and 37 <= radius <= 48
                        and abs(radius / measured_radius - 1) < .10)
        semantic = p.get('proposal') == 'SAM3_wooden_circular_pad'
        # Radial polarity distinguishes a light wooden pad from a dark disc.
        # Sample around the observed boundary; never generate another center.
        contrasts = []
        for delta in (-2., 0., 2.):
            values = []
            for offset in (-4., 4.):
                xy = center + (radius + delta + offset) * direction
                values.append(cv2.remap(gray, xy[:, 0].astype(np.float32).reshape(1, -1),
                                        xy[:, 1].astype(np.float32).reshape(1, -1),
                                        cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE).ravel())
            contrasts.append(values[0] - values[1])
        contrast = np.max(contrasts, axis=0)
        supported = contrast > 18
        sector_support = supported.reshape(12, 6).mean(axis=1)
        visible_fraction = float(supported.mean())
        sectors = int(np.count_nonzero(sector_support >= .5))
        boundary_ok = visible_fraction >= .70 and sectors >= 9
        accepted = bool(scale_ok and boundary_ok and
                        (semantic or p.get('proposal') == 'independent_outer_scale'
                         or radius >= 37))
        record = dict(center=p['center'], radius=radius, semantic=semantic,
                      observed_row_peers=len(row_peers), observed_radius=measured_radius,
                      scale_ok=scale_ok, boundary_fraction=visible_fraction,
                      supported_sectors=sectors, median_contrast=float(np.median(contrast)),
                      accepted=accepted)
        audit.append(record)
        if accepted:
            p['pad'] = True
            p['qualification'] = 'visible_radial_boundary_and_observed_peer_geometry'
            p['boundary_evidence'] = record
    return audit
