"""RGB-screened support placement before left-arm re-observation/regrasp.

Independent support-and-regrasp implementation, conceptually informed by the
reviewed PhysicalRSI PourBallsRGB flow (CONTROL_LOGIC_REVIEW.txt). No external
source code, regression coefficients or instance positions are copied. RGB
clearance is only against detected bottles, NOT a complete collision proof.
"""
import numpy as np


def staging_candidates(source, detections, excluded=()):
    radius=float(max(source.size)/2)
    if not np.isfinite(np.r_[source.xy,radius]).all() or not .015<radius<.055:
        return
    others=[v for k,v in detections.items() if k.startswith('bottle') and v is not source]
    # Central overlap of the two public robot workspaces, on the tabletop.
    candidates=[np.array([x,y]) for x in (-.04,0.,.04) for y in (-.10,-.02,.06)]
    candidates.sort(key=lambda xy:float(np.linalg.norm(xy-source.xy)))
    for xy in candidates:
        if np.linalg.norm(xy-source.xy)<.09:continue
        if any(np.linalg.norm(xy-old)<.06 for old in excluded):continue
        # A bottle's plan-view half diagonal also covers lying detections.
        if any(np.linalg.norm(xy-v.xy) < radius+np.linalg.norm(v.size)/2+.04 for v in others):
            continue
        if any(np.linalg.norm(xy-np.array([x,-.45]))>.55 for x in (-.3,.3)):
            continue
        yield xy
    return


def staging_site(source,detections,excluded=()):
    return next(staging_candidates(source,detections,excluded),None)
