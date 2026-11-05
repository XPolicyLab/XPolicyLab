"""Pure initial-RGB/public-template scheduling; not a grasp/success predictor.

Keep target assignment and every ANR4 transfer unchanged. Put one-leg routes
before staged routes and same-half routes before cross-midline routes. Remaining
risk is a smooth geometric heuristic, not fitted from episode scores or IDs.
"""
import cv2
import numpy as np


def template_risk(template):
    """Centroid support/concavity of the unchanged ANR4 template-origin grasp.

    Raster coordinates are static mesh coordinates in metres. The 20 mm centre
    window is only a scheduling prior; neither grasp point nor yaw is changed.
    Empty/unknown shape gets maximum risk, not rejection of a reachable object.
    """
    mask=np.zeros((128,128),np.uint8)
    for polygon in sorted(template.get("polygons",[]),key=lambda p:p["hole"]):
        points=np.rint(np.asarray(polygon["xy"])*1000+64).astype(np.int32)
        cv2.fillPoly(mask,[points],0 if polygon["hole"] else 1)
    points=cv2.findNonZero(mask)
    if points is None: return 1.
    hull=np.zeros_like(mask)
    cv2.fillConvexPoly(hull,cv2.convexHull(points),1)
    solidity=float(mask.sum())/max(int(hull.sum()),1)
    support=float(mask[54:75,54:75].mean())
    return float(np.clip(.6*(1-support)+.4*(1-solidity),0,1))


def route_features(route,templates):
    legs=route["legs"]
    shape=template_risk(templates.get(str(route["digit"]["digit"]),{}))
    crossing=any(float(l["xy"][0])*float(l["goal"][0])<0 for l in legs)
    length=sum(float(np.linalg.norm(np.asarray(l["goal"])-l["xy"])) for l in legs)
    radial=0.
    turn=0.
    for leg in legs:
        base=np.array([-.3 if leg["side"]=="left" else .3,-.45])
        radial=max(radial,*(float(np.linalg.norm(np.asarray(p)-base)) for p in (leg["xy"],leg["goal"])))
        turn+=abs(float(np.arctan2(np.sin(leg["yaw"]),np.cos(leg["yaw"]))))
    # Radial distance is a coarse reach-margin proxy, NOT a joint/collision
    # certificate. Unknown grasp confidence never causes a planning abstention.
    confidence=float(np.clip(route["digit"].get("confidence",1.),0,1))
    risk=.30*shape+.30*radial/.5+.20*length/.5+.10*turn/np.pi+.10*(1-confidence)
    return dict(staged=len(legs)>1,cross_midline=bool(crossing),template_risk=shape,
                route_length_m=length,max_radial_reach_m=radial,turn_radians=turn,
                detection_confidence=confidence,geometric_risk=float(risk))


def order_transfers(routes,templates):
    decorated=[]
    for route in routes:
        features=route_features(route,templates)
        decorated.append(dict(route,schedule_features=features))
    # No hard-coded digit priorities. target_index is only a stable final tie
    # break and remains the original descending assignment, never a new target.
    return sorted(decorated,key=lambda r:(r["schedule_features"]["staged"],
        r["schedule_features"]["cross_midline"],round(r["schedule_features"]["geometric_risk"],9),r["target_index"]))
