"""Upright open-finger descent using released X5 geometry, not scene truth."""
import numpy as np

def upright_descent_clearance(height,radius):
    # Public ARX.usd link6-frame collision bounds: closed finger inner faces
    # at Y +/- .00040; each published prismatic stroke opens .044 m.
    # Palm ends at X=.084, while the nominal grasp TCP is X=.145.
    # Bottle public-size fit supplies its maximum radius and half-height.
    if not np.isfinite([height,radius]).all() or not .075<height<.15 or radius<=0:
        return None
    finger_gap=.0444-radius
    palm_gap=.145-.084-radius
    if min(finger_gap,palm_gap)<.003:return None
    return dict(hover_height=2*height+.045,
                open_finger_side_clearance=finger_gap,palm_clearance=palm_gap,
                source='public ARX.usd collision bounds and .044 finger stroke',
                warning='nominal upright body clearance, not world collision validation')
