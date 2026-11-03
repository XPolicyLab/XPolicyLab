"""Bounded key insertion using visible bow geometry and observed motion.

A sustained motion stop is a sensorless contact hypothesis, not proof of seating:
a concealed jam can also stop motion. No lock internal dimensions are supplied.
"""
import numpy as np
from scipy.spatial.transform import Rotation
from . import GeometryUnavailable
from .key_outline import observed_key_tip


def observed_bow_boundary(outline):
    points=np.asarray(outline,float)
    observed_key_tip(points)  # Validate the observed outline and shaft extent.
    low,high=points[:,2].min(),points[:,2].max()
    if not low<0<high:raise GeometryUnavailable('Observed bow origin must lie inside the silhouette')
    levels=np.linspace(low,high,257)[1:-1]
    widths=[]
    for z in levels:
        xs=[]
        for a,b in zip(points,np.roll(points,-1,axis=0)):
            if (a[2]<=z<b[2]) or (b[2]<=z<a[2]):
                xs.append(a[0]+(b[0]-a[0])*(z-a[2])/(b[2]-a[2]))
        widths.append(max(xs)-min(xs) if len(xs)>=2 else 0.)
    widths=np.asarray(widths)
    middle=int(np.argmin(abs(levels)))
    width=float(np.interp(0.,levels,widths))
    if width<=1e-8:raise GeometryUnavailable('Observed bow width unresolved')
    # Follow only the broad region containing the observed bow origin. Teeth
    # farther along the shaft do not define the bow's insertion boundary.
    threshold=width/2
    index=middle
    while index>0 and widths[index]>=threshold:index-=1
    if index==0 or widths[index+1]<=widths[index]:
        raise GeometryUnavailable('No resolved bow-to-shaft transition')
    fraction=(threshold-widths[index])/(widths[index+1]-widths[index])
    boundary=float(levels[index]+fraction*(levels[index+1]-levels[index]))
    if boundary>=0 or boundary-low<4*(levels[1]-levels[0]):
        raise GeometryUnavailable('Visible shaft below bow unresolved')
    # Clip the full polygon, preserving interpolated points at the boundary.
    bow=[]
    for a,b in zip(points,np.roll(points,-1,axis=0)):
        inside_a,inside_b=a[2]>=boundary,b[2]>=boundary
        if inside_a:bow.append(a)
        if inside_a!=inside_b:bow.append(a+(b-a)*(boundary-a[2])/(b[2]-a[2]))
    return np.asarray(bow),{'boundary_local_z':boundary,'bow_width_m':width,
                           'source':'current_visible_bow_to_shaft_transition',
                           'half_width_boundary_assumed':True}


def observed_insertion_offset(outline,rotation,*,clearance=.003):
    """Lowest permitted root before the inferred bow reaches the aperture plane.

    The clearance is a control margin; this is a travel bound, not a seat depth.
    """
    bow,evidence=observed_bow_boundary(outline)
    rotation=np.asarray(rotation,float)
    if (rotation.shape!=(3,3) or not np.isfinite(rotation).all()
            or not np.allclose(rotation.T@rotation,np.eye(3),atol=1e-5)
            or not np.isclose(np.linalg.det(rotation),1.,atol=1e-5)
            or not np.isfinite(clearance) or clearance<0):
        raise GeometryUnavailable('Rigid insertion orientation and nonnegative control margin required')
    return float(-(bow@rotation.T)[:,2].min()+clearance)


def observed_approach_offset(outline,rotation,*,clearance=.02):
    """Keep the visible key above the aperture by a control clearance."""
    points=np.asarray(outline,float)
    observed_key_tip(points)
    rotation=np.asarray(rotation,float)
    if (rotation.shape!=(3,3) or not np.isfinite(rotation).all()
            or not np.allclose(rotation.T@rotation,np.eye(3),atol=1e-5)
            or not np.isclose(np.linalg.det(rotation),1.,atol=1e-5)
            or not np.isfinite(clearance) or clearance<0):
        raise GeometryUnavailable('Rigid approach orientation and nonnegative margin required')
    return float(-(points@rotation.T)[:,2].min()+clearance)


def insertion_feedback(memory,current_key,target_key,hole,outline):
    """Evidence for a sustained stop after observed insertion and attempted descent.

    Uses the existing millimeter-scale controller tolerances. Does not accept
    reaching the geometric travel bound, a stationary pre-insertion pose, or
    obvious lateral/angular misalignment as seating evidence.
    """
    current=np.asarray(current_key,float);target=np.asarray(target_key,float);hole=np.asarray(hole,float)
    if any(t.shape!=(4,4) or not np.isfinite(t).all() for t in (current,target,hole)):
        raise GeometryUnavailable('Finite measured and commanded key transforms required')
    if any(not np.allclose(t[3],[0,0,0,1]) or not np.allclose(t[:3,:3].T@t[:3,:3],np.eye(3),atol=1e-5)
           or not np.isclose(np.linalg.det(t[:3,:3]),1.,atol=1e-5) for t in (current,target,hole)):
        raise GeometryUnavailable('Proper rigid key transforms required')
    tip=observed_key_tip(outline)
    position=current[:3,3]
    history=memory.setdefault('positions',[])
    history.append(position.tolist());del history[:-8]
    memory['highest_root_z']=max(float(position[2]),memory.get('highest_root_z',float(position[2])))
    inserted=float(hole[2,3]-(current[:3,:3]@tip+position)[2])
    descent=float(memory['highest_root_z']-position[2])
    gap=float(position[2]-target[2,3])
    lateral=float(np.linalg.norm(position[:2]-target[:2,3]))
    angle=float(Rotation.from_matrix(target[:3,:3].T@current[:3,:3]).magnitude())
    window=np.asarray(history)
    stopped=bool(len(window)==8 and np.ptp(window[:,2])<.0005 and np.linalg.norm(np.ptp(window[:,:2],axis=0))<.001)
    ready=bool(stopped and inserted>.003 and descent>.006 and gap>.003 and lateral<.003 and angle<np.deg2rad(1))
    return {'sustained_stop_after_insertion':ready,'tip_below_aperture_m':inserted,
            'observed_descent_m':descent,'remaining_command_descent_m':gap,
            'lateral_error_m':lateral,'angle_error_rad':angle,'window_samples':len(history),
            'travel_bound_reached':bool(gap<=.002),
            'source':'observed_EE_motion_and_episode_key_outline',
            'seating_proven':False,'hidden_jam_not_excluded':True}
