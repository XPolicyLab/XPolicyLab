"""X33 observation-quality and execution diagnostics; no task/result inputs."""
import itertools
import math
import numpy as np
from .xylophone_hexashell import pair_offset, confirm_cones


def qualified_cones(rows, nominal):
    """Check conditioning BEFORE interpreting pair scatter as contradiction.

    The widest relative-camera baseline is chosen geometrically, never by its
    fit residual. All original rows still participate in the final verdict.
    Unknown is not consistent; the separate all-point RGB fit must still pass.
    """
    if len(rows) < 4:
        return dict(state='unknown', reason='x33_insufficient_cones')
    a,b=max(itertools.combinations(rows,2), key=lambda p:
            np.linalg.norm(p[0]['local'][:3,3]-p[1]['local'][:3,3]))
    baseline=float(np.linalg.norm(a['local'][:3,3]-b['local'][:3,3]))
    if baseline < .025:
        return dict(state='unknown', reason='x33_weak_relative_baseline', baseline_m=baseline)
    try:
        base,_,_=pair_offset(a,b)
        sensitivity=0.
        for index,axis,sign in itertools.product((0,1),(0,1),(-1,1)):
            changed=[dict(a,fit=dict(a['fit'])),dict(b,fit=dict(b['fit']))]
            fit=changed[index]['fit'];pixel=fit['pixel'].copy();pixel[axis]+=sign
            direction=np.linalg.solve(fit['intrinsics'],[*pixel,1.])
            fit['axis']=direction/np.linalg.norm(direction)
            offset,_,_=pair_offset(*changed)
            sensitivity=max(sensitivity,float(np.linalg.norm(offset-base)))
    except (ValueError,np.linalg.LinAlgError):
        return dict(state='unknown',reason='x33_conditioning_not_measurable',baseline_m=baseline)
    if sensitivity > .006:
        return dict(state='unknown',reason='x33_uncertain_cones_not_slip',
                    one_pixel_sensitivity_m=sensitivity,baseline_m=baseline)
    result=confirm_cones(rows,nominal)
    return dict(result,quality_checked_before_contradiction=True,
                baseline_m=baseline,conditioning_sensitivity_m=sensitivity)


class ExecutionWindow:
    """Actual error trend, not nominal IK speed or evaluator contact state."""
    def __init__(self):
        self.phase=None;self.rows=[]

    def observe(self,phase,position_error,angle):
        if phase!=self.phase:self.phase=phase;self.rows=[]
        self.rows.append((float(position_error),float(angle)))
        self.rows=self.rows[-17:]
        if len(self.rows)<17:return dict(ready=False,stalled=False,progressing=False)
        old=np.median(self.rows[:4],axis=0);new=np.median(self.rows[-4:],axis=0)
        gains=old-new
        # Either still-needed component may converge; small pose oscillations
        # and rotational convergence must not be confused with static contact.
        progressing=bool((position_error>.003 and gains[0]>.001) or
                         (angle>.035 and gains[1]>.008))
        eta=[]
        for error,tolerance,gain in zip((position_error,angle),(.003,.035),gains):
            eta.append(0. if error<=tolerance else
                       (error-tolerance)/(gain/13.) if gain>0 else None)
        bounded_eta=None if any(x is None for x in eta) else int(math.ceil(max(eta)))
        return dict(ready=True,stalled=not progressing,progressing=progressing,
                    position_gain_m=float(gains[0]),angle_gain_rad=float(gains[1]),
                    estimated_remaining_actions=bounded_eta,
                    estimate_is_not_completion_guarantee=True)


def allow_more_actions(age,step,trend,reserve,phase):
    """Never reset elapsed age or expand the official 500-action episode."""
    if age<=85:return True
    eta=trend.get('estimated_remaining_actions')
    cap=160 if phase.startswith('xy_approach') else 125
    return bool(age<cap and trend.get('progressing') and eta is not None
                and step+eta+reserve<500)
