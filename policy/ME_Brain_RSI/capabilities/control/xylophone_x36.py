"""X36: X35 plus grasp-side preference and a binding that can correct the grasp.

Flipped grasps (head on the EE +z side) left the mallet head-up in two X35
episodes and it was dropped or never reached the band; unflipped grasps kept
the head down in every observed case, so the ranking now penalises flipped
candidates before cost. The head search radius for probe observations grows
from 18 to 60 px and the anchor may differ from the planar prior by up to the
shaft length, so a pinch that landed away from the intended 5 cm point is
measured instead of assumed. A relaxed point fit precedes the nominal
fallback, and lateral hover moves are slowed to limit axial sliding of the
frictionless shaft.
"""
import numpy as np
from .xylophone_x35 import XylophonePolicy as X35,JOINT_RATE_PER_STEP,NOMINAL_SAG
from .xylophone_anchor import fit_moving_anchor

HEAD_SEARCH_PX=60.
ANCHOR_PRIOR_TOLERANCE=.09
RELAXED_MAX_SIGMA=.03
UNFLIPPED_YAW_EXTRA_DEG=20.
HOVER_STEP=.009


def rank_tuple(audit,flipped,alternative_index):
    margin=audit.get('min_joint_margin_rad') or 0.
    travel=audit.get('joint_travel_rad')
    cost=audit['nominal_steps'] if travel is None else max(audit['nominal_steps'],travel/JOINT_RATE_PER_STEP)
    return (0 if margin>=.30 else 1,1 if flipped else 0,cost,alternative_index)


class XylophonePolicy(X35):
    HEAD_SEARCH_PX=HEAD_SEARCH_PX

    def reset(self):
        super().reset()
        self.program_notes.update(candidate='X36_unflipped_grasp_wide_head_search')

    def _rank_grasp(self,audit,alternative_index):
        # Legacy index = yaw_variant*2 + probe_variant; yaw variant 1 flips the gripper.
        return rank_tuple(audit,alternative_index>=2,alternative_index)

    def _rank_compact(self,audit,yaw_extra,alternative_index):
        flipped=abs(((yaw_extra+180.)%360.)-180.)>UNFLIPPED_YAW_EXTRA_DEG
        return rank_tuple(audit,flipped,alternative_index)

    def _translation_bound(self,w,current,anchor):
        move=super()._translation_bound(w,current,anchor)
        if w.name.startswith('xy_key_') and w.name.endswith('_hover') and w.name!='xy_key_0_hover':
            move=min(move,HOVER_STEP)
        return move

    def _bind(self):
        visible=[s for s in self.samples if s[1] is not None]
        ees=[e for e,p in visible];pixels=[p for e,p in visible]
        point=fit_moving_anchor(self.perception.camera,ees,pixels,stable=True)
        source='measured_rgb_point_fit'
        if point['state']!='consistent' and len(visible)>=3:
            relaxed=fit_moving_anchor(self.perception.camera,ees,pixels,max_sigma_m=RELAXED_MAX_SIGMA,stable=False)
            self.program_notes['x36_relaxed_point_binding']={k:v for k,v in relaxed.items() if k!='ee_anchor'}
            if relaxed['state']=='consistent':
                point={**relaxed,'strict_state':point['state'],'strict_reason':point.get('reason')};source='relaxed_rgb_point_fit'
        measured={k:v for k,v in point.items() if k!='ee_anchor'}
        self.program_notes['x34_all_point_binding']=measured
        offset=None if point['state']!='consistent' else float(np.linalg.norm(point['ee_anchor']-self.nominal_anchor))
        if offset is not None and offset<=ANCHOR_PRIOR_TOLERANCE:
            self.anchor=np.asarray(point['ee_anchor'],float).copy()
            self.binding={**measured,'source':source,'prior_offset_m':offset}
            self.reason='xy_x34_bound_measured_pending_keyboard'
        else:
            self.anchor=np.asarray(self.nominal_anchor,float).copy();self.anchor[0]+=NOMINAL_SAG
            self.binding={'state':'nominal_fallback','measured_state':point['state'],'measured_reason':point.get('reason'),
                          'prior_offset_m':offset,'source':'grasp_geometry_prior','sag_compensation_m':NOMINAL_SAG}
            self.reason='xy_x34_bound_nominal_pending_keyboard'
        self.validation_observation_pose=np.asarray(self.state_now[self.side+'_ee_pose']).copy()
        self.stage='keyboard'
