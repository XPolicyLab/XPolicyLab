"""Experimental bounded proprioceptive contact-height adaptation.
A pose plateau is only a contact hypothesis, never independent task-success evidence.
Native scoring and broader qualification remain mandatory before promotion.
"""
import json
from dataclasses import replace
import numpy as np
from .trace_press_model import TracePolicy
from .observed_press_model import PROBE_BUDGET_M,Model as CandidateModel
from rdj_master.tasks.press_by_number import cap as baseline


def plateau_candidate(history, actual, anchor):
    if len(history)<3:return None
    recent=np.asarray(history[-3:]);actual=np.asarray(actual);target=np.asarray(anchor.retry_down_pose)
    if np.ptp(recent[:,2])>.0015 or np.max(np.linalg.norm(recent[:,:2]-recent[-1,:2],axis=1))>.002:return None
    difference=float(actual[2]-target[2])
    if not .0015<difference<PROBE_BUDGET_M+.035:return None
    reference=target.copy();reference[2]=actual[2]
    if not baseline.pose_reached(actual,reference,position_tolerance_m=.005,rotation_tolerance_deg=5.):return None
    return float(actual[2])

class ContactPolicy(TracePolicy):
    def reset(self):
        self._contact_histories={};self._contact_calibrated=set();self._contact_initial={};self._trace_phases={};self._contact_endpoints={}
        return super().reset()

    def _observe(self,env_idx,observation):
        slot=self._slots.get(env_idx)
        if slot is not None:
            label=slot.session.current_label;phase=slot.session.phase
            histories=getattr(self,'_contact_histories',{});calibrated=getattr(self,'_contact_calibrated',set());key=(env_idx,label)
            if label is not None and phase in {'down','retry_down'} and key not in calibrated:
                arm=baseline.BUTTON_ARM_ASSIGNMENTS[label];actual=np.asarray(observation['state'][arm+'_ee_pose'],dtype=float)
                history=histories.setdefault(key,[]);history.append(actual[:3].tolist());anchor=slot.anchors[label]
                initial=self._contact_initial.setdefault(key,float(anchor.tip_pose[2]))
                height=plateau_candidate(history,actual,anchor)
                tip_height=None if height is None else height-baseline.PRESS_TOOL_TIP_OFFSET_M
                if height is not None and initial-PROBE_BUDGET_M<=tip_height<=initial+.035:
                    # Retain observed surface position; do not infer released height
                    # by adding an assumed native joint travel to this endpoint.
                    endpoint=list(anchor.retry_down_pose);endpoint[2]=height;endpoint=tuple(endpoint)
                    updated=replace(anchor,down_pose=endpoint,retry_down_pose=endpoint)
                    slot.anchors[label]=updated;slot.session._anchors[label]=updated
                    slot.session._down_target_pose=endpoint;calibrated.add(key)
                    endpoints=getattr(self,'_contact_endpoints',{});endpoints[key]=height;self._contact_endpoints=endpoints
                    with (self.output/'contact-hypotheses.jsonl').open('a') as f:
                        f.write(json.dumps({'env':env_idx,'label':label,'observed_surface_height':initial,
                            'observed_endpoint_ee_z':height,'actual_ee_pose':actual.tolist(),
                            'evidence':'bounded proprioceptive plateau; not button state or task success',
                            'promoted':False})+'\n')
            elif phase not in {'down','retry_down'}:
                histories.pop(key,None);calibrated.discard(key)
            self._contact_histories=histories;self._contact_calibrated=calibrated
        super()._observe(env_idx,observation)

class Model(CandidateModel):
    def __init__(self,model_cfg):
        from rdj_rgb_adapters.press_by_number import Model as IncumbentModel
        IncumbentModel.__init__(self,model_cfg);self._core=ContactPolicy(model_cfg)
