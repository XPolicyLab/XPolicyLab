"""ANS16: ANS15 with the digit-2 grasp chosen by reach (top arc, else across-height pinch) and fewer pointless repairs.

Observed digit-2 grasps: middle-stroke notch 0/3 held through the carry,
bottom bar 1/2 (ANS12 held, ANS14 dropped mid-carry), across-height pinch 3/4
(one 90 deg spin at release). The top-arc pinch keeps the heavy arc at the
fingers but needs 13 mm more reach, so it is used when the TCP stays within
REACH_SAFE and the across-height pinch otherwise. The 7 rotates 24-30 deg in
any centre grasp; repairing below 37 deg only re-rolls its position.
"""
import numpy as np
from . import ans9_policy as _ans9
from .ans8_policy import REACH_SAFE
from .ans10_policy import Policy as ANS10
from .ans15_policy import Policy as ANS15

GRASP_OFFSETS={2:((0.,.013),),7:((0.,.016),)}
REPAIR_ANGLE=.65


class Policy(ANS15):
    def _grasp_offset(self,side,digit,xy,goal,yaw):
        c,s=np.cos(yaw),np.sin(yaw);rot=np.array([[c,-s],[s,c]])
        for candidate in GRASP_OFFSETS.get(digit,()):
            offset=np.asarray(candidate,float)
            if max(self.reach(side,np.asarray(xy,float)+rot@offset),self.reach(side,np.asarray(goal,float)+offset))<=REACH_SAFE:return offset
        return None

    def transfer(self,side,xy,goal,yaw,height,label):
        digit=int(label.split('_')[0]);tilted=tuple(np.round(np.asarray(goal,float),3).tolist()) in self.tilted_goals
        if digit==2 and not tilted and self._grasp_offset(side,2,xy,goal,yaw) is None:
            _ans9.Y_PINCH_DIGITS.add(2)
            try:ANS10.transfer(self,side,xy,goal,yaw,height,label)
            finally:_ans9.Y_PINCH_DIGITS.discard(2)
            self.grasp_variants[-1]['fallback']='y_pinch_inner_pad';return
        super().transfer(side,xy,goal,yaw,height,label)

    def _repair_plan(self,obs,context,found):
        if found is not None and abs(float(found['yaw']))<REPAIR_ANGLE and float(np.linalg.norm(np.asarray(found['xy'])-context['target']))<.015:return None
        return super()._repair_plan(obs,context,found)

    def diagnostics(self):
        result=super().diagnostics()
        result.update(candidate='ANS16',grasp_offsets={str(k):[list(o) for o in v] for k,v in GRASP_OFFSETS.items()},repair_thresholds=dict(angle_rad=REPAIR_ANGLE,centre_m=.015))
        return result
