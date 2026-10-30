"""ANS17: ANS16 plus angle-only repair for a digit spun on its pad, and quieter hold carry-over.

The across-height pinch of the 2 once spun the released digit by ~90 deg
(ANS9 layout 20). ANS7's repair gate rejects yaw errors above 0.80 rad, so such
a spin was never repaired although the digit sits centred on its pad. An
angle-only repair (centre error < 15 mm) now accepts yaw errors up to 1.75 rad.
The ANS13 carry-over is only applied once two wrist observations exist.
"""
import numpy as np
from .ans7_policy import Policy as ANS7
from .ans16_policy import Policy as ANS16

SPIN_REPAIR_ANGLE=1.75
SPIN_REPAIR_CENTRE=.015


class Policy(ANS16):
    def _evidence(self,obs,context,observations):
        result=super()._evidence(obs,context,observations)
        if result.get('carried_over') and len(result['wrist_observations'])<2:
            # One frame is never enough to decide; keep the raw status until the second.
            result['local_status']='unknown';result.pop('carried_over',None)
            if self.grasp_checks:self.grasp_checks[-1].update(status='unknown',carried_over=False)
            if self.carry_overs:self.carry_overs.pop()
        return result

    def _repair_plan(self,obs,context,found):
        if found is None:return None
        error=float(np.linalg.norm(np.asarray(found['xy'])-context['target']));angle=abs(float(found['yaw']))
        if error<SPIN_REPAIR_CENTRE and .80<angle<=SPIN_REPAIR_ANGLE and found['confidence']>=.70:
            # Spun on the pad: ANS7's 0.80 rad gate would refuse; rebuild its leg with the measured yaw.
            identity=context['digit']
            if identity in self.repaired or '_stage' in context['label'] or '_repair' in context['label']:return None
            side=context['side'];xy=np.asarray(found['xy']);goal=np.asarray(context['target'])
            if (side=='left' and max(xy[0],goal[0])>.06) or (side=='right' and min(xy[0],goal[0])<-.06):return None
            saved_plan,saved_transfers=self.plan,self.transfers;self.plan=[];self.transfers=[]
            try:
                self.transfer(side,xy,goal,float(found['yaw']),context['height'],str(identity)+'_repair')
                proposed=self.plan;transfer=self.transfers[0]
            finally:self.plan,self.transfers=saved_plan,saved_transfers
            if self.step+self._remaining_cost(obs,proposed)>=980:return None
            if not self._checked_path(obs,side,[w['pose'] for w in proposed]):return None
            for w in proposed:w['checked_slow']=True
            return dict(plan=proposed,transfer=transfer,kind='spin_angle_only')
        return super()._repair_plan(obs,context,found)

    def diagnostics(self):
        result=super().diagnostics()
        result.update(candidate='ANS17',spin_repair=dict(angle_max_rad=SPIN_REPAIR_ANGLE,centre_max_m=SPIN_REPAIR_CENTRE))
        return result
