"""ANS10: ANS9 with a leaner step budget per transfer leg.

ANS9 dev5 layout 5 ran out of the 1050 steps after four of five digits with
three staging hand-offs (~140 steps per leg). Free motion without a held digit
moves faster, the descent/lowering use coarse steps until the final precision
band, and the steady release dwells are shortened. Grasp, carry and release
geometry are unchanged.
"""
import numpy as np
from .ans9_policy import Policy as ANS9
from .geometry import bounded_pose

FREE_STEP=.024                     # approach/retreat/home without a held digit
PRE_DESCENT_CLEARANCE=.026         # coarse descent stops 26 mm above the grasp height
FREE_PREFIXES=('approach_','retreat_','home_','putback_retreat_','putback_home_','empty_home_','unreachable_home_')


class Policy(ANS9):
    def transfer(self,side,xy,goal,yaw,height,label):
        super().transfer(side,xy,goal,yaw,height,label)
        own={w['name'].split('_')[0]:w for w in self.plan if w['name'].split('_',1)[1]==label}
        descend=own['descend'];i=next(k for k,w in enumerate(self.plan) if w is descend)
        pre=dict(descend,name='predescend_'+label,pose=descend['pose'].copy(),dwell=1)
        pre['pose'][2]+=PRE_DESCENT_CLEARANCE
        self.plan.insert(i,pre)
        own['lower'].update(translation_step=self.calibration['max_translation_step'],dwell=1)
        own['place']['dwell']=4;own['softopen']['dwell']=3;own['release']['dwell']=5
        own['clearance'].update(translation_step=self.calibration['precision_translation_step'],dwell=1)
        if 'home' in own:own['home']['dwell']=2

    def _act(self,obs):
        active=self.hold is None and self.pending is None and self.gate is None and self.auxiliary is None and self.planned and self.index<len(self.plan)
        w=self.plan[self.index] if active else None
        action=super()._act(obs)
        if w is not None and self.index<len(self.plan) and self.plan[self.index] is w and w['name'].startswith(FREE_PREFIXES) and w['gripper']>=.5 and not w.get('checked_slow') and not w.get('steady') and self.hold is None and self.auxiliary is None and self.pending is None:
            key=w['side']+'_ee_pose'
            action[key]=bounded_pose(np.asarray(obs['state'][key],float),w['pose'],FREE_STEP,self.calibration['max_rotation_step'])
        return action

    def diagnostics(self):
        result=super().diagnostics()
        result.update(candidate='ANS10',free_motion_step=FREE_STEP,pre_descent_clearance=PRE_DESCENT_CLEARANCE)
        return result
