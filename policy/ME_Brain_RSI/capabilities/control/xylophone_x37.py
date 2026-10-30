"""X37: X36 plus stall handling that never parks and re-plans unreachable keys.

A same-key retreat waypoint stalled for 86 actions in one eval15 episode because
it was not a movable phase; a hit that stopped after most of its descent was
retried although something was already in contact; and the farthest key of one
layout was out of reach for the chosen stroke orientation while the vertical
functional route from the actual joint state could still have solved it.
"""
import numpy as np
from .xylophone_x36 import XylophonePolicy as X36
from .xylophone_feedback import ExecutionWindow

CONTACT_DESCENT=.02
MAX_KEY_REPLANS=2


class XylophonePolicy(X36):
    MOVABLE_PREFIXES=X36.MOVABLE_PREFIXES+('xy_recover_',)

    def reset(self):
        super().reset()
        self.key_replans=0
        self.program_notes.update(candidate='X37_contact_lift_replan_unreachable_keys')

    def _recover_stalled_hit(self,w,current,anchor,action,evidence):
        guard=evidence.get('contact_guard',{}) if isinstance(evidence,dict) else {}
        if not guard.get('qualified') and (guard.get('descent_m') or 0.)>=CONTACT_DESCENT:
            # Most of the stroke was executed before the plateau: the tool rests
            # on something. Repeating the same descent cannot change the check.
            evidence={**evidence,'contact_guard':{**guard,'qualified':True,'x37_descent_contact':True}}
        super()._recover_stalled_hit(w,current,anchor,action,evidence)

    def _recover_motion(self,w,current,action,evidence):
        if w.name.startswith('xy_key_') and self.key_replans<MAX_KEY_REPLANS and self.keys is not None:
            key=int(w.name.split('_')[2]);remaining=list(range(key,len(self.keys)))
            anchor=self.anchor if self.anchor is not None else self.nominal_anchor
            route=self._endpoint_checked_strokes(w.side,anchor,(),key_indices=remaining) if anchor is not None else None
            self.motion_recoveries.append(dict(step=self.step,phase=w.name,evidence=evidence,keys=remaining,
                task_success=False,action='replan_remaining_keys_vertical' if route else 'replan_rejected'))
            if route is not None:
                self.key_replans+=1
                action[w.side+'_ee_pose']=np.asarray(current).copy()
                self._set_plan(route,'done');self.reason='xy_x37_replanned_remaining_keys';self.execution=ExecutionWindow();return
        super()._recover_motion(w,current,action,evidence)
