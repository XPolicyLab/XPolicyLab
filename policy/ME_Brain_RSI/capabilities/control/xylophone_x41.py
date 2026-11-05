"""X41: X40 plus a vertical correction learned from the first contact plateau.

A hit that plateaus after most of its descent means the real head is lower than
the model by at least the remaining modelled distance. Because touching the
cylindrical keys pushes the head sideways, the remaining hover/hit/lift targets
are raised by that distance plus a 4 mm margin (cumulative cap 25 mm), so later
keys are struck from inside the beat band without contact. The X37 contact lift
still ends the current key.
"""
import numpy as np
from .xylophone_x40 import XylophonePolicy as X40

CONTACT_MARGIN=.004
MAX_CONTACT_RAISE=.025
MIN_DESCENT_FOR_RAISE=.02


class XylophonePolicy(X40):
    def reset(self):
        super().reset()
        self.contact_raise=0.;self.contact_raise_events=[]
        self.program_notes.update(candidate='X41_contact_height_correction',contact_raise_events=self.contact_raise_events)

    def _recover_stalled_hit(self,w,current,anchor,action,evidence):
        guard=evidence.get('contact_guard',{}) if isinstance(evidence,dict) else {}
        descent=guard.get('descent_m') or 0.;error=guard.get('head_error_m')
        if descent>=MIN_DESCENT_FOR_RAISE and error is not None and len(error)==3:
            dz=float(error[2])
            raise_by=min(max(dz,0.)+CONTACT_MARGIN,MAX_CONTACT_RAISE-self.contact_raise)
            if raise_by>0:
                for remaining in self.plan[self.index+1:]:
                    if remaining.name.startswith('xy_key_'):remaining.pose[2]+=raise_by
                self.contact_raise+=raise_by
                self.contact_raise_events.append(dict(step=self.step,phase=w.name,modelled_above_target_m=dz,
                                                      raised_m=raise_by,cumulative_m=self.contact_raise))
        super()._recover_stalled_hit(w,current,anchor,action,evidence)
