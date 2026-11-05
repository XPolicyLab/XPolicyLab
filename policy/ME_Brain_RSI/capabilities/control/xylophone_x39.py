"""X39: stuck hovers are lowered before any re-plan; hexashell hits centred in the band.

X37's vertical re-plan replaced a working route on native49 when the first hover
froze 2 cm short at its reach limit, yet the same re-plan struck the last key on
eval15 layouts 1 and 17 where the hexashell orientation ran out of reach. X39
keeps it as the second step: a stalled key hover first retries the same key with
the head 2 cm lower (still >= 3.6 cm above the hit height, so the lift stays
valid), audited like every other waypoint; only then the X37/X34 recovery runs.
"""
import numpy as np
from . import xylophone_hexashell as hexmod
from .xylophone_hexashell import hx
from .xylophone_x38 import XylophonePolicy as X38
from .xylophone_feedback import ExecutionWindow
from .geometry import pose_matrix,matrix_quat

LOWERED_HOVER=.02
# Beat band is hit_z+[.010,.036]; the imported .023 gap put the head 5 mm below
# the top of the band, so an upward slide of the shaft missed silently.
CENTERED_GAP=.017


def tool_chain_centered(pose,rotation,ee_anchor,tcp_offset,targets):
    tcp=pose.copy();tcp[:3,3]+=pose[:3,:3]@tcp_offset
    local=ee_anchor-tcp_offset
    frames=hx.make_chain(tcp,rotation,local,targets,CENTERED_GAP,.04,0,None)
    result=[]
    for name,target in frames:
        ee=target.copy();ee[:3,3]-=ee[:3,:3]@tcp_offset
        if ":" in name:
            phase,index=name.split(":")
            name="xy_key_"+index+"_"+{"above":"hover","descend":"hit","rise":"lift"}[phase]
        else:name="xy_hexashell_"+name
        result.append((name,ee))
    return result


class XylophonePolicy(X38):
    def reset(self):
        super().reset()
        self.lowered_hovers=set()
        self.program_notes.update(candidate='X39_lowered_hover_centred_gap')

    def _choose_strokes(self,side,rotation,anchor,prefix=(),reserve=60):
        original=hexmod.tool_chain;hexmod.tool_chain=tool_chain_centered
        try:return super()._choose_strokes(side,rotation,anchor,prefix,reserve)
        finally:hexmod.tool_chain=original

    def _recover_motion(self,w,current,action,evidence):
        if (w.name.startswith('xy_key_') and w.name.endswith('_hover') and w.name not in self.lowered_hovers
                and self.index+1<len(self.plan) and self.plan[self.index+1].name==w.name[:-5]+'hit'):
            self.lowered_hovers.add(w.name)
            anchor=self.anchor if self.anchor is not None else self.nominal_anchor
            frame=pose_matrix(w.pose);frame[2,3]-=LOWERED_HOVER
            lowered=np.r_[frame[:3,3],matrix_quat(frame[:3,:3])]
            w_low=type(w)(w.side,lowered,w.gripper,w.name,w.dwell,w.anchor_offset)
            route=[w_low]+list(self.plan[self.index+1:])
            accepted=anchor is not None and self._audit(w.side,route,anchor,reserve=60,local_only=True)
            self.motion_recoveries.append(dict(step=self.step,phase=w.name,evidence=evidence,task_success=False,
                action='lowered_hover' if accepted else 'lowered_hover_rejected',
                audit=dict(self.program_notes.get('path_audit',{}))))
            if accepted:
                action[w.side+'_ee_pose']=np.asarray(current).copy()
                self.plan=self.plan[:self.index]+route;self.age=self.dwell=0;self.execution=ExecutionWindow()
                self.reason='xy_x39_lowered_hover_retry';return
        super()._recover_motion(w,current,action,evidence)
