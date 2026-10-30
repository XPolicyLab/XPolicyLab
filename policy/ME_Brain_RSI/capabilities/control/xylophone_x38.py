"""X38: X37 with online re-binding during the stroke sequence; hover speed restored.

The frictionless shaft slides in the pinch during lateral moves, so a binding
measured right after the lift drifts by the time of later keys (eval15 layouts
17 and 28 stalled from key 3 on). Every reached key hover/lift records the head
pixel in the head camera; when at least three recent observations fit a
consistent anchor that differs from the current one by 6-30 mm, the anchor and
the remaining waypoints are updated (at most twice per episode). The 9 mm hover
limit of X36 is dropped: on seed0 layout1 it was the only difference from a
successful X35 episode.
"""
import numpy as np
from .xylophone_x37 import XylophonePolicy as X37
from .xylophone_x35 import XylophonePolicy as X35
from .xylophone_policy import head_pixel
from .xylophone_anchor import fit_moving_anchor
from .geometry import pose_matrix,matrix_quat

REBIND_WINDOW=6
REBIND_MIN_VISIBLE=3
REBIND_MIN_SHIFT=.006
REBIND_MAX_SHIFT=.03
REBIND_MAX_SIGMA=.03
MAX_REBINDS=2


class XylophonePolicy(X37):
    def reset(self):
        super().reset()
        self.stroke_samples=[];self.rebind_events=[];self.rebinds=0
        self.program_notes.update(candidate='X38_online_rebinding',rebind_events=self.rebind_events)

    def _translation_bound(self,w,current,anchor):
        return X35._translation_bound(self,w,current,anchor)

    def _observe_stroke_head(self,w,current):
        rgb=self.current_vision.get('cam_head',{}).get('color')
        if rgb is None or self.anchor is None:return
        ee=pose_matrix(current)
        predicted=self.perception.camera.project([ee[:3,:3]@self.anchor+ee[:3,3]])[0]
        pixel=head_pixel(np.asarray(rgb),predicted,self.HEAD_SEARCH_PX)
        self.stroke_samples.append((ee,None if pixel is None else np.asarray(pixel,float)))
        self.stroke_samples=self.stroke_samples[-REBIND_WINDOW:]
        visible=[(e,p) for e,p in self.stroke_samples if p is not None]
        if len(visible)<REBIND_MIN_VISIBLE or self.rebinds>=MAX_REBINDS:return
        fit=fit_moving_anchor(self.perception.camera,[e for e,p in visible],[p for e,p in visible],
                              max_sigma_m=REBIND_MAX_SIGMA,stable=False)
        event=dict(step=self.step,phase=w.name,visible=len(visible),state=fit['state'],reason=fit.get('reason'))
        if fit['state']=='consistent':
            new=np.asarray(fit['ee_anchor'],float);shift=float(np.linalg.norm(new-self.anchor))
            event.update(shift_m=shift,one_pixel_sigma_m=fit.get('one_pixel_sigma_m'))
            if REBIND_MIN_SHIFT<=shift<=REBIND_MAX_SHIFT:
                old=self.anchor.copy()
                for remaining in self.plan[self.index+1:]:
                    frame=pose_matrix(remaining.pose)
                    target=frame[:3,:3]@old+frame[:3,3]
                    remaining.pose=np.r_[target-frame[:3,:3]@new,matrix_quat(frame[:3,:3])]
                    remaining.anchor_offset=new
                self.anchor=new;self.rebinds+=1;self.stroke_samples=[]
                self.binding={**self.binding,'rebound':self.rebinds,'rebound_shift_m':shift}
                event['applied']=True
        self.rebind_events.append(event)

    def _advance(self,w,current,how):
        if (how in ('reached','soft_arrival') and w.name.startswith('xy_key_')
                and w.name.endswith(('_hover','_lift')) and self.anchor is not None):
            self._observe_stroke_head(w,current)
        super()._advance(w,current,how)
