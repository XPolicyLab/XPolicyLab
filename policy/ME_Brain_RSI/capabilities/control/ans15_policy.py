"""ANS15: ANS14 plus a bounded model identity poll before planning.

eval15 (ANS10) lost two of four failed layouts to digit identities the local
template matcher got wrong at 0.70-0.85 confidence (8 read for 5, 8 and 5
swapped) while the model's initial review named the right digits from the
wrist views. The model now reports the digit at each detected position; a
disagreement is adopted only when the model is confident (>=0.85), the local
match is not (<0.88) and the corrected set stays unique. Only the assignment of
digits to pads changes; no hold, observation or put-back path is added.
"""
import numpy as np
from .ans14_policy import Policy as ANS14
from .identity_check import IdentityChecker

IDENTITY_MODEL_MIN=.85
IDENTITY_LOCAL_MAX=.88


class Policy(ANS14):
    def __init__(self,task,calibration,supervisor=None,identity_checker=None):
        super().__init__(task,calibration,supervisor)
        self.identity_checker=identity_checker or IdentityChecker()
        self.identity_overrides=[];self._latest_obs=None

    def _act(self,obs):
        self._latest_obs=obs
        return super()._act(obs)

    def _verify_identities(self,rgb,digits):
        obs=self._latest_obs;z=self.calibration['table_z']+.014
        entries=[]
        for i,d in enumerate(digits):
            uv=self.perception.camera.project([[d['xy'][0],d['xy'][1],z]])[0]
            entries.append(dict(index=i,head_pixel=[int(round(uv[0])),int(round(uv[1]))],local_digit=int(d['digit']),local_confidence=round(float(d.get('confidence',0.)),3)))
        frames=[('head',rgb)]
        if obs is not None:
            frames+=[(name,obs['vision'][name]['color']) for name in ('cam_left_wrist','cam_right_wrist') if name in obs.get('vision',{})]
        result=self.identity_checker.check(self.step,entries,frames)
        if result is None:return digits
        proposed=[int(d['digit']) for d in digits];changes=[]
        for item in result:
            i=item['index'];local=digits[i]
            if item['digit']!=int(local['digit']) and item['confidence']>=IDENTITY_MODEL_MIN and float(local.get('confidence',1.))<IDENTITY_LOCAL_MAX:
                proposed[i]=item['digit']
                changes.append(dict(step=self.step,index=i,xy=np.round(local['xy'],3).tolist(),local_digit=int(local['digit']),local_confidence=round(float(local.get('confidence',0.)),3),
                    model_digit=item['digit'],model_confidence=round(item['confidence'],3)))
        if not changes:return digits
        if len(set(proposed))!=len(proposed):
            self.identity_overrides.append(dict(step=self.step,rejected='duplicate_identities',changes=changes));return digits
        self.identity_overrides.extend(changes)
        return [dict(d,digit=proposed[i],identity_overridden=proposed[i]!=int(d['digit'])) for i,d in enumerate(digits)]

    def build(self,rgb):
        detect=self.perception.detect
        digits,targets=detect(rgb)
        if len(digits) in (4,5) and len(targets)==len(digits):
            digits=self._verify_identities(rgb,digits)
        self.perception.detect=lambda _rgb:(digits,targets)
        try:super().build(rgb)
        finally:self.perception.detect=detect

    def diagnostics(self):
        result=super().diagnostics()
        result.update(candidate='ANS15',identity_overrides=self.identity_overrides,identity_records=self.identity_checker.records,
            identity_thresholds=dict(model_min=IDENTITY_MODEL_MIN,local_max=IDENTITY_LOCAL_MAX))
        return result
