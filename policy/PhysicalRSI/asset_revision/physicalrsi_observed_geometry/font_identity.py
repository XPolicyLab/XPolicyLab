"""Scale-free digit identity evidence from generic fonts, never object meshes."""
import numpy as np
from .generic_digits import normalize, silhouette_bank


class FontDigitMatcher:
    def __init__(self):
        self.bank=silhouette_bank()
        self.templates=np.stack([item[2] for item in self.bank])
        self.digits=np.array([item[0] for item in self.bank])
        self.angles=np.array([item[1] for item in self.bank])

    def _scores(self,mask):
        mask=np.asarray(mask,bool)
        if mask.ndim!=2 or not mask.any():
            return np.zeros(len(self.bank)),np.ones(len(self.bank),int)
        glyph=normalize(mask)
        union=np.logical_or(self.templates,glyph).sum((1,2))
        scores=np.logical_and(self.templates,glyph).sum((1,2))/np.maximum(union,1)
        return scores,union

    def fit(self,mask,xy=None):
        # xy is accepted only for compatibility with the former projector API.
        # It has no effect on identity, scale or the generic templates.
        scores,union=self._scores(mask);rows=[]
        for digit in range(10):
            indices=np.flatnonzero(self.digits==digit)
            best=int(indices[np.argmax(scores[indices])])
            rows.append(dict(digit=digit,yaw_deg=int(self.angles[best]),
                             confidence=float(scores[best]),union_pixels=int(union[best])))
        rows.sort(key=lambda item:-item['confidence'])
        return dict(**rows[0],margin=rows[0]['confidence']-rows[1]['confidence'],
                    quantization_margin=1./max(1,min(rows[0]['union_pixels'],rows[1]['union_pixels'])),ranking=rows)

    def refine(self,mask,xy,original):
        fit=self.fit(mask)
        use=bool(fit['confidence']>original['confidence'] and fit['margin']>fit['quantization_margin'])
        identity={key:fit[key] for key in ('digit','yaw_deg','confidence')} if use else dict(original)
        return dict(**identity,view_conditioned_fit=dict(original=dict(original),used=use,**fit,
                    source='generic_font_normalized_silhouette',uses_object_geometry=False))

    def validate_identity(self,mask,record):
        scores,_=self._scores(mask)
        indices=np.flatnonzero(self.digits==int(record['digit']))
        if not len(indices):return dict(valid=False,reason='unknown_digit',metric_size_validated=False)
        delta=np.abs(self.angles[indices]-float(record['yaw_deg']))
        indices=indices[delta==delta.min()]
        score=float(scores[indices].max())
        return dict(valid=score>.60,iou=score,threshold=.60,
                    source='generic_font_normalized_identity',scale_invariant=True,
                    metric_size_validated=False,uses_object_geometry=False)
