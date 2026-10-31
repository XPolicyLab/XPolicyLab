"""Conservative translation-only RGB mask propagation for association search regions."""
import numpy as np
from scipy.signal import fftconvolve
from scipy.ndimage import shift as translate


def propagate_mask(previous,current,mask,max_shift=48):
    if previous.shape!=current.shape or mask.shape!=previous.shape[:2] or not mask.any():return None
    ys,xs=np.where(mask);h,w=mask.shape
    x0,x1=max(0,xs.min()-4),min(w,xs.max()+5)
    y0,y1=max(0,ys.min()-4),min(h,ys.max()+5)
    gray=lambda rgb:np.asarray(rgb,dtype=float).mean(axis=2)
    template=gray(previous[y0:y1,x0:x1]);template-=template.mean()
    norm=np.linalg.norm(template)
    if norm<1e-6:return None # A featureless patch cannot localize a unique match.
    sx0,sx1=max(0,x0-max_shift),min(w,x1+max_shift)
    sy0,sy1=max(0,y0-max_shift),min(h,y1+max_shift)
    search=gray(current[sy0:sy1,sx0:sx1]);ones=np.ones(template.shape)
    sums=fftconvolve(search,ones,mode='valid');squares=fftconvolve(search*search,ones,mode='valid')
    corr=fftconvolve(search,template[::-1,::-1],mode='valid')
    denom=norm*np.sqrt(np.maximum(0,squares-sums*sums/template.size))
    scores=np.divide(corr,denom,out=np.full_like(corr,-1.),where=denom>1e-6)
    py,px=np.unravel_index(np.argmax(scores),scores.shape);best=float(scores[py,px])
    competitors=scores.copy();competitors[max(0,py-3):py+4,max(0,px-3):px+4]=-1.
    if best<.80 or best-float(competitors.max())<.03:return None
    dx,dy=int(sx0+px-x0),int(sy0+py-y0)
    propagated=translate(mask.astype(np.uint8),(dy,dx),order=0,mode='constant',cval=0)>0
    if not propagated.any():return None
    yy,xx=np.where(propagated)
    return {'mask':propagated,'bbox_2d':[int(xx.min()),int(yy.min()),int(xx.max()+1),int(yy.max()+1)],
            'translation_pixels':[dx,dy],'score':best,'method':'translation_only_rgb_ncc','status':'predicted'}


class MaskPropagation:
    def __init__(self):self.previous=None;self.camera=None;self.tracks={}

    def predict(self,image,camera,now,sample_index,memory):
        if self.previous is None or self.camera!=camera or self.previous.shape!=image.shape:return
        for eid,track in list(self.tracks.items()):
            if sample_index-track['observed_sample']>3:continue
            prediction=propagate_mask(self.previous,image,track['mask'])
            if prediction is None:continue
            track['mask']=prediction.pop('mask')
            prediction.update(time=now,camera=camera,image_size=[image.shape[1],image.shape[0]],
                observation_id=track['observation_id'],observed_sample=track['observed_sample'])
            memory.entities[eid]['tracking_prediction']=prediction
            memory.log('mask_search_prediction',{'entity_id':eid,**prediction})

    def accept(self,image,camera,observations,entity_ids,sample_index,memory):
        from PIL import Image
        from pathlib import Path
        if self.camera!=camera:self.tracks={}
        for obs,eid in zip(observations,entity_ids):
            if eid is None:continue
            path=Path(obs.get('mask_path') or '__missing__')
            if not path.is_file():continue
            with Image.open(path) as im:mask=np.asarray(im)>0
            if mask.shape!=image.shape[:2]:continue
            existing=self.tracks.get(eid)
            if existing and existing['observed_sample']==sample_index:mask=mask|existing['mask']
            self.tracks[eid]={'mask':mask,'observed_sample':sample_index,'observation_id':obs['observation_id']}
        self.previous=image.copy();self.camera=camera
