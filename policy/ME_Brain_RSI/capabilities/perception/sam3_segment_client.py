"""SAM3 file-service client (segmentation runs in a separate process that watches a shared directory)."""
import os,time,uuid
from pathlib import Path
import numpy as np
import cv2


def sam_root():
    directory = os.environ.get('SNR_SAM_ROOT') or os.environ.get('SND_SAM_ROOT')
    if not directory:
        raise ValueError('Set SND_SAM_ROOT to the shared SAM3 request directory')
    return Path(directory)


def segment(rgb,prompts,min_score=0.5,timeout=180.,min_area=25,root=None,prefer_larger=False):
    """List of dict(mask: bool HxW, score) sorted by score, duplicates (>75 % overlap of the smaller) removed: the higher
    score survives, or with prefer_larger the larger mask (a part of an object, e.g. its round head, must not replace the
    whole object) unless it is more than 4x larger (a merge with a neighbour)."""
    root=Path(root) if root else sam_root()
    token=uuid.uuid4().hex
    tmp=root/(token+'.request.writing')
    with tmp.open('wb') as f:
        np.savez_compressed(f,rgb=np.ascontiguousarray(np.asarray(rgb)[...,:3],dtype=np.uint8),prompts=np.asarray(list(prompts)))
    tmp.replace(root/(token+'.request.npz'))
    out=root/(token+'.response.npz');err=root/(token+'.error.json')
    deadline=time.monotonic()+timeout
    while not out.exists():
        if err.exists():raise RuntimeError('SAM3 service error: '+err.read_text()[:300])
        if time.monotonic()>deadline:raise TimeoutError('SAM3 segmentation response timed out')
        time.sleep(0.05)
    for _ in range(20):
        try:
            with np.load(out) as src:masks=src['masks'].astype(bool);scores=np.asarray(src['scores'],float)
            break
        except Exception:time.sleep(0.1)
    else:raise RuntimeError('SAM3 response unreadable')
    keep=[]
    for i in np.argsort(-scores):
        m=masks[i]
        if scores[i]<min_score or m.sum()<min_area:continue
        n,lab,stats,_=cv2.connectedComponentsWithStats(m.astype(np.uint8))
        if n>2:m=lab==(1+int(np.argmax(stats[1:,cv2.CC_STAT_AREA])))
        area=int(m.sum());dup=None
        for k in keep:
            ka=int(k['mask'].sum())
            if np.count_nonzero(m&k['mask'])/max(1,min(area,ka))>0.75:dup=k;break
        if dup is None:keep.append(dict(mask=m,score=float(scores[i])));continue
        if prefer_larger and area>int(dup['mask'].sum()) and area<=4*int(dup['mask'].sum()):dup['mask']=m;dup['score']=max(dup['score'],float(scores[i]))
    return keep
