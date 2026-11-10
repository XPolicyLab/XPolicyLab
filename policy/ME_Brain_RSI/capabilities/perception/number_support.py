"""Canonical measured foreground, independent of glyph identity/arithmetic."""
import cv2,numpy as np

def canonical(rgb,masks):
 hsv=cv2.cvtColor(rgb,cv2.COLOR_RGB2HSV);colored=(hsv[:,:,1]>100)&(hsv[:,:,2]>65);proposals=[];evidence=[]
 for i,m in enumerate(masks):
  clean=((m>0)&colored).astype('uint8');n,labels,stats,_=cv2.connectedComponentsWithStats(clean)
  if n<=1:continue
  j=1+int(np.argmax(stats[1:,cv2.CC_STAT_AREA]));body=labels==j;area=int(body.sum())
  ev=dict(source_index=i,raw_pixels=int((m>0).sum()),colored_pixels=int(clean.sum()),canonical_pixels=area)
  evidence.append(ev)
  if area>=15:proposals.append((area,body,ev))
 kept=[]
 for area,body,ev in sorted(proposals,key=lambda x:x[0],reverse=True):
  overlap=next((k for k,(other,_) in enumerate(kept) if np.count_nonzero(body&other)/area>=.90),None)
  if overlap is not None:ev['contained_duplicate_of']=kept[overlap][1]['source_index'];continue
  kept.append((body,ev))
 return [body.astype('uint8')*255 for body,_ in kept],evidence
