"""RGB depression evidence independent of equation content and instance location."""
import cv2,numpy as np
def depressions(rgb,mask):
 ys,xs=np.nonzero(mask)
 if not len(xs):return dict(components=[],paired=False)
 x0,x1=xs.min(),xs.max()+1;y0,y1=ys.min(),ys.max()+1
 m=(mask[y0:y1,x0:x1]>0).astype('uint8');crop=rgb[y0:y1,x0:x1]
 # Counter detection uses the outer silhouette; the glyph may already have holes removed.
 contours,_=cv2.findContours(m,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
 filled=np.zeros_like(m);cv2.drawContours(filled,contours,-1,1,-1)
 m=filled
 light=cv2.cvtColor(crop,cv2.COLOR_RGB2LAB)[:,:,0]
 k=max(5,int(round(min(m.shape)*.35))|1)
 baseline=cv2.morphologyEx(light,cv2.MORPH_CLOSE,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(k,k)))
 residual=cv2.subtract(baseline,light)
 inner=cv2.erode(m,np.ones((5,5),np.uint8))>0
 threshold=float(cv2.threshold(residual[inner],0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)[0]) if inner.any() else 255
 dark=((residual>max(8,threshold))&inner).astype('uint8')
 n,labels,stats,centers=cv2.connectedComponentsWithStats(dark)
 distance=cv2.distanceTransform(np.pad(m,1),cv2.DIST_L2,5)[1:-1,1:-1]
 comps=[]
 for j in range(1,n):
  area=int(stats[j,cv2.CC_STAT_AREA]);clearance=float(distance[labels==j].min())
  if area<4 or clearance<max(3,min(m.shape)*.1):continue
  comps.append(dict(area=area,center=centers[j].tolist(),clearance=clearance))
 paired=False
 if len(comps)==2:
  separation=float(np.linalg.norm(np.array(comps[0]['center'])-comps[1]['center']))
  paired=separation>max(m.shape)*.2
 return dict(components=comps,paired=paired,otsu=threshold,kernel=k)


def visible_counters(rgb,mask):
 """Remove measured enclosed dark interiors, without assigning a digit label."""
 ys,xs=np.nonzero(mask)
 if not len(xs):return mask
 x0,x1=xs.min(),xs.max()+1;y0,y1=ys.min(),ys.max()+1
 m=mask[y0:y1,x0:x1];crop=rgb[y0:y1,x0:x1]
 contours,_=cv2.findContours(m,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
 filled=np.zeros_like(m);cv2.drawContours(filled,contours,-1,255,-1)
 inner=cv2.erode(filled,np.ones((5,5),np.uint8))>0
 if not inner.any():return mask
 light=cv2.cvtColor(crop,cv2.COLOR_RGB2LAB)[:,:,0]
 threshold=cv2.threshold(light[inner],0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)[0]
 dark=((light<threshold)&inner).astype('uint8')
 distance=cv2.distanceTransform(np.pad((filled>0).astype('uint8'),1),cv2.DIST_L2,5)[1:-1,1:-1]
 n,labels,stats,_=cv2.connectedComponentsWithStats(dark)
 result=mask.copy();local=result[y0:y1,x0:x1]
 for i in range(1,n):
  if stats[i,cv2.CC_STAT_AREA]>=4 and float(distance[labels==i].min())>=max(3,min(m.shape)*.1):local[labels==i]=0
 return result
