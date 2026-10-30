"""Independent current-RGB outer silhouette; reference is observed at actual close."""
import cv2,numpy as np
from .contour import ellipse_points,distance
from .vision_gate import appearance_mask,wrist_unproject,wrist_project
class Silhouette:
 def __init__(self,ring,rgb,pose):
  self.appearance=ring.appearance;self.saturation=ring.saturation;self.bound=ring.bound
  self.ellipse=cv2.fitEllipse(ring.reference['points'][360:].astype('float32')[:,None])
  self.reference=ellipse_points(self.ellipse);self.world=wrist_unproject(self.reference,pose,ring.z)
  self.center=np.asarray(self.ellipse[0]);self.rays=ellipse_points(self.ellipse,72)-self.center
  self.pose=np.asarray(pose);self.streak=0;self.confirmed=False
  self.observe(rgb)
 def observe(self,rgb):
  hsv=cv2.cvtColor(rgb,cv2.COLOR_RGB2HSV);mask=appearance_mask(rgb,self.appearance)&(cv2.GaussianBlur(hsv[:,:,1],(5,5),1)>self.saturation)
  mask=cv2.morphologyEx(mask.astype('uint8'),cv2.MORPH_CLOSE,np.ones((3,3),np.uint8)).astype(bool)
  smooth=cv2.GaussianBlur(rgb,(5,5),1).astype(float)
  points=[];angles=[];scales=np.linspace(.75,1.25,101)
  for i,v in enumerate(self.rays):
   pp=np.rint(self.center+scales[:,None]*v).astype(int)
   if np.any(pp<0) or np.any(pp[:,0]>=rgb.shape[1]) or np.any(pp[:,1]>=rgb.shape[0]):continue
   color=mask[pp[:,1],pp[:,0]]
   values=smooth[pp[:,1],pp[:,0]]
   contrast=np.zeros(len(scales));contrast[3:-3]=np.linalg.norm(values[6:]-values[:-6],axis=1)
   crossings=[j for j in range(4,len(scales)-4) if color[j-3:j].all() and contrast[j]>20 and contrast[j]>=max(contrast[j-2:j+3])]
   if not crossings:continue
   for j in crossings:points.append(self.center+scales[j]*v);angles.append(i)
  self.debug_points=points;self.debug_angles=angles;self.debug_mask=mask
  if len(points)<50:raise ValueError('silhouette_boundary_support:'+str(len(points)))
  pts=np.asarray(points,np.float32);angles=np.asarray(angles);rng=np.random.default_rng(0);fits=[]
  for subset in [np.arange(len(pts))]+[rng.choice(len(pts),12,replace=False) for _ in range(128)]:
   el=cv2.fitEllipse(pts[subset,None]);center,axes,_=el
   if min(axes)<=0 or np.linalg.norm(np.asarray(center)-self.center)>self.bound:continue
   ratio=np.sort(axes)/np.sort(self.ellipse[1])
   if np.any(ratio<.85) or np.any(ratio>1.15):continue
   from scipy.spatial import cKDTree
   residual=cKDTree(ellipse_points(el)).query(pts)[0];good=residual<3
   unique=np.zeros(len(good),bool)
   for angle in np.unique(angles):
    ids=np.where((angles==angle)&good)[0]
    if len(ids):unique[ids[np.argmin(residual[ids])]]=True
   fits.append((int(unique.sum()),-float(np.median(residual[unique])) if unique.any() else -999,unique,el))
  if not fits:raise ValueError('silhouette_no_geometric_fit')
  _,_,good,el=max(fits,key=lambda x:x[:2])
  residual=cKDTree(ellipse_points(el)).query(pts)[0]
  quadrants=[int(np.sum(good&(angles//18==q))) for q in range(4)]
  if good.sum()<50 or min(quadrants)<9:raise ValueError('silhouette_inlier_coverage:'+str([int(good.sum()),quadrants]))
  ratio=np.sort(el[1])/np.sort(self.ellipse[1])
  if np.any(ratio<.85) or np.any(ratio>1.15) or np.linalg.norm(np.asarray(el[0])-self.center)>self.bound:raise ValueError('silhouette_shape_drift')
  return {'points':ellipse_points(el),'support':int(good.sum()),'quadrants':quadrants,'fit_error_px':float(np.median(residual[good]))}
 def update(self,rgb,pose,corroborated):
  obs=self.observe(rgb);points=obs.pop('points');attached=distance(points,self.reference)
  # Recompute current curve for static comparison from same observed boundary fit.
  # Store no synthetic texture features; reference world points only encode the static hypothesis.
  static=distance(points,wrist_project(self.world,pose))
  passed=attached<self.bound and static>attached+6
  self.streak=self.streak+1 if passed and corroborated and pose[2]-self.pose[2]>.012 else 0
  if self.streak>=2:self.confirmed=True
  return dict(obs,passed=passed,confirmed=self.confirmed,attached_error_px=attached,static_error_px=static)
