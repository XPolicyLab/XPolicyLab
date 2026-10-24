"""Robot projection raster only; extracted corrected near-plane CPU rasterizer."""
import numpy as np
from numba import njit
@njit
def _raster_triangles(vertices,faces,K,depth,labels,label):
 for face in faces:
  a,b,c=vertices[face[0]],vertices[face[1]],vertices[face[2]]
  if min(a[2],b[2],c[2])<.001-1e-10: continue
  x0,y0=K[0,0]*a[0]/a[2]+K[0,2],K[1,1]*a[1]/a[2]+K[1,2]
  x1,y1=K[0,0]*b[0]/b[2]+K[0,2],K[1,1]*b[1]/b[2]+K[1,2]
  x2,y2=K[0,0]*c[0]/c[2]+K[0,2],K[1,1]*c[1]/c[2]+K[1,2]
  denom=(y1-y2)*(x0-x2)+(x2-x1)*(y0-y2)
  if abs(denom)<1e-10:continue
  for v in range(max(0,int(np.floor(min(y0,y1,y2)))),min(depth.shape[0],int(np.ceil(max(y0,y1,y2)))+1)):
   for u in range(max(0,int(np.floor(min(x0,x1,x2)))),min(depth.shape[1],int(np.ceil(max(x0,x1,x2)))+1)):
    w0=((y1-y2)*(u-x2)+(x2-x1)*(v-y2))/denom
    w1=((y2-y0)*(u-x2)+(x0-x2)*(v-y2))/denom;w2=1-w0-w1
    if min(w0,w1,w2)>=-1e-8:
     z=1/(w0/a[2]+w1/b[2]+w2/c[2])
     if z<depth[v,u]:depth[v,u]=z;labels[v,u]=label

@njit
def raster(vertices,faces,K,depth,labels,label):
 # Sutherland-Hodgman clipping at the optical near plane. A clipped triangle
 # has at most four vertices; fan triangulation preserves its surface.
 triangles=np.empty((len(faces)*2,3,3),np.float64);n=0
 for face in faces:
  poly=np.empty((4,3),np.float64);count=0
  previous=vertices[face[2]];prev_inside=previous[2]>=.001
  for index in range(3):
   current=vertices[face[index]];inside=current[2]>=.001
   if inside!=prev_inside:
    t=(.001-previous[2])/(current[2]-previous[2]);poly[count]=previous+t*(current-previous);count+=1
   if inside:poly[count]=current;count+=1
   previous=current;prev_inside=inside
  for i in range(1,count-1):
   triangles[n,0]=poly[0];triangles[n,1]=poly[i];triangles[n,2]=poly[i+1];n+=1
 if n:
  _raster_triangles(triangles[:n].reshape((n*3,3)),np.arange(n*3).reshape((n,3)),K,depth,labels,label)
