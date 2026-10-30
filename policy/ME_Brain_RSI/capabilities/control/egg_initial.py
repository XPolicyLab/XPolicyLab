"""Recover separate fixed-size source shells from their visible RGB cores."""
import math
import cv2
import numpy as np


def _sample_boundary(points,count=48):
    """Resample the observed convex boundary without inventing another object."""
    points=np.asarray(points,float).reshape(-1,2)
    points=np.vstack([points,points[0]])
    lengths=np.r_[0.,np.cumsum(np.linalg.norm(np.diff(points,axis=0),axis=1))]
    along=np.linspace(0.,lengths[-1],count,endpoint=False)
    return np.column_stack([np.interp(along,lengths,points[:,axis]) for axis in range(2)])


def complete_initial_sources(ctx,observed):
    """Keep four image-supported source identities even when touching masks merge."""
    image=np.asarray(ctx.vision['cam_head']['color'],np.uint8)[...,:3]
    h,s,v=cv2.split(cv2.cvtColor(image,cv2.COLOR_RGB2HSV))
    workspace=ctx.perception.workspace_mask(image.shape)>0
    gray=cv2.cvtColor(image,cv2.COLOR_RGB2GRAY).astype(float)
    yy,xx=np.indices(image.shape[:2])
    chosen=[]
    selected_mask=None
    candidate_count=0
    for brightness in (220,200,180):
        mask=((h>=10)&(h<35)&(s<75)&(v>brightness)&workspace).astype(np.uint8)*255
        mask=cv2.morphologyEx(mask,cv2.MORPH_OPEN,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3)))
        distance=cv2.distanceTransform(mask,cv2.DIST_L2,5)
        maxima=(distance>=cv2.dilate(distance,np.ones((17,17),np.uint8))-1e-5)&(distance>4.)
        count,labels,stats,centers=cv2.connectedComponentsWithStats(maxima.astype(np.uint8))
        candidates=[]
        for component in range(1,count):
            pixel=centers[component];u,w=np.round(pixel).astype(int)
            radius=float(distance[w,u])
            if radius>16.:
                continue
            inside=(xx-pixel[0])**2+(yy-pixel[1])**2<max(3.,radius*.7)**2
            texture=float(np.std(gray[inside]))
            position=ctx.perception.project(pixel,ctx.table_z+.051,'cam_head',ctx.state,image.shape)
            if -.45<position[0]<.45 and -.34<position[1]<.025:
                candidates.append(dict(pixel=pixel.copy(),position=position.copy(),
                    radius=radius,score=radius-.08*texture))
        selected=[]
        for item in sorted(candidates,key=lambda item:item['score'],reverse=True):
            if all(np.linalg.norm(item['pixel']-other['pixel'])>11. for other in selected):
                selected.append(item)
            if len(selected)==4:
                break
        if len(selected)>len(chosen):
            chosen=selected;selected_mask=mask;candidate_count=len(candidates)
        if len(chosen)==4:
            break
    ctx.memory['egg_initial_completion']=dict(original_contours=len(observed),
        recovered_sources=len(chosen),candidate_count=candidate_count,
        source='public_rgb_fixed_shell_cores',source_measurement_step=int(ctx.steps),
        follows_grasped_objects=False,private_geometry_used=False)
    if len(chosen)<4:
        return observed
    centers=np.asarray([item['pixel'] for item in chosen])
    ownership=np.argmin((xx[...,None]-centers[:,0])**2+(yy[...,None]-centers[:,1])**2,axis=-1)
    sources=[]
    for index,item in enumerate(chosen):
        pixel=item['pixel'];radius=max(12.,2.*item['radius'])
        owned=(selected_mask>0)&(ownership==index)&((xx-pixel[0])**2+(yy-pixel[1])**2<radius**2)
        points=np.c_[xx[owned],yy[owned]].astype(np.float32)
        hull=cv2.convexHull(points)[:,0,:]
        contour=_sample_boundary(hull)
        world=np.asarray([ctx.perception.project(point,ctx.table_z+.051,'cam_head',ctx.state,image.shape)[:2]
                          for point in contour])
        centered=world-world.mean(0)
        _,axes=np.linalg.eigh(centered.T@centered)
        axis=axes[:,-1];yaw=math.atan2(axis[1],axis[0])%math.pi
        position=item['position'].copy();position[2]=ctx.table_z+.029
        close=sorted(observed,key=lambda other:np.linalg.norm(np.asarray(other['position'])[:2]-position[:2]))
        if close and np.linalg.norm(np.asarray(close[0]['position'])[:2]-position[:2])<.020:
            yaw=float(close[0]['yaw'])
        bbox=cv2.boundingRect(hull)
        x,y,width,height=bbox
        sources.append(dict(name='egg',kind='egg',position=position,yaw=yaw,
            pixel=tuple(pixel),contour=contour,bbox=bbox,size=(.058,.042,.058),
            camera='cam_head',confidence=.55,crop=image[y:y+height,x:x+width].copy(),
            mean_rgb=image[owned].mean(0),area=float(np.count_nonzero(owned)),
            initial_source_model_completed=True,completion_basis='visible_shell_core_and_public_fixed_size'))
    return sources
