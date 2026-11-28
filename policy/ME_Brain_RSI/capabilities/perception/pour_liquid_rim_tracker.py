import cv2,numpy as np
def fit_rim(rgb,ellipse,hue=None,reference=None):
    center,axes,angle=ellipse
    center=np.array(center,float);axes=np.array(axes,float)
    center_fraction=.35 if reference is None else .18
    hsv=cv2.cvtColor(rgb,cv2.COLOR_RGB2HSV);theta=np.linspace(0,2*np.pi,96,endpoint=False)
    rot=np.array([[np.cos(np.deg2rad(angle)),-np.sin(np.deg2rad(angle))],[np.sin(np.deg2rad(angle)),np.cos(np.deg2rad(angle))]])
    rays=np.c_[np.cos(theta)*axes[0]/2,np.sin(theta)*axes[1]/2]@rot.T
    if hue is None:
        uv=np.round(center+rays*1.1).astype(int);valid=(uv[:,0]>=0)&(uv[:,0]<rgb.shape[1])&(uv[:,1]>=0)&(uv[:,1]<rgb.shape[0]);colors=hsv[uv[valid,1],uv[valid,0]];colors=colors[(colors[:,1]>70)&(colors[:,2]>45)]
        if len(colors)<10:raise ValueError('No colored lip initialization')
        bins=np.bincount(colors[:,0],minlength=180);hue=int(bins.argmax())
    diff=np.abs(hsv[:,:,0].astype(float)-hue);color=(np.minimum(diff,180-diff)<10)&(hsv[:,:,1]>80)&(hsv[:,:,2]>35)
    points=[];angles=[]
    for i,direction in enumerate(rays):
        rs=np.linspace(.55,1.6,70);uv=np.round(center+rs[:,None]*direction).astype(int);valid=(uv[:,0]>=0)&(uv[:,0]<rgb.shape[1])&(uv[:,1]>=0)&(uv[:,1]<rgb.shape[0])
        vals=np.zeros(len(rs),bool);vals[valid]=color[uv[valid,1],uv[valid,0]]
        crossings=[k for k in range(3,len(rs)-3) if vals[k:k+3].all() and not vals[k-3:k].any()]
        if not crossings:continue
        k=min(crossings,key=lambda k:abs(rs[k]-1));points.append(center+rs[k]*direction);angles.append(i)
    if len(points)<35:raise ValueError('Lip arc insufficient')
    points=np.asarray(points,np.float32);best=None;rng=np.random.default_rng(0)
    if reference is None:reference=ellipse
    reference_axes=np.array(reference[1]);ref_major=max(reference_axes)
    for trial in range(70):
        sample=points if trial==0 else points[rng.choice(len(points),min(8,len(points)),replace=False)]
        try:c,a,ang=cv2.fitEllipse(sample.reshape(-1,1,2))
        except cv2.error:continue
        if min(a)<2 or max(a)>ref_major*1.3 or max(a)<ref_major*.65:continue
        if np.linalg.norm(np.array(c)-center)>max(axes)*center_fraction:continue
        rr=np.array([[np.cos(np.deg2rad(ang)),-np.sin(np.deg2rad(ang))],[np.sin(np.deg2rad(ang)),np.cos(np.deg2rad(ang))]])
        norm=(points-np.array(c))@rr/(np.array(a)/2);error=np.abs(np.linalg.norm(norm,axis=1)-1);good=error<.055
        coverage=len(set(np.asarray(angles)[good]//24))
        if good.sum()<32 or coverage<3:continue
        cost=int(good.sum())-10*float(np.median(error[good]))
        if best is None or cost>best[0]:best=(cost,good)
    if best is None:raise ValueError('Lip ellipse uncertain')
    good=best[1];ellipse=cv2.fitEllipse(points[good].reshape(-1,1,2))
    c,a,ang=ellipse
    if not np.isfinite(np.r_[c,a,ang]).all() or min(a)<max(2.,min(reference_axes)*.5) or max(a)>ref_major*1.3 or max(a)<ref_major*.65:raise ValueError('Refined lip scale uncertain')
    if np.linalg.norm(np.asarray(c)-center)>max(axes)*center_fraction:raise ValueError('Refined lip center uncertain')
    rr=np.array([[np.cos(np.deg2rad(ang)),-np.sin(np.deg2rad(ang))],[np.sin(np.deg2rad(ang)),np.cos(np.deg2rad(ang))]])
    error=np.abs(np.linalg.norm((points-np.asarray(c))@rr/(np.asarray(a)/2),axis=1)-1)
    good=error<.055;coverage=len(set(np.asarray(angles)[good]//24))
    if good.sum()<32 or coverage<3:raise ValueError('Refined lip support uncertain')
    return ellipse,hue,dict(points=int(good.sum()),coverage=coverage,residual=float(np.median(error[good])))


class RimTracker:
    def __init__(self,vision,state,side,anchor,radius,upright):
        from .geometry import Camera,quat_matrix
        from . import wristcam
        self.side=side;self.radius=radius
        pose=np.asarray(state[f'{side}_ee_pose']);rotation=quat_matrix(pose[3:])
        camera=Camera(dict(K=wristcam.K,camera_to_world_opengl=wristcam.camera_to_world(pose),resolution=wristcam.RESOLUTION))
        ee=np.eye(4);ee[:3,:3]=rotation;ee[:3,3]=pose[:3]
        self.camera_to_ee=np.linalg.inv(ee)@camera.c2w
        self.prior_anchor=np.asarray(anchor).copy()
        local=np.linalg.inv(self.camera_to_ee)@np.r_[anchor,1.]
        self.depth=-float(local[2])
        if self.depth<=radius:raise ValueError('Prior stereo depth too close')
        self.k=camera.k.copy();self.cv_to_gl=camera.cv_to_gl.copy()
        point=pose[:3]+rotation@self.prior_anchor;self.prior_pixel=camera.project([point])[0]
        theta=np.linspace(0,2*np.pi,64,endpoint=False)
        rim=point+(np.c_[np.cos(theta),np.sin(theta),np.zeros(len(theta))]*radius)@(rotation@upright.T).T
        projected=camera.project(rim)
        if not np.isfinite(projected).all():raise ValueError('Initial rim projection invalid')
        ellipse=cv2.fitEllipse(projected.astype(np.float32).reshape(-1,1,2))
        rgb=vision[f'cam_{side}_wrist']['color']
        self.ellipse,self.hue,info=fit_rim(rgb,ellipse)
        self.reference=self.ellipse
        self.centers=[];self.last_info=info

    def update(self,rgb,current_anchor):
        try:ellipse,hue,info=fit_rim(rgb,self.ellipse,self.hue,self.reference)
        except ValueError:
            self.centers=[]
            raise
        candidates=(self.centers+[np.asarray(ellipse[0])])[-3:]
        center=np.median(candidates,axis=0)
        if np.max(np.linalg.norm(np.asarray(candidates)-center,axis=1))>max(self.reference[1])*.12:
            self.centers=[]
            raise ValueError('Rim feature unstable')
        pixel_delta=center-self.prior_pixel
        delta_cv=np.r_[pixel_delta/self.k.diagonal()[:2]*self.depth,0.]
        delta=self.camera_to_ee[:3,:3]@self.cv_to_gl@delta_cv
        if np.linalg.norm(delta)>2*self.radius:
            self.centers=[]
            raise ValueError('Rim transverse correction outside measured mouth support')
        self.centers=candidates
        if len(candidates)<3:raise ValueError('Awaiting consecutive rim observations')
        self.ellipse=ellipse;self.hue=hue
        target=self.prior_anchor+delta
        step=target-np.asarray(current_anchor);norm=np.linalg.norm(step)
        if norm>.001:step*=.001/norm
        return np.asarray(current_anchor)+step,dict(pixel=center.tolist(),pixel_delta=pixel_delta.tolist(),target_anchor=target.tolist(),increment=step.tolist(),prior_depth_m=self.depth,depth_updated=False,measurement='Wrist image-plane feature correction at initial stereo depth; depth and changed rim normal are not observed',**info)
