"""RGB-only perception for sort_nesting_dolls_by_size (head camera object blobs + known-profile silhouette fits).

Everything is public geometry: per-height width profiles of the 15 released matryoshka_dolls assets (indices 0-4 and 5-9
are round dolls with two paint schemes, 10-14 are rubber ducks; rank = index % 5, rank 0 is the tallest) and the released
head-camera calibration. No layout, depth or object pose is read.
"""
import json
from pathlib import Path
import numpy as np
import cv2
from .geometry import Camera,quat_matrix
from . import wristcam

TABLE_Z=0.765
WORKSPACE=dict(x=(-0.55,0.55),y=(-0.36,0.02))          # table footprint the policy may act on
IMAGE_REGION=dict(x=(-0.56,0.56),y=(-0.37,0.24))        # object pixels (tops of 13 cm objects project onto farther table points)
TYPES={'t0':0,'t1':5,'duck':10}                        # asset index = base + rank
GEOMETRY=json.loads((Path(__file__).resolve().parents[1]/'configs/doll_geometry.json').read_text())
HEIGHT={r:GEOMETRY[str(r)]['height'] for r in range(5)}


def profile(kind,rank):return GEOMETRY[str(TYPES[kind]+rank)]


def base_radius(kind,rank):
    p=profile(kind,rank)['profile'][0];return 0.25*(p['xw']+p['yw'])


def footprint_radius(kind,rank):
    """Half of the largest horizontal extent (collision reasoning)."""
    e=profile(kind,rank)['extents'];return 0.5*max(e[0],e[1])


def profile_points(kind,rank,xy,n=12):
    """3-D surface samples of an upright object of this kind/rank standing at xy (axis-aligned, yaw 0)."""
    ang=np.linspace(0,2*np.pi,n,endpoint=False);ca,sa=np.cos(ang),np.sin(ang)
    pts=[]
    for s in profile(kind,rank)['profile']:
        pts.append(np.c_[xy[0]+s['xc']+0.5*s['xw']*ca,xy[1]+s['yc']+0.5*s['yw']*sa,np.full(n,TABLE_Z+s['z'])])
    return np.vstack(pts)


def components(mask,min_area=1):
    n,labels,stats,centers=cv2.connectedComponentsWithStats(mask.astype(np.uint8),connectivity=8)
    out=[]
    for j in range(1,n):
        area=int(stats[j,cv2.CC_STAT_AREA])
        if area<min_area:continue
        out.append(dict(area=area,center=centers[j].copy(),box=(int(stats[j,0]),int(stats[j,1]),int(stats[j,2]),int(stats[j,3])),label=j))
    return out,labels


def merge_touching(blobs,labels,gap=4):
    """Union blobs whose bounding boxes come within `gap` pixels (one object split by a dark paint band)."""
    parent=list(range(len(blobs)))
    def find(i):
        while parent[i]!=i:parent[i]=parent[parent[i]];i=parent[i]
        return i
    for i in range(len(blobs)):
        for j in range(i+1,len(blobs)):
            x0,y0,w0,h0=blobs[i]['box'];x1,y1,w1,h1=blobs[j]['box']
            if x0-gap<x1+w1 and x1-gap<x0+w0 and y0-gap<y1+h1 and y1-gap<y0+h0:parent[find(i)]=find(j)
    groups={}
    for i,b in enumerate(blobs):groups.setdefault(find(i),[]).append(b)
    out=[];relabel=labels.copy()
    for k,(root,members) in enumerate(groups.items(),start=1):
        for b in members:relabel[labels==b['label']]=k
        xs=[b['box'][0] for b in members];ys=[b['box'][1] for b in members]
        xe=[b['box'][0]+b['box'][2] for b in members];ye=[b['box'][1]+b['box'][3] for b in members]
        area=sum(b['area'] for b in members)
        center=sum(b['center']*b['area'] for b in members)/area
        out.append(dict(area=area,center=center,box=(min(xs),min(ys),max(xe)-min(xs),max(ye)-min(ys)),label=k))
    return out,relabel


def local_iou(mask,sil,band_px=6):
    """IoU restricted to a band around the silhouette: touching neighbours do not count against the fit."""
    k=2*band_px+1;band=cv2.dilate(sil,np.ones((k,k),np.uint8)).astype(bool)
    inter=np.count_nonzero(sil.astype(bool)&mask);union=np.count_nonzero(sil.astype(bool)|(mask&band))
    return inter/max(union,1)


def fit_silhouette(mask,project,kind,rank,init_xy,steps=(0.008,0.004,0.002,0.001),band_px=6):
    """Maximise the local IoU of the known upright silhouette over table XY (coordinate descent)."""
    h,w=mask.shape
    def score(xy):
        pix=project(profile_points(kind,rank,xy))
        if pix is None or not np.all(np.isfinite(pix)):return -1.
        x0=int(max(0,np.floor(pix[:,0].min())-band_px-4));x1=int(min(w,np.ceil(pix[:,0].max())+band_px+5))
        y0=int(max(0,np.floor(pix[:,1].min())-band_px-4));y1=int(min(h,np.ceil(pix[:,1].max())+band_px+5))
        if x1-x0<3 or y1-y0<3:return -1.
        sil=hull_mask(pix,(y1-y0,x1-x0),(x0,y0))
        return local_iou(mask[y0:y1,x0:x1],sil,band_px)
    best=np.asarray(init_xy,float);val=score(best)
    for dp in steps:
        improved=True
        while improved:
            improved=False
            for move in ((dp,0),(-dp,0),(0,dp),(0,-dp)):
                xy=best+np.array(move);v=score(xy)
                if v>val+1e-6:best=xy;val=v;improved=True
    return dict(xy=best.tolist(),iou=float(val))


def wood_mask(hsv,hm):
    H=hsv[...,0].astype(int);S=hsv[...,1].astype(int);V=hsv[...,2].astype(int)
    return (H>=hm-6)&(H<=hm+7)&(S>=60)&(S<=205)&(V>=45)&(V<=238)


def hull_mask(pixels,shape,offset=(0,0)):
    pts=np.round(np.asarray(pixels)-np.asarray(offset)).astype(np.int32)
    hull=cv2.convexHull(pts.reshape(-1,1,2))
    m=np.zeros(shape,np.uint8);cv2.fillConvexPoly(m,hull,1);return m


class HeadVision:
    """Table objects from the fixed head camera: wood-colour background model, blobs, known-profile silhouette fits."""
    def __init__(self,calibration):
        self.cam=Camera(calibration['head'])
        h,w=self.cam.resolution[1],self.cam.resolution[0]
        uu,vv=np.meshgrid(np.arange(w),np.arange(h))
        pts=np.c_[uu.ravel(),vv.ravel(),np.ones(h*w)]
        rays=(pts@np.linalg.inv(self.cam.k).T)@self.cam.cv_to_gl@self.cam.c2w[:3,:3].T
        origin=self.cam.c2w[:3,3]
        with np.errstate(divide='ignore',invalid='ignore'):
            distance=(TABLE_Z-origin[2])/rays[:,2]
        world=origin+distance[:,None]*rays
        ok=np.isfinite(distance)&(distance>0)
        self.table=(ok&(world[:,0]>WORKSPACE['x'][0])&(world[:,0]<WORKSPACE['x'][1])&(world[:,1]>WORKSPACE['y'][0])&(world[:,1]<WORKSPACE['y'][1])).reshape(h,w)
        self.region=(ok&(world[:,0]>IMAGE_REGION['x'][0])&(world[:,0]<IMAGE_REGION['x'][1])&(world[:,1]>IMAGE_REGION['y'][0])&(world[:,1]<IMAGE_REGION['y'][1])).reshape(h,w)
        self.log={}

    def project(self,points):return self.cam.project(points)

    def robot_mask(self,ee_poses,shape):
        """Pixels covered by the white link6 shells at the current EE poses (public URDF extents), dilated for the dark fingers."""
        m=np.zeros(shape,np.uint8)
        if not ee_poses:return m
        local=np.array([[x,y,z] for x in (-0.012,0.16) for y in (-0.092,0.092) for z in (-0.045,0.045)])
        for pose in ee_poses.values():
            pose=np.asarray(pose,float);R=quat_matrix(pose[3:]);pts=pose[:3]+local@R.T
            pix=self.project(pts)
            if not np.all(np.isfinite(pix)):continue
            hull=cv2.convexHull(np.round(pix).astype(np.int32).reshape(-1,1,2));cv2.fillConvexPoly(m,hull,1)
        return cv2.dilate(m,np.ones((9,9),np.uint8))

    def object_mask(self,rgb,ee_poses):
        hsv=cv2.cvtColor(np.ascontiguousarray(rgb[...,:3]),cv2.COLOR_RGB2HSV)
        H=hsv[...,0].astype(int);S=hsv[...,1].astype(int);V=hsv[...,2].astype(int)
        robot=self.robot_mask(ee_poses,rgb.shape[:2]).astype(bool)
        sample=self.table&~robot
        hm=float(np.median(H[sample]));sm=float(np.median(S[sample]));vm=float(np.median(V[sample]))
        # wood is a tight HSV box (12 frames: H 6-10, S 71-184, V 115-179 at the 1st/99th percentiles); anything else on
        # the table is object (gold/green/cream doll paint, yellow duck); shadows stay wood through the low V bound
        wood=wood_mask(hsv,hm)
        obj=~wood&self.region&~robot
        obj=cv2.morphologyEx(obj.astype(np.uint8),cv2.MORPH_OPEN,np.ones((3,3),np.uint8))
        obj=cv2.morphologyEx(obj,cv2.MORPH_CLOSE,np.ones((9,9),np.uint8))
        self.log['wood']=dict(h=hm,s=sm,v=vm)
        return obj.astype(bool)

    def local_iou(self,mask,sil):return local_iou(mask,sil)

    def fit(self,mask,kind,rank,init_xy,steps=(0.008,0.004,0.002,0.001)):
        return fit_silhouette(mask,self.project,kind,rank,init_xy,steps)

    def seeds(self,blobs,labels,kind):
        """Candidate base positions per rank: blob bottom edge (visible base) and blob top edge (object top, for an
        object partly hidden behind a neighbour); wide blobs contribute several columns."""
        out={r:[] for r in range(5)}
        for b in blobs:
            x,y,w,h=b['box'];sub=labels[y:y+h,x:x+w]==b['label']
            cols=np.nonzero(sub.any(0))[0]
            if len(cols)==0:continue
            n=max(1,int(round(w/28.)));cuts=np.linspace(cols[0],cols[-1],n+2)[1:-1] if n>1 else [0.5*(cols[0]+cols[-1])]
            for cu in cuts:
                c=int(round(cu));lo=max(0,c-6);hi=min(w,c+7);band=sub[:,lo:hi]
                rows=np.nonzero(band.any(1))[0]
                if len(rows)==0:continue
                u=x+c;vb=y+rows.max();vt=y+rows.min()
                for r in range(5):
                    try:
                        pb=self.cam.pixel_to_plane((u,vb),TABLE_Z);out[r].append(pb[:2]+np.array([0.,base_radius(kind,r)]))
                        pt=self.cam.pixel_to_plane((u,vt),TABLE_Z+HEIGHT[r]);out[r].append(pt[:2])
                    except ValueError:pass
        return out

    def refit(self,rgb,ee_poses,kind,rank,expected_xy,max_shift=0.04):
        """Re-localise one known object near an expected position (after a hand-off or a nudge); None when unconvincing."""
        mask=self.object_mask(rgb,ee_poses)
        f=fit_silhouette(mask,self.project,kind,rank,np.asarray(expected_xy,float),steps=(0.006,0.003,0.0015,0.001))
        shift=float(np.linalg.norm(np.asarray(f['xy'])-np.asarray(expected_xy)))
        ok=f['iou']>=0.6 and shift<=max_shift
        self.log['refit']=dict(valid=bool(ok),iou=round(f['iou'],3),shift_mm=round(1000*shift,1),xy=[round(v,4) for v in f['xy']])
        return np.asarray(f['xy']) if ok else None

    def detect(self,rgb,ee_poses):
        """Five table objects -> kind (t0/t1/duck), rank per object and table XY.

        Greedy largest-first fits with explained pixels removed, so touching or partly hidden objects are separated;
        the kind whose five silhouettes explain the mask best wins."""
        mask=self.object_mask(rgb,ee_poses)
        blobs,labels=components(mask,min_area=60)
        blobs,labels=merge_touching(blobs,labels,gap=4)
        blobs=[b for b in blobs if b['area']>=100]
        blobs.sort(key=lambda b:-b['area']);blobs=blobs[:8]
        out=dict(blobs=[dict(area=b['area'],box=b['box']) for b in blobs])
        if not blobs:out['error']='no_blobs';return out
        results={}
        for kind in TYPES:
            remaining=mask.copy();seeds=self.seeds(blobs,labels,kind);assign=[];total=0.
            for r in range(5):
                cands=[c for c in seeds[r] if np.all(np.isfinite(c))]
                if not cands:assign.append(dict(rank=r,xy=[np.nan,np.nan],iou=0.));continue
                # coarse pre-score to keep the fits cheap
                scored=[]
                for c in cands:
                    pix=self.project(profile_points(kind,r,c))
                    if not np.all(np.isfinite(pix)):continue
                    h,w=mask.shape;x0=int(max(0,pix[:,0].min()-10));x1=int(min(w,pix[:,0].max()+11));y0=int(max(0,pix[:,1].min()-10));y1=int(min(h,pix[:,1].max()+11))
                    if x1-x0<3 or y1-y0<3:continue
                    scored.append((self.local_iou(remaining[y0:y1,x0:x1],hull_mask(pix,(y1-y0,x1-x0),(x0,y0))),tuple(c)))
                scored.sort(reverse=True)
                best=None
                for _,c in scored[:3]:
                    f=self.fit(remaining,kind,r,np.array(c))
                    if best is None or f['iou']>best['iou']:best=f
                if best is None:assign.append(dict(rank=r,xy=[np.nan,np.nan],iou=0.));continue
                assign.append(dict(rank=r,**best));total+=best['iou']
                sil=hull_mask(self.project(profile_points(kind,r,best['xy'])),mask.shape)
                remaining&=~cv2.dilate(sil,np.ones((5,5),np.uint8)).astype(bool)
            results[kind]=dict(score=float(total),assign=assign)
        kind=max(results,key=lambda k:results[k]['score'])
        objs=results[kind]['assign']
        out.update(kind=kind,score=results[kind]['score'],objects=objs,alternatives={k:round(v['score'],3) for k,v in results.items()})
        bad=[o['rank'] for o in objs if not np.all(np.isfinite(o['xy'])) or o['iou']<0.35]
        if bad:out['error']='weak_fits';out['weak']=bad
        return out


class WristVision:
    """Refine one object's table XY from the wrist camera at a hover: known-silhouette fit against the non-wood mask."""
    def __init__(self):self.log={}

    def fit(self,rgb,ee_pose,kind,rank,expected_xy,max_shift=0.035):
        c2w=wristcam.camera_to_world(ee_pose)
        hsv=cv2.cvtColor(np.ascontiguousarray(rgb[...,:3]),cv2.COLOR_RGB2HSV)
        H=hsv[...,0].astype(int);S=hsv[...,1].astype(int);V=hsv[...,2].astype(int)
        woodish=(S>=60)&(V>=90)&(V<=230)&(H<=25)
        hm=float(np.median(H[woodish])) if np.count_nonzero(woodish)>2000 else 9.
        # wood keeps its hue/saturation in the fingers' shadow, so no brightness floor here; the black fingers are excluded
        wood=(H>=hm-6)&(H<=hm+7)&(S>=60)&(S<=205)&(V<=238)
        black=(V<40)|((S<50)&(V<90))
        mask=~wood&~black
        h,w=mask.shape
        # keep only pixels whose ray lands near the expected object at mid height (other objects, robot body, scenery out)
        uu,vv=np.meshgrid(np.arange(w),np.arange(h))
        pts,valid=wristcam.pixel_to_plane(c2w,np.c_[uu.ravel(),vv.ravel()],TABLE_Z+0.5*HEIGHT[rank])
        near=valid&(np.linalg.norm(pts[:,:2]-np.asarray(expected_xy),axis=1)<0.08)
        mask&=near.reshape(h,w)
        mask=cv2.morphologyEx(mask.astype(np.uint8),cv2.MORPH_OPEN,np.ones((3,3),np.uint8)).astype(bool)
        if np.count_nonzero(mask)<150:
            self.log=dict(valid=False,reason='no_object_pixels',pixels=int(np.count_nonzero(mask)));return None
        def project(points):
            pix,depth=wristcam.project(c2w,points)
            if np.any(depth<=0.02):return None
            return pix
        f=fit_silhouette(mask,project,kind,rank,np.asarray(expected_xy,float),steps=(0.006,0.003,0.0015),band_px=10)
        shift=float(np.linalg.norm(np.asarray(f['xy'])-np.asarray(expected_xy)))
        ok=f['iou']>=0.40 and shift<=max_shift
        self.log=dict(valid=bool(ok),iou=round(f['iou'],3),shift_mm=round(1000*shift,1),xy=[round(v,4) for v in f['xy']],pixels=int(np.count_nonzero(mask)),hm=hm)
        return np.asarray(f['xy']) if ok else None
