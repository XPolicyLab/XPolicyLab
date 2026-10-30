"""RGB-only game measurements using public camera and object dimensions.

Mahjong feature extraction and board geometry were adapted from the public
Lxm6wuec skill release (rdj_kong b44fea680452539a, rdj_tic 160f6e1b5c00bc22).
This module reads pixels only. No simulator, reward state, or object labels are
imported or consumed. Camera constants describe the public arx_x5 calibration.
"""
import itertools
import cv2
import numpy as np
from scipy.signal import find_peaks

F=640*10/22.212
CAM=np.array([0.,-.41,1.308])
R=np.array([[1.,0,0],[0,np.cos(np.pi/6),-np.sin(np.pi/6)],[0,np.sin(np.pi/6),np.cos(np.pi/6)]])
TABLE=.765
HEIGHT=.0647
WIDTH=.0449
THICK=.0328
BOARD_SURFACE_Z = TABLE+.016  # Public checkerboard geometry thickness.

def project(p):
    q=(np.asarray(p)-CAM)@R
    return np.stack([320+F*q[...,0]/-q[...,2],240-F*q[...,1]/-q[...,2]],axis=-1)

def plane(uv,z):
    u,v=np.asarray(uv)
    ray=R@np.array([(u-320)/F,-(v-240)/F,-1.])
    return CAM+ray*((z-CAM[2])/ray[2])

def quad_crop(rgb,q):
    return cv2.warpPerspective(rgb,cv2.getPerspectiveTransform(np.float32(q),np.float32([[0,0],[47,0],[47,63],[0,63]])),(48,64))

def feature(crop, adaptive=False):
    # Remove borders before selecting printed pixels; hue keeps green/blue/red
    # distinctions that binary registered descriptors discarded.
    hsv=cv2.cvtColor(crop,cv2.COLOR_RGB2HSV)
    colored=(hsv[:,:,1]>max(25,float(np.percentile(hsv[7:-7,5:-5,1],65))))&(hsv[:,:,2]>80)
    interior=hsv[9:-10,9:-9]
    black_cut=140
    if adaptive or np.count_nonzero(interior[:,:,1]>65)<8:
        level,_=cv2.threshold(interior[:,:,2],0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        black_cut=min(210,float(level))
    black=(hsv[:,:,2]<black_cut)&(hsv[:,:,1]<100)
    ink=colored|black
    ink[:9]=False;ink[-10:]=False;ink[:,:9]=False;ink[:,-9:]=False
    ys,xs=np.where(ink)
    if len(xs)<8:return None
    x0,x1=xs.min(),xs.max()+1;y0,y1=ys.min(),ys.max()+1
    masks=[]
    for a,b in [(0,18),(165,180),(28,90),(90,140)]:
        masks.append(ink&colored&(hsv[:,:,0]>=a)&(hsv[:,:,0]<b))
    masks=[masks[0]|masks[1],masks[2],masks[3],ink&black]
    return np.stack([cv2.resize(m[y0:y1,x0:x1].astype(np.float32),(18,26),interpolation=cv2.INTER_AREA) for m in masks])

def balanced(f):
    if f is None:return None
    norms=np.linalg.norm(f,axis=(1,2),keepdims=True)
    return np.where(norms>.5,f/np.maximum(norms,1e-6),0.)

def similarity(a,b):
    if a is None or b is None:return 0.
    best=0.
    for bb in (b,b[:,::-1,::-1]):
        for dy in (-1,0,1):
            for dx in (-1,0,1):
                v=np.roll(bb,(dy,dx),(1,2))
                best=max(best,float((a*v).sum()/max(np.linalg.norm(a)*np.linalg.norm(v),1e-6)))
    return best

def white_mask(rgb):
    h=cv2.cvtColor(rgb,cv2.COLOR_RGB2HSV)
    return ((h[:,:,1]<75)&(h[:,:,2]>135)).astype(np.uint8)*255

def row_instances(rgb):
    mask=cv2.morphologyEx(white_mask(rgb),cv2.MORPH_CLOSE,np.ones((3,3),np.uint8))
    cs,_=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    rows=[c for c in cs if cv2.boundingRect(c)[2]>5*cv2.boundingRect(c)[3] and cv2.contourArea(c)>1000]
    if not rows:return [],None
    x,y,w,h=cv2.boundingRect(max(rows,key=cv2.contourArea))
    gray=cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY).astype(float)
    # A clean strip through tile top faces exposes seams, unlike ink-heavy fronts.
    strip=gray[y+3:y+10,x:x+w]
    darkness=255-strip.mean(0)
    peaks,_=find_peaks(darkness,prominence=8,distance=max(8,int(h*.4)))
    if len(peaks)<5:return [],{'bbox':[x,y,w,h],'error':'insufficient_seams'}
    spacing=float(np.median(np.diff(peaks)))
    # Front/top face fold is the strongest common downward luminance edge.
    profile=gray[y+5:y+int(h*.8),x+int(w*.15):x+int(w*.85)].mean(1)
    drops=np.where(np.diff(profile)<-8)[0]
    if not len(drops):return [],{'bbox':[x,y,w,h],'error':'no_face_fold'}
    fold=y+5+int(drops[0])+1
    # Track each top seam to the fold using small local minima, no fixed tile IDs.
    seams=[]
    for p in peaks:
        pts=[]
        for yy in range(y+3,min(fold,y+12)):
            lo=max(x,int(x+p-3));hi=min(x+w,int(x+p+4))
            xx=lo+int(np.argmin(gray[yy,lo:hi]));pts.append((yy,xx))
        fit=np.polyfit(np.array(pts)[:,0]-y,np.array(pts)[:,1],1)
        seams.append(float(np.polyval(fit,fold-y)))
    # A visible outside edge is allowed; missing occluded edges are not invented.
    # Only add a boundary supported by the current visible white contour.
    contour_edges=[]
    for yy in range(fold-1,fold+2):
        visible=np.flatnonzero(mask[yy,x:x+w])
        if len(visible):contour_edges.append(x+int(visible[0]))
    if contour_edges:
        first=float(np.median(contour_edges))
        if .7*spacing<seams[0]-first<1.3*spacing:seams.insert(0,first)
    last=float(x+w-1)
    if .7*spacing<last-seams[-1]<1.3*spacing:seams.append(last)
    found=[]
    for l,r in zip(seams,seams[1:]):
        if not .7*spacing<r-l<1.35*spacing:continue
        a=plane([l,fold],TABLE+HEIGHT);b=plane([r,fold],TABLE+HEIGHT)
        bottom_a=a.copy();bottom_a[2]=TABLE
        bottom_b=b.copy();bottom_b[2]=TABLE
        q=np.array([[l,fold],[r,fold],project(bottom_b),project(bottom_a)])
        crop=quad_crop(rgb,q)
        f=feature(crop)
        if f is None:continue
        center=(a+b)/2;center[1]+=THICK/2;center[2]=TABLE+HEIGHT/2
        found.append({'quad':q,'crop':crop,'feature':f,'xyz':center})
    return found,{'bbox':[x,y,w,h],'fold':fold,'seams':seams,'spacing':spacing}

def flat_faces(rgb,row_info):
    if row_info is None:return []
    mask=white_mask(rgb)
    cs,_=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    out=[]
    for c in cs:
        a=cv2.contourArea(c);x,y,w,h=cv2.boundingRect(c)
        if not 140<a<1800 or min(w,h)<10 or max(w,h)/min(w,h)>2.8:continue
        if y+h>=row_info['bbox'][1]-8:continue
        rect=cv2.minAreaRect(c);box=cv2.boxPoints(rect)
        s=box.sum(1);d=np.diff(box,axis=1).ravel()
        q=np.array([box[s.argmin()],box[d.argmin()],box[s.argmax()],box[d.argmax()]])
        # Canonical long edge maps to vertical tile dimension in table plane.
        xyzs=np.array([plane(uv,TABLE+THICK) for uv in q])
        if np.linalg.norm(xyzs[1]-xyzs[0])>np.linalg.norm(xyzs[2]-xyzs[1]):q=np.roll(q,1,axis=0)
        crop=quad_crop(rgb,q);f=feature(crop)
        if f is not None:out.append({'quad':q,'crop':crop,'feature':f,'center':np.mean(q,axis=0),'xyz':plane(np.mean(q,axis=0),TABLE+THICK)})
    return out

def group_match(fs, target):
    scores=np.array([similarity(f,target) for f in fs])
    eligible=np.flatnonzero(scores>.64).tolist()
    pair_cache={}
    def pair(i,j):
        if (i,j) not in pair_cache:pair_cache[i,j]=similarity(fs[i],fs[j])
        return pair_cache[i,j]
    groups=[]
    for ids in itertools.combinations(eligible,3):
        if min(pair(i,j) for i,j in itertools.combinations(ids,2))<.64:continue
        groups.append((float(scores[list(ids)].mean()),ids))
    groups.sort(reverse=True)
    if not groups:return None
    best,ids=groups[0]
    alternatives=[score for score,other in groups[1:] if set(other).isdisjoint(ids)]
    gap=best-max(alternatives,default=0.)
    if gap<=.08:return None
    return {'scores':scores,'matches':list(ids),'gap':gap,'valid':True}

def _printed_parts(crop, separate=False):
    """Mahjong fallback for flat/front lighting and small printed numerals.

    The original descriptor remains the first choice. Relative paper brightness
    prevents the dim upright face from losing neutral ink. When parts are
    separate, each printed color is registered independently, so a shared red
    glyph cannot hide a different small black numeral. Whole-face registration
    preserves the blue border outline. These descriptors use observed pixels.
    """
    trim=9 if separate else 7
    hsv=cv2.cvtColor(crop,cv2.COLOR_RGB2HSV).astype(float)
    paper=float(np.percentile(hsv[trim:-trim,trim:-trim,2],90))
    colored=(hsv[:,:,1]>25)&(hsv[:,:,2]>paper*.35)
    black=(hsv[:,:,2]<paper*(.88 if separate else .72))&(hsv[:,:,1]<100)
    masks=[colored&((hsv[:,:,0]<18)|(hsv[:,:,0]>=165)),
           colored&(hsv[:,:,0]>=28)&(hsv[:,:,0]<90),
           colored&(hsv[:,:,0]>=90)&(hsv[:,:,0]<145),black]
    for mask in masks:
        mask[:trim]=False;mask[-trim:]=False
        mask[:,:trim]=False;mask[:,-trim:]=False
    common=np.logical_or.reduce(masks)
    if np.count_nonzero(common)<8:return None
    features=[]
    for mask in masks:
        extent=mask if separate else common
        ys,xs=np.where(extent)
        if np.count_nonzero(mask)<8:
            features.append(np.zeros((26,18),np.float32));continue
        part=mask[ys.min():ys.max()+1,xs.min():xs.max()+1].astype(np.float32)
        part=cv2.resize(part,(18,26),interpolation=cv2.INTER_AREA)
        # Match the resolution of the low, oblique upright face; retain all
        # colors rather than relaxing the match or ambiguity requirements.
        features.append(cv2.GaussianBlur(part,(0,0),1.5 if separate else 1.))
    return balanced(np.stack(features))

def _printed_parts_match(own,face,separate):
    fs=[_printed_parts(o['crop'],separate) for o in own]
    target=_printed_parts(face['crop'],separate)
    scores=np.array([similarity(f,target) for f in fs])
    rank=np.argsort(-scores)
    if len(rank)<4:return None
    ids=rank[:3].tolist()
    gap=float(scores[rank[2]]-scores[rank[3]])
    if scores[rank[2]]<=.64 or gap<=.08:return None
    if min(similarity(fs[i],fs[j]) for i,j in itertools.combinations(ids,2))<=.64:return None
    return {'scores':scores,'matches':ids,'gap':gap,'valid':True}

def _relative_ink(crop):
    """Register all ink together, preserving numeral/glyph relationships.

    RGB deficits relative to the observed paper retain weak, blurred neutral
    strokes that hard black segmentation can lose. Registering each color
    independently is useful for lighting, but can confuse small numerals if it
    discards their position relative to a shared colored glyph.
    """
    image=crop[9:-9,9:-9].astype(np.float32)
    paper=np.percentile(image.reshape(-1,3),90,axis=0)
    deficit=np.clip(1-image/np.maximum(paper,1),0,1)
    printed=np.max(deficit,axis=2)>.08
    ys,xs=np.where(printed)
    if len(xs)<8:return None
    values=np.maximum(deficit-.08,0)[ys.min():ys.max()+1,xs.min():xs.max()+1]
    return cv2.resize(values,(18,26),interpolation=cv2.INTER_AREA).transpose(2,0,1)

def _relative_ink_match(own,face):
    fs=[_relative_ink(o['crop']) for o in own]
    target=_relative_ink(face['crop'])
    scores=np.array([similarity(f,target) for f in fs]);rank=np.argsort(-scores)
    if len(rank)<6:return None
    ids=rank[:3].tolist()
    if scores[rank[2]]<=.64:return None
    if min(similarity(fs[i],fs[j]) for i,j in itertools.combinations(ids,2))<=.64:return None
    # Compare complete three-tile groups, including the closest remaining
    # group even when its individual scores are below the acceptance gate.
    # The original group matcher also uses a mean-group ambiguity margin.
    gap=float(scores[rank[:3]].mean()-scores[rank[3:6]].mean())
    if gap<=.08:return None
    return {'scores':scores,'matches':ids,'gap':gap,'valid':True}

def _raster_spatial_feature(crop):
    """Retain shared ink positions while tolerating subpixel rasterization.

    Color-channel presence is established before smoothing. The low-resolution
    upright faces require a small registration filter, but no color may vanish
    just because its norm decreases under that filter.
    """
    f=feature(crop,adaptive=True)
    if f is None:return None
    f=balanced(f)
    smooth=np.stack([cv2.GaussianBlur(channel,(0,0),.5) for channel in f])
    norms=np.linalg.norm(smooth,axis=(1,2),keepdims=True)
    return smooth/np.maximum(norms,1e-6)

def _raster_spatial_match(own,face):
    fs=[_raster_spatial_feature(o['crop']) for o in own]
    target=_raster_spatial_feature(face['crop'])
    scores=np.array([similarity(f,target) for f in fs]);rank=np.argsort(-scores)
    if len(rank)<4:return None
    ids=rank[:3].tolist();gap=float(scores[rank[2]]-scores[rank[3]])
    if scores[rank[2]]<=.64 or gap<=.08:return None
    if min(similarity(fs[i],fs[j]) for i,j in itertools.combinations(ids,2))<=.64:return None
    return {'scores':scores,'matches':ids,'gap':gap,'valid':True}

class Reader:
    def __init__(self):
        self.initial=None;self.last=None;self.stable=0;self.binding=None;self.pending_key=None;self.pending_center=None;self.pending_ticks=0
        self.observations=0
    def observe(self,rgb):
        self.observations+=1
        own,info=row_instances(rgb);flat=flat_faces(rgb,info)
        if self.initial is None:self.initial=rgb.copy()
        small=cv2.resize(rgb,(40,30),interpolation=cv2.INTER_AREA).astype(float)
        motion=100 if self.last is None else float(np.abs(small-self.last).mean())
        self.stable=self.stable+1 if motion<.8 else 0;self.last=small
        candidates=[]
        for f in flat:
            q=f['quad'];m=np.zeros(rgb.shape[:2],np.uint8);cv2.fillConvexPoly(m,np.int32(q),1)
            change=float(np.abs(rgb.astype(float)-self.initial.astype(float))[m>0].mean())
            scores=np.array([similarity(o['feature'],f['feature']) for o in own]);rank=np.argsort(-scores)
            gap=float(scores[rank[2]]-scores[rank[3]]) if len(rank)>3 else 0
            matching=rank[:3].tolist()
            confidence= len(rank)>3 and scores[rank[2]]>.64 and gap>.08
            candidates.append({'face':f,'change':change,'scores':scores,'matches':matching,'gap':gap,'valid':bool(change>18 and confidence)})
        good=[c for c in candidates if c['valid']]
        if not good:
            # Preserve the successful descriptor; retry unresolved RGB only.
            alternate=[feature(o['crop'],adaptive=True) for o in own]
            for c in candidates:
                scores=np.array([similarity(f,feature(c['face']['crop'],adaptive=True)) for f in alternate])
                rank=np.argsort(-scores)
                gap=float(scores[rank[2]]-scores[rank[3]]) if len(rank)>3 else 0
                valid=bool(c['change']>18 and len(rank)>3 and scores[rank[2]]>.64 and gap>.08)
                if valid:c.update(scores=scores,matches=rank[:3].tolist(),gap=gap,valid=True,descriptor='adaptive_black')
            good=[c for c in candidates if c['valid']]

        if not good:
            alternate=[balanced(feature(o['crop'],adaptive=True)) for o in own]
            for c in candidates:
                target=balanced(feature(c['face']['crop'],adaptive=True))
                scores=np.array([similarity(f,target) for f in alternate]);rank=np.argsort(-scores)
                gap=float(scores[rank[2]]-scores[rank[3]]) if len(rank)>3 else 0
                valid=bool(c['change']>18 and len(rank)>3 and scores[rank[2]]>.64 and gap>.08)
                if valid:c.update(scores=scores,matches=rank[:3].tolist(),gap=gap,valid=True,descriptor='balanced_channels')
            good=[c for c in candidates if c['valid']]
        # Explicitly compare three independently observed, mutually similar tiles.
        # Adaptive evidence may correct a falsely confident red-only match.
        if self.stable>=8:
            for c in candidates:
                if c['change']<=18:continue
                for mode in ('adaptive','original','balanced'):
                    fs=[feature(o['crop'],adaptive=mode!='original') for o in own]
                    target=feature(c['face']['crop'],adaptive=mode!='original')
                    if mode=='balanced':fs=[balanced(f) for f in fs];target=balanced(target)
                    result=group_match(fs,target)
                    if result is not None:
                        c.update(result,descriptor='group_'+mode)
                        break
            good=[c for c in candidates if c['valid']]
        # Give the original measurements a full observation window first.
        # The fallback must not change a binding already supported by them.
        if not good and self.stable>=8 and self.observations>=70:
            for c in candidates:
                if c['change']<=18:continue
                structure=_relative_ink_match(own,c['face'])
                if structure is not None:
                    c.update(structure,descriptor='relative_ink_structure')
                    continue
                results=[_printed_parts_match(own,c['face'],separate)
                         for separate in (False,True)]
                accepted=[(separate,result) for separate,result in zip((False,True),results)
                          if result is not None]
                # If two independent registrations confidently disagree, wait
                # for another observation rather than choosing a descriptor.
                keys={tuple(sorted(result['matches'])) for _,result in accepted}
                if len(keys)!=1:continue
                # Independent color crops can discard a small numeral's
                # arrangement. Even below the full-structure confidence gate,
                # they must agree with its closest observed three-tile group.
                whole_target=_relative_ink(c['face']['crop'])
                whole_scores=np.array([similarity(_relative_ink(o['crop']),whole_target)
                                       for o in own])
                whole_key=tuple(sorted(np.argsort(-whole_scores)[:3].tolist()))
                if whole_key not in keys:
                    # Weak whole-RGB evidence can mix near-equal red/black
                    # groups. Parts must then agree with a second confident
                    # descriptor that retains the numeral/glyph arrangement.
                    # The strict similarity, ambiguity and pair gates remain.
                    spatial=_raster_spatial_match(own,c['face'])
                    if spatial is None or tuple(sorted(spatial['matches'])) not in keys:continue
                separate,result=max(accepted,key=lambda item:item[1]['gap'])
                c.update(result,descriptor='printed_parts' if separate else 'printed_outline')
            good=[c for c in candidates if c['valid']]
        binding=None
        if len(good)==1 and self.stable>=8:
            c=good[0];key=tuple(sorted(c['matches']));center=c['face']['center']
            same=key==self.pending_key and self.pending_center is not None and np.linalg.norm(center-self.pending_center)<1.
            self.pending_ticks=self.pending_ticks+1 if same else 1
            self.pending_key=key;self.pending_center=center.copy()
            if self.pending_ticks>=4:
                binding={'own':own,'targets':[own[i] for i in c['matches']],'discard':c['face'],'matches':c['matches']}
        else:
            self.pending_key=None;self.pending_center=None;self.pending_ticks=0
        return own,info,candidates,binding

def draw(rgb,own,info,candidates,binding):
    canvas=rgb.copy()
    for i,o in enumerate(own):
        color=(0,255,255) if binding and i in binding['matches'] else (255,70,40)
        cv2.polylines(canvas,[np.int32(o['quad'])],True,color,1)
        cv2.putText(canvas,str(i),tuple(np.int32(o['quad'][0])+[0,-3]),cv2.FONT_HERSHEY_SIMPLEX,.35,color,1)
    for c in candidates:
        cv2.polylines(canvas,[np.int32(c['face']['quad'])],True,(0,255,0),1)
    return canvas

def _stock_stacked_support(rgb,face):
    """Confirm two visible tile-body layers beneath a complete measured back.

    A reflected dark patch on the other stock back can break its white contour.
    The grasped face still has to pass every original top-plane measurement.
    Two white side-face layers, separated by green rims at the public tile
    thickness, provide an independent visible stack measurement. A loose blank
    on the tabletop does not expose this four-band support pattern.
    """
    xy=np.array([plane(uv,TABLE+2*THICK)[:2] for uv in face['quad']])
    edges=np.roll(xy,-1,axis=0)-xy
    long_edge=edges[int(np.argmax(np.linalg.norm(edges,axis=1)))]
    along=long_edge/max(np.linalg.norm(long_edge),1e-6)
    toward=np.array([-along[1],along[0]])
    center=np.asarray(face['xyz'],float)
    if np.dot(toward,CAM[:2]-center[:2])<0:toward=-toward
    front=center[:2]+toward*WIDTH/2
    hsv=cv2.cvtColor(rgb,cv2.COLOR_RGB2HSV)
    white=(hsv[:,:,1]<75)&(hsv[:,:,2]>135)
    green=(hsv[:,:,0]>30)&(hsv[:,:,0]<90)&(hsv[:,:,1]>65)
    evidence={}
    for name,height,mask,rim in [('lower_white',.5,white,False),
                                 ('middle_green',1.,green,True),
                                 ('upper_white',1.5,white,False),
                                 ('top_green',2.,green,True)]:
        observed=[]
        for distance in np.linspace(-.018,.018,5):
            point=np.r_[front+distance*along,TABLE+height*THICK]
            u,v=np.rint(project(point)).astype(int)
            if not (1<=u<rgb.shape[1]-1 and 1<=v<rgb.shape[0]-1):return None
            # A one-pixel vertical registration window resolves rasterization
            # of thin green rims; white side faces must occupy the full patch.
            value=bool(mask[v-1:v+2,u].any()) if rim else float(mask[v-1:v+2,u-1:u+2].mean())>=.8
            observed.append(value)
        evidence[name]=float(np.mean(observed))
        if evidence[name]<.8:return None
    return evidence

def declaration_stock(rgb,row_info):
    """Visible face-down stock: green-rimmed blank planar faces, nearest own row.
    This is a spatial role from the V3 declaration procedure, not a tile identity.
    The face is occluded initially; post-lift RGB must establish physical carrying.
    """
    if row_info is None:return None
    hsv=cv2.cvtColor(rgb,cv2.COLOR_RGB2HSV)
    white=white_mask(rgb);cs,_=cv2.findContours(white,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    found=[]
    for c in cs:
        area=cv2.contourArea(c);x,y,w,h=cv2.boundingRect(c)
        if not 150<area<900 or min(w,h)<10 or not 1.3<w/h<3.5 or y+h>=row_info['bbox'][1]:continue
        inner=np.zeros(white.shape,np.uint8);cv2.drawContours(inner,[c],-1,1,-1)
        ring=(cv2.dilate(inner,np.ones((7,7),np.uint8))-inner)>0
        green=(hsv[:,:,0]>30)&(hsv[:,:,0]<90)&(hsv[:,:,1]>65)
        ratio=float(green[ring].mean())
        if ratio<.40:continue
        # Reject side panels: a top back has a large unprinted white interior.
        er=cv2.erode(inner,np.ones((5,5),np.uint8))>0
        if not er.any() or float((hsv[:,:,1][er]<25).mean())<.8:continue
        hull=cv2.convexHull(c)
        q=cv2.approxPolyDP(hull,.025*cv2.arcLength(hull,True),True).reshape(-1,2).astype(float)
        if len(q)!=4:continue
        xy=np.array([plane(uv,TABLE+2*THICK)[:2] for uv in q])
        edges=np.roll(xy,-1,axis=0)-xy
        lengths=np.linalg.norm(edges,axis=1)
        units=edges/np.maximum(lengths[:,None],1e-6)
        # A side face fails horizontal-plane orthogonality; clipped tops fail
        # opposite-edge agreement. Dimensions allow the inset white back.
        if np.max(np.abs(np.sum(units*np.roll(units,-1,axis=0),axis=1)))>.22:continue
        if max(abs(lengths[0]-lengths[2]),abs(lengths[1]-lengths[3]))>.007:continue
        short,long=sorted([float((lengths[0]+lengths[2])/2),float((lengths[1]+lengths[3])/2)])
        if not .025<short<.043 or not .043<long<.061:continue
        # Center of the planar quadrilateral, projected back to the image.
        center_xyz=np.r_[xy.mean(axis=0),TABLE+2*THICK]
        found.append({'quad':q,'center':project(center_xyz),'xyz':center_xyz,'ring_green':ratio,
                      'top_edge_lengths':lengths,'top_orthogonality':float(np.max(np.abs(np.sum(units*np.roll(units,-1,axis=0),axis=1))))})
    # Stock is a visibly paired/stacked arrangement; do not grasp an arbitrary blank.
    paired=[f for f in found if any(f is not g and np.linalg.norm(f['center']-g['center'])<65 for g in found)]
    if not paired:
        # Keep the complete top geometry and establish the stacked role using
        # the currently visible support, rather than an obscured neighbor back.
        for face in found:
            support=_stock_stacked_support(rgb,face)
            if support is not None:
                face['stacked_support_rgb']=support
                paired.append(face)
    if not paired:return None
    f=max(paired,key=lambda o:o['center'][1])
    return f


def board_ray(u, v, z):
    f = 640 * 10 / 22.212
    up = -(v - 240) / f
    d = (z - 1.308) / (0.5 * up - 0.866025403784)
    return np.array([d * (u - 320) / f, -0.41 + d * (0.866025403784 * up + 0.5), z])

def perceive(im):
    hsv = cv2.cvtColor(im, cv2.COLOR_RGB2HSV)
    yellow = cv2.inRange(hsv, np.array([18, 65, 55]), np.array([42, 255, 255]))
    dark = cv2.inRange(hsv, np.array([0, 0, 0]), np.array([179, 255, 65]))
    dark = cv2.morphologyEx(dark, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    contours, _ = cv2.findContours(dark, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if 65 < w < 230 and 45 < h < 180 and (140 < x < 400) and (100 < y < 330):
            candidates.append((cv2.contourArea(c), c))
    if not candidates:
        return None
    c = max(candidates, key=lambda a: a[0])[1]
    poly = cv2.approxPolyDP(c, 0.035 * cv2.arcLength(c, True), True).reshape(-1, 2)
    if len(poly) != 4:
        return None
    s = poly.sum(1)
    d = np.diff(poly, axis=1).ravel()
    corners = np.float32([poly[s.argmin()], poly[d.argmin()], poly[s.argmax()], poly[d.argmax()]])
    H = cv2.getPerspectiveTransform(np.float32([[0, 0], [3, 0], [3, 3], [0, 3]]), corners)
    uv = cv2.perspectiveTransform(np.float32([[[j + 0.5, i + 0.5] for i in range(3) for j in range(3)]]), H)[0]
    cells = [board_ray(*pt, BOARD_SURFACE_Z) for pt in uv]
    warp = cv2.getPerspectiveTransform(corners, np.float32([[0, 0], [120, 0], [120, 120], [0, 120]]))
    board = cv2.warpPerspective(hsv, warp, (120, 120))
    ink = cv2.inRange(board, np.array([18, 65, 20]), np.array([42, 255, 255]))
    occupied = [bool(np.count_nonzero(ink[r * 40 + 10:r * 40 + 30, c * 40 + 10:c * 40 + 30]) > 20) for r in range(3) for c in range(3)]
    gray = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
    bottom = int(corners[:, 1].max()) + 8
    gray[:bottom] = 0
    gray[min(im.shape[0], bottom + 95):] = 0
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 22, param1=60, param2=16, minRadius=8, maxRadius=17)
    rings = []
    shadow_yellow = cv2.inRange(hsv, np.array([18, 55, 20]), np.array([45, 255, 255]))
    if circles is not None:
        for u, v, r in circles[0]:
            if v - r < bottom or v + r > bottom + 95:
                continue
            x0, x1 = (max(0, int(u - r - 2)), min(im.shape[1], int(u + r + 3)))
            y0, y1 = (max(0, int(v - r - 2)), min(im.shape[0], int(v + r + 3)))
            yy, xx = np.mgrid[y0:y1, x0:x1]
            radial = np.sqrt((xx - u) ** 2 + (yy - v) ** 2)
            support = np.count_nonzero((shadow_yellow[y0:y1, x0:x1] > 0) & (radial > 0.6 * r) & (radial < 1.1 * r))
            if support >= 20:
                rings.append([float(u), float(v)])
    # Hough gradients can lock onto one illuminated side of a hollow ring.
    # Refine against its colored outer ellipse; the real probe showed a 4 px
    # bias on the leftmost ring, enough to degrade a 50 mm pinch grasp.
    stock_mask = shadow_yellow.copy()
    stock_mask[:bottom] = 0
    stock_mask[min(im.shape[0], bottom+95):] = 0
    stock_contours, _ = cv2.findContours(stock_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    ellipses = []
    for contour in stock_contours:
        x, y, width, height = cv2.boundingRect(contour)
        if len(contour) < 5 or not (15 < width < 40 and 15 < height < 40):
            continue
        if cv2.contourArea(contour) < 120:
            continue
        center, axes, _ = cv2.fitEllipse(contour)
        if min(axes) < 15 or max(axes) > 40 or max(axes)/min(axes) > 1.6:
            continue
        ellipses.append(list(map(float, center)))
    if ellipses:
        refined = []
        for pixel in rings:
            near = [point for point in ellipses if np.linalg.norm(np.asarray(point)-pixel) < 9]
            refined.append(min(near, key=lambda point: np.linalg.norm(np.asarray(point)-pixel)) if near else pixel)
        for point in ellipses:
            if all(np.linalg.norm(np.asarray(point)-pixel) > 12 for pixel in refined):
                refined.append(point)
        rings = refined
    x0 = max(0, int(corners[:, 0].min()))
    x1 = min(im.shape[1], int(corners[:, 0].max()) + 1)
    y0 = max(0, int(corners[:, 1].min()) - 25)
    y1 = min(im.shape[0], int(corners[:, 1].max()) + 1)
    roi = hsv[y0:y1, x0:x1]
    robot_fraction = float(np.mean((roi[:, :, 1] < 65) & (roi[:, :, 2] > 110)))
    return dict(board_clear=robot_fraction < 0.03, robot_fraction=robot_fraction, corners=corners.tolist(), uv=uv.tolist(), cells=[x.tolist() for x in cells], occupied=occupied, rings=sorted(rings))
