"""GP2 RGB instance support. No object labels, layout or depth truth inputs."""
import cv2
import numpy as np


def rgb_components(rgb, box):
    """Separate RGB foreground pieces inside a language-grounded proposal.

    A detection rectangle is NOT an instance mask: it may include a nearby cup
    or gripper. GrabCut separates appearance from tabletop, then semantic crop
    ranking in the caller chooses among disconnected pieces.
    """
    rgb = np.asarray(rgb)
    a,b,c,d = np.rint(box).astype(int)
    a,c = np.clip([a,c],1,rgb.shape[1]-2)
    b,d = np.clip([b,d],1,rgb.shape[0]-2)
    if c-a < 5 or d-b < 5: raise ValueError('Degenerate instance proposal')
    mask = np.zeros(rgb.shape[:2],np.uint8)
    bg,fg = np.zeros((1,65),float),np.zeros((1,65),float)
    cv2.setRNGSeed(0)
    try:
        cv2.grabCut(rgb,mask,(int(a),int(b),int(c-a+1),int(d-b+1)),bg,fg,5,cv2.GC_INIT_WITH_RECT)
    except cv2.error as error:
        raise ValueError('RGB instance segmentation failed') from error
    foreground = np.isin(mask,[cv2.GC_FGD,cv2.GC_PR_FGD]).astype('uint8')
    foreground = cv2.morphologyEx(foreground,cv2.MORPH_OPEN,np.ones((2,2),'uint8'))
    n,labels,stats,centres = cv2.connectedComponentsWithStats(foreground)
    result = []
    for index in range(1,n):
        x,y,w,h,area = stats[index]
        # Tiny high-contrast decorations can outscore an entire object in CLIP.
        # Require nontrivial support relative to the grounded proposal.
        if area < max(20,.04*(c-a)*(d-b)) or min(w,h)<3: continue
        support = labels==index
        yy,xx = np.nonzero(support)
        centrality = np.exp(-.5*(((xx-(a+c)/2)/max(3,.3*(c-a)))**2+
                               ((yy-(b+d)/2)/max(3,.3*(d-b)))**2))
        if float(centrality.max())<.20: continue
        result.append(dict(mask=support,bbox=[int(x),int(y),int(x+w),int(y+h)],
                           area=int(area),centrality=float(centrality.max())))
    if not result: raise ValueError('No isolated RGB target support')
    return result


def masked_crop(rgb, support, box):
    a,b,c,d = np.asarray(box,int)
    a,b=max(0,a-3),max(0,b-3)
    c,d=min(rgb.shape[1],c+3),min(rgb.shape[0],d+3)
    crop = np.asarray(rgb[b:d,a:c]).copy()
    crop[~support[b:d,a:c]] = 128
    return crop


def supported_grasp(z, support, box, calibration, camera):
    """Choose a grasp on the selected RGB instance, not the largest depth blob."""
    yy,xx = np.nonzero(support)
    finite = np.isfinite(z[yy,xx])
    height = z[yy,xx]-calibration['table_z']
    valid = finite & (height>.002) & (height<.22)
    if valid.sum()<12: raise ValueError('Instance has insufficient RGB depth support')
    # Robust object-height estimate. A single dark pixel must not send the
    # gripper below the tabletop; a bright outlier must not lift the estimate.
    surface = calibration['table_z']+float(np.percentile(height[valid],75))
    surface = float(np.clip(surface,calibration['table_z']+.015,calibration['table_z']+.20))
    a,b,c,d = box
    v,u = np.indices(support.shape)
    central = np.exp(-.5*(((u-(a+c)/2)/max(3,.30*(c-a)))**2+
                         ((v-(b+d)/2)/max(3,.30*(d-b)))**2))
    # RGB masks can fill a ring's central hole. Keep the semantic instance but
    # prefer its elevated RGB-estimated surface, never a neighbouring blob.
    elevated = support & np.isfinite(z) & (z>calibration['table_z']+.010) & (z<calibration['table_z']+.22)
    point_support=elevated if int(elevated.sum())>=12 else support
    distance = cv2.distanceTransform(np.pad(point_support.astype('uint8'),1),cv2.DIST_L2,5)[1:-1,1:-1]
    score = np.minimum(distance,10.)*central
    score[~point_support]=0
    row,col = np.unravel_index(score.argmax(),score.shape)
    # Conservative association gate: retain the former depth-interior point
    # when it really belongs to this semantic instance. Replace it only if it
    # falls on clutter/outside support; do not gratuitously move a valid grasp.
    a0,b0,c0,d0=np.rint(box).astype(int)
    a0,c0=np.clip([a0,c0],1,z.shape[1]-2)
    b0,d0=np.clip([b0,d0],1,z.shape[0]-2)
    old=((z[b0:d0+1,a0:c0+1]>calibration['table_z']+.012)&
         (z[b0:d0+1,a0:c0+1]<calibration['table_z']+.24)).astype('uint8')
    old=cv2.morphologyEx(old,cv2.MORPH_OPEN,np.ones((3,3),'uint8'))
    old_distance=cv2.distanceTransform(np.pad(old,1),cv2.DIST_L2,5)[1:-1,1:-1]
    old_row,old_col=np.unravel_index(old_distance.argmax(),old_distance.shape)
    old_row,old_col=int(old_row+b0),int(old_col+a0)
    if old_distance.max()>=2 and support[old_row,old_col] and distance[old_row,old_col]>=1.5:
        row,col=old_row,old_col
    if distance[row,col]<1.5: raise ValueError('No interior RGB grasp point')
    xyz = camera.pixel_to_plane([int(col),int(row)],surface)
    points = np.array([camera.pixel_to_plane([int(x),int(y)],surface)[:2]
                       for x,y in zip(xx[::max(1,len(xx)//300)],yy[::max(1,len(xx)//300)])])
    values,vectors = np.linalg.eigh(np.cov(points,rowvar=False))
    major = vectors[:,-1]
    elongation = float(np.sqrt(max(values[-1],1e-8)/max(values[0],1e-8)))
    yaw = float(np.arctan2(major[1],major[0])) if elongation>1.35 else 0.
    yaw=(yaw+np.pi/2)%np.pi-np.pi/2
    # GP4 thin-object contact: retain 5 mm nominal fingertip/table clearance.
    # GP3's 10 mm clearance can pass above thin tools. This remains an RGB /
    # static-geometry estimate, not an object-specific depth or force reading.
    finger_extra = float(np.linalg.norm(np.asarray(calibration['contact_tip_offset_in_ee'])-
                                       np.asarray(calibration['tool_offset_in_ee'])))
    floor = calibration['table_z']+finger_extra+.005
    contact = max(floor,calibration['table_z']+min(.10,.60*(surface-calibration['table_z'])))
    if not (-.49<xyz[0]<.49 and -.27<xyz[1]<.15): raise ValueError('Instance outside workspace')
    return xyz[:2].tolist(),surface,float(contact),yaw,[int(col),int(row)],elongation
