"""Unknown rigid boxes fitted to calibrated RGB silhouettes, without a table prior."""
from itertools import combinations, product
import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
from . import GeometryUnavailable
from .multiview import triangulate, _boundary_distance
from .primitives import fixed_boundary


def image_clipped_polygon(poly, width, height):
    """Clip a projected silhouette at image edges without quantizing derivatives."""
    poly=np.asarray(poly,float)
    for axis,bound,sign in [(0,0.,1),(0,width-1.,-1),(1,0.,1),(1,height-1.,-1)]:
        if len(poly)==0:break
        output=[]
        for a,b in zip(poly,np.roll(poly,-1,axis=0)):
            a_inside=sign*(a[axis]-bound)>=0;b_inside=sign*(b[axis]-bound)>=0
            if a_inside:output.append(a)
            if a_inside!=b_inside:
                output.append(a+(b-a)*((bound-a[axis])/(b[axis]-a[axis])))
        poly=np.asarray(output,float).reshape(-1,2)
    if len(poly)<3:raise GeometryUnavailable('Projected box is outside the image')
    return poly


def fit_box_views(views, *, previous_rotation=None, measured_dimensions=None, previous_center=None, max_rms_px=2.):
    """Views contain Camera and visible contour; optional dimensions come from this episode.

    New objects are initialized upright, with unknown XYZ, yaw and all three
    dimensions. A previously measured rigid box may rotate freely during tracking.
    """
    if not np.isfinite(max_rms_px) or max_rms_px<=0:
        raise GeometryUnavailable('Positive finite image-space fit tolerance required')
    if len(views)<2 and not (len(views)==1 and previous_rotation is not None and measured_dimensions is not None and previous_center is not None):
        raise GeometryUnavailable('New box geometry needs stereo; monocular tracking requires an observed episode shape and pose')
    prepared=[]
    for camera, contour in views:
        contour=np.asarray(contour,float).reshape(-1,2)
        if len(contour)<3 or not np.isfinite(contour).all():continue
        hull=cv2.convexHull(contour.astype('float32')).reshape(-1,2).astype(float)
        if cv2.contourArea(hull.astype('float32'))<8:continue
        simplified=cv2.approxPolyDP(hull.astype('float32'),.5,True).reshape(-1,2).astype(float)
        if len(simplified)>=3:hull=simplified
        prepared.append((camera,hull,fixed_boundary(hull),hull.mean(0)))
    if not prepared:raise GeometryUnavailable('No resolved box silhouettes')
    centers=[]
    for a,b in combinations(prepared,2):
        try:point,_=triangulate(a[0],b[0],a[3],b[3],max_error=8.);centers.append(point)
        except GeometryUnavailable:continue
    if previous_center is not None:
        center=np.asarray(previous_center,float)
        if center.shape!=(3,) or not np.isfinite(center).all():
            raise GeometryUnavailable('Invalid previous observed box center')
    elif centers:
        center=np.median(centers,axis=0)
    else:
        raise GeometryUnavailable('No consistent box correspondences')
    observed_sizes=[]
    for camera,hull,_,_ in prepared:
        depth=camera.project([center])[1][0]
        observed_sizes.extend(np.ptp(hull,axis=0)*depth/np.array([camera.k[0,0],camera.k[1,1]]))
    scale=float(np.median(observed_sizes))
    if not np.isfinite(scale) or scale<=0:
        raise GeometryUnavailable('No measurable visible box size')
    fixed=None if measured_dimensions is None else np.asarray(measured_dimensions,float)
    if fixed is not None and (fixed.shape!=(3,) or not np.isfinite(fixed).all() or np.any(fixed<=0)):
        raise GeometryUnavailable('Invalid dimensions from episode observation')
    free_rotation=previous_rotation is not None
    prior=np.eye(3) if not free_rotation else np.asarray(previous_rotation,float)
    unit=np.asarray(list(product([-.5,.5],repeat=3)))
    angle_count=3 if free_rotation else 1
    def decode(values):
        rotation=(Rotation.from_rotvec(values[3:6]).as_matrix()@prior if free_rotation
                  else Rotation.from_euler('z',values[3]).as_matrix())
        dimensions=np.exp(values[3+angle_count:]) if fixed is None else fixed
        return values[:3],rotation,dimensions
    def residual(values):
        position,rotation,dimensions=decode(values)
        points=(unit*dimensions)@rotation.T+position
        errors=[]
        for camera,hull,boundary,_ in prepared:
            pixels=camera.project(points)[0]
            indices=cv2.convexHull(pixels.astype('float32'),returnPoints=False).ravel()
            poly=image_clipped_polygon(pixels[indices],camera.width,camera.height)
            errors.extend(_boundary_distance(boundary,poly))
            errors.extend(_boundary_distance(fixed_boundary(poly),hull))
            # Corners constrain depth better than unsigned edge distances alone.
            # The term is used only for a resolved low-complexity contour.
            if len(hull)<=6:
                if len(poly)==len(hull):
                    matched=min((np.roll(poly,shift,axis=0) for shift in range(len(poly))),
                                key=lambda proposal:np.sum((proposal-hull)**2))
                else:
                    matched=poly[np.argmin(np.linalg.norm(hull[:,None]-poly[None],axis=-1),axis=1)]
                errors.extend((.5*(matched-hull)).ravel())
        return np.asarray(errors)
    size_seed=np.log(np.full(3,scale*.7)) if fixed is None else np.empty(0)
    size_low=np.log(np.full(3,scale*.08)) if fixed is None else np.empty(0)
    size_high=np.log(np.full(3,scale*3)) if fixed is None else np.empty(0)
    lower=np.r_[center-scale*3,np.full(angle_count,-np.pi),size_low]
    upper=np.r_[center+scale*3,np.full(angle_count,np.pi),size_high]
    seeds=[np.zeros(3)] if free_rotation else [np.array([yaw]) for yaw in np.linspace(0,np.pi/2,4,endpoint=False)]
    seeds=[np.r_[center,angles,size_seed] for angles in seeds]
    if free_rotation and fixed is not None:
        pnp_seeds=[]
        local=unit*fixed
        for camera,hull,_,_ in prepared:
            if np.any(hull[:,0]<=1) or np.any(hull[:,1]<=1) or np.any(hull[:,0]>=camera.width-2) or np.any(hull[:,1]>=camera.height-2):continue
            # Small pose changes can change which corners form the silhouette.
            # Enumerate nearby visibility patterns; dimensions remain episode measurements.
            patterns=set()
            for delta in product([-.2,0.,.2],repeat=3):
                rotation=Rotation.from_rotvec(delta).as_matrix()@prior
                projected=camera.project(local@rotation.T+center)[0]
                indices=tuple(cv2.convexHull(projected.astype('float32'),returnPoints=False).ravel())
                if len(indices)==len(hull) and len(indices)>=4:patterns.add(indices)
            optical=np.diag([1.,-1.,-1.])@camera.transform[:3,:3].T
            for pattern in sorted(patterns):
                indices=np.asarray(pattern)
                for shift in range(len(hull)):
                    target=np.ascontiguousarray(np.roll(hull,shift,axis=0))
                    rotation_vector=cv2.Rodrigues(optical@prior)[0]
                    translation=(optical@(center-camera.transform[:3,3])).reshape(3,1)
                    try:
                        ok,rvec,tvec=cv2.solvePnP(local[indices],target,camera.k,None,
                                                 rotation_vector,translation,True,flags=cv2.SOLVEPNP_ITERATIVE)
                    except cv2.error:continue
                    if not ok:continue
                    recovered_rotation=optical.T@cv2.Rodrigues(rvec)[0]
                    recovered_center=camera.transform[:3,3]+optical.T@tvec.ravel()
                    seed=np.r_[recovered_center,Rotation.from_matrix(recovered_rotation@prior.T).as_rotvec()]
                    if np.all(seed>lower) and np.all(seed<upper):pnp_seeds.append(seed)
        if pnp_seeds:
            seeds=sorted(pnp_seeds+seeds,key=lambda seed:np.mean(residual(seed)**2))[:3]
        # A clipped face loses corner correspondences. Camera-facing starts
        # avoid relying on one possibly biased previous monocular rotation.
        for camera,hull,_,_ in prepared:
            clipped=(np.any(hull[:,0]<=1) or np.any(hull[:,1]<=1) or
                     np.any(hull[:,0]>=camera.width-2) or np.any(hull[:,1]>=camera.height-2))
            if not clipped:continue
            for angle in (-.7,-.35,0.,.35,.7):
                candidate=camera.transform[:3,:3]@Rotation.from_euler('z',angle).as_matrix()
                seed=np.r_[center,Rotation.from_matrix(candidate@prior.T).as_rotvec()]
                if np.all(seed>lower) and np.all(seed<upper):seeds.append(seed)
    solutions=[]
    for seed in seeds:
        try:
            fit=least_squares(residual,seed,bounds=(lower,upper),
                              loss='soft_l1',f_scale=.75,max_nfev=300,diff_step=1e-4,x_scale='jac')
        except (GeometryUnavailable,ValueError,FloatingPointError):continue
        if fit.success and np.isfinite(fit.fun).all():solutions.append(fit)
    if not solutions:raise GeometryUnavailable('No converged box silhouette fit')
    best=min(solutions,key=lambda fit:fit.cost)
    rms=float(np.sqrt(np.mean(best.fun**2)))
    if rms>max_rms_px:raise GeometryUnavailable('Box silhouettes are inconsistent or occluded')
    position,rotation,dimensions=decode(best.x)
    return {'center':position,'rotation':rotation,'dimensions':dimensions,'rms_px':rms,
            'source':'current_calibrated_rgb_box_silhouettes'}


def fit_supported_box_pose(camera, contour, plane, dimensions, previous_center, previous_rotation):
    """Track an episode-measured box resting on an episode-observed plane."""
    dimensions=np.asarray(dimensions,float)
    plane=np.asarray(plane,float)
    if dimensions.shape!=(3,) or plane.shape!=(3,) or not np.isfinite(np.r_[dimensions,plane]).all() or np.any(dimensions<=0):
        raise GeometryUnavailable('Finite observed dimensions and support plane required')
    normal=np.r_[-plane[:2],1.];normal/=np.linalg.norm(normal)
    axis=np.array([1.,0.,0.]);axis-=normal*(axis@normal);axis/=np.linalg.norm(axis)
    basis=np.column_stack([axis,np.cross(normal,axis),normal])
    level=plane[2]/np.linalg.norm(np.r_[-plane[:2],1.])+dimensions[2]/2
    center=np.asarray(previous_center,float);previous_rotation=np.asarray(previous_rotation,float)
    local_rotation=basis.T@previous_rotation
    yaw=np.arctan2(local_rotation[1,0],local_rotation[0,0])
    seed=np.r_[center@basis[:,:2],yaw]
    hull=cv2.convexHull(np.asarray(contour,np.float32).reshape(-1,2)).reshape(-1,2).astype(float)
    if len(hull)<3 or cv2.contourArea(hull.astype('float32'))<8:
        raise GeometryUnavailable('Insufficient supported box contour')
    boundary=fixed_boundary(hull)
    local=np.asarray(list(product([-.5,.5],repeat=3)))*dimensions
    def decode(values):
        position=basis[:,:2]@values[:2]+normal*level
        rotation=basis@Rotation.from_euler('z',values[2]).as_matrix()
        pixels=camera.project(local@rotation.T+position)[0]
        poly=pixels[cv2.convexHull(pixels.astype('float32'),returnPoints=False).ravel()]
        return position,rotation,poly
    def residual(values):
        poly=decode(values)[2]
        return np.r_[_boundary_distance(boundary,poly),_boundary_distance(fixed_boundary(poly),hull)]
    radius=float(np.max(dimensions))*3
    fit=least_squares(residual,seed,bounds=(seed-[radius,radius,np.pi],seed+[radius,radius,np.pi]),
                      loss='soft_l1',f_scale=.75,max_nfev=200,diff_step=1e-4)
    rms=float(np.sqrt(np.mean(fit.fun**2)))
    if not fit.success or not np.isfinite(rms) or rms>2:
        raise GeometryUnavailable('Current contour is inconsistent with observed support geometry')
    position,rotation,_=decode(fit.x)
    return {'center':position,'rotation':rotation,'dimensions':dimensions,'rms_px':rms,
            'source':'current_rgb_episode_measured_box_on_observed_support'}
