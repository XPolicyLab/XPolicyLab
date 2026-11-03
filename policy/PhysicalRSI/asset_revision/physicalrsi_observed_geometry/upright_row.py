"""Dimensions inferred from a visible upright row, without stored object sizes.

Assumes a straight vertical front, a horizontal top, and a lower silhouette
edge on the observed support. Visible boundaries are approximate; this does
not reconstruct hidden surfaces or certify the assumptions from one view.
"""
import numpy as np
from . import GeometryUnavailable
from .support_projection import support_point


def observe_upright_row(camera, mask, bbox, fold, seams, support_plane):
    mask=np.asarray(mask,bool);plane=np.asarray(support_plane,float)
    seams=np.asarray(seams,float)
    if (mask.shape!=camera.image.shape[:2] or plane.shape!=(3,)
            or not np.isfinite(plane).all() or seams.ndim!=1
            or len(seams)<4 or not np.isfinite(seams).all() or np.any(np.diff(seams)<=0)):
        raise GeometryUnavailable('Calibrated row mask, observed support and ordered seams required')
    x,y,w,h=map(int,bbox);fold=float(fold)
    if x<=0 or y<=0 or x+w>=camera.width or y+h>=camera.height or not y<fold<y+h:
        raise GeometryUnavailable('Complete visible row bounds and internal face fold required')
    origin=camera.transform[:3,3]
    def intersect(pixels,normal,offset):
        pixels=np.asarray(pixels,float)
        rays=camera.unproject(pixels,np.ones(len(pixels)))-origin
        denominator=rays@normal
        if np.any(abs(denominator)<1e-8):raise GeometryUnavailable('Row rays parallel to inferred surface')
        distances=(offset-origin@normal)/denominator
        if np.any(distances<=0):raise GeometryUnavailable('Inferred row surface behind camera')
        return origin+distances[:,None]*rays
    # Use interior columns to avoid confusing the slanted side with the front.
    margin=max(2,int(np.median(np.diff(seams))/2))
    bottom=[]
    for u in range(x+margin,x+w-margin):
        ys=np.flatnonzero(mask[int(np.ceil(fold)):y+h,u])+int(np.ceil(fold))
        if len(ys):bottom.append([u,int(ys[-1])])
    if len(bottom)<8:raise GeometryUnavailable('Insufficient visible row lower edge')
    normal_support=np.r_[-plane[:2],1.]
    base=intersect(bottom,normal_support,plane[2])
    center=base.mean(0)
    _,spread,vh=np.linalg.svd(base[:,:2]-center[:2],full_matrices=False)
    if spread[0]<1e-8:raise GeometryUnavailable('Degenerate row baseline')
    normal=np.r_[-vh[0,1],vh[0,0],0.]
    distances=np.abs((base-center)@normal)
    # Robust fit in pixel-derived metric units, never a known object scale.
    pixel_metric=float(np.median(camera.project(base)[1])/np.sqrt(camera.k[0,0]*camera.k[1,1]))
    selected=distances<=max(2*pixel_metric,float(np.median(distances)+3*np.median(abs(distances-np.median(distances)))))
    if selected.sum()<8:raise GeometryUnavailable('No consistent row lower edge')
    base=base[selected];center=base.mean(0)
    _,_,vh=np.linalg.svd(base[:,:2]-center[:2],full_matrices=False)
    normal=np.r_[-vh[0,1],vh[0,0],0.]
    rms=float(np.sqrt(np.mean(((base-center)@normal)**2)))
    if rms>2*pixel_metric:raise GeometryUnavailable('Row front is not consistent with a straight lower edge')
    top_pixels=np.c_[seams,np.full(len(seams),fold)]
    tops=intersect(top_pixels,normal,center@normal)
    heights=tops[:,2]-(tops[:,:2]@plane[:2]+plane[2])
    height=float(np.median(heights))
    height_variation=float(np.max(abs(heights-height)))
    # First-order pixel sensitivity for the fold; lower-edge fitting adds its
    # own uncertainty. These bounds do not include segmentation/model bias.
    fold_variation=[]
    for delta in (-1.,1.):
        perturbed=intersect(top_pixels+[0,delta],normal,center@normal)
        fold_variation.append(float(np.max(abs(perturbed[:,2]-tops[:,2]))))
    height_uncertainty=2*max(fold_variation)+2*pixel_metric+rms
    if height<=3*height_uncertainty or height_variation>height_uncertainty:
        raise GeometryUnavailable('Row height unresolved at current pixel resolution')
    front_pixels=[];back_pixels=[]
    for left,right in zip(seams,seams[1:]):
        u=int(round((left+right)/2))
        if not x<=u<x+w:continue
        ys=np.flatnonzero(mask[y:int(np.floor(fold)),u])+y
        if len(ys):front_pixels.append([u,fold]);back_pixels.append([u,int(ys[0])])
    if len(front_pixels)<3:raise GeometryUnavailable('Insufficient visible row top faces')
    front=intersect(front_pixels,normal,center@normal)
    back=intersect(back_pixels,np.array([0.,0.,1.]),float(np.median(tops[:,2])))
    signed=(back-front)@normal
    if not (np.all(signed>0) or np.all(signed<0)):
        raise GeometryUnavailable('Inconsistent front/rear ordering')
    thicknesses=abs(signed);thickness=float(np.median(thicknesses))
    widths=np.linalg.norm(np.diff(tops,axis=0),axis=1);width=float(np.median(widths))
    thickness_uncertainty=2*pixel_metric+float(np.median(abs(thicknesses-thickness)))
    if thickness<=3*thickness_uncertainty or width<=6*pixel_metric:
        raise GeometryUnavailable('Row thickness or seam width unresolved')
    return {'height_m':height,'thickness_m':thickness,'width_m':width,
            'height_uncertainty_m':height_uncertainty,'thickness_uncertainty_m':thickness_uncertainty,
            'pixel_metric_m':pixel_metric,'source':'current_RGB_upright_row_and_observed_support',
            'evidence':{'bottom_points':len(base),'bottom_line_rms_m':rms,'top_faces':len(front),
                        'height_variation_m':height_variation,'uses_object_assets':False,
                        'assumptions':['lower_silhouette_on_support','vertical_planar_front','horizontal_top'],
                        'uncertainty_excludes_segmentation_and_shape_bias':True}}
