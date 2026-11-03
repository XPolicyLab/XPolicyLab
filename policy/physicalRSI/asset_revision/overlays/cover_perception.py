"""Cover geometry from current calibrated silhouettes, without asset dimensions."""
import cv2
import numpy as np
from physicalrsi_observed_geometry import Camera, GeometryUnavailable
from physicalrsi_observed_geometry.free_cuboid import fit_upright_cuboid
from physicalrsi_observed_geometry.primitives import fit_supported_primitive
from physicalrsi_observed_geometry.supported_frustum import fit_supported_frustum


def masks(rgb):
    hsv=cv2.cvtColor(np.asarray(rgb,np.uint8),cv2.COLOR_RGB2HSV);h,s,v=cv2.split(hsv)
    ranges={'red':(((h<5)|(h>175))&(s>140)&(v>190)), 'green':((h>=40)&(h<=90)), 'blue':((h>=95)&(h<=135)), 'cup':((h>=20)&(h<40))}
    groups={}
    for kind,sel in ranges.items():
        raw=(sel&(s>75)&(v>85)).astype('uint8')
        contours,_=cv2.findContours(raw,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        contours=[c.reshape(-1,2) for c in contours if cv2.contourArea(c)>(150 if kind=='cup' else 12) and .4 < cv2.boundingRect(c)[2]/cv2.boundingRect(c)[3] < 2.5]
        if kind in ('cup','red') and contours:
            # Preserve the original strong-color detections; grow each only
            # through connected weaker color to recover highlights and shadows.
            # Unseeded weak-color regions do not become new object instances.
            weak_hue=((h<5)|(h>175)) if kind=='red' else sel
            _,labels=cv2.connectedComponents((weak_hue&(s>75/2)&(v>85)).astype('uint8'))
            selected=set()
            for contour in contours:
                seed=np.zeros(raw.shape,'uint8');cv2.fillPoly(seed,[contour.astype('int32')],1)
                selected.update(int(label) for label in np.unique(labels[(seed>0)&(raw>0)]) if label)
            contours=[]
            for label in sorted(selected):
                pieces,_=cv2.findContours((labels==label).astype('uint8'),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
                contours.extend(c.reshape(-1,2) for c in pieces)
        groups[kind]=contours
    return groups


def _silhouette(camera,contour):
    mask=np.zeros(camera.image.shape[:2],'uint8')
    cv2.fillPoly(mask,[np.asarray(contour,'int32')],1)
    return mask.astype(bool)


def _complete(mask):
    return not (mask[0].any() or mask[-1].any() or mask[:,0].any() or mask[:,-1].any())


def initial_scene(observation):
    cameras={name:Camera.from_observation(observation,name) for name in observation['vision']}
    if 'cam_head' not in cameras:raise GeometryUnavailable('Current head calibration required')
    groups={name:masks(camera.image) for name,camera in cameras.items()}
    counts={kind:len(items) for kind,items in groups['cam_head'].items()}
    if counts!={'red':1,'green':1,'blue':1,'cup':3}:
        raise GeometryUnavailable(f'RGB instance counts {counts}')
    observed={}
    for color in ('red','green','blue'):
        views=[]
        for name,camera in cameras.items():
            contours=groups[name][color]
            if len(contours)==1:
                mask=_silhouette(camera,contours[0])
                if _complete(mask):views.append((camera,mask))
        if len(views)>=2:
            observed[color]=fit_upright_cuboid(views)
    if len(observed)<2:
        raise GeometryUnavailable('Two independent multiview block bases required for cover support')
    levels=np.array([fit['bottom_z'] for fit in observed.values()])
    if np.ptp(levels)>.005:
        raise GeometryUnavailable('Observed block bases disagree about horizontal support')
    # Horizontal support is the original controller's operating assumption;
    # its level now comes from independent fitted block bases, not a constant.
    plane=np.array([0.,0.,float(np.mean(levels))])
    support={'normal':np.array([0.,0.,1.]),'offset':-plane[2],
             'source':'independent_calibrated_cuboid_bases','base_spread_m':float(np.ptp(levels))}
    head=cameras['cam_head'];fitted={};blocks={}
    for color in ('red','green','blue'):
        if color in observed:
            fit=observed[color]
            geometry={'pose':fit['center'].tolist(),'dims':fit['dimensions'].tolist(),
                      'rms_px':fit['rms_px'],'source':fit['source']}
        else:
            fit=fit_supported_primitive(head,_silhouette(head,groups['cam_head'][color][0]),plane,'cube')
            geometry={'pose':list(fit['pose']),'dims':list(fit['dims']),
                      'rms_px':fit['rms_px'],'source':'current_silhouette_and_observed_support'}
        geometry['support_plane']=support
        fitted[color]=[geometry];blocks[color]=np.asarray(geometry['pose'][:2])
    fitted['cup']=[]
    for contour in groups['cam_head']['cup']:
        fit=fit_supported_frustum([(head,_silhouette(head,contour))],plane)
        radius=max(fit['bottom_radius'],fit['top_radius'])
        center=fit['base']+fit['rotation'][:,2]*fit['height']/2
        fitted['cup'].append({'pose':center.tolist(),'dims':[2*radius,2*radius,fit['height']],
                              'support_plane':support,'rms_px':fit['rms_px'],
                              'source':fit['source'],'shape_hypothesis':fit['shape_hypothesis']})
    fitted['cup'].sort(key=lambda g:g['pose'][0])
    cups=[np.asarray(g['pose'][:2]) for g in fitted['cup']]
    return blocks,cups,fitted
