"""Condition a stereo observation using live hand-eye calibration and episode geometry."""
import cv2
import numpy as np
from scipy.spatial.transform import Rotation
from . import Camera, GeometryUnavailable
from .multiview import projection


def conditioned_wrist_pose(observation, side, target_pose, root_local, axis_local, length):
    target_pose=np.asarray(target_pose,float);root_local=np.asarray(root_local,float);axis_local=np.asarray(axis_local,float)
    if (side not in ('left','right') or target_pose.shape!=(7,) or root_local.shape!=(3,)
            or axis_local.shape!=(3,) or not np.isfinite(np.r_[target_pose,root_local,axis_local,length]).all()
            or length<=0 or np.linalg.norm(axis_local)<1e-8):
        raise GeometryUnavailable('Finite observed attachment and target pose required')
    axis_local=axis_local/np.linalg.norm(axis_local)
    head=Camera.from_observation(observation,'cam_head')
    wrist=Camera.from_observation(observation,f'cam_{side}_wrist')
    ee=np.asarray(observation['state'][f'{side}_ee_pose'],float)
    current=Rotation.from_quat(ee[[4,5,6,3]]).as_matrix()
    mount=current.T@wrist.transform[:3,:3]
    offset=current.T@(wrist.transform[:3,3]-ee[:3])
    nominal=Rotation.from_quat(target_pose[[4,5,6,3]]).as_matrix()
    def condition(rotation):
        transform=np.eye(4);transform[:3,:3]=rotation@mount;transform[:3,3]=target_pose[:3]+rotation@offset
        moved=Camera(wrist.image,wrist.k,transform)
        root=target_pose[:3]+rotation@root_local;cap=root+length*(rotation@axis_local)
        planes=[]
        for camera in (head,moved):
            pixels,_=camera.project([root,cap])
            if np.linalg.norm(pixels[1]-pixels[0])<3 or np.max(np.abs(pixels))>1e6:
                raise GeometryUnavailable('Projected centerline is unresolved')
            if not cv2.clipLine((0,0,camera.width,camera.height),tuple(np.rint(pixels[0]).astype(int)),tuple(np.rint(pixels[1]).astype(int)))[0]:
                raise GeometryUnavailable('Projected centerline is outside the image')
            line=np.cross(np.r_[pixels[0],1],np.r_[pixels[1],1]);plane=projection(camera).T@line
            norm=np.linalg.norm(plane[:3])
            if norm<1e-10:raise GeometryUnavailable('Degenerate projected centerline')
            planes.append(plane[:3]/norm)
        singular=np.linalg.svd(planes,compute_uv=False)
        return float(singular[1]/singular[0])
    try:before=condition(nominal)
    except GeometryUnavailable:before=0.
    if before>=.05:return None
    choices=[]
    for angle in (.15,-.15,.3,-.3):
        for axis in np.eye(3):
            rotation=Rotation.from_rotvec(axis*angle).as_matrix()@nominal
            try:score=condition(rotation)
            except GeometryUnavailable:continue
            if score>=.075 and score>2*before:
                pose=np.r_[target_pose[:3],Rotation.from_matrix(rotation).as_quat()[[3,0,1,2]]]
                choices.append((abs(angle),-score,pose,score))
    if not choices:return None
    _,_,pose,after=min(choices,key=lambda item:item[:2])
    return {'pose':pose.tolist(),'condition_before':before,'condition_after':after,
            'source':'live_camera_calibration_and_episode_attachment'}
