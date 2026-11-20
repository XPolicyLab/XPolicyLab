"""Original resampling, tool geometry and pose-repair bounds; contact width remains unknown."""
import numpy as np
from scipy.spatial.transform import Rotation
from skills.shared.geometry.x5_public_frame_model.implementation import rotation, quaternion, pose, matrix

def wrist_for_tip(tip,q,offset=.155):
    return np.r_[np.asarray(tip,float)-rotation(q)[:,0]*offset,q]


def downward(closing_yaw,tilt=0.,outward=(0.,1.)):
    y=np.array([np.cos(closing_yaw),np.sin(closing_yaw),0.])
    outward=np.asarray(outward,float);outward/=max(np.linalg.norm(outward),1e-8)
    x=np.r_[np.sin(np.deg2rad(tilt))*outward,-np.cos(np.deg2rad(tilt))]
    y-=x*np.dot(x,y);y/=np.linalg.norm(y)
    return quaternion(np.column_stack([x,y,np.cross(x,y)]))


def resample(path,current,max_joint_step=.075,speed=1.8,min_controls=5):
    path=np.vstack([np.asarray(current),np.asarray(path)])
    length=np.r_[0.,np.cumsum(np.max(np.abs(np.diff(path,axis=0)),axis=1))]
    count=max(min_controls,int(np.ceil(length[-1]/max_joint_step)),int(np.ceil((len(path)-1)/(10*speed))))
    if count>250:raise RuntimeError(f'Planned route exceeds per-phase budget: {count}')
    times=np.linspace(0,length[-1],count+1)[1:]
    if length[-1]<1e-8:return np.repeat(path[-1:],min_controls,axis=0)
    return np.stack([np.interp(times,length,path[:,joint]) for joint in range(6)],axis=1)


def lateral_park_pose(frame,arm):
    """Public X5 fingertip clearance at the outer front corner of the table."""
    side=-1. if arm=='left' else 1.
    return wrist_for_tip(np.array([side*.43,-.385,frame.mounting_z+.10]),
                         downward(side*np.pi/2))


def wrist_look_pose(frame,rig,xy,yaw,height=.22,*,normal=True,tilt_deg=0.,openings=None):
    if normal:
        camera_rotation=(Rotation.from_euler('z',yaw).as_matrix() @
                         Rotation.from_euler('x',tilt_deg,degrees=True).as_matrix())
        forward=-camera_rotation[:,2]
    else:
        q=downward(yaw);r=rotation(q)
        forward=r@(-rig.wrist_from_link6[:3,2])
    if forward[2]>=-.2:raise ValueError('Observation optical axis must face the mounting plane')
    aim=np.r_[np.asarray(xy),frame.mounting_z+.019]
    center=aim-forward*(height/-forward[2])
    if normal:
        camera=np.eye(4);camera[:3,:3]=camera_rotation;camera[:3,3]=center
        target=pose(camera@np.linalg.inv(rig.wrist_from_link6))
    else:
        target=np.r_[center-r@rig.wrist_from_link6[:3,3],q]
    transform=matrix(target)
    openings=np.full(2,.044) if openings is None else np.abs(np.asarray(openings))
    clearance=[]
    for index,(origin,axis,triangles) in enumerate(rig.fingers):
        local=triangles@origin[:3,:3].T+origin[:3,3]+axis*openings[index]
        world=local@transform[:3,:3].T+transform[:3,3]
        clearance.append(float(world[...,2].min()-frame.mounting_z))
    if min(clearance)<.07:raise ValueError('Observation public finger mesh clearance below 7 cm')
    return target


def needs_pose_repair(xy,destination,yaw):
    # Public official limits are 20 mm and 45 degrees. Leave 6 mm and
    # 13 degrees for RGB localization error without gratuitous transfers.
    return (np.linalg.norm(np.asarray(xy)-destination)>.014 or
            abs(yaw)>np.deg2rad(32.))


def supported_place_stop(actual,target):
    """Allow only a small upward contact offset, followed by RGB release verification."""
    delta=np.asarray(actual[:3])-np.asarray(target[:3])
    return (np.linalg.norm(delta[:2])<=.004
            and -.001<=delta[2]<=.012 and np.linalg.norm(delta)<=.014)
