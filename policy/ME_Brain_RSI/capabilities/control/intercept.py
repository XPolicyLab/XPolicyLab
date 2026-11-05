"""CVP3 future interception and public-geometry approach checks, never scene truth."""
import numpy as np
from scipy.spatial.transform import Rotation
from .geometry import tcp_to_ee,quat_matrix,bounded_pose,pose_matrix,matrix_quat
from .kinematics import SerialArm


def side_pose(xy,z,yaw,calibration,tilt=0.):
    """CPP24: fingers pitched toward +x (about world y). The TCP sits 0.145*sin(tilt) to the
    RIGHT of the EE, so the left arm reaches the held basket's interior from its own side."""
    r=Rotation.from_euler('y',-tilt).as_matrix()@Rotation.from_euler('z',yaw).as_matrix()@Rotation.from_euler('y',np.pi/2).as_matrix()
    return tcp_to_ee([*xy,z],r,calibration['tool_offset_in_ee'])


def down_pose(xy,z,yaw,calibration,tilt=0.):
    r=Rotation.from_euler('x',tilt).as_matrix()@Rotation.from_euler('z',yaw).as_matrix()@Rotation.from_euler('y',np.pi/2).as_matrix()
    return tcp_to_ee([*xy,z],r,calibration['tool_offset_in_ee'])


def grasp_face(basket):
    """Select once; a nearly 45-degree fit must not switch walls mid-grasp."""
    a=basket['yaw'];normals=np.array([[1,0],[-1,0],[0,1],[0,-1]],float)
    r=np.array([[np.cos(a),-np.sin(a)],[np.sin(a),np.cos(a)]])
    return int(np.argmin((normals@r.T)[:,1]))


def front_grasp(basket,face=None):
    a=basket['yaw'];long=np.array([np.cos(a),np.sin(a)]);short=np.array([-np.sin(a),np.cos(a)])
    n,d=[(long,.175),(-long,.175),(short,.118),(-short,.118)][grasp_face(basket) if face is None else face]
    # CPP2: pinch 3 cm deeper than CPP1 (.118): tips ~5 cm below the rim.
    # Public mesh end-wall rim is 0.12746m above bottom; 0.21863m is
    # the raised handle, NOT the end wall. Pinch just below the rim;
    # public finger tips extend another 12.57mm below this TCP.
    return np.asarray(basket['xy'])+n*d,basket['bottom_z']+.088,float(np.arctan2(n[1],n[0])-np.pi/2)


HANDLE_TOP=.21863       # raised central handle top above the basket bottom (public mesh)
HANDLE_PINCH_DEPTH=.035  # bar sits this far toward the palm from the grasp centre: closed fingers hook under it
HANDLE_X_OFFSET=.04      # hold the arch 4 cm toward +x: leaves the basket's left half free for the left-arm drop (arch only ~5 mm lower there)


def handle_grasp(basket,tilt=0.):
    """CPP6: hook the central handle bar; the grasp centre is offset ALONG THE TILTED FINGER AXIS.

    down_pose tilts the gripper about the world x axis, so the finger axis is
    (0, sin t, -cos t) and the closing axis is tilted with it (one finger higher).
    Placing the TCP straight below the bar top (CPP4/5) moved the bar ~4 cm along
    that closing axis, under the upper finger. The TCP must be bar_top + d*finger_axis.
    """
    a=basket['yaw'];long=np.array([np.cos(a),np.sin(a)]);short=np.array([-np.sin(a),np.cos(a)])
    n=-long if long[1]>0 else long
    along=short if short[0]>0 else -short
    bar=np.asarray(basket['xy'])+along*HANDLE_X_OFFSET
    xy=bar+np.array([0.,HANDLE_PINCH_DEPTH*np.sin(tilt)])
    z=basket['bottom_z']+HANDLE_TOP-HANDLE_PINCH_DEPTH*np.cos(tilt)
    return xy,z,float(np.arctan2(n[1],n[0])-np.pi/2)


def hover_z(basket,cal,tilt=0.):
    # The lowest public finger tip, not just the EE/TCP centre, clears the rim.
    return (basket['bottom_z']+basket['height']+.020+
            (cal['contact_tip_offset_in_ee'][0]-cal['tool_offset_in_ee'][0])*np.cos(tilt)+.045*abs(np.sin(tilt)))


def rotation_error(a,b):
    return float(Rotation.from_matrix(quat_matrix(a[3:]).T@quat_matrix(b[3:])).magnitude())


def tool_gap(pose,basket,center):
    offsets=np.array([[x,y,0.] for x in (0.,.086,.15757) for y in (-.045,.045)])
    points=pose[:3]+offsets@quat_matrix(pose[3:]).T
    a=basket['yaw'];r=np.array([[np.cos(a),-np.sin(a)],[np.sin(a),np.cos(a)]])
    local=(points[:,:2]-center)@r
    inside=(np.abs(local[:,0])<.205)&(np.abs(local[:,1])<.145)
    return float((points[inside,2]-(basket['bottom_z']+basket['height'])).min()) if inside.any() else None


def check_approach(arm,current,joints,waypoints,basket,velocity,cal,time_offset=0):
    """Local IK continuation and conservative moving basket/tool envelope.

    Public static model only: not a full mesh/world/self-collision certificate.
    It rejects disconnected pose-only IK branches and nominal tool/rim crossings.
    """
    q=np.asarray(joints,float);pose=np.asarray(current,float);samples=0;min_tip_gap=10.
    initial_xy=np.asarray(basket['xy']);v=np.asarray(velocity)
    for goal in waypoints:
        for _ in range(70):
            if np.linalg.norm(pose[:3]-goal[:3])<1e-6 and rotation_error(pose,goal)<1e-6:break
            pose=bounded_pose(pose,goal,.024,.20);samples+=1
            fit=arm.inverse(pose_matrix(pose),initial=q,restarts=0)
            qnext=np.asarray(fit['joints'])
            if (fit['position_error_m']>.006 or fit['orientation_error_rad']>.08
                    or np.max(np.abs(qnext-q))>.65):return None
            q=qnext
            center=initial_xy+v*min(150,time_offset+2*samples)
            gap=tool_gap(pose,basket,center)
            if gap is not None:
                min_tip_gap=min(min_tip_gap,gap)
                if gap<.018:return None
        else:return None
    return dict(samples=samples,min_nominal_tool_gap=min_tip_gap if min_tip_gap<10 else None,warning='Static local IK/tool envelope only; not full collision or dynamic proof')


def plan_intercept(basket,velocity,obs,cal,step,check=True):
    """Select a future reachable arm/window; never chase from initial side alone."""
    velocity=np.asarray(velocity,float)
    if velocity.shape!=(2,) or not np.isfinite(velocity).all() or np.linalg.norm(velocity)>.009:return None
    candidates=[]
    for horizon in (50,65,80,95,110,125,140,160,180,200):
        future=dict(basket,xy=(np.asarray(basket['xy'])+velocity*horizon).tolist())
        xy,z,yaw=handle_grasp(future)
        if abs(xy[0])>.57 or not -.24<xy[1]<.25:continue  # bar position gate; the tilt offset is applied per candidate
        for side in ('right',):
            if not ((-.58<xy[0]<.10) if side=='left' else (-.10<xy[0]<.58)):continue
            actual=np.asarray(obs['state'][side+'_ee_pose']);arm=SerialArm.public_x5(side)
            q=np.asarray(obs['state'].get(side+'_arm_joint_state',np.zeros(6)))
            # Released arm's zero posture is folded/singular. Use small public
            # FK increments to unfold behind the belt before Cartesian turning.
            escape_q=q.copy();escape_q[1]=.30;escape_q[2]=.50
            escape=[]
            n=max(1,int(np.ceil(np.max(np.abs(escape_q-q))/.055)))
            if n>24:continue
            for fraction in np.linspace(1/n,1,n):
                t=arm.forward(q+(escape_q-q)*fraction)
                escape.append(np.r_[t[:3,3],matrix_quat(t[:3,:3])].tolist())
            escaped=np.asarray(escape[-1])
            escape_duration=sum(max(np.linalg.norm(np.asarray(b)[:3]-np.asarray(a)[:3])/.016,rotation_error(a,b)/.10)
                for a,b in zip([actual.tolist()]+escape[:-1],escape))
            gaps=[tool_gap(np.asarray(p),basket,np.asarray(basket['xy'])+velocity*escape_duration*(i+1)/n)
                  for i,p in enumerate(escape)]
            if any(g is not None and g<.018 for g in gaps):continue
            yaws=sorted((yaw,yaw+np.pi),key=lambda a:rotation_error(actual,down_pose(xy,z,a,cal)))
            for angle,tilt in [(a,t) for a in yaws for t in (np.deg2rad(40),np.deg2rad(50),np.deg2rad(60),np.deg2rad(30))]:
                xy,z,_=handle_grasp(future,tilt)
                # Rotate behind the belt, lift there, then translate above rim.
                base_x=-.3 if side=='left' else .3
                prepare_xy=[base_x+(-.025 if side=='left' else .025),-.30+cal['tool_offset_in_ee'][0]*np.sin(tilt)]
                prepare=down_pose(prepare_xy,hover_z(future,cal,tilt),angle,cal,tilt)
                raised=down_pose(prepare_xy,hover_z(future,cal,tilt),angle,cal,tilt)
                hz=hover_z(future,cal,tilt)
                # CPP9: hover behind the grasp point along the tilted finger axis; the descent then
                # runs along the fingers and the bar stays centred between them (a vertical
                # descent sweeps the bar across the tilted closing axis by dz*sin(tilt)).
                hover=down_pose(xy-np.array([0.,(hz-z)*np.tan(tilt)]),hz,angle,cal,tilt)
                duration=escape_duration+max(np.linalg.norm(escaped[:3]-prepare[:3])/.016,rotation_error(escaped,prepare)/.10)
                duration+=np.linalg.norm(raised[:3]-prepare[:3])/.016+np.linalg.norm(hover[:3]-raised[:3])/.016
                duration+=(hover_z(future,cal,tilt)-z)/.010+12+8
                if duration>horizon+7:continue
                fit=arm.inverse(pose_matrix(hover),initial=q,restarts=3)
                if fit['position_error_m']>.008 or fit['orientation_error_rad']>.08:continue
                # CPP42: a hook far from the right shoulder cannot lift (eval15 s0 ep3: hook x=.05 stalled at 4 cm); prefer x>=.22
                # CPP49: prefer 60 deg (lower EE for the same hook height at the handoff)
                candidates.append((horizon+.1*duration+2*abs(tilt-np.deg2rad(60))+200*max(0.,.22-xy[0]),dict(holder=side,yaw=angle,tilt=tilt,xy=xy.tolist(),grasp_z=z,grasp='handle',
                    grasp_face=grasp_face(basket),anchor_yaw=float(basket['yaw']),
                    prepare=prepare.tolist(),raised=raised.tolist(),hover=hover.tolist(),
                    escape=escape,escape_joints=escape_q.tolist(),escape_duration=float(escape_duration),
                    intercept_step=step+horizon,estimated_steps=float(duration),future_basket=future,
                    original_basket=dict(basket),velocity=velocity.tolist()),escaped,escape_q,arm))
    for _,plan,actual,q,arm in sorted(candidates,key=lambda x:x[0]):
        verified=check_approach(arm,actual,q,[np.asarray(plan[k]) for k in ('prepare','raised','hover')],basket,velocity,cal,time_offset=plan['escape_duration']) if check else {'not_checked':True}
        if verified is not None:plan['approach_check']=verified;return plan
    return None


def preview_envelope_safe(pose,right,over_belt=False):
    """Nominal separated tool envelope; not full collision proof.

    over_belt (CPP39): the tracking standby hovers in the target's lane upstream of the basket
    (x<-.10), 8+ cm above the belt; the separation from the right arm is still required."""
    pose=np.asarray(pose,float);right=np.asarray(right,float)
    if pose.shape!=(7,) or right.shape!=(7,) or not np.isfinite(np.r_[pose,right]).all():return False
    points=pose[:3]+np.array([[x,y,0.] for x in (0.,.086,.15757) for y in (-.045,.045)])@quat_matrix(pose[3:]).T
    if over_belt:
        return bool(points[:,0].max()<-.10 and points[:,0].min()>-.61
                    and points[:,1].max()<.22 and points[:,1].min()>-.55
                    and points[:,2].min()>.845 and points[:,2].max()<1.40
                    and np.linalg.norm(pose[:3]-right[:3])>.28)
    return bool(points[:,0].max()<-.14 and points[:,0].min()>-.61
                and points[:,1].max()<-.17 and points[:,1].min()>-.55
                and points[:,2].min()>.89 and points[:,2].max()<1.40
                and np.linalg.norm(pose[:3]-right[:3])>.28)


def left_preview_path(obs,cal):
    """Bounded public-FK unfold, then IK-checked high open standby lane.

    Requires the robot's actual joint state. No fabricated zero state or
    unrestricted Cartesian rotation from the folded singular configuration.
    """
    q=np.asarray(obs['state'].get('left_arm_joint_state',[]),float)
    current=np.asarray(obs['state']['left_ee_pose']);right=obs['state']['right_ee_pose']
    if q.shape!=(6,) or not np.isfinite(q).all():return None
    arm=SerialArm.public_x5('left');t=arm.forward(q)
    if np.linalg.norm(t[:3,3]-current[:3])>.025:return None
    if not preview_envelope_safe(current,right):return None
    escape=q.copy();escape[1]=.30;escape[2]=.50
    n=max(1,int(np.ceil(np.max(np.abs(escape-q))/.035)))
    if n>30:return None
    path=[]
    for fraction in np.linspace(1/n,1,n):
        t=arm.forward(q+fraction*(escape-q));pose=np.r_[t[:3,3],matrix_quat(t[:3,:3])]
        if not preview_envelope_safe(pose,right):return None
        path.append(pose.tolist())
    q=escape;pose=np.array(path[-1]);goal=down_pose([-.30,-.32],1.07,0.,cal)
    for _ in range(90):
        if np.linalg.norm(pose[:3]-goal[:3])<1e-6 and rotation_error(pose,goal)<1e-6:break
        pose=bounded_pose(pose,goal,.006,.06)
        if not preview_envelope_safe(pose,right):return None
        fit=arm.inverse(pose_matrix(pose),initial=q,restarts=0);new=np.asarray(fit['joints'])
        if fit['position_error_m']>.006 or fit['orientation_error_rad']>.06 or np.max(np.abs(new-q))>.50:return None
        q=new;path.append(pose.tolist())
    else:return None
    # Validate endpoints of the small standby lane before using target-x tracking.
    for x in (-.44,-.22):
        target=down_pose([x,-.32],1.07,0.,cal)
        fit=arm.inverse(pose_matrix(target),initial=q,restarts=0)
        if fit['position_error_m']>.006 or fit['orientation_error_rad']>.06:return None
    return path
