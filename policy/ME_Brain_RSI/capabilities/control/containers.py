"""Stateful RGB-driven container manipulation strategies."""
import math
import numpy as np

from ..motion import down_quat, tcp_to_link, interpolate_pose
from .containers_vision import (color_masks, detect_box, detect_organize, detect_pack_at, detect_pack_objects,
                                detect_stack_base, detect_stack_blocks, detect_stack_known_source, image,
                                models, observe_pack_held, observe_stack_hole, project_points, refine_hole,
                                silhouette)


def _rotate(xy, angle):
    c,s=math.cos(angle),math.sin(angle)
    return np.array([[c,-s],[s,c]]) @ np.asarray(xy,float)


def _up(point, height):
    value=np.asarray(point,float).copy();value[2]+=height
    return value


def _center(item):
    info,_=models();bounds=np.asarray(info[item['model_key']]['bounds'])
    point=np.asarray(item['position'],float).copy()
    point[:2]+=_rotate(bounds.mean(0)[:2],item['yaw'])
    return point


def _log(ctx,event,**details):
    events=ctx.memory.setdefault('container_events',[])
    events.append(dict(step=ctx.steps,event=event,**details))


def _stack_hole_precise(ctx,hole):
    if hole is None:return False
    point=np.asarray(hole['position'],float)
    uv=project_points(ctx,[point,point+[.001,0,0],point+[0,.001,0]],hole['camera'])
    jacobian=(uv[1:]-uv[0]).T/.001
    scale=float(np.linalg.svd(jacobian,compute_uv=False).min())
    metric_error=hole['fit_error']/max(scale,1e-6)
    hole['metric_error']=metric_error
    # Reject model fits less precise than the public hole-to-peg clearance.
    return metric_error<hole['radius']-.0049


def _park(ctx):
    """Expose the objects initially hidden behind the two fingers."""
    ctx.stage='clear_initial_view'
    yield from ctx.move_both([-.48,-.36,ctx.table_z+.245],
                             [.48,-.36,ctx.table_z+.245],
                             down_quat(0),down_quat(0),1.0,1.0,duration=18)
    yield from ctx.hold(3)


def _clear_arm(ctx,arm,duration=12):
    point=np.array([-.48 if arm=='left' else .48,-.36,ctx.table_z+.245])
    yield from _move(ctx,arm,point,down_quat(0),1.0,duration=duration)


def _reached(ctx,arm,position,tolerance=.004):
    return np.linalg.norm(np.asarray(ctx.tcp(arm))-np.asarray(position))<tolerance


def _move(ctx,arm,position,quat,grip=None,duration=12,precision=.004):
    yield from ctx.move(arm,position,quat,grip,duration=duration)
    yield from ctx.hold(1 if ctx.task_name=='play_stacking_toy' else 2)
    if not _reached(ctx,arm,position,precision):
        yield from ctx.move(arm,position,quat,grip,duration=4)
        yield from ctx.hold(3)
    error=float(np.linalg.norm(ctx.tcp(arm)-np.asarray(position)))
    if error>max(.012,precision*5):
        raise RuntimeError(f'container motion did not reach target: {arm} {error:.4f} m')


def _grasp_yaw(item):
    if item.get('group')==0:
        # The star is left/right symmetric but its upward and downward tips
        # have different extents. Clamp its symmetric axis to avoid pushing
        # the hole away from the commanded tool center.
        return float(item['yaw']-math.pi/2)
    size=np.asarray(item['size'])
    yaw=float(item['yaw']+(math.pi/2 if size[1]>size[0] else 0))
    return yaw


def _pack_grasp_height(item):
    if item['kind']=='car' and item['model_key']=='car_00007':
        # Fixed public cooked hulls expose broad, nearly vertical side
        # panels at this lower line; the full tip clears the table by 7.44 mm.
        return .020
    return max(.016,item['height']*(.65 if item['kind']=='car' else .52))


def _pack_contact_offset(item):
    # The public car7 mesh-volume centroid lies at local Y=-3.15 mm.
    # A 4 mm rearward line excludes its mirror frames and puts that centroid
    # within both door contact patches. Other variants use their center line.
    if item['kind']=='car' and item['model_key']=='car_00007':
        return np.array([0.,-.004])
    return np.zeros(2)


def _pack_source(ctx,item):
    point=np.asarray(item['position'],float).copy()
    point[:2]+=_rotate(_pack_contact_offset(item),item['yaw'])
    point[2]=ctx.table_z+_pack_grasp_height(item)
    return point


def _pack_expected_contact(item,point,body_yaw):
    point=np.asarray(point,float).copy()
    point[:2]+=_rotate(_pack_contact_offset(item),body_yaw)
    return point


def _car_grip(item,grasp_height):
    # Pinch the central side panels gently, above the sloping underbody. The
    # public gripper has gap=0.803 mm+2*(-10 mm+54 mm*normalized_command).
    if item['model_key']=='car_00007':
        # Public cooked car7 hulls and full finger geometry first contact
        # at q=.02618/.02625 m at the 20 mm side-panel line. Table-supported
        # closure adds 9 mm preload; RGB still decides attachment/stability.
        return float((.02617948-.009+.010)/.054)
    _,data=models();points=data[item['model_key']+'_points']
    narrow=int(np.argmin(np.asarray(item['size'])[:2]));long=1-narrow
    line=_pack_contact_offset(item)[long]
    contact=points[(np.abs(points[:,long]-line)<.009)&
                   (points[:,2]>grasp_height-.01257)&(points[:,2]<grasp_height-.0025)]
    if len(contact)>20:
        width=np.quantile(contact[:,narrow],.95)-np.quantile(contact[:,narrow],.05)
    else:width=float(item['size'][narrow])*.92
    return float(np.clip((width-.002-.000803+.020)/.108,.15,.95))


def _pack_adjust(ctx,arm,points,yaw):
    held=ctx.memory.get('pack_held')
    if held is None or held['arm']!=arm:return [np.asarray(p,float).copy() for p in points],yaw
    yaw+=held['source_yaw']-held['body_yaw']
    offset=np.asarray(held['offset'],float).copy()
    offset[:2]=_rotate(offset[:2],yaw-held['grasp_yaw'])
    correction=np.array([0.,0.,held['nominal_z']])-offset
    adjusted=[np.asarray(p,float)+correction for p in points]
    if min(p[2] for p in adjusted)<ctx.table_z+.0155:
        raise RuntimeError('Observed car grasp requires a placement below the safe finger height')
    _log(ctx,'car_placement_compensated',offset=offset.tolist(),correction=correction.tolist(),yaw=yaw)
    return adjusted,float(yaw)


def _pack_refresh(ctx,arm,points,yaw):
    held=ctx.memory.get('pack_held')
    if held is None or held['arm']!=arm:return points,yaw
    expected_yaw=held['body_yaw']+yaw-held['grasp_yaw']
    item=dict(model_key=held['model_key'],height=held['height'],yaw=expected_yaw)
    body=observe_pack_held(ctx,item,arm,held['grasp_height'])
    if body is None:
        raise RuntimeError('Car body lost its measurable horizontal grasp during carry')
    new_offset=body['position']-ctx.tcp(arm)
    old_offset=np.asarray(held['offset'],float).copy()
    old_offset[:2]=_rotate(old_offset[:2],yaw-held['grasp_yaw'])
    new_yaw=yaw+expected_yaw-body['yaw']
    rotated=new_offset.copy();rotated[:2]=_rotate(rotated[:2],new_yaw-yaw)
    correction=old_offset-rotated
    held.update(offset=new_offset.tolist(),body_yaw=body['yaw'],source_yaw=expected_yaw,grasp_yaw=yaw)
    _log(ctx,'car_carry_body',offset=new_offset.tolist(),error=body['fit_error'],
         correction=correction.tolist(),yaw=body['yaw'])
    return [np.asarray(p,float)+correction for p in points],float(new_yaw)


def _path_ok(ctx,arm,points):
    """Check every carried Cartesian segment with the public arm model."""
    from ..robot_collision import PublicSphereCheck
    collision=PublicSphereCheck()
    reference=np.asarray(ctx.state[f'{arm}_arm_joint_state'],float)
    start=tcp_to_link(*points[0])
    first=ctx.kinematics.solve(arm,start,current_joints=reference,
                  position_tol=.0005,rotation_tol=.003,max_starts=12,max_nfev=160)
    if first['status']!='Success':return False
    if collision.check(ctx.kinematics,arm,first['joint_value'])['self_gap']<-.001:return False
    reference=np.asarray(first['joint_value'])
    for xyz,quat in points[1:]:
        goal=tcp_to_link(xyz,quat)
        count=max(8,int(math.ceil(np.linalg.norm(goal[:3]-start[:3])/.015)))
        progress=0.;increment=1./count
        while progress<1.-1e-9:
            trial=min(1.,progress+increment)
            target=interpolate_pose(start,goal,trial)
            result=ctx.kinematics.solve(arm,target,current_joints=reference,
                      position_tol=.0005,rotation_tol=.003,max_starts=4,max_nfev=120)
            if result['status']!='Success':
                result=ctx.kinematics.solve(arm,target,current_joints=reference,
                          position_tol=.0005,rotation_tol=.003,max_starts=12,max_nfev=180)
            joints=np.asarray(result['joint_value'])
            if result['status']!='Success' or np.max(np.abs(joints-reference))>.35:
                increment*=.5
                if increment<1e-4:return False
                continue
            if collision.check(ctx.kinematics,arm,joints)['self_gap']<-.001:return False
            reference=joints;progress=trial;increment=min(1./count,increment*1.5)
        start=goal
    return True


def _choose(ctx,source,destination,yaw,delta=0.0,prefer=None,allow_cross=True,
            clearance=.10,lift_height=.12,approach_point=None,full_path=False,
            source_clearance=.095,destination_hover=None,allowed_arms=None,
            allowed_yaws=None):
    """Filter endpoint grasp choices through public FK, using measured joints."""
    source=np.asarray(source,float);destination=np.asarray(destination,float)
    hover=_up(destination,clearance) if destination_hover is None else np.asarray(destination_hover,float)
    arms=[prefer or ctx.arm_for(source)]
    arms.append('right' if arms[0]=='left' else 'left')
    if allowed_arms is not None:arms=[arm for arm in arms if arm in allowed_arms]
    if not getattr(ctx,'kinematics',None):
        return arms[0],float(yaw),float(yaw+delta)
    options=[]
    for arm in arms:
        angles=[yaw,yaw+math.pi]
        if allow_cross:
            angles += [yaw+math.pi/2,yaw-math.pi/2]
        if allowed_yaws is not None:
            angles=list(allowed_yaws)
        for angle in angles:
            cost=0.0
            endpoints=[(source,down_quat(angle)),(_up(source,source_clearance),down_quat(angle)),
                       (_up(source,lift_height),down_quat(angle)),
                       (destination,down_quat(angle+delta)),(hover,down_quat(angle+delta))]
            if approach_point is not None:
                endpoints.append((np.asarray(approach_point,float),down_quat(angle+delta)))
            for xyz,q in endpoints:
                result=ctx.kinematics.solve(arm,tcp_to_link(xyz,q),
                          current_joints=ctx.state[f'{arm}_arm_joint_state'],
                          position_tol=.0005,rotation_tol=.003,max_starts=12,max_nfev=160)
                if result['status']!='Success':
                    break
                cost+=result['joint_distance']
            else:
                if full_path:
                    grasp=down_quat(angle);place=down_quat(angle+delta)
                    path=[(ctx.tcp(arm),ctx.state[f'{arm}_ee_pose'][3:]),
                          (_up(source,source_clearance),grasp),(source,grasp),(_up(source,lift_height),grasp)]
                    if approach_point is not None:path.append((approach_point,place))
                    path.extend([(hover,place),(destination,place)])
                    if not _path_ok(ctx,arm,path):continue
                options.append((cost,arm,angle,angle+delta))
        if options:
            break
    if not options:
        raise RuntimeError('No feasible RGB-derived grasp and placement endpoints')
    _,arm,grasp_yaw,place_yaw=min(options)
    return arm,float(grasp_yaw),float(place_yaw)


def _source_patch(ctx,item):
    rgb=image(ctx,item.get('camera','cam_head'))
    x,y,w,h=map(int,item['bbox'])
    return rgb[max(0,y):min(rgb.shape[0],y+h),max(0,x):min(rgb.shape[1],x+w)].copy()


def _changed(ctx,item,before):
    after=_source_patch(ctx,item)
    if after.shape!=before.shape or not before.size:
        return True
    import cv2
    def material(rgb):
        masks=color_masks(rgb)
        if 'color' in item:
            return masks[item['color']]
        hsv=cv2.cvtColor(rgb,cv2.COLOR_RGB2HSV)
        gray=(hsv[...,1]<70)
        if item['kind']=='garage':
            return gray|masks['green']|masks['yellow']|masks['purple']|masks['pink']
        if item['kind']=='mouse':
            return gray|masks['purple']|masks['blue']
        return gray
    old=int(np.count_nonzero(material(before)))
    current=int(np.count_nonzero(material(after)))
    if old>=8:
        return current<old*.48
    return float(np.mean(np.abs(after.astype(float)-before.astype(float))))>18.0


def _pack_retry_source(ctx,item,arm,grasp_yaw,approach,lift):
    """Keep the same contact line relative to an RGB-remeasured car."""
    fresh=detect_pack_at(ctx,'car',item['position'],item['model_key'],yaw_hint=item['yaw'])
    if fresh is None:
        raise RuntimeError('RGB car source missing after a failed grasp')
    # A failed close can translate and yaw the car. Preserve the selected
    # parallel jaw direction while moving its door contact with the body.
    change=math.atan2(math.sin(fresh['yaw']-item['yaw']),math.cos(fresh['yaw']-item['yaw']))
    fresh['yaw']=float(item['yaw']+change)
    grasp_yaw=float(grasp_yaw+change)
    quat=down_quat(grasp_yaw)
    source=_pack_source(ctx,fresh)
    path=[(ctx.tcp(arm),ctx.state[f'{arm}_ee_pose'][3:]),
          (_up(source,approach),quat),(source,quat),(_up(source,lift),quat)]
    if not _path_ok(ctx,arm,path):
        raise RuntimeError('Remeasured car contact has no continuous retry path')
    _log(ctx,'car_source_remeasured',position=fresh['position'].tolist(),yaw=fresh['yaw'],
         shift=(fresh['position']-np.asarray(item['position'])).tolist(),yaw_change=change,
         error=fresh['fit_error'],grasp_yaw=grasp_yaw)
    return fresh,source,grasp_yaw,quat


def _pack_camera_shapes(ctx, arm, joints, opened):
    from . import organize as geometry
    frames=geometry._sphere_geometry().frames(ctx.kinematics.base_transforms[arm],joints)
    result={}
    for link,(vertices,normals,edges) in geometry._robot_neighbor_geometry().items():
        if opened and link.startswith('finger'):
            vertices,normals,edges,_=geometry._open_hand_geometry()[1+int(link[-1])]
        matrix=frames['link6'] if link.startswith('finger') else frames[link]
        result[link]=(vertices@matrix[:3,:3].T+matrix[:3,3],normals@matrix[:3,:3].T)
    return result

def _pack_camera_pair_gap(first,second):
    a,na=first;b,nb=second
    broad=float(np.maximum(b.min(0)-a.max(0),a.min(0)-b.max(0)).max())
    if broad>=.008:return broad
    axes=np.vstack([na,nb]);pa=a@axes.T;pb=b@axes.T
    # Positive separating-face gaps prove clearance; ambiguous pairs reject.
    return float(np.maximum(pb.min(0)-pa.max(0),pa.min(0)-pb.max(0)).max())

def _pack_observe_body_development(ctx,item,arm,grasp_height):
    from . import organize as geometry
    from ..motion import quat_matrix
    if arm!='right' or not ctx.numeric_ik:
        raise RuntimeError('Held-car observation development needs the numeric right-arm grasp')
    height=item['height'];tcp=ctx.tcp(arm)
    aim=np.r_[tcp[:2],tcp[2]+height/2-grasp_height]
    box=detect_box(ctx)
    if box is None:raise RuntimeError('No current RGB box for observation-path exclusion')
    others=[x for x in detect_pack_objects(ctx,box) if x['kind']!='car']
    prior=dict(item,position=aim,object_root=aim-[0,0,height/2])
    obstacles=tuple(others+[prior])
    q=geometry._tilt(20.,0.);rotation=quat_matrix(q)
    # Public wrist mount, aimed at the unconfirmed body prior from the side.
    local=np.array([-.0605,0.,.051]);camera_offset=rotation@local
    from scipy.spatial.transform import Rotation
    direction=rotation@Rotation.from_euler('y',.3491-math.radians(80)).as_matrix()@np.array([0.,0.,-1.])
    z=ctx.table_z+.335
    distance=(aim[2]-z-camera_offset[2])/direction[2]
    point=aim-camera_offset-distance*direction;point[2]=z
    plans=[];original=ctx.kinematics.solve
    def capture(*args,**kwargs):
        result=original(*args,**kwargs)
        if result['status']=='Success':plans.append(np.asarray(result['joint_value']).copy())
        return result
    ctx.kinematics.solve=capture
    try:
        pose,final,guard=geometry._check_carry_path(ctx,'left',point,q,box,
            'held_car_empty_camera',samples=88,record=False,obstacles=obstacles,fingers_open=True)
    finally:ctx.kinematics.solve=original
    if len(plans)!=88:raise RuntimeError('Camera path does not match its checked continuous joint sequence')
    fixed=_pack_camera_shapes(ctx,'right',ctx.state['right_arm_joint_state'],False)
    minimum=float('inf');pair=None
    for joints in plans:
        for a,first in _pack_camera_shapes(ctx,'left',joints,True).items():
            for b,second in fixed.items():
                gap=_pack_camera_pair_gap(first,second)
                if gap<minimum:minimum,pair=gap,(a,b)
    if minimum<.008:raise RuntimeError('Empty camera path approaches the grasping arm')
    old=dict(ctx.state)
    try:
        ctx.obs['state']=dict(old,left_ee_pose=pose,left_arm_joint_state=final)
        _,clouds=models();uv=project_points(ctx,
            clouds[item['model_key']+'_points']@np.array([
                [math.cos(item['yaw']),-math.sin(item['yaw']),0.],
                [math.sin(item['yaw']),math.cos(item['yaw']),0.],[0.,0.,1.]]).T+
                aim-[0.,0.,height/2],'cam_left_wrist')
        coverage=float(((uv[:,0]>15)&(uv[:,0]<625)&(uv[:,1]>15)&(uv[:,1]<465)).mean())
    finally:ctx.obs['state']=old
    if coverage<.995:raise RuntimeError('New wrist view would still clip the public body prior')
    ctx.stage='observe_held_car_with_empty_wrist'
    _log(ctx,'held_car_observation_plan',point=point.tolist(),quat=q.tolist(),guard=guard,
         inter_arm_gap=minimum,inter_arm_pair=pair,coverage=coverage,
         unconfirmed_body_prior=True,development_only=True)
    # Execute exactly the checked Cartesian branch, bypassing any free-space
    # joint shortcut. Two adjacent samples per command keep the step bounded.
    ctx.gripper_targets['left']=1.0
    for joints in plans[1::2]:
        ctx.joint_targets['left']=joints.copy()
        yield ctx.hold_action()
    yield from ctx.hold(4)
    if np.linalg.norm(ctx.tcp('left')-point)>.006:
        raise RuntimeError('Empty camera did not physically reach the checked viewpoint')
    ctx.stage='measure_held_car_from_empty_wrist'
    _log(ctx,'held_car_observation_acquired',development_only=True,held_body_accepted=False)
    yield from ctx.hold(16)
    raise RuntimeError('Development RGB view acquired; held-car pose is still unaccepted')


def _pick(ctx,item,source,arm,yaw,duration=11,quat=None,approach=.095,lift=.12):
    quat=down_quat(yaw) if quat is None else np.asarray(quat,float)
    ctx.memory.pop('pack_held',None)
    if (item['kind']=='car' and item['model_key']=='car_00007' and
            ctx.stage=='clear_car_before_box_move'):
        # Stop the initial inspection lift below the background shoe's
        # overlapping wrist projection. Its RGB-derived floor remains
        # 60 mm above the table; all held-body/stability gates still apply.
        lift=min(lift,.060)
        if not _path_ok(ctx,arm,[(ctx.tcp(arm),ctx.state[f'{arm}_ee_pose'][3:]),
                                (_up(source,approach),quat),(source,quat),
                                (_up(source,lift),quat)]):
            raise RuntimeError('Car observation lift has no continuous public path')
        _log(ctx,'car_observation_lift_height',height=lift)
    closing=_car_grip(item,source[2]-ctx.table_z) if item['kind']=='car' else 0.0
    before=_source_patch(ctx,item)
    yield from _move(ctx,arm,_up(source,approach),quat,1.0,duration)
    yield from _move(ctx,arm,source,quat,1.0,max(8,duration-2))
    yield from ctx.grip(arm,closing,duration=7)
    yield from ctx.hold(3)
    yield from _move(ctx,arm,_up(source,lift),quat,closing,16 if item['kind']=='car' else duration)
    yield from ctx.hold(3)
    held=_changed(ctx,item,before)
    if item['kind']=='car' and not held:
        # The mesh gives a starting gap, while RGB determines whether that
        # gap actually contacts the side panels. Close in small increments;
        # commanding zero immediately previously tipped the car on its end.
        for retry in range(3):
            item,source,yaw,quat=_pack_retry_source(ctx,item,arm,yaw,approach,lift)
            before=_source_patch(ctx,item)
            closing=max(.15,closing-.04)
            _log(ctx,'car_grip_retry',attempt=retry+1,grip=closing)
            yield from _move(ctx,arm,_up(source,approach),quat,1.0,duration=6)
            yield from _move(ctx,arm,source,quat,1.0,duration=8,precision=.001)
            yield from ctx.grip(arm,closing,duration=7)
            yield from ctx.hold(3)
            yield from _move(ctx,arm,_up(source,lift),quat,closing,duration=16)
            yield from ctx.hold(3)
            held=_changed(ctx,item,before)
            if held:break
    if held and item['kind']=='car':
        body=observe_pack_held(ctx,item,arm,source[2]-ctx.table_z)
        # The cooked-panel grasp was preloaded on the table, not in air.
        preloaded=item['model_key']=='car_00007'
        if body is None and item['model_key']=='car_00007':
            yield from _pack_observe_body_development(ctx,item,arm,source[2]-ctx.table_z)
        if body is not None:
            for verification in range(1 if preloaded else 2):
                if verification:
                    closing=max(.15,closing-.04)
                    _log(ctx,'car_hold_preload',grip=closing)
                    yield from ctx.grip(arm,closing,duration=6)
                    yield from ctx.hold(3)
                    body=observe_pack_held(ctx,item,arm,source[2]-ctx.table_z)
                    if body is None:
                        _log(ctx,'car_preload_unmeasured',grip=closing)
                        break
                first_offset=body['position']-ctx.tcp(arm)
                first_yaw=body['yaw']
                yield from ctx.hold(3)
                stable=observe_pack_held(ctx,item,arm,source[2]-ctx.table_z)
                change=None if stable is None else stable['position']-ctx.tcp(arm)-first_offset
                yaw_change=None if stable is None else math.atan2(
                        math.sin(stable['yaw']-first_yaw),math.cos(stable['yaw']-first_yaw))
                if (stable is None or np.linalg.norm(change[:2])>.003 or
                        abs(change[2])>.003 or abs(yaw_change)>.04):
                    _log(ctx,'car_hold_unstable',offset_change=None if change is None else change.tolist(),
                         yaw_change=yaw_change)
                    body=None
                    break
                else:
                    body=stable
                    _log(ctx,'car_hold_stable',offset_change=change.tolist(),yaw_change=yaw_change,
                         components=body.get('components',1),error=body['fit_error'])
        if body is None:
            held=False
            _log(ctx,'car_hold_unmeasured',grip=closing)
        else:
            offset=body['position']-ctx.tcp(arm)
            ctx.memory['pack_held']=dict(arm=arm,kind='car',offset=offset.tolist(),
                    source_yaw=item['yaw'],body_yaw=body['yaw'],grasp_yaw=yaw,
                    model_key=item['model_key'],height=item['height'],grasp_height=source[2]-ctx.table_z,
                    nominal_z=item['height']/2-(source[2]-ctx.table_z))
            _log(ctx,'car_held_body',offset=offset.tolist(),yaw=body['yaw'],error=body['fit_error'],grip=closing)
    _log(ctx,'hold_check',kind=item['kind'],held=held)
    return held


def _place(ctx,arm,target,yaw,clearance=.12,duration=12,quat=None,hover=None,correct_held=True):
    hover=_up(target,clearance) if hover is None else np.asarray(hover,float)
    if correct_held:
        (target,hover),yaw=_pack_adjust(ctx,arm,[target,hover],yaw)
    quat=down_quat(yaw) if quat is None else np.asarray(quat,float)
    if ctx.memory.get('pack_held') is not None:
        if not _path_ok(ctx,arm,[(ctx.tcp(arm),ctx.state[f'{arm}_ee_pose'][3:]),(hover,quat),(target,quat)]):
            raise RuntimeError('Observed car offset has no continuous placement path')
    holding=ctx.gripper_targets[arm]
    # Native RGB confirmed a held car slipped during the simultaneous large
    # translation and wrist turn. Keep its measured grasp and geometric
    # checks, and give this free-space transport more settling time.
    held_car=ctx.memory.get('pack_held')
    transport_duration=max(duration,32) if held_car is not None and held_car['arm']==arm else duration
    if transport_duration!=duration:
        _log(ctx,'car_transport_duration',duration=transport_duration)
    yield from _move(ctx,arm,hover,quat,holding,transport_duration)
    if ctx.memory.get('pack_held') is not None:
        (target,hover),yaw=_pack_refresh(ctx,arm,[target,hover],yaw)
        quat=down_quat(yaw)
        if min(target[2],hover[2])<ctx.table_z+.0155:
            raise RuntimeError('Measured car carry offset requires an unsafe finger height')
        if not _path_ok(ctx,arm,[(ctx.tcp(arm),ctx.state[f'{arm}_ee_pose'][3:]),(hover,quat),(target,quat)]):
            raise RuntimeError('Measured car carry offset has no continuous contact path')
        yield from _move(ctx,arm,hover,quat,holding,duration=4,precision=.001)
    yield from _move(ctx,arm,target,quat,holding,duration)
    yield from ctx.grip(arm,1.0,duration=6)
    yield from ctx.hold(4)
    yield from _move(ctx,arm,hover,quat,1.0,10)
    ctx.memory.pop('pack_held',None)


def _pack_shape_polygon(item, xy=None, yaw=None):
    import cv2
    _,clouds=models()
    angle=item['yaw'] if yaw is None else yaw
    c,s=math.cos(angle),math.sin(angle)
    points=clouds[item['model_key']+'_points'][:,:2]@np.array([[c,-s],[s,c]]).T
    points+=np.asarray(item['position'])[:2] if xy is None else np.asarray(xy)[:2]
    return cv2.convexHull(points.astype(np.float32)).reshape(-1,2).astype(float)


def _pack_polygon_gap(a,b):
    edges=np.vstack([np.roll(a,-1,axis=0)-a,np.roll(b,-1,axis=0)-b])
    axes=edges[:,[1,0]]*[-1,1]
    axes=axes[np.linalg.norm(axes,axis=1)>1e-8]
    axes/=np.linalg.norm(axes,axis=1)[:,None]
    first,second=a@axes.T,b@axes.T
    return float(np.maximum(first.min(0)-second.max(0),second.min(0)-first.max(0)).max())


def _pack_box_sweep_polygon(box):
    import cv2
    info,clouds=models()
    mean=np.asarray(info[box['model_key']]['bounds']).mean(0)[:2]
    start=_center(box)[:2];finish=np.array([0.,-.17])
    points=clouds[box['model_key']+'_points'][:,:2]-mean
    values=[]
    for fraction in np.linspace(0.,1.,33):
        angle=box['yaw']*(1-fraction);c,s=math.cos(angle),math.sin(angle)
        values.append(points@np.array([[c,-s],[s,c]]).T+
                      start*(1-fraction)+finish*fraction)
    return cv2.convexHull(np.vstack(values).astype(np.float32)).reshape(-1,2).astype(float)


def _pack_choose_source(ctx,source,destination,yaw,delta=0.,**kwargs):
    """Keep the original source path, then check lower reachable lifts."""
    for approach,lift in ((.095,.12),(.065,.075),(.055,.055)):
        settings=dict(kwargs,source_clearance=approach,lift_height=lift)
        try:
            arm,grasp,place=_choose(ctx,source,destination,yaw,delta,**settings)
        except RuntimeError:
            continue
        return arm,grasp,place,approach,lift
    raise RuntimeError('No continuous RGB packing source/lift route')


def _pack_clear_plan(ctx,box,item,original_xy,delta):
    """Select a measured free buffer outside the conservative box sweep."""
    source=_pack_source(ctx,item);swept=_pack_box_sweep_polygon(box)
    neighbours=[x for x in detect_pack_objects(ctx,box) if x['kind']!=item['kind']]
    sign=1. if ctx.arm_for(source)=='right' else -1.
    candidates=[original_xy,[sign*.30,-.35],[sign*.38,-.35],[sign*.42,-.31]]
    for xy in candidates:
        root_yaw=item['yaw']+delta
        shape=_pack_shape_polygon(item,xy,root_yaw)
        box_gap=_pack_polygon_gap(shape,swept)
        other_gap=min((_pack_polygon_gap(shape,_pack_shape_polygon(x))
                       for x in neighbours),default=float('inf'))
        if box_gap<.012 or other_gap<.015:continue
        buffer=np.r_[xy,source[2]]
        try:
            arm,grasp,place,approach,lift=_pack_choose_source(ctx,source,
                _pack_expected_contact(item,buffer,root_yaw),_grasp_yaw(item),delta,
                allow_cross=False,clearance=.10,full_path=True)
        except RuntimeError:continue
        _log(ctx,'box_sweep_buffer_plan',kind=item['kind'],buffer=buffer.tolist(),
             box_gap=box_gap,neighbour_gap=other_gap,approach=approach,lift=lift)
        return source,buffer,arm,grasp,place,approach,lift
    raise RuntimeError('No reachable RGB buffer outside the packing box sweep')


def _pack_choice(ctx,item,source,target,center,floor,prefer=None):
    # The public box's front wall is 96 mm tall. A 20 mm object-bottom
    # margin suffices when entering from the front, avoiding its side flaps.
    minimum=max(.035,.096+.020-floor)
    clearances=sorted(set([minimum,minimum+.01,minimum+.02,.11,.12,.13,.14,.18]))
    delta=float(item['target_yaw']-item['yaw'])
    for clearance in clearances:
        if clearance<minimum-1e-9:continue
        front=_up(target,clearance);front[1]=center[1]-.15
        # Enter above the front interior, then descend and move toward the
        # rear row together. A high hover over the back of the box can require
        # an otherwise unnecessary, discontinuous arm branch change.
        entry=_up(target,clearance);entry[1]=min(target[1],center[1]-.04)
        try:
            arm,grasp,place,approach,lift=_pack_choose_source(ctx,source,_pack_expected_contact(item,target,item['target_yaw']),
                              _grasp_yaw(item),delta,prefer=prefer,
                              allow_cross=max(item['size'][:2])<.074,clearance=clearance,
                              approach_point=_pack_expected_contact(item,front,item['target_yaw']),
                              destination_hover=_pack_expected_contact(item,entry,item['target_yaw']),
                              full_path=True)
        except RuntimeError:
            continue
        return arm,grasp,place,clearance,front,entry,approach,lift
    raise RuntimeError('No collision-clear reachable RGB box placement path')


def _prepare_pack_box(ctx,box):
    """Clear the front sweep, then remeasure an actually repositioned box."""
    center=_center(box)
    buffers=ctx.memory.setdefault('pack_buffers',{})
    # The central box sweep passes the original hammer and car. Keep both
    # outside its measured front/side walls and retain their directed yaw.
    for kind,xy in (('hammer',[-.10,-.35]),('car',[.18,-.35])):
        item=next((entry for entry in detect_pack_objects(ctx,box) if entry['kind']==kind),None)
        if item is None:raise RuntimeError(f'RGB {kind} missing before clearing the box sweep')
        sweep_gap=_pack_polygon_gap(_pack_shape_polygon(item),_pack_box_sweep_polygon(box))
        if sweep_gap>.020:
            _log(ctx,'box_sweep_source_clear',kind=kind,gap=sweep_gap)
            continue
        source=_pack_source(ctx,item)
        if item['model_key']=='car_00007':
            # This foreground patch permits the door grasp to turn halfway
            # before a supported regrasp and the final box entry.
            xy=[.26,-.34]
        buffer=np.r_[xy,source[2]]
        # Split the car7 half-turn at a supported regrasp and remeasure its
        # actual resulting pose before planning the second rotation.
        clearing_delta=(item['target_yaw']-item['yaw'])*.5 if item['model_key']=='car_00007' else 0.
        source,buffer,arm,grasp,place,approach,lift=_pack_clear_plan(ctx,box,item,xy,clearing_delta)
        ctx.stage=f'clear_{kind}_before_box_move'
        held=yield from _pick(ctx,item,source,arm,grasp,approach=approach,lift=lift)
        if not held:raise RuntimeError(f'{kind} did not leave its RGB source before box movement')
        yield from _place(ctx,arm,buffer,place,clearance=.10)
        yield from _clear_arm(ctx,arm)
        moved=detect_pack_at(ctx,kind,buffer,item['model_key'])
        if moved is None:
            raise RuntimeError(f'RGB {kind} buffer is not available for a later regrasp')
        buffers[kind]={key:moved[key] for key in ('position','yaw','model_key','fit_error')}
        _log(ctx,'box_sweep_cleared',kind=kind,position=moved['position'].tolist(),error=moved['fit_error'])
    box=detect_box(ctx)
    if box is None:raise RuntimeError('RGB box missing after clearing its front')
    center=_center(box)
    # Pinch the thin front wall along its normal. Keeping the TCP at 85 mm
    # engages its upper edge without reaching the 182 mm side flaps.
    rim_offset=np.array([0.,-.084])
    rim=center.copy();rim[:2]+=_rotate(rim_offset,box['yaw']);rim[2]=ctx.table_z+.085
    desired=center.copy();desired[:2]=[0.,-.17]
    target=desired.copy();target[:2]+=rim_offset;target[2]=rim[2]
    arm,grasp,place=_choose(ctx,rim,target,box['yaw'],-box['yaw'],allow_cross=False,
                           clearance=.055,lift_height=.055,source_clearance=.055,full_path=True)
    ctx.stage='bring_empty_box_forward'
    # A lifted box still overlaps its original large image crop. Verify the
    # actual resulting pose below rather than treating crop overlap as empty.
    yield from _pick(ctx,box,rim,arm,grasp,approach=.055,lift=.055)
    yield from _place(ctx,arm,target,place,clearance=.055)
    yield from _clear_arm(ctx,arm)
    measured=detect_box(ctx)
    if measured is None:raise RuntimeError('RGB box missing after forward movement')
    measured_center=_center(measured)
    if np.linalg.norm(measured_center[:2]-desired[:2])>.045:
        raise RuntimeError('RGB box did not reach the central packing workspace')
    if abs(math.sin(measured['yaw']))>math.sin(math.radians(30)):
        raise RuntimeError('RGB box long axis remains misaligned after its rim regrasp')
    _log(ctx,'box_repositioned',before=center.tolist(),after=measured_center.tolist(),
         yaw=measured['yaw'],error=measured['fit_error'])
    return measured


def pack_objects_into_box(ctx):
    """Fit each public object, preserve directed yaw, and pack distinct rows."""
    yield from _park(ctx)
    ctx.stage='measure_box'
    box=detect_box(ctx)
    if box is None:
        raise RuntimeError('RGB box measurement missing')
    box=yield from _prepare_pack_box(ctx,box)
    center=_center(box)
    ctx.memory['box_measurement']={key:box[key] for key in ('position','yaw','fit_error')}
    completed=set()
    # Two bottom rows keep the box's reachable front area open. Thin tools
    # rest on the previously placed shoe/car instead of reaching its back row.
    rows={'shoe':-.044,'car':.026,'electric_toothbrush':-.044,'hammer':.026}
    support_top={}
    for kind in ('shoe','car','electric_toothbrush','hammer'):
        for attempt in range(2):
            if ctx.steps>1120:
                break
            ctx.stage=f'measure_{kind}_{attempt}'
            buffer=ctx.memory.get('pack_buffers',{}).get(kind)
            if buffer is not None:
                recovered=detect_pack_at(ctx,kind,buffer['position'],buffer['model_key'])
                found=[] if recovered is None else [recovered]
            else:
                found=[item for item in detect_pack_objects(ctx,box) if item['kind']==kind]
            if not found:
                _log(ctx,'object_missing',kind=kind)
                break
            item=min(found,key=lambda obj:obj['fit_error'])
            grasp_height=_pack_grasp_height(item)
            source=_pack_source(ctx,item)
            target=center.copy();target[1]+=rows[kind]
            floor=.007
            if kind in ('electric_toothbrush','hammer'):
                support='shoe' if kind=='electric_toothbrush' else 'car'
                if support not in support_top:
                    raise RuntimeError(f'RGB packing support {support} has not been placed')
                floor=support_top[support]+.005
            target[2]=ctx.table_z+floor+grasp_height
            delta=float(item['target_yaw']-item['yaw'])
            try:
                arm,grasp_yaw,place_yaw,clearance,front,entry,approach,lift=_pack_choice(ctx,item,source,target,center,floor)
            except RuntimeError:
                # A right-hand source and a left-hand box need a tabletop
                # handoff, not a knowingly unreachable cross-body carry.
                buffer=np.array([.025,-.35,ctx.table_z+grasp_height])
                other=detect_pack_objects(ctx,box)
                for xy in ([.025,-.35],[-.28,-.27],[.30,-.27]):
                    candidate=np.r_[xy,buffer[2]]
                    if all(x['kind']==kind or np.linalg.norm(candidate[:2]-x['position'][:2])>.12 for x in other):
                        buffer=candidate;break
                # A near half-turn of the hammer crosses a wrist limit while
                # held. Split it at a supported tabletop regrasp, where the
                # empty gripper can select the opposite parallel jaw pose.
                buffer_delta=delta*.5 if kind=='hammer' else delta
                arm,grasp_yaw,buffer_yaw,approach,lift=_pack_choose_source(ctx,source,
                                                 _pack_expected_contact(item,buffer,item['yaw']+buffer_delta),
                                                 _grasp_yaw(item),buffer_delta,
                                                 allow_cross=max(item['size'][:2])<.074,clearance=.13,
                                                 full_path=True)
                ctx.stage=f'pick_{kind}_for_handoff'
                held=yield from _pick(ctx,item,source,arm,grasp_yaw,approach=approach,lift=lift)
                if not held:
                    yield from ctx.grip(arm,1.0,5)
                    yield from _clear_arm(ctx,arm)
                    continue
                ctx.stage=f'handoff_{kind}'
                yield from _place(ctx,arm,buffer,buffer_yaw,clearance=.13)
                yield from _clear_arm(ctx,arm)
                remeasured=detect_pack_at(ctx,kind,buffer,item['model_key'])
                if remeasured is None:
                    raise RuntimeError(f'RGB {kind} missing after tabletop handoff')
                item=remeasured
                source=_pack_source(ctx,item)
                delta=float(item['target_yaw']-item['yaw'])
                opposite='left' if arm=='right' else 'right'
                arm,grasp_yaw,place_yaw,clearance,front,entry,approach,lift=_pack_choice(ctx,item,source,target,center,floor,
                                                                               prefer=opposite)
                _log(ctx,'tabletop_handoff',kind=kind,buffer=buffer.tolist(),arm=arm)
            ctx.stage=f'pick_{kind}_{attempt}'
            held=yield from _pick(ctx,item,source,arm,grasp_yaw,approach=approach,lift=lift)
            if not held:
                yield from ctx.grip(arm,1.0,5)
                yield from _clear_arm(ctx,arm)
                continue
            ctx.stage=f'place_{kind}'
            (target,front,entry),place_yaw=_pack_adjust(ctx,arm,[target,front,entry],place_yaw)
            if ctx.memory.get('pack_held') is not None:
                q=down_quat(place_yaw)
                if not _path_ok(ctx,arm,[(ctx.tcp(arm),ctx.state[f'{arm}_ee_pose'][3:]),
                                        (front,q),(entry,q),(target,q)]):
                    raise RuntimeError('RGB car body compensation has no continuous box entry path')
            # Approach from the front so the carried object clears the 96 mm
            # box walls without crossing the taller open side flaps.
            yield from _move(ctx,arm,front,down_quat(place_yaw),ctx.gripper_targets[arm],duration=12)
            yield from _place(ctx,arm,target,place_yaw,clearance=clearance,hover=entry,correct_held=False)
            yield from _clear_arm(ctx,arm)
            _log(ctx,'placed',kind=kind,target=target.tolist(),yaw=place_yaw,clearance=clearance)
            support_top[kind]=floor+item['height']
            completed.add(kind)
            break
    ctx.memory['packed_objects']=sorted(completed)
    ctx.stage='home'
    yield from ctx.home(duration=28)
    yield from ctx.hold(12)


def _stack_cardinality(ctx):
    """Infer the shape-to-pole assignment from the ten visible source blocks."""
    items=detect_stack_blocks(ctx,camera='cam_head')
    counts=[sum(item['group']==group for item in items) for group in range(4)]
    return counts,items


def _stack_cached_plan(ctx,plan,*args,**kwargs):
    """Reuse only identical public IK inputs within one actual observation."""
    from copy import deepcopy
    from ..kinematics import (JOINT_ORIGINS,JOINT_ORIGIN_ROLLS,JOINT_AXES,
                             JOINT_LOWER,JOINT_UPPER)
    from ..motion import TCP_OFFSET
    original=ctx.kinematics
    if original is None:return plan(ctx,*args,**kwargs)
    def exact(value):return np.asarray(value,dtype=float).tobytes()
    # Include the measured reference and fixed public geometry, then discard
    # the entire cache at the next observation. No rounded pose/state reuse.
    signature=(id(original),id(ctx.obs),ctx.steps,float(ctx.table_z),
               tuple((arm,exact(ctx.state[f'{arm}_arm_joint_state']),
                       exact(ctx.state[f'{arm}_ee_pose']),
                       exact(original.base_transforms[arm]),exact(original.initial_joints[arm]))
                      for arm in ('left','right')),
               exact(JOINT_ORIGINS),exact(JOINT_ORIGIN_ROLLS),tuple(JOINT_AXES),
               exact(JOINT_LOWER),exact(JOINT_UPPER),exact(TCP_OFFSET))
    cache=getattr(ctx,'_stack_ik_cache',None)
    if cache is None or cache.signature!=signature:
        class ExactIK:
            def __init__(self):
                self.signature=signature;self.results={};self.hits=0;self.misses=0
            def __getattr__(self,name):return getattr(original,name)
            def solve(self,arm,target_pose,current_joints=None,tcp=False,
                      position_tol=.002,rotation_tol=.02,max_starts=12,max_nfev=160,
                      diagnostic_candidates=False):
                key=(arm,exact(target_pose),None if current_joints is None else exact(current_joints),
                     bool(tcp),float(position_tol),float(rotation_tol),int(max_starts),int(max_nfev),
                     bool(diagnostic_candidates))
                if key in self.results:
                    self.hits+=1
                    return deepcopy(self.results[key])
                result=original.solve(arm,target_pose,current_joints=current_joints,tcp=tcp,
                                      position_tol=position_tol,rotation_tol=rotation_tol,
                                      max_starts=max_starts,max_nfev=max_nfev,
                                      diagnostic_candidates=diagnostic_candidates)
                self.misses+=1
                if len(self.results)>=4096:self.results.pop(next(iter(self.results)))
                self.results[key]=deepcopy(result)
                return result
        cache=ExactIK();ctx._stack_ik_cache=cache
    ctx.kinematics=cache
    try:return plan(ctx,*args,**kwargs)
    finally:ctx.kinematics=original


def _stack_choice(ctx,source,target,yaw,group,hover_height,minimum_hover=None,only_arm=None,
                  allowed_yaws=None):
    return _stack_cached_plan(ctx,_stack_choice_plan,source,target,yaw,group,
                              hover_height,minimum_hover,only_arm,allowed_yaws)


def _stack_choice_plan(ctx,source,target,yaw,group,hover_height,minimum_hover=None,only_arm=None,
                       allowed_yaws=None):
    # Sources near the rear/center can be reachable at contact but outside
    # the physical branch's reach at a 95 mm lift. Verify the whole motion
    # from the actual arm pose and lower the initial lift before carrying
    # toward the more reachable base, without relaxing IK precision.
    clearances=[hover_height]
    if minimum_hover is not None:
        # A 50 mm margin above the final contact point is unnecessary when
        # the explicit peg/held-body constraint already clears the whole
        # object by 20 mm. Keep successful heights, then reduce only a path
        # that cannot reach the top layers on its physical arm branch.
        clearances+= [max(.035,minimum_hover),max(.020,minimum_hover)]
    for clearance in dict.fromkeys(clearances):
        for lift in (.095,.075,.055):
            approach=min(.065,lift)
            try:
                arm,angle,_=_choose(ctx,source,target,yaw,lift_height=lift,
                             clearance=clearance,source_clearance=approach,
                             allow_cross=group!=0,full_path=True,prefer=only_arm,
                             allowed_arms=None if only_arm is None else (only_arm,),
                             allowed_yaws=allowed_yaws)
                return arm,angle,approach,lift,clearance
            except RuntimeError:
                continue
    raise RuntimeError('No continuous RGB-derived stacking grasp, lift and insertion path')


def _stack_item_plan(ctx,item,level,peg,only_arm=None,allowed_yaws=None):
    source=np.asarray(item['position'],float).copy()
    grasp_offset=_rotate([0,.0092566],item['yaw']) if item['group']==0 else np.zeros(2)
    source[:2]+=grasp_offset;source[2]=ctx.table_z+.016
    target=np.asarray(peg,float).copy();target[2]=ctx.table_z+.011+level*item['height']+.016
    preflight_target=target.copy();preflight_target[:2]+=grasp_offset
    minimum=max(.020,peg[2]+.020+.016-target[2])
    choice=_stack_choice(ctx,source,preflight_target,_grasp_yaw(item),item['group'],
                         max(.05,minimum),minimum,only_arm=only_arm,allowed_yaws=allowed_yaws)
    return source,grasp_offset,target,choice


def _stack_empty_retry_yaws(item,failed_yaws):
    """After a confirmed empty lift, try the rectangle's other opposed faces."""
    if item['group']!=1 or not failed_yaws:return None
    # The public rectangular block has two perpendicular opposed face pairs.
    # Changing between them avoids repeating a jaw sweep that closed on a
    # neighbouring block/peg or missed the intended piece. Its current RGB
    # body yaw defines the new face; no layout identity is consulted.
    info,_=models();bounds=np.asarray(info[item['model_key']]['bounds'],float)
    widths=np.diff(bounds,axis=0)[0,:2]
    if min(widths)<=0 or max(widths)<min(widths)*1.5:return None
    angles=[]
    for angle in (float(item['yaw']),float(item['yaw'])+math.pi/2):
        if all(abs(math.sin(angle-failed))>math.sqrt(.5) for failed in failed_yaws):
            angles.extend((angle,angle+math.pi))
    return angles


def _stack_yaw_change(item,body_yaw):
    """Preserve a grasp face across equivalent public-shape RGB yaw fits."""
    # The rectangle has two equivalent faces and the yellow square four.
    # A circular body has no meaningful grasp yaw; the asymmetric star does.
    folds={0:1,1:2,2:4,3:0}[item['group']]
    if not folds:return 0.0
    difference=float(body_yaw)-float(item['yaw'])
    return math.atan2(math.sin(folds*difference),math.cos(folds*difference))/folds


def _stack_source_evidence(ctx,item,arm,references):
    if not (item.get('wrist_recovered') or item.get('memory_recovered')):
        return item,_source_patch(ctx,item)
    import cv2
    # The opposite wrist stays fixed throughout this pick. A moving detector
    # camera cannot itself supply evidence that the source disappeared.
    other='cam_right_wrist' if arm=='left' else 'cam_left_wrist'
    candidates=[]
    for camera in (other,'cam_head'):
        polygon=silhouette(ctx,item['model_key'],item['position'][:2],item['yaw'],camera)
        bbox=cv2.boundingRect(polygon.astype(np.float32));x,y,w,h=bbox
        rgb=image(ctx,camera)
        if x+w<=0 or y+h<=0 or x>=rgb.shape[1] or y>=rgb.shape[0]:continue
        check=dict(item,camera=camera,bbox=bbox)
        patch=_source_patch(ctx,check)
        pixels=np.count_nonzero(color_masks(patch)[item['color']]) if patch.size else 0
        if pixels>=8:candidates.append((pixels,check,patch))
    if candidates:
        _,check,patch=max(candidates,key=lambda candidate:candidate[0])
        return check,patch
    # Both fixed views can be occluded while the active close view still
    # sees the aperture. Compare against its positive earlier head RGB
    # observation; the separate current/lifted aperture checks remain required.
    reference=references.get(item.get('source_id'))
    if reference is None:raise RuntimeError('Recovered stacking source has no positive fixed-view RGB reference')
    check,patch=reference
    _log(ctx,'stack_source_reference_from_memory',source_id=item['source_id'],camera=check['camera'])
    return check,patch.copy()


def _stack_free_frames(ctx,arm,point,quat,duration):
    start=np.asarray(ctx.state[f'{arm}_ee_pose'],float);goal=tcp_to_link(point,quat)
    if min(ctx.tcp(arm)[2],point[2])<ctx.table_z+.08:return duration
    angle=2*math.acos(float(np.clip(abs(np.dot(start[3:]/np.linalg.norm(start[3:]),quat)),0,1)))
    return max(duration,int(math.ceil(np.linalg.norm(goal[:3]-start[:3])/.020)),
               int(math.ceil(angle/.18)))


def _stack_park_plan(ctx,arm):
    return _stack_cached_plan(ctx,_stack_park_plan_uncached,arm)


def _stack_park_plan_uncached(ctx,arm):
    """Clear the idle wrist on a continuous public branch without folding it."""
    measured=np.asarray(ctx.state[f'{arm}_ee_pose'][3:],float)
    for x,y,height in ((.48,-.36,.245),(.43,-.38,.225)):
        point=np.array([-x if arm=='left' else x,y,ctx.table_z+height])
        for quat in (down_quat(0),measured,down_quat(math.pi/2),down_quat(-math.pi/2)):
            if _path_ok(ctx,arm,[(ctx.tcp(arm),measured),(point,quat)]):
                return point,quat
    raise RuntimeError('Idle stacking wrist has no clear continuous retreat path')


def _stack_refined_lift(ctx,arm,source,quat,lift):
    """Recheck the wrist-refined contact instead of executing the coarse head pose."""
    # The held hole-to-TCP offset is measured after closure/lift. Its separate
    # carry check below uses that measurement rather than a nominal offset.
    for height in dict.fromkeys([lift]+[h for h in (.075,.055) if h<lift]):
        if _path_ok(ctx,arm,[(ctx.tcp(arm),ctx.state[f'{arm}_ee_pose'][3:]),
                             (source,quat),(_up(source,height),quat)]):
            if height!=lift:_log(ctx,'stack_refined_lift',arm=arm,before=lift,after=height)
            return height
    raise RuntimeError('Wrist-refined stacking source has no continuous lift path')


def _stack_carry_clear(ctx,item,start,hover,offset,base,done,pole_for_group):
    """Conservatively sweep the RGB-measured held body past the public pegs/stacks."""
    info,data=models()
    radius=float(np.linalg.norm(data[item['model_key']+'_points'][:,:2],axis=1).max())
    obstacles=[(np.asarray(peg)[:2],peg[2],.0049) for peg in base['pegs']]
    for group,count in enumerate(done):
        if not count:continue
        key=f'stackingblocks_{[0,2,1,3][group]:05d}'
        block_radius=float(np.linalg.norm(data[key+'_points'][:,:2],axis=1).max())
        peg=np.asarray(base['pegs'][pole_for_group[group]])
        obstacles.append((peg[:2],ctx.table_z+.011+count*info[key]['height'],block_radius))
    for fraction in np.linspace(0,1,61):
        top=(1-fraction)*np.asarray(start)+fraction*np.asarray(hover)+offset
        bottom=top[2]-item['height']
        for xy,height,obstacle_radius in obstacles:
            if np.linalg.norm(top[:2]-xy)<radius+obstacle_radius+.001 and bottom<height+.003:
                return False
    return True


def _stack_final_frames(ctx,item,plan,park=None):
    """Count the last nominal route with the unchanged public numeric controller."""
    from copy import copy
    if not ctx.numeric_ik:return None
    preview=copy(ctx)
    preview.obs=dict(ctx.obs)
    preview.obs['state']={key:np.asarray(value,float).copy() for key,value in ctx.state.items()}
    preview.joint_targets={arm:np.asarray(value,float).copy() for arm,value in ctx.joint_targets.items()}
    preview.gripper_targets=dict(ctx.gripper_targets)
    def frames(actions):
        count=0
        for action in actions:
            for side in ('left','right'):
                joints=np.asarray(action[f'{side}_arm_joint_state'],float).copy()
                preview.state[f'{side}_arm_joint_state']=joints
                preview.state[f'{side}_ee_pose']=preview.kinematics.forward(side,joints)
            preview.steps+=1;count+=1
        return count
    park_frames=0
    if park is not None:
        other,ready,park_quat=park
        park_frames=frames(_move(preview,other,ready,park_quat,1.0,duration=8))
    source,grasp_offset,target,choice=plan
    arm,yaw,approach,lift,clearance=choice;quat=down_quat(yaw)
    target=np.asarray(target,float).copy();target[:2]+=grasp_offset
    hover=_up(target,clearance)
    count=frames(_move(preview,arm,_up(source,approach),quat,1.0,duration=8))
    count+=frames(_move(preview,arm,source,quat,1.0,duration=8,precision=.001))
    count+=frames(preview.grip(arm,0.0,duration=6))+frames(preview.hold(2))
    count+=frames(_move(preview,arm,_up(source,lift),quat,0.0,duration=8))
    count+=frames(_move(preview,arm,hover,quat,0.0,duration=10,precision=.001))
    count+=frames(_move(preview,arm,target,quat,0.0,duration=12,precision=.0007))
    count+=frames(preview.hold(3))+frames(preview.grip(arm,1.0,duration=4))+frames(preview.hold(3))
    retract=hover.copy();retract[2]=max(retract[2],preview.table_z+.09)
    count+=frames(_move(preview,arm,retract,quat,1.0,duration=8))
    return count,park_frames

def _stack_insert_settle(ctx,arm,target):
    """Keep at most the original three insert holds; stop on measured stability."""
    if not ctx.numeric_ik:
        yield from ctx.hold(3)
        return
    target=np.asarray(target,float)
    for wait in range(3):
        before_tcp=np.asarray(ctx.tcp(arm),float).copy()
        before_joints=np.asarray(ctx.state[f'{arm}_arm_joint_state'],float).copy()
        before_grip=float(ctx.state[f'{arm}_ee_joint_state'][0])
        yield from ctx.hold(1)
        tcp=np.asarray(ctx.tcp(arm),float)
        error=float(np.linalg.norm(tcp-target))
        tcp_shift=float(np.linalg.norm(tcp-before_tcp))
        joint_shift=float(np.max(np.abs(np.asarray(ctx.state[f'{arm}_arm_joint_state'])-before_joints)))
        grip_shift=abs(float(ctx.state[f'{arm}_ee_joint_state'][0])-before_grip)
        # This only removes a redundant waiting action. It never accepts a
        # farther insertion target, changes a command, or shortens the waits
        # after opening the gripper. Moving or uncertain feedback keeps all
        # three original holds, including cases resting against a peg.
        if (error<.0005 and tcp_shift<.00005 and
                joint_shift<.0005 and grip_shift<.001):
            break
    _log(ctx,'stack_insert_settle',frames=wait+1,error=error,
         tcp_shift=tcp_shift,joint_shift=joint_shift,grip_shift=grip_shift)


def play_stacking_toy(ctx):
    """Place the ten measured hole centers over the four measured pegs."""
    initial_counts,initial_items=_stack_cardinality(ctx)
    initial_references=[(item,_source_patch(ctx,item)) for item in initial_items]
    yield from _park(ctx)
    counts,items=_stack_cardinality(ctx)
    manifest=list(items)
    if sorted(counts)!=[1,2,3,4]:
        # Only fuse the fixed head view: wrist illumination can change the
        # blue/purple labels. The source pieces have not been touched yet.
        combined=list(items)
        for item in initial_items:
            if not any(np.linalg.norm(item['position'][:2]-other['position'][:2])<.012
                       for other in combined):
                combined.append(item)
        counts=[sum(item['group']==group for item in combined) for group in range(4)]
        manifest=combined
    if sorted(counts)!=[1,2,3,4]:
        raise RuntimeError(f'RGB source cardinalities do not identify all four stacking groups: {counts}')
    manifest=[dict(item,source_id=index) for index,item in enumerate(manifest)]
    references={}
    for item in manifest:
        reference_candidates=[(item,_source_patch(ctx,item))]+[(prior,patch) for prior,patch in initial_references
                if prior['group']==item['group'] and np.linalg.norm(prior['position'][:2]-item['position'][:2])<.012]
        reference_candidates=[(np.count_nonzero(color_masks(patch)[item['color']]),prior,patch)
                for prior,patch in reference_candidates if patch.size]
        if reference_candidates:
            pixels,prior,patch=max(reference_candidates,key=lambda candidate:candidate[0])
            if pixels>=8:references[item['source_id']]=(prior,patch.copy())
    used_sources=set();failed_memory_sources=set()
    # The public task has four capacities, but assigns the shapes to them
    # differently between layouts. Four pieces use the longest pole, etc.
    pole_for_group=[4-count for count in counts]
    ctx.memory['stack_assignment']=dict(counts=counts,pole_for_group=pole_for_group,
                                         initial_counts=initial_counts)
    _log(ctx,'stack_shape_assignment',counts=counts,pole_for_group=pole_for_group)
    ctx.stage='measure_stack_base'
    base=detect_stack_base(ctx)
    if base is None:
        raise RuntimeError('RGB stacking base measurement missing')
    ctx.memory['stack_base']={key:base[key] for key in ('position','yaw','fit_error','pegs')}
    done=[0,0,0,0]
    handled=[]
    total_retries=0
    empty_grasp_faces={}
    previous_arm=None
    previous_quat=None
    manipulation_costs=[]
    # Short peg first is a cheap calibration check, then long groups.
    for group in sorted(range(4),key=lambda group:counts[group]):
        view_retries=0
        while done[group]<counts[group]:
            reserve=37
            estimate=max(85,int(np.median(manipulation_costs[-3:])) if manipulation_costs else 95)
            last_numeric=10-sum(done)==1 and ctx.numeric_ik
            if ctx.steps+estimate+reserve>1200 and not last_numeric:
                ctx.memory['stack_completed']=done.copy()
                raise RuntimeError(f'Stacking action budget exhausted with {10-sum(done)} RGB pieces remaining')
            ctx.stage=f'measure_stack_group_{group}_{done[group]}'
            items=detect_stack_blocks(ctx)
            options=[item for item in items if item['group']==group and
                     min(np.linalg.norm(item['position'][:2]-peg[:2]) for peg in base['pegs'])>.034]
            if not options:
                # These source identities came from the original head RGB.
                # Fit their actual current images in already available wrists
                # instead of spending 20--30 action frames clearing a view.
                for prior in manifest:
                    if prior['group']!=group:continue
                    for camera in ('cam_left_wrist','cam_right_wrist'):
                        recovered=detect_stack_known_source(ctx,prior,camera)
                        if recovered is None:continue
                        if min(np.linalg.norm(recovered['position'][:2]-peg[:2]) for peg in base['pegs'])<=.034:continue
                        if not any(np.linalg.norm(recovered['position'][:2]-other['position'][:2])<.012 for other in options):
                            options.append(recovered)
            if not options:
                # The same-episode RGB memory supplies a viewing waypoint,
                # never permission to contact an unseen object. Verify its
                # current aperture after the normal pregrasp approach below.
                options=[dict(prior,memory_recovered=True) for prior in manifest
                         if prior['group']==group and prior['source_id'] not in used_sources and
                         prior['source_id'] not in failed_memory_sources]
            if not options:
                if view_retries<2 and ctx.steps+estimate+reserve+35<1200:
                    view_retries+=1
                    _log(ctx,'clear_stack_view',group=group)
                    if previous_arm is None:
                        yield from _park(ctx)
                    else:
                        # The inactive arm was already parked at the last
                        # switch. Clear the wrist that just covered the peg.
                        ctx.stage='clear_stack_view'
                        ready,park_quat=_stack_park_plan(ctx,previous_arm)
                        yield from _move(ctx,previous_arm,ready,park_quat,1.0,duration=8)
                    continue
                _log(ctx,'stack_piece_missing',group=group,count=done[group])
                break
            grasp_height=.016
            peg=np.asarray(base['pegs'][pole_for_group[group]],float)
            unused=[prior for prior in manifest if prior['group']==group and prior['source_id'] not in used_sources]
            for candidate in options:
                if 'source_id' not in candidate and unused:
                    candidate['source_id']=min(unused,key=lambda prior:
                            np.linalg.norm(prior['position'][:2]-candidate['position'][:2]))['source_id']
            ordered=sorted(options,key=lambda obj:np.linalg.norm(obj['position'][:2]-ctx.tcp(previous_arm)[:2])
                           if previous_arm else obj['fit_error'])
            selected=None
            # Try all measured sources on the current arm before committing
            # to a costly switch. Every proposal retains the full path checks.
            for preferred in ([previous_arm,None] if previous_arm else [None]):
                for candidate in ordered:
                    failed_yaws=empty_grasp_faces.get(candidate.get('source_id'),[])
                    allowed_yaws=_stack_empty_retry_yaws(candidate,failed_yaws)
                    if allowed_yaws==[]:continue
                    try:plan=_stack_item_plan(ctx,candidate,done[group],peg,preferred,allowed_yaws)
                    except RuntimeError:continue
                    if allowed_yaws is not None:candidate=dict(candidate,empty_grasp_retry=True)
                    selected=(candidate,plan);break
                if selected:break
            if selected is None:raise RuntimeError('No remaining RGB stacking source has a continuous placement path')
            item,(source,grasp_offset,target,choice)=selected
            arm,yaw,approach,lift,hover_height=choice
            if item.get('empty_grasp_retry'):
                _log(ctx,'stack_empty_grasp_face_retry',group=group,
                     source_id=item['source_id'],previous_yaws=failed_yaws,grasp_yaw=yaw)
            quat=down_quat(yaw)
            approach_frames=_stack_free_frames(ctx,arm,_up(source,approach),quat,8)
            estimate+=max(0,approach_frames-20)
            park_estimate=0
            if previous_arm is not None and previous_arm!=arm:
                ready,park_quat=_stack_park_plan(ctx,previous_arm)
                park_estimate=_stack_free_frames(ctx,previous_arm,ready,park_quat,8)+8
            if last_numeric:
                park=None if not park_estimate else (previous_arm,ready,park_quat)
                try:exact_frames=_stack_final_frames(ctx,item,(source,grasp_offset,target,choice),park)
                except RuntimeError:exact_frames=None
                if exact_frames is not None:
                    estimate,park_estimate=exact_frames
                    _log(ctx,'stack_final_numeric_budget',piece_frames=estimate,park_frames=park_estimate,
                         reserve_home=reserve,mode='public_numeric_ideal')
            _log(ctx,'stack_action_budget',remaining=10-sum(done),estimated_piece=estimate,
                 estimated_park=park_estimate,reserve_home=reserve,arm=arm,
                 source=item['position'].tolist())
            if ctx.steps+estimate+park_estimate+reserve>1200:
                ctx.memory['stack_completed']=done.copy()
                raise RuntimeError('Measured stacking travel leaves insufficient action budget for the next piece and home')
            if item.get('wrist_recovered'):
                _log(ctx,'stack_source_from_wrist',group=group,camera=item['camera'],
                     position=item['position'].tolist(),error=item['fit_error'])
            if item.get('memory_recovered'):
                _log(ctx,'stack_source_view_from_memory',group=group,source_id=item['source_id'],
                     position=item['position'].tolist())
            if previous_arm is not None and previous_arm!=arm:
                # Adjacent pegs are only 64 mm apart. Leaving the other wrist
                # over them physically blocks the next arm's carry.
                ctx.stage='clear_other_stack_arm'
                yield from _move(ctx,previous_arm,ready,park_quat,1.0,duration=8)
            ctx.stage=f'pregrasp_stack_{group}_{done[group]}'
            manipulation_start=ctx.steps
            recovered=item.get('wrist_recovered') or item.get('memory_recovered')
            retry_face=item.get('empty_grasp_retry',False)
            if not recovered:check_item,before=_stack_source_evidence(ctx,item,arm,references)
            yield from _move(ctx,arm,_up(source,approach),quat,1.0,duration=8)
            hole_estimate=source.copy();hole_estimate[:2]-=grasp_offset
            refined=observe_stack_hole(ctx,item,arm,ctx.table_z+item['height'],hole_estimate)
            if (recovered or retry_face) and refined is None:
                if item.get('memory_recovered'):
                    failed_memory_sources.add(item['source_id'])
                    _log(ctx,'stack_memory_source_not_observed',source_id=item['source_id'],group=group)
                    previous_arm=arm;previous_quat=quat
                    continue
                raise RuntimeError('Recovered RGB source has no current aperture in the active wrist')
            if refined is not None:
                if recovered or retry_face:
                    change=_stack_yaw_change(item,refined['body_yaw'])
                    item=dict(item,yaw=item['yaw']+change)
                    yaw+=change;quat=down_quat(yaw)
                    grasp_offset=_rotate([0,.0092566],item['yaw']) if group==0 else np.zeros(2)
                source[:2]=refined['position'][:2]+grasp_offset
                if recovered:
                    position=np.asarray(item['position'],float).copy();position[:2]=refined['position'][:2]
                    item=dict(item,position=position)
                _log(ctx,'source_hole',group=group,position=refined['position'].tolist(),
                     radius=refined['radius'],error=refined['fit_error'],
                     body_yaw=refined['body_yaw'],grasp_yaw=yaw)
                lift=_stack_refined_lift(ctx,arm,source,quat,lift)
            if recovered:check_item,before=_stack_source_evidence(ctx,item,arm,references)
            yield from _move(ctx,arm,source,quat,1.0,duration=8,precision=.001)
            yield from ctx.grip(arm,0.0,duration=6)
            yield from ctx.hold(2)
            yield from _move(ctx,arm,_up(source,lift),quat,0.0,duration=8)
            held=_changed(ctx,check_item,before)
            if not held:
                total_retries+=1
                _log(ctx,'stack_grasp_retry',group=group)
                actual_grip=float(np.asarray(ctx.state[f'{arm}_ee_joint_state']).reshape(-1)[0])
                if group==1 and item.get('source_id') is not None and actual_grip<.02:
                    empty_grasp_faces.setdefault(item['source_id'],[]).append(float(yaw))
                    _log(ctx,'stack_empty_grasp_face',group=group,source_id=item['source_id'],
                         grasp_yaw=yaw,actual_grip=actual_grip)
                yield from ctx.grip(arm,1.0,4)
                yield from ctx.home(arm,duration=8)
                if total_retries>2:
                    break
                continue
            # Measure the carried hole-to-TCP offset rather than assuming closure centered it.
            tcp=ctx.tcp(arm)
            held_top=tcp[2]+item['height']-grasp_height
            hole_estimate=tcp.copy();hole_estimate[:2]-=grasp_offset
            hole=observe_stack_hole(ctx,item,arm,held_top,hole_estimate,optimize_z=True)
            if hole is not None and not _stack_hole_precise(ctx,hole):
                _log(ctx,'held_hole_uncertain',group=group,error=hole['fit_error'],metric_error=hole['metric_error'])
                hole=None
            if hole is None and (group==0 or recovered or retry_face):
                # A same-colored source behind a lifted piece can join its
                # contour. Recover a clear view before accepting its offset.
                # Source departure alone never authorizes insertion.
                for sideways,forward in ((.06,.12),(.02,.08)):
                    tcp=ctx.tcp(arm)
                    view=tcp.copy()
                    view[0]=np.clip(view[0]+(-sideways if arm=='left' else sideways),-.45,.45)
                    view[1]=max(-.36,view[1]-forward)
                    view_offset=np.r_[-grasp_offset,item['height']-grasp_height]
                    if not _stack_carry_clear(ctx,item,tcp,view,view_offset,base,done,pole_for_group):continue
                    if not _path_ok(ctx,arm,[(tcp,quat),(view,quat)]):continue
                    ctx.stage='measure_held_stack_in_clear_view'
                    yield from _move(ctx,arm,view,quat,0.0,duration=6,precision=.001)
                    tcp=ctx.tcp(arm)
                    hole_estimate=tcp.copy();hole_estimate[:2]-=grasp_offset
                    hole=observe_stack_hole(ctx,item,arm,tcp[2]+item['height']-grasp_height,
                                            hole_estimate,optimize_z=True)
                    if hole is not None and _stack_hole_precise(ctx,hole):
                        _log(ctx,'held_stack_view_recovered',group=group,error=hole['fit_error'],
                             value_floor=hole['value_floor'],position=hole['position'].tolist())
                        break
                    hole=None
                if hole is None:
                    raise RuntimeError('The carried stacking hole could not be measured within its public bore clearance')
            if (recovered or retry_face) and hole is None:
                raise RuntimeError('Recovered stacking piece did not remain measurable with the lifted wrist')
            held_offset=np.r_[-grasp_offset,item['height']-grasp_height] if hole is None else hole['position']-tcp
            offset=held_offset[:2]
            if hole is not None:
                _log(ctx,'held_hole',group=group,position=hole['position'].tolist(),
                     tcp=tcp.tolist(),radius=hole['radius'],error=hole['fit_error'])
            if np.linalg.norm(offset)>.020:
                raise RuntimeError('Carried stacking hole offset exceeds the visible object radius')
            target[:2]-=offset
            desired_top=ctx.table_z+.011+(done[group]+1)*item['height']
            target[2]=desired_top-held_offset[2]
            hover=None
            minimum_hover=max(.020,peg[2]+item['height']+.020-held_offset[2]-target[2])
            for clearance in dict.fromkeys([hover_height,max(.035,minimum_hover),minimum_hover]):
                candidate=_up(target,clearance)
                candidate[2]=max(candidate[2],peg[2]+item['height']+.020-held_offset[2])
                if not _stack_carry_clear(ctx,item,ctx.tcp(arm),candidate,held_offset,base,done,pole_for_group):
                    continue
                if _path_ok(ctx,arm,[(ctx.tcp(arm),ctx.state[f'{arm}_ee_pose'][3:]),
                                     (candidate,quat),(target,quat)]):
                    hover=candidate;hover_height=clearance;break
            if hover is None:
                raise RuntimeError('RGB-held stacking body has no clear continuous insertion path')
            ctx.stage=f'insert_stack_{group}_{done[group]}'
            yield from _move(ctx,arm,hover,quat,0.0,duration=10,precision=.001)
            # Re-measure after carry: closure and acceleration can shift the
            # object relative to the two pads.
            tcp=ctx.tcp(arm)
            hole_estimate=tcp.copy();hole_estimate[:2]+=held_offset[:2]
            hover_hole=observe_stack_hole(ctx,item,arm,tcp[2]+held_offset[2],hole_estimate,optimize_z=True)
            if hover_hole is not None and not _stack_hole_precise(ctx,hover_hole):
                _log(ctx,'carry_hole_uncertain',group=group,error=hover_hole['fit_error'],metric_error=hover_hole['metric_error'])
                hover_hole=None
            if hover_hole is not None:
                new_offset=hover_hole['position']-tcp
                if np.linalg.norm(new_offset[:2])<.020:
                    held_offset=new_offset
                    offset=held_offset[:2]
                    target[:2]=peg[:2]-offset
                    target[2]=desired_top-held_offset[2]
                    _log(ctx,'carry_hole_offset',group=group,offset=offset.tolist(),
                         error=hover_hole['fit_error'],method=hover_hole['method'])
            # Peg tips in the same close view provide a final metric XY correction.
            # The independent base fit is more reliable than arbitrary white
            # circular texture regions. Aperture observations above determine
            # the held offset; a future peg servo must use the visible aperture.
            yield from _move(ctx,arm,target,quat,0.0,duration=12,precision=.0007)
            yield from _stack_insert_settle(ctx,arm,target)
            yield from ctx.grip(arm,1.0,duration=4)
            yield from ctx.hold(3)
            retract=hover.copy();retract[2]=max(retract[2],ctx.table_z+.09)
            yield from _move(ctx,arm,retract,quat,1.0,duration=8)
            # Retracting above the base leaves the scattered pieces visible.
            # Move straight to the next measured source, without a ready detour.
            handled.append((group,np.asarray(item['position'])[:2].copy()))
            done[group]+=1
            if 'source_id' in item:used_sources.add(item['source_id'])
            manipulation_costs.append(ctx.steps-manipulation_start)
            view_retries=0
            previous_arm=arm;previous_quat=quat
            _log(ctx,'stack_release',group=group,pole=pole_for_group[group],level=done[group],
                 target=target.tolist(),offset=offset.tolist(),source_lift=lift,
                 hover_clearance=hover_height)
    ctx.memory['stack_completed']=done
    if done!=counts:
        raise RuntimeError(f'Stacking incomplete: RGB counts {counts}, placed {done}')
    ctx.stage='home'
    yield from ctx.home(duration=25)
    yield from ctx.hold(12)




def organize_table(ctx):
    """Move the alarm, push the keyboard, and place mouse and upright figurine."""
    yield from _park(ctx)
    ctx.stage='measure_organize'
    scene=detect_organize(ctx)
    completed=[]
    alarm=scene.get('alarm');drawer=scene.get('drawer')
    if alarm is not None and drawer is not None:
        source=np.asarray(alarm['position'],float).copy()
        source[2]=ctx.table_z+alarm['height']*.52
        info,_=models();bounds=np.asarray(info[drawer['model_key']]['bounds'])
        # A front-rim placement stays on the measured drawer top while avoiding
        # a long reach toward the center/back of the cabinet.
        local=bounds.mean(0)[:2]
        local[1]=bounds[0,1]+alarm['size'][1]/2+.007
        target=np.asarray(drawer['position'],float).copy()
        target[:2]+=_rotate(local,drawer['yaw'])
        target[2]=ctx.table_z+drawer['height']+alarm['height']*.52+.002
        choices=[]
        for arm_candidate in ('left','right'):
            for q in (np.array([1.,0,0,0]),np.array([0.,0,0,1.])):
                cost=0.0
                for position in (source,_up(source,.12),target,_up(target,.08)):
                    result=ctx.kinematics.solve(arm_candidate,tcp_to_link(position,q),
                                  current_joints=ctx.state[f'{arm_candidate}_arm_joint_state'],
                                  position_tol=.001,rotation_tol=.008,max_starts=8,max_nfev=140)
                    if result['status']!='Success':break
                    cost+=result['joint_distance']
                else:choices.append((cost,arm_candidate,q))
        if not choices:
            raise RuntimeError('No feasible side grasp and drawer top placement for alarm')
        _,arm,quat=min(choices,key=lambda entry:entry[0])
        ctx.stage='pick_alarm'
        held=yield from _pick(ctx,alarm,source,arm,0.0,quat=quat)
        if held:
            ctx.stage='place_alarm_on_drawer'
            yield from _place(ctx,arm,target,0.0,clearance=.08,quat=quat)
            completed.append('alarm')
        yield from ctx.home(arm,duration=18)
    ctx.stage='measure_keyboard'
    scene=detect_organize(ctx)
    keyboard=scene.get('keyboard');frame=scene.get('frame_position')
    if keyboard is not None and frame is not None:
        center=_center(keyboard)
        size=np.asarray(keyboard['size'])
        # Opposed two-finger contact pushes the rear edge without needing a 160 mm jaw.
        source=center.copy();source[1]-=size[1]/2+.012;source[2]=ctx.table_z+.016
        target=np.asarray(frame,float).copy();target[1]-=size[1]/2+.012;target[2]=source[2]
        dx=min(size[0]*.27,.09)
        left=source+[-dx,0,0];right=source+[dx,0,0]
        ctx.stage='push_keyboard'
        yield from ctx.move_both(_up(left,.075),_up(right,.075),down_quat(0),down_quat(0),0.0,0.0,15)
        yield from ctx.move_both(left,right,down_quat(0),down_quat(0),0.0,0.0,12)
        for fraction in np.linspace(.2,1.0,5):
            point=(1-fraction)*source+fraction*target
            yield from ctx.move_both(point+[-dx,0,0],point+[dx,0,0],
                                    down_quat(0),down_quat(0),0.0,0.0,8)
        yield from ctx.move_both(_up(target+[-dx,0,0],.11),_up(target+[dx,0,0],.11),
                                 down_quat(0),down_quat(0),1.0,1.0,12)
        yield from ctx.home(duration=18)
        completed.append('keyboard')
    for kind in ('mouse','garage'):
        ctx.stage=f'measure_{kind}'
        scene=detect_organize(ctx)
        item=scene.get(kind)
        destination=scene.get('mousemat') if kind=='mouse' else scene.get('cushion_position')
        if item is None or destination is None:
            _log(ctx,'organize_measurement_missing',kind=kind)
            continue
        source=np.asarray(item['position'],float).copy()
        delta=-item['yaw'] if kind=='mouse' else 0.0
        grasp_fraction=.52 if kind=='mouse' else .78
        offset=np.zeros(2)
        if kind=='garage':
            _,data=models();cloud=data[item['model_key']+'_points']
            head=cloud[(cloud[:,2]>item['height']*.70)&(cloud[:,2]<item['height']*.87)]
            if len(head):
                offset=(head.min(0)[:2]+head.max(0)[:2])/2
                source[:2]+=_rotate(offset,item['yaw'])
        source[2]=ctx.table_z+item['height']*grasp_fraction
        target=_center(destination) if isinstance(destination,dict) else np.asarray(destination,float).copy()
        target[:2]+=_rotate(offset,item['yaw']+delta)
        target[2]=ctx.table_z+(0.007 if kind=='mouse' else .0001)+item['height']*grasp_fraction+.002
        arm,yaw,goal=_choose(ctx,source,target,_grasp_yaw(item),delta=delta,
                            allow_cross=kind=='garage' or max(item['size'][:2])<.074)
        ctx.stage=f'pick_{kind}'
        held=yield from _pick(ctx,item,source,arm,yaw,duration=14)
        if held:
            ctx.stage=f'place_{kind}'
            yield from _place(ctx,arm,target,goal,clearance=.13,duration=15)
            completed.append(kind)
        yield from ctx.home(arm,duration=18)
    ctx.memory['organized_objects']=completed
    ctx.stage='home'
    yield from ctx.home(duration=25)
    yield from ctx.hold(18)


TASKS={'pack_objects_into_box':pack_objects_into_box,
       'play_stacking_toy':play_stacking_toy,'organize_table':organize_table}
