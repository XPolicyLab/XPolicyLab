"""Dependent controller edits: consume geometry measured in the current episode."""
import ast

STAGING_DEFINITION = """def _staging_point(policy,env_idx,geometries,order,target_index):
    target=geometries[order[target_index]]
    support=geometries[order[target_index-1]]
    try:
        plane=np.asarray(policy.rgb_observed_shapes[env_idx][order[target_index]]['support_plane'],float)
    except (AttributeError,KeyError) as error:
        raise StackBlocksEnvelopeError('Observed support plane required for staging') from error
    if plane.shape!=(3,) or not np.isfinite(plane).all():
        raise StackBlocksEnvelopeError('Invalid observed staging support plane')
    delta=np.asarray(support.center[:2])-target.center[:2]
    distance=float(np.linalg.norm(delta))
    if distance<1e-8:return None
    perpendicular=np.array([-delta[1],delta[0]])/distance
    midpoint=(np.asarray(target.center[:2])+support.center[:2])/2
    radius=float(np.linalg.norm(target.half_extents[:2]))
    options=[]
    for offset in (0.,-2*radius,2*radius):
        xy=midpoint+offset*perpendicular
        z=float(plane[:2]@xy+plane[2]+np.linalg.norm(np.r_[-plane[:2],1.])*target.half_extents[2])
        point=np.r_[xy,z]
        clearance=min(np.linalg.norm(xy-g.center[:2])-radius-np.linalg.norm(g.half_extents[:2])
                      for label,g in geometries.items() if label!=order[target_index])
        if clearance>0:options.append((float(clearance),point))
    return max(options,key=lambda entry:entry[0])[1] if options else None
"""


def replace_assignment(text,name,replacement):
    lines=text.splitlines(keepends=True)
    nodes=[n for n in ast.walk(ast.parse(text)) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id==name for t in n.targets)]
    for node in sorted(nodes,key=lambda n:n.lineno,reverse=True):
        lines[node.lineno-1:node.end_lineno]=[' '*node.col_offset+replacement+'\n']
    return ''.join(lines)


def controller_edit(path,text,remove):
    if path.endswith('/nesting_rgb/model.py'):
        tree=ast.parse(text)
        episode=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='EpisodePolicy')
        initialize=next(n for n in episode.body if isinstance(n,ast.FunctionDef) and n.name=='__init__')
        reader_assignment=next(n for n in initialize.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Attribute) and t.attr=='reader' for t in n.targets))
        initialize.body[initialize.body.index(reader_assignment)+1:initialize.body.index(reader_assignment)+1]=ast.parse('import copy\nself.reader = copy.copy(self.reader)\nself.reader.head_size = copy.deepcopy(self.reader.head_size)').body
        reset=next(n for n in episode.body if isinstance(n,ast.FunctionDef) and n.name=='reset')
        reset.body[0:0]=ast.parse('from physicalrsi_observed_geometry.view_history import SupportViewHistory\nself.observed_support_views = SupportViewHistory()\nself.calibrated_obs = None\nself.geometry.calibrated_obs = None\nself.joint_control.reset()').body
        update=next(n for n in episode.body if isinstance(n,ast.FunctionDef) and n.name=='update_obs')
        update.body[0:0]=ast.parse('self.observed_support_views.observe(obs)\nself.calibrated_obs = obs\nself.geometry.calibrated_obs = obs').body
        reset.body.insert(0,ast.parse('self.reader.head_size.camera = None').body[0])
        update.body[0:0]=ast.parse('from physicalrsi_observed_geometry import Camera\nself.reader.head_size.camera = Camera.from_observation(obs)').body
        text=ast.unparse(ast.fix_missing_locations(tree))+'\n'
    elif path.endswith('/nesting_rgb/head_size.py'):
        tree=ast.parse(text)
        for node in tree.body:
            if isinstance(node,ast.ImportFrom) and node.module=='contours':
                node.names=[name for name in node.names if name.name not in {'CAM','ROT','FX'}]
        descriptors=[n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=='descriptor']
        if len(descriptors)!=2:raise ValueError('Expected two nesting head projection functions')
        for descriptor in descriptors:
            descriptor.body=ast.parse('''from physicalrsi_observed_geometry import GeometryUnavailable
if getattr(self, 'camera', None) is None:
    raise GeometryUnavailable('Current head camera calibration required')
uv,_=self.camera.project(g.points[rank]+[xy[0],xy[1],TABLE])
return g.descriptor(cv2.convexHull(uv.astype(np.float32)))''').body
        text=ast.unparse(ast.fix_missing_locations(tree))+'\n'
    elif path.endswith('/nesting_rgb/object_observer.py'):
        tree=ast.parse(text)
        for node in tree.body:
            if isinstance(node,ast.ImportFrom) and node.module=='contours':
                node.names=[name for name in node.names if name.name not in {'CAM','ROT','FX','ray'}]
        hull=next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=='hull')
        hull.body=ast.parse('''points=self.head_size.geometry.points[rank]+[xy[0],xy[1],TABLE]
uv,_=self.head_size.camera.project(points)
return cv2.convexHull(uv.astype(np.float32))''').body
        text=ast.unparse(ast.fix_missing_locations(tree))+'\n'
        text=text.replace('np.zeros((480, 640), np.uint8)', 'np.zeros((self.head_size.camera.height, self.head_size.camera.width), np.uint8)')
        text=text.replace("depth = abs((TABLE - CAM[2]) / ray(*doll['bottom_pixel'])[2])", "camera = self.head_size.camera\n        origin = camera.transform[:3, 3]\n        direction = camera.unproject([doll['bottom_pixel']], [1.])[0] - origin\n        depth = abs((TABLE - origin[2]) / direction[2])\n        focal = float(np.sqrt(camera.k[0, 0] * camera.k[1, 1]))")
        text=text.replace('depth / FX','depth / focal')
    elif path.endswith('/nesting_rgb/wrist_geometry.py'):
        tree=ast.parse(text)
        tree.body=[n for n in tree.body if not (isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id in {'F','MOUNT_T','MOUNT_R'} for t in n.targets))]
        cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='WristGeometry')
        hull=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='hull')
        hull.body=ast.parse('''from physicalrsi_observed_geometry.wrist_camera import current_wrist_camera
from physicalrsi_observed_geometry import GeometryUnavailable
camera=current_wrist_camera(getattr(self, 'calibrated_obs', None), ee)
# Object profile and support-height migration remains a separate pending change.
points=self.points[rank]+np.array([xy[0],xy[1],xy[2] if len(xy)>2 else .765])
try:
    uv,depth=camera.project(points)
except GeometryUnavailable:
    return None
if np.min(depth)<.002:return None
hull=cv2.convexHull(uv.astype(np.float32))
frame=np.array([[0,0],[camera.width-1,0],[camera.width-1,camera.height-1],[0,camera.height-1]],np.float32)
_,clipped=cv2.intersectConvexConvex(hull,frame)
return clipped''').body
        text=ast.unparse(ast.fix_missing_locations(tree))+'\n'
    elif path.endswith('/rdj_store_tools/model.py'):
        text=text.replace("self.reader.read(self.obs['vision']['cam_head'])", "self.reader.read(self.obs['vision']['cam_head'], observation=self.calibrated_obs)")
        tree=ast.parse(text)
        episode=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='EpisodePolicy')
        update=next(n for n in episode.body if isinstance(n,ast.FunctionDef) and n.name=='update_obs')
        update.body.insert(0,ast.parse('self.calibrated_obs=obs').body[0])
        count=0
        for node in ast.walk(episode):
            if isinstance(node,ast.Call) and isinstance(node.func,ast.Name) and node.func.id=='qualify':
                node.args.append(ast.parse('self.calibrated_obs',mode='eval').body);count+=1
        if count!=1:raise ValueError('Expected one store route qualification call')
        snapshot=next(n for n in episode.body if isinstance(n,ast.FunctionDef) and n.name=='snapshot')
        class SnapshotRGB(ast.NodeTransformer):
            def visit_Call(self,node):
                if isinstance(node.func,ast.Attribute) and isinstance(node.func.value,ast.Name) and node.func.value.id=='cv2' and node.func.attr=='imwrite':
                    replacement=ast.parse('Image.fromarray(img).save(filename)',mode='eval').body
                    replacement.args=[node.args[0]]
                    return replacement
                return self.generic_visit(node)
        SnapshotRGB().visit(snapshot)
        tree.body.insert(0,ast.ImportFrom(module='PIL',names=[ast.alias(name='Image')],level=0))
        text=ast.unparse(ast.fix_missing_locations(tree))+'\n'
    elif path.endswith('/rdj_store_tools/transport.py'):
        tree=ast.parse(text)
        qualify=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='qualify')
        qualify.args.args.append(ast.arg(arg='observation'))
        count=0
        for node in ast.walk(qualify):
            if isinstance(node,ast.Call) and isinstance(node.func,ast.Name) and node.func.id=='ik':
                node.args.append(ast.Name(id='observation',ctx=ast.Load()));count+=1
        if count!=1:raise ValueError('Expected one store numerical IK call')
        text=ast.unparse(ast.fix_missing_locations(tree))+'\n'
    elif path.endswith('/rdj_v1_stack_blocks_by_language/vendor/rdj_master/tasks/stack_blocks/stage_controller.py'):
        tree = ast.parse(text)
        methods = [node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == '_ordered_moves']
        if len(methods) != 1:
            raise ValueError('Expected one language-stack candidate ordering method')
        method = methods[0]
        lines = text.splitlines(keepends=True)
        lines[method.body[0].lineno-1:method.end_lineno] = [
            '        # Preserve planner ranking by feasibility and observed carried orientation.\n',
            '        return moves\n',
        ]
        text = ''.join(lines)
    elif path.endswith('/deposit_policy/deploy.py'):
        text=text.replace('{"color": obs["vision"][name]["color"]}',
                          '{key: obs["vision"][name][key] for key in ("color", "intrinsic_matrix", "extrinsic_matrix")}')
    elif path.endswith('/deposit_policy/model.py'):
        text=text.replace('from .policy_ik import PolicyIK\n', '')
        text=text.replace('        self.frames.ik = PolicyIK()\n', '')
        text=text.replace('self.frames.states.clear()', 'self.frames.reset()')
        text=text.replace("self.frames.states[index] = obs['state']", "self.frames.bind_observation(index, obs)")
        text=text.replace("        envelope = _build_envelope(",
            "        inspection = self.core._perception.inspection_envelope(self.frames, index, obs)\n"
            "        if inspection is not None:\n"
            "            self.core._executor._slots.pop(index, None)\n"
            "            return inspection\n"
            "        if index in self.core._perception.inspected:\n"
            "            self.core._perception.inspected.remove(index)\n"
            "            self.core._executor._slots.pop(index, None)\n"
            "            self.counts[index] = 0\n"
            "        envelope = _build_envelope(")
    elif path.endswith('/deposit_policy/stages.py'):
        text=remove(text, ['_HANDOFF_SOURCE_SEMANTIC_POSES'])
        text=text.replace('def _receiver_handoff_pose(\n', 'def _receiver_handoff_pose(\n    frames: Any,\n')
        text=text.replace('source_pose = _HANDOFF_SOURCE_SEMANTIC_POSES[source_arm]',
                          'source_pose = _ee_pose_to_semantic(frames.get_robot_by_arm_name(source_arm), frames.fk(source_arm, _HANDOFF_SOURCE_JOINT_TARGETS[source_arm]))')
        text=text.replace('receiver_pose = _HANDOFF_SOURCE_SEMANTIC_POSES[bank_arm]',
                          'receiver_pose = _ee_pose_to_semantic(frames.get_robot_by_arm_name(bank_arm), frames.fk(bank_arm, _HANDOFF_SOURCE_JOINT_TARGETS[bank_arm]))')
    elif path.endswith('/rdj_insert_policy/planning.py'):
        text=text.replace('runtime.ik.solve(arm, seed_joints, pose)',
            'runtime.ik.solve(arm, seed_joints, pose, observed_state=runtime.observations[env_idx]["state"], support=runtime.perception.support_geometry(env_idx), env_idx=env_idx)')
        text=remove(text,['_TUBE_GRASP_LOCAL','_EXPECTED_ATTACHMENT_X'])
        text=text.replace('def _pickup_contacts(tube_pose: Sequence[float])','def _pickup_contacts(tube_pose: Sequence[float], tube_length: float)')
        text=text.replace('object_rotation @ _TUBE_GRASP_LOCAL','object_rotation @ np.asarray([0.0, tube_length / 2, 0.0])')
        text=text.replace('slot_poses: Sequence[Sequence[float]]) -> tuple[_TubeCandidate, ...]:','slot_poses: Sequence[Sequence[float]], tube_length: float) -> tuple[_TubeCandidate, ...]:')
        text=text.replace('_pickup_contacts(tube_pose)','_pickup_contacts(tube_pose, tube_length)')
        text=retain_insert_pose_choices(continuous_insert_solutions(search_insert_axial_orientations(remove_insert_attachment_prior(text))))
    elif path.endswith('/rdj_insert_policy/policy.py'):
        text=text.replace('RgbInsertTubesPerception()','RgbInsertTubesPerception(self.robot_geometry)')
        text=text.replace('        self.perception.observe(env_idx, obs)',
            '        _was_ready = self.perception.is_ready(env_idx)\n'
            '        self.perception.observe(env_idx, obs)\n'
            '        if not _was_ready and self.perception.is_ready(env_idx) and self.plans.get(env_idx, {}).get("skill_choice") == "infeasible":\n'
            '            self.plans.pop(env_idx, None)\n'
            '            self.contexts.pop(env_idx, None)\n'
            '            self.executor._slots.pop(env_idx, None)\n'
            '            self.action_counts[env_idx] = 0')
        text=text.replace('self.action_counts[0] += len(actions)', 'self.action_counts[0] += len(actions)\n        _trace_insert_actions(self, 0, actions)')
        text=text.replace('self.action_counts[env_idx] += len(chunk)', 'self.action_counts[env_idx] += len(chunk)\n            _trace_insert_actions(self, env_idx, chunk)')
        text = insert_revision_imports(text, "from physicalrsi_observed_geometry.insert_obstacles import activate_completed_placements\n")
        text = text.replace('self.contexts = {}', 'self.contexts = {}\n        self.ik.placed_tube_scenes = {}')
        text = text.replace('_refresh_after_lift(self, env_idx, plan, observation)', 'activate_completed_placements(self, env_idx, plan)\n        _refresh_after_lift(self, env_idx, plan, observation)')
        text += INSERT_ACTION_TRACE

    elif path.endswith('/rdj_insert_policy/refresh.py'):
        text=remove(text,['_HELD_ROOT_IN_GRIPPER','_CALIBRATED_ROOT_RESIDUAL'])
        text=text.replace("'slot_pose': list(slot_poses[item.slot_index])", "'slot_pose': list(item.slot_pose)")
        text=text.replace('    tube_poses, slot_poses = _initial(value, env_idx)', '''    tube_poses, slot_poses = _initial(value, env_idx)
    tube_lengths = _numbers(value.get('tube_lengths'), len(tube_poses), 'observed tube lengths')
    if any(length <= 0 for length in tube_lengths):
        raise InsertTubesEnvelopeError('Observed tube lengths must be positive')''')
        text=text.replace("f'tube{index}', pose, slot_poses)","f'tube{index}', pose, slot_poses, tube_lengths[index])")
        text=text.replace("_pickup_contacts(tube_poses[int(item.label.removeprefix('tube'))])", "_pickup_contacts(tube_poses[int(item.label.removeprefix('tube'))], tube_lengths[int(item.label.removeprefix('tube'))])")
        text=text.replace("'slot_index': item.slot_index,", "'slot_index': item.slot_index, 'tube_length': tube_lengths[int(item.label.removeprefix('tube'))],")
        text=text.replace("    for context in contexts[env_idx].values():", """    for context in contexts[env_idx].values():
        context['held_root_in_gripper'] = (_rotation(context['pickup_contact'][3:]).T @ (np.asarray(context['tube_pose'][:3]) - np.asarray(context['pickup_contact'][:3]))).tolist()""")
        text=text.replace('_HELD_ROOT_IN_GRIPPER.tolist()', 'context["held_root_in_gripper"]')
        text=text.replace('_HELD_ROOT_IN_GRIPPER', 'np.asarray(context["held_root_in_gripper"])')
        text=text.replace('0.1404','context["tube_length"]')
        text=text.replace('root_residual = _CALIBRATED_ROOT_RESIDUAL[arm].copy()', "context['realigned'] = True\n        return  # No unseen cap correction from fixture-specific residuals.")
        text=text.replace("('calibrated' if cap_xy is None else 'ready')", "'ready'")
        text=late_insert_observation(condition_insert_observation(stereo_insert_delivery(commit_insert_axis_on_feasible_replan(text))))
        text=text.replace('assignment = _best_assignment(groups)', "print('[insert_candidates]', env_idx, [{'count': len(group), 'slots': sorted({item.slot_index for item in group})} for group in groups], flush=True)\n    assignment = _best_assignment(groups)")
        text = insert_revision_imports(text, "from physicalrsi_observed_geometry.insert_obstacles import observed_assignment\nfrom .planning import _MIN_SLOT_DISTANCE\nfrom .ik import observed_base\nfrom .geometry import semantic_pose_to_ee\n")
        text = text.replace('assignment = _best_assignment(groups)', 'assignment = observed_assignment(runtime, env_idx, groups, value, robot, _stages, _pickup_contacts, _delivery_poses, _MIN_SLOT_DISTANCE, observed_base, lambda pose: semantic_pose_to_ee(runtime.robot_geometry, pose))')
    elif path.endswith('/deposit_policy/planning.py'):
        # Only pickup waypoints are used here; the slot argument is irrelevant to those.
        text=replace_assignment(text,'nominal_slot','nominal_slot = list(coin_pose)')
        text=text.replace('_receiver_handoff_pose(source_arm, bank_arm, coin_in_source)', '_receiver_handoff_pose(frames, source_arm, bank_arm, coin_in_source)')
        text=deposit_preinsert_feedback(text)
    elif path.endswith('/rdj_v1_stack_blocks_by_language/joint_action.py'):
        original_call = 'joints = policy.ik.solve(arm, current, target.tolist())'
        assert text.count(original_call) == 1, 'Language stack joint conversion caller changed'
        text = text.replace(original_call, '''support = getattr(policy, "rgb_observed_support", {}).get(index)
                if not support:
                    raise ValueError("Current observed support required for stack joint conversion")
                joints = policy.ik.solve(arm, current, target.tolist(), observed_state=state, support=support, env_idx=index)''')
    elif path.endswith('/rdj_v1_stack_blocks_by_language/planning.py'):
        original_call='joints = policy.ik.solve(robot.arm, policy.observations[env_idx]["state"][f"{robot.arm}_arm_joint_state"], _semantic_to_ee(robot, pose))'
        assert text.count(original_call)==1, 'Language stack IK caller changed'
        observed_call='support = getattr(policy, "rgb_observed_support", {}).get(env_idx)\n    if not support:\n        raise StackBlocksEnvelopeError("Current observed support required for stack IK")\n    joints = policy.ik.solve(robot.arm, policy.observations[env_idx]["state"][f"{robot.arm}_arm_joint_state"], _semantic_to_ee(robot, pose), observed_state=policy.observations[env_idx]["state"], support=support, env_idx=env_idx)'
        text=text.replace(original_call,observed_call)
        lines=text.splitlines(keepends=True)
        node=next(n for n in ast.parse(text).body if isinstance(n,ast.FunctionDef) and n.name=='_staging_point')
        lines[node.lineno-1:node.end_lineno]=[STAGING_DEFINITION]
        text=''.join(lines)
        text=text.replace('_staging_point(geometries,order,1)', '_staging_point(policy,env_idx,geometries,order,1)')
        text=text.replace('_staging_point(geometries,state.order,state.target_index)', '_staging_point(policy,env_idx,geometries,state.order,state.target_index)')
    elif path.endswith('/rdj_v1_stack_blocks_by_language/perception_state.py'):
        text=text.replace('g.rotation[:]=np.asarray(d.rotation)', 'g.rotation[:]=np.asarray(d.rotation)\n            g.half_extents[:]=np.asarray(d.half_extents)\n            g.minimum[:]=g.center-g.half_extents\n            g.maximum[:]=g.center+g.half_extents')
        text=text.replace('<=.0175 and abs(target.center[2]-.7825)', '<=xy_threshold and abs(target.center[2]-state.stage_target[2])')
    elif path.endswith('/rdj_v1_stack_blocks_by_language/batch_rgb.py'):
        text=text.replace('HISTORY = (','HISTORY = (\'rgb_observed_support\',\'rgb_observed_shapes\',').replace('HISTORY=(',"HISTORY=('rgb_observed_support','rgb_observed_shapes',")
        text=text.replace('            getattr(policy, name)[index] = value', '            existing = getattr(policy, name).get(index)\n            if name != \"rgb_observed_support\" or existing != value:\n                getattr(policy, name)[index] = value')
    return text


def tighten_press_completion(text):
    """Use measured-endpoint precision, not an approach tolerance, for press completion."""
    tree=ast.parse(text)
    function=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='_facts')
    changed=0
    for node in ast.walk(function):
        if isinstance(node,ast.Call) and isinstance(node.func,ast.Name) and node.func.id=='pose_reached':
            for keyword in node.keywords:
                if keyword.arg=='position_tolerance_m':
                    keyword.value=ast.Constant(.0002)
                    changed+=1
    if changed!=1:raise ValueError('Expected exactly one press completion pose gate')
    lines=text.splitlines(keepends=True)
    lines[function.lineno-1:function.end_lineno]=[ast.unparse(function)+'\n']
    return ''.join(lines)


def remove_insert_attachment_prior(text):
    """Keep the observed pickup transform instead of calibrated attachment offsets."""
    tree=ast.parse(text)
    function=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='_delivery_poses')
    function.body=[node for node in function.body if not (
        isinstance(node,ast.If) and any(isinstance(child,ast.Name) and child.id=='expected_attachment_x' for child in ast.walk(node.test)))]
    for node in ast.walk(tree):
        if isinstance(node,ast.Call) and isinstance(node.func,ast.Name) and node.func.id=='_delivery_poses':
            node.args=node.args[:3]
            node.keywords=[item for item in node.keywords if item.arg!='expected_attachment_x']
    return ast.unparse(tree)+'\n'


SLOT_ORIENTATIONS = """def _slot_orientations(slot_pose):
    # Circular openings constrain insertion axis, not axial yaw.
    # Search equivalent cylinder orientations without changing measured geometry.
    rotation = _rotation(slot_pose[3:])
    for yaw in tuple(np.arange(8) * np.pi / 4):
        c, s = np.cos(yaw), np.sin(yaw)
        axial = np.asarray([[c,-s,0.],[s,c,0.],[0.,0.,1.]])
        yield list(slot_pose[:3]) + _quaternion(rotation @ axial)
"""


def search_insert_axial_orientations(text):
    tree=ast.parse(text)
    candidate=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='_TubeCandidate')
    candidate.body.append(ast.parse('slot_pose: tuple[float, ...]').body[0])
    function=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='_tube_candidates')
    loops=[n for n in ast.walk(function) if isinstance(n,ast.For) and isinstance(n.target,ast.Tuple)
           and any(isinstance(t,ast.Name) and t.id=='slot_index' for t in n.target.elts)]
    if len(loops)!=1:raise ValueError('Expected one observed slot candidate loop')
    loop=loops[0]
    for name in loop.target.elts:
        if isinstance(name,ast.Name) and name.id=='slot_pose':name.id='observed_slot_pose'
    nested=ast.parse('for slot_pose in _slot_orientations(observed_slot_pose):\n    pass').body[0]
    nested.body=loop.body;loop.body=[nested]
    calls=[n for n in ast.walk(function) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='_TubeCandidate']
    if len(calls)!=1:raise ValueError('Expected one insert candidate constructor')
    calls[0].keywords.append(ast.keyword(arg='slot_pose',value=ast.parse('tuple(slot_pose)',mode='eval').body))
    tree.body.extend(ast.parse(SLOT_ORIENTATIONS).body)
    return ast.unparse(ast.fix_missing_locations(tree))+'\n'


CONTINUOUS_INSERT_SOLVE = """def _solve(runtime, env_idx, arm, poses, seed_joints=None):
    seed = runtime.observations[env_idx]['state'][f'{arm}_arm_joint_state'] if seed_joints is None else seed_joints
    joints = {}
    for name, pose in poses.items():
        target = _joint_target_seeded(runtime, env_idx, arm, pose, seed)
        if target is None:
            return None
        joints[name] = target
        seed = target
    return joints
"""


def continuous_insert_solutions(text):
    tree=ast.parse(text)
    solve=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='_solve')
    tree.body[tree.body.index(solve)]=ast.parse(CONTINUOUS_INSERT_SOLVE).body[0]
    changed=0
    for node in ast.walk(tree):
        if isinstance(node,ast.Call) and isinstance(node.func,ast.Name) and node.func.id=='_solve' and len(node.args)==4 and isinstance(node.args[-1],ast.Name) and node.args[-1].id=='delivery':
            node.keywords.append(ast.keyword(arg='seed_joints',value=ast.parse("pickup_joints['lift']",mode='eval').body));changed+=1
    if changed!=1:raise ValueError('Expected one pickup-to-delivery IK chain')
    return ast.unparse(ast.fix_missing_locations(tree))+'\n'


def commit_insert_axis_on_feasible_replan(text):
    """Keep cached attachment geometry consistent with the accepted motion plan."""
    tree=ast.parse(text)
    function=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='_refresh_delivery')
    def assigns(node,key):
        return (isinstance(node,ast.Assign) and len(node.targets)==1
                and isinstance(node.targets[0],ast.Subscript)
                and isinstance(node.targets[0].value,ast.Name) and node.targets[0].value.id=='context'
                and isinstance(node.targets[0].slice,ast.Constant) and node.targets[0].slice.value==key)
    updates=[n for n in function.body if assigns(n,'axis_in_gripper')]
    commits=[n for n in function.body if assigns(n,'refreshed')]
    if len(updates)!=1 or len(commits)!=1:raise ValueError('Expected one successful insert replan commit')
    update=updates[0];function.body.remove(update)
    function.body.insert(function.body.index(commits[0]),update)
    return ast.unparse(ast.fix_missing_locations(tree))+'\n'


def stereo_insert_delivery(text):
    """Observe again before descent; commit observed root only with feasible IK."""
    replacements={
        "if not isinstance(context, dict) or context.get('refreshed') is True:":
        "if not isinstance(context, dict) or (context.get('refreshed') is True and not (isinstance(axis_estimate, Mapping) and axis_estimate.get('retry_at_visible_stage', False))):",
        "np.asarray(context['held_root_in_gripper'])":
        "np.asarray(_numbers(axis_estimate.get('root_in_gripper', context['held_root_in_gripper']), 3, 'observed held root'))",
        "context['axis_in_gripper'] = axis_in_gripper.tolist()":
        "context['axis_in_gripper'] = axis_in_gripper.tolist()\n    if 'root_in_gripper' in axis_estimate:\n        context['held_root_in_gripper'] = list(axis_estimate['root_in_gripper'])\n        context['stereo_refreshed'] = True",
        "if action_count != elapsed or not str(stage['name']).endswith(':lift'):":
        "if action_count != elapsed or not str(stage['name']).endswith((':lift', ':above_slot', ':preinsert')):",
        "axis_estimate = [runtime.perception.held_axis(env_idx, side, estimated_yaw)][0]":
        "axis_estimate = (runtime.perception.held_pose(env_idx, side, contexts[label]['tube_length']) if str(stage['name']).endswith((':above_slot', ':preinsert')) else runtime.perception.held_axis(env_idx, side, estimated_yaw))",
    }
    # Target only _refresh_delivery and _refresh_after_lift, preserving other consumers.
    tree=ast.parse(text);lines=text.splitlines(keepends=True)
    for node in sorted((n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ['_refresh_delivery','_refresh_after_lift']),key=lambda n:n.lineno,reverse=True):
        part=''.join(lines[node.lineno-1:node.end_lineno])
        for old,new in replacements.items():
            if old in part:part=part.replace(old,new)
        lines[node.lineno-1:node.end_lineno]=[part]
    result=''.join(lines)
    if 'retry_at_visible_stage' not in result or 'runtime.perception.held_pose(' not in result:raise ValueError('Missing stereo insertion hook')
    ast.parse(result)
    return result


def condition_insert_observation(text):
    old='        _refresh_delivery(runtime, env_idx, plan, label, official_observation, axis_estimate)'
    if text.count(old)!=1:raise ValueError('Expected one insertion observation-stage hook')
    text=text.replace(old,old+"\n        if str(stage['name']).endswith(':lift'):\n            _condition_above_slot(runtime, env_idx, plan, label, official_observation)")
    return text+INSERT_OBSERVATION_POSE


INSERT_OBSERVATION_POSE = '''
def _condition_above_slot(runtime, env_idx, plan, label, observation):
    from physicalrsi_observed_geometry import GeometryUnavailable
    from physicalrsi_observed_geometry.inspection_pose import conditioned_wrist_pose
    from physicalrsi_skill_library.geometry import forward_kinematics
    from .ik import observed_base
    context = runtime.contexts[env_idx][label]
    arm = context['arm']
    stage = next((s for s in plan['stages'] if s['name'] == label + ':above_slot'), None)
    if stage is None:
        return
    key = arm + '_arm_joint_state'
    seed = stage['target'][key]
    base = observed_base(runtime.ik.chain, arm, observation['state'])
    nominal = np.asarray(forward_kinematics(runtime.ik.chain, base, seed))
    semantic = _ee_pose_to_semantic(runtime.robot_geometry, nominal)
    physical_rotation = _rotation(nominal[3:])
    semantic_rotation = _rotation(semantic[3:])
    root_local = physical_rotation.T @ (np.asarray(semantic[:3]) + semantic_rotation @ np.asarray(context['held_root_in_gripper']) - nominal[:3])
    axis_local = physical_rotation.T @ semantic_rotation @ np.asarray(context['axis_in_gripper'])
    try:
        proposal = conditioned_wrist_pose(observation, arm, nominal, root_local, axis_local, context['tube_length'])
    except GeometryUnavailable:
        return
    if proposal is None:
        return
    joints = runtime.ik.solve(arm, seed, proposal['pose'], observed_state=observation['state'], support=runtime.perception.support_geometry(env_idx), env_idx=env_idx)
    if joints is None:
        print(f'[insert_observation] env={env_idx} tube={label} unreachable; retained planned view', flush=True)
        return
    target = {key: list(value) for key, value in stage['target'].items()}
    target[arm + '_arm_joint_state'] = list(joints)
    stage['target'] = target
    context['observation_pose'] = proposal
    print(f'[insert_observation] env={env_idx} tube={label} condition={proposal["condition_before"]:.5f}->{proposal["condition_after"]:.5f}', flush=True)
'''


INSERT_ACTION_TRACE = """
def _trace_insert_actions(runtime, env_idx, actions):
    import json
    import numpy as np
    capture = getattr(runtime.perception, '_trace_paths', {}).get(env_idx)
    if capture is None:
        return
    slot = runtime.executor._slots[env_idx]
    stage = slot.stages[slot.stage_index] if slot.stage_index < len(slot.stages) else None
    label = stage['name'].split(':', 1)[0] if stage is not None else None
    record = {'action_count': runtime.action_counts[env_idx],
              'stage_index': slot.stage_index, 'stage_actions': slot.stage_actions,
              'stage': stage, 'actions': actions,
              'observed_state': runtime.observations[env_idx]['state'],
              'context': runtime.contexts.get(env_idx, {}).get(label)}
    with capture.with_suffix('.actions.jsonl').open('a') as stream:
        stream.write(json.dumps(record, default=lambda value: np.asarray(value).tolist())+'\\n')
"""


def retain_insert_slot_choices(text):
    """Preserve every feasible observed slot, not only the cheapest grasp variants.

    Assignment constraints depend on slot identity and position, and its objective
    is additive candidate cost. The cheapest candidate per slot dominates all
    other candidates for that slot without losing assignment feasibility.
    """
    tree=ast.parse(text)
    function=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='_tube_candidates')
    returns=[n for n in function.body if isinstance(n,ast.Return)]
    if len(returns)!=1 or ast.unparse(returns[0].value)!='tuple(candidates[:12])':
        raise ValueError('Expected the legacy insertion candidate truncation')
    index=function.body.index(returns[0])
    function.body[index:index+1]=ast.parse("""best_by_slot = {}
for candidate in candidates:
    previous = best_by_slot.get(candidate.slot_index)
    if previous is None or candidate.cost < previous.cost:
        best_by_slot[candidate.slot_index] = candidate
return tuple(best_by_slot.values())
""").body
    return ast.unparse(ast.fix_missing_locations(tree))+'\n'


LATE_INSERT_OBSERVATION = """def _refresh_observed_insertion(runtime, env_idx, plan, label, observation, *, guide_only):
    context = runtime.contexts[env_idx][label]
    estimate = runtime.perception.held_pose(env_idx, context['arm'], context['tube_length'])
    if estimate.get('visible') is not True:
        return False
    phases = ('guide', 'retreat') if guide_only else ('realign', 'guide', 'retreat')
    return bool(_refresh_delivery(runtime, env_idx, plan, label, observation, estimate, delivery_phases=phases))
"""


def late_insert_observation(text):
    """Reobserve slipping attachments before contact correction and final descent."""
    tree=ast.parse(text)
    functions={n.name:n for n in tree.body if isinstance(n,ast.FunctionDef)}
    refresh=functions['_refresh_delivery']
    refresh.args.args.append(ast.arg(arg='delivery_phases'))
    refresh.args.defaults.append(ast.Constant(value=None))
    poses=next(n for n in refresh.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='poses' for t in n.targets))
    refresh.body.insert(refresh.body.index(poses)+1,ast.parse("""if delivery_phases is not None:
    poses = {name: poses[name] for name in delivery_phases}""").body[0])
    assignments=[n for n in ast.walk(refresh) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='joint_name' for t in n.targets)]
    if len(assignments)!=1:raise ValueError('Expected one delivery stage mapping')
    assignments[0].value=ast.parse("({'settle_align': 'realign', 'release': 'guide'}.get(phase, phase) if delivery_phases is not None else ('align' if phase == 'settle_align' else phase))",mode='eval').body
    refresh.body.append(ast.Return(value=ast.Constant(value=True)))
    for name,flag,guide_only in [('_refresh_after_align','realigned',False),('_refresh_before_guide','guided',True)]:
        function=functions[name]
        loop=next(n for n in function.body if isinstance(n,ast.For))
        guard=next(n for n in loop.body if isinstance(n,ast.If) and flag in ast.unparse(n.test))
        hook=ast.parse(f"if _refresh_observed_insertion(runtime, env_idx, plan, label, official_observation, guide_only={guide_only}):\n    context[{flag!r}] = True\n    return").body[0]
        loop.body.insert(loop.body.index(guard)+1,hook)
    tree.body.extend(ast.parse(LATE_INSERT_OBSERVATION).body)
    return ast.unparse(ast.fix_missing_locations(tree))+'\n'


def retain_insert_pose_choices(text):
    """Object collision compatibility depends on pose, so retain all IK alternatives."""
    tree = ast.parse(text)
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == '_tube_candidates')
    returns = [n for n in function.body if isinstance(n, ast.Return)]
    if len(returns) != 1 or ast.unparse(returns[0].value) != 'tuple(candidates[:12])':
        raise ValueError('Expected legacy insertion candidate truncation')
    returns[0].value = ast.parse('tuple(candidates)', mode='eval').body
    return ast.unparse(ast.fix_missing_locations(tree)) + '\n'


def insert_revision_imports(text, imports):
    tree = ast.parse(text)
    after = 0
    for index, node in enumerate(tree.body):
        if ((index == 0 and isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant)
             and isinstance(node.value.value, str))
                or (isinstance(node, ast.ImportFrom) and node.module == '__future__')):
            after = node.end_lineno
        else:
            break
    lines = text.splitlines(keepends=True)
    lines.insert(after, imports)
    return ''.join(lines)


def deposit_preinsert_feedback(text):
    text = insert_revision_imports(text,
        'from physicalrsi_observed_geometry import GeometryUnavailable\n'
        'from physicalrsi_observed_geometry.slot_servo import stage_window\n'
        'from .rgb_perception import near_slot_correction\n')
    # Retain the attachment actually measured after pickup/handoff in this env.
    text = text.replace('        context["delivery_bound"] = True',
        '        context["measured_coin_local"] = coin_in_gripper.tolist()\n'
        '        context["delivery_bound"] = True')
    # Use line boundaries so the cross-side replacement cannot hit same-side.
    text = text.replace('\n    context["delivery_bound"] = True',
        '\n    context["measured_coin_local"] = coin_in_receiver.tolist()\n'
        '    context["delivery_bound"] = True')
    build = next(node for node in ast.parse(text).body
                 if isinstance(node, ast.FunctionDef) and node.name == '_build_envelope')
    refresh = [node for node in build.body if isinstance(node, ast.If)
               and any(isinstance(child, ast.Call) and isinstance(child.func, ast.Name)
                       and child.func.id == '_refresh_cross_insert' for child in ast.walk(node))]
    if len(refresh) != 1:
        raise ValueError('Expected one deposit insertion refresh condition')
    lines = text.splitlines(keepends=True)
    lines.insert(refresh[0].lineno - 1,
        '    _near_insert_feedback(frames, env_idx, plan, observation, action_count)\n')
    text = ''.join(lines)
    text = text.replace('and action_count >= int(context["insert_refresh_action"])',
        'and stage_window(plan["stages"], "slot_insert")[0] <= action_count\n'
        '        < stage_window(plan["stages"], "release")[0]')
    return text + DEPOSIT_NEAR_FEEDBACK


DEPOSIT_NEAR_FEEDBACK = '''

def _near_insert_feedback(frames, env_idx, plan, observation, action_count):
    context = plan.get("context", {})
    if not context.get("delivery_bound") or context.get("near_insert_corrected"):
        return
    start, end = stage_window(plan["stages"], "slot_align")
    # Leave five joint-control updates for the bounded adjustment before descent.
    if not max(start, end - 5) <= action_count < end:
        return
    local = context.get("measured_coin_local")
    if local is None:
        return
    arm = str(context["bank_arm"])
    robot = frames.get_robot_by_arm_name(f"{arm}_arm")
    current = _ee_pose_to_semantic(robot, observation["state"][f"{arm}_ee_pose"])
    expected = np.asarray(current[:3]) + _rotation(current[3:]) @ np.asarray(local)
    slot_pose = context["slot_poses"][int(context["slot_index"])]
    try:
        correction = near_slot_correction(observation, arm, slot_pose, expected)
    except GeometryUnavailable:
        return
    poses = _rgb_bank_waypoints(slot_pose, local)
    joints = {}
    for name in ("slot_align", "slot_insert"):
        pose = list(poses[name])
        pose[:3] = (np.asarray(pose[:3]) + correction["translation"]).tolist()
        target = _joint_target(frames, env_idx, arm, pose)
        if target is None:
            return  # Commit both stages together only after both IK solves.
        joints[name] = target
    for stage in plan["stages"]:
        name = stage["name"]
        if name in {"slot_align", "slot_insert", "release"}:
            stage["target"] = _target(stage["target"], arm_joints={arm: joints["slot_insert" if name == "release" else name]})
    context["near_insert_corrected"] = True
    context["near_insert_translation"] = correction["translation"].tolist()
    print(f"[deposit_coin rgb] env={env_idx} near_insert action={action_count} error_px={correction['transverse_error_px']:.3f}", flush=True)
'''
