"""Remove object-name geometry overrides derived from privileged trajectories."""
import ast


def remove_pickup_geometry_overrides(text):
    tree=ast.parse(text);remove=[]
    constants={'_CALIBRATED_PRIMARY_GEOMETRY','_SKILL_FIRST_CALIBRATED_KEYWORDS','_PRIVILEGED_PRIMARY_KEYWORDS'}
    functions={'_calibrated_geometry','_uses_privileged_primary'}
    for node in tree.body:
        if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id in constants for t in node.targets):remove.append(node)
        elif isinstance(node,ast.FunctionDef) and node.name in functions:remove.append(node)
    calls=[]
    for node in ast.walk(tree):
        if isinstance(node,ast.Assign) and isinstance(node.value,ast.Call) and isinstance(node.value.func,ast.Name) and node.value.func.id=='_calibrated_geometry':
            remove.append(node);calls.append(node)
        elif isinstance(node,ast.If):
            if any(isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='_uses_privileged_primary' for n in ast.walk(node.test)):
                remove.append(node)
            elif ast.dump(node.test)==ast.dump(ast.parse('calibrated is not None',mode='eval').body):remove.append(node)
    if len(remove)!=8 or len(calls)!=1:raise ValueError('Unexpected pickup geometry override structure')
    lines=text.splitlines(keepends=True)
    for node in sorted(remove,key=lambda n:n.lineno,reverse=True):del lines[node.lineno-1:node.end_lineno]
    after=''.join(lines)
    if any(token in after for token in constants|functions):raise ValueError('Pickup geometry override reference remains')
    compile(after,'revised pickup','exec')
    return after


def replace_pickup_depth_scale(text):
    """Keep RGB grounding/morphology, obtain metric geometry from current views."""
    tree = ast.parse(text)
    removed = {'_PIXEL_V', '_PIXEL_U', '_RAY_UP', '_RAY_Z',
               '_TABLE_FORWARD_DISTANCE', '_TABLE_FIT_PIXELS'}
    lines = text.splitlines(keepends=True)
    nodes = [n for n in tree.body if isinstance(n, ast.Assign)
             and any(isinstance(x, ast.Name) and x.id in removed
                     for target in n.targets for x in ast.walk(target))]
    if len(nodes) != 5:
        raise ValueError('Unexpected pickup depth calibration constants')
    for node in sorted(nodes, key=lambda n: n.lineno, reverse=True):
        del lines[node.lineno-1:node.end_lineno]
    text = ''.join(lines)
    text = text.replace('from sklearn.linear_model import LinearRegression, RANSACRegressor\n', '')
    old = '                predicted_depth[len(grounded)],\n                box,\n'
    if text.count(old) != 1:
        raise ValueError('Unexpected pickup affordance call')
    text = text.replace(old, old + '                observation=observations[len(grounded)],\n')
    start = text.index('    calibration = RANSACRegressor(', text.index('def _depth_affordance('))
    end = text.index('    x0, y0, x1, y1 =', start)
    text = text[:start] + '''    from physicalrsi_observed_geometry.calibrated_depth_scene import calibrated_depth_scene

    scene = calibrated_depth_scene(observation, predictor=lambda image: predicted_depth)
    height, width = scene.depth.shape
    vv, uu = np.mgrid[:height, :width]
    points = scene.camera.unproject(
        np.column_stack((uu.ravel(), vv.ravel())), scene.depth.ravel()
    ).reshape(height, width, 3)
    world_z = points[:, :, 2]
    a, b, c = scene.depth_calibration_report['observed_plane']
    # Signed height above the observed support, not a task-specific world Z.
    support_distance = (world_z - a*points[:, :, 0] - b*points[:, :, 1] - c) / np.sqrt(1+a*a+b*b)
''' + text[end:]
    old = '    box: tuple[float, float, float, float],\n) -> tuple[tuple[float, float], float, tuple[float, float], float]:'
    if text.count(old) != 1:
        raise ValueError('Unexpected pickup affordance signature')
    text = text.replace(old, old.replace('\n)', '\n    *, observation,\n)'))
    text = text.replace('min(_IMAGE_WIDTH - 1, x1)', 'min(width - 1, x1)')
    text = text.replace('min(_IMAGE_HEIGHT - 1, y1)', 'min(height - 1, y1)')
    text = text.replace('    crop_z = world_z[y0 : y1 + 1, x0 : x1 + 1]\n    foreground = (crop_z > 0.778).astype(np.uint8)',
                        '    crop_height = support_distance[y0 : y1 + 1, x0 : x1 + 1]\n    foreground = (crop_height > 0.013).astype(np.uint8)')
    text = text.replace('float(np.clip(world_z[v, u], 0.78, 0.95))', 'float(world_z[v, u])')
    if any(token in text for token in removed):
        raise ValueError('Legacy table depth scale reference remains')
    compile(text, 'observed pickup depth', 'exec')
    return text


def replace_pickup_projection(text):
    """Thread an observation-local camera through every pickup projection call."""
    old='from .rgb_perception import GroundedClipPerceptor, GroundedTarget, pixel_to_world_on_height'
    if text.count(old)!=1:raise ValueError('Unexpected pickup projection import')
    text=text.replace(old, 'from .rgb_perception import GroundedClipPerceptor, GroundedTarget\nfrom physicalrsi_observed_geometry import Camera\nfrom physicalrsi_observed_geometry.camera_projection import pixel_to_world_on_height')
    old='''            projection_z = (
                detection.surface_z
                if detection.surface_z > 0.90
                else _TARGET_CENTER_PLANE_Z_M
            )'''
    if text.count(old)!=1:raise ValueError('Unexpected pickup target-height branch')
    text=text.replace(old,'            camera = Camera.from_observation(observation)\n            projection_z = detection.surface_z')
    text=text.replace('world_z=_TARGET_CENTER_PLANE_Z_M','world_z=detection.surface_z')
    tree=ast.parse(text);lines=text.splitlines(keepends=True)
    # Remove now-unused absolute projection plane; grasp heights are separate.
    nodes=[n for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='_TARGET_CENTER_PLANE_Z_M' for t in n.targets)]
    if len(nodes)!=1:raise ValueError('Unexpected pickup projection-plane constant')
    del lines[nodes[0].lineno-1:nodes[0].end_lineno]
    text=''.join(lines);tree=ast.parse(text);lines=text.splitlines(keepends=True)
    offsets=[0]
    for line in lines:offsets.append(offsets[-1]+len(line))
    edits=[];names={'_plans','_plan_from_skill','_principal_axis_quaternion'}
    definitions=0;calls=0
    for node in ast.walk(tree):
        if isinstance(node,ast.FunctionDef) and node.name in names:
            definitions+=1
            closing=next(i for i in range(node.lineno-1,node.body[0].lineno-1) if ') ->' in lines[i])
            indent=lines[closing][:len(lines[closing])-len(lines[closing].lstrip())]
            edits.append((offsets[closing],indent+'    *, camera,\n'))
        if isinstance(node,ast.Call):
            name=node.func.id if isinstance(node.func,ast.Name) else node.func.attr if isinstance(node.func,ast.Attribute) else None
            if name in names|{'pixel_to_world_on_height'}:
                calls+=1;position=offsets[node.end_lineno-1]+node.end_col_offset-1
                before=text[:position].rstrip();separator='' if before.endswith(',') else ','
                edits.append((position,separator+' camera=camera'))
    if definitions!=3 or calls!=9:raise ValueError(f'Unexpected pickup camera propagation: {definitions} definitions, {calls} calls')
    for position,value in sorted(edits,reverse=True):text=text[:position]+value+text[position:]
    if '_TARGET_CENTER_PLANE_Z_M' in text:raise ValueError('Pickup projection height remains')
    compile(text,'observed pickup projection','exec')
    return text


def replace_pickup_target_region(text):
    """Keep affordance morphology but scope it to the current segmented target."""
    text=text.replace('import numpy as np\n','import numpy as np\nfrom physicalrsi_observed_geometry.pickup_target import observe_pickup_target\n',1)
    old='                observation=observations[len(grounded)],\n'
    if text.count(old)!=1:raise ValueError('Unexpected pickup observation call')
    text=text.replace(old,old+'                target_mask=observe_pickup_target(np.asarray(images[len(grounded)]), description, box),\n')
    old='    *, observation,\n'
    if text.count(old)!=1:raise ValueError('Unexpected observed affordance signature')
    text=text.replace(old,'    *, observation, target_mask,\n')
    old='    foreground = (crop_height > 0.013).astype(np.uint8)'
    if text.count(old)!=1:raise ValueError('Unexpected depth foreground threshold')
    text=text.replace(old,'    foreground = target_mask[y0:y1+1, x0:x1+1].astype(np.uint8)')
    # Metric depth is still aligned to observed support; target identity is RGB-only.
    start=text.index("    a, b, c = scene.depth_calibration_report['observed_plane']",text.index('def _depth_affordance('))
    end=text.index('    x0, y0, x1, y1 =',start)
    text=text[:start]+text[end:]
    text=text.replace('    crop_height = support_distance[y0 : y1 + 1, x0 : x1 + 1]\n','')
    old='''        u = round((x0 + x1) * 0.5)
        v = round((y0 + y1) * 0.5)'''
    if text.count(old)!=1:raise ValueError('Unexpected pickup fallback point')
    text=text.replace(old,'''        if not foreground.any():
            raise GeneralPickupRgbError("Target mask lost during grasp morphology")
        local_v, local_u = np.unravel_index(int(distance.argmax()), distance.shape)
        u, v = x0 + int(local_u), y0 + int(local_v)''')
    compile(text,'target-scoped pickup depth','exec')
    return text


def replace_pickup_skill_offsets(text):
    """Do not transfer absolute world-XY corrections from old grasp examples."""
    tree=ast.parse(text)
    fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='_plan_from_skill')
    node=fn.body[0]
    if not isinstance(node,ast.If) or ast.unparse(node.test)!='skill.contact_pixel is None':
        raise ValueError('Unexpected pickup skill contact selection')
    lines=text.splitlines(keepends=True)
    lines[node.lineno-1:node.end_lineno]=['    from physicalrsi_observed_geometry.pickup_target import observed_pickup_contact\n',
        '    contact_x, contact_y = observed_pickup_contact(detection, camera, preferred_pixel=skill.contact_pixel)\n']
    text=''.join(lines)
    if 'skill.delta_xy' in text:raise ValueError('Pickup still applies old world-coordinate offsets')
    compile(text,'observed pickup skill contact','exec');return text
