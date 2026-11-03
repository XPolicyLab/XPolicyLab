"""Migrate key skill wrist calibration to per-observation arguments.

Object shapes and fixed heights are separate outstanding
migrations. This edit alone does not qualify insert_key as prior-free.
"""
import ast


def edit_key_camera(path, source):
    name=path.rsplit('/',1)[-1]
    if '/insert_key/' not in path or name not in {'model.py','side_visible.py','tooth_pose.py','hole_servo.py','tail_yaw.py'}:
        return source
    tree=ast.parse(source)
    def statements(code):return ast.parse(code).body
    def assigned(node):
        return {t.id for t in getattr(node,'targets',[]) if isinstance(t,ast.Name)}
    drop={'model.py':{'WRIST_CAL','WRIST_K','WRIST_KI','WRIST_REL','K','KI','C','O'},'side_visible.py':{'cal','CAM','K'},'tooth_pose.py':{'cal','CAM','K','TOCV'},'hole_servo.py':{'cal','REL','K'},'tail_yaw.py':set()}[name]
    tree.body=[n for n in tree.body if not assigned(n)&drop and not (name=='tail_yaw.py' and isinstance(n,ast.ImportFrom) and n.module=='hole_servo')]
    if name=='model.py':
        tree.body[1:1]=statements('from physicalrsi_observed_geometry.wrist_camera import ee_relative_camera\nfrom physicalrsi_observed_geometry import Camera, GeometryUnavailable')
        for n in tree.body:
            if isinstance(n,ast.FunctionDef) and n.name in {'ray','perceive'}:
                n.args.kwonlyargs.append(ast.arg(arg='camera'));n.args.kw_defaults.append(None)
                if n.name=='ray':
                    n.body=statements('''origin = camera.transform[:3, 3]
direction = camera.unproject([uv], [1.])[0] - origin
if not np.isfinite(z) or abs(direction[2]) < 1e-9:
    raise GeometryUnavailable('Cannot intersect observed camera ray with support plane')
distance = (z - origin[2]) / direction[2]
if distance <= 0:
    raise GeometryUnavailable('Requested plane is behind the observed camera')
return origin + direction * distance''')
            if isinstance(n,ast.FunctionDef) and n.name in {'wrist_center','measured_rotation'}:
                n.args.kwonlyargs.append(ast.arg(arg='calibration'));n.args.kw_defaults.append(None)
                if n.name=='wrist_center':
                    # Existing function uses USD camera axes.
                    n.body[0:2]=statements('relative_camera, K = calibration\nT = mat(ee) @ relative_camera')
                else:
                    start=next(i for i,x in enumerate(n.body) if assigned(x)=={'cal'})
                    n.body[start:start+3]=statements('T, K = calibration')
        class Calls(ast.NodeTransformer):
            def visit_Call(self,node):
                self.generic_visit(node)
                if not isinstance(node.func,ast.Name):return node
                f=node.func.id
                if f=='ray':
                    node.keywords.append(ast.keyword(arg='camera',value=ast.Name(id='camera',ctx=ast.Load())))
                elif f=='perceive':
                    node.keywords.append(ast.keyword(arg='camera',value=ast.parse("Camera.from_observation(_frame_values['o'])",mode='eval').body))
                if f in {'wrist_center','source_key_pose','shaft_held','wrist_tail_yaw','wrist_hole_centre'}:
                    ee='actual' if f=='wrist_center' else "s[a + '_ee_pose']" if f=='source_key_pose' else "s[b + '_ee_pose']"
                    value=ast.parse('ee_relative_camera(o, '+ee+')',mode='eval').body
                    node.keywords.append(ast.keyword(arg='calibration',value=value))
                elif f=='metric_lobe_centre':
                    node.args=node.args[:3]+[ast.Starred(value=ast.parse("ee_relative_camera(o, s[b + '_ee_pose'])",mode='eval').body,ctx=ast.Load())]
                return node
        tree=Calls().visit(tree)
    else:
        entry={'side_visible.py':'held','tooth_pose.py':'estimate','hole_servo.py':'centre','tail_yaw.py':'observe'}[name]
        fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==entry)
        fn.args.kwonlyargs.append(ast.arg(arg='calibration'));fn.args.kw_defaults.append(None)
        code='CAM, K = calibration' if name in {'side_visible.py','tooth_pose.py'} else 'REL, K = calibration'
        if name=='tooth_pose.py':code+='\nTOCV = CV @ np.linalg.inv(CAM)'
        fn.body[0:0]=statements(code)
    output=ast.unparse(ast.fix_missing_locations(tree))+'\n'
    assert 'wrist_calibration.json' not in output
    return output
