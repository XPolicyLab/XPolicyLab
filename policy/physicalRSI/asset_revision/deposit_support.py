"""Remove deposit IK's benchmark table while retaining its solver settings."""
import ast


def edit_deposit_ik(text):
    old='self.robot_cfg, self.scene_model = self._build_robot_and_scene_cfg(data, 0.74-0.765)'
    if text.count(old)!=1:
        raise ValueError('Expected one deposit fixed-table initialization')
    text=text.replace('def __init__(self):','def __init__(self, scene_model):',1)
    text=text.replace(old,'self.robot_cfg, self.scene_model = self._build_robot_and_scene_cfg(data, scene_model)')
    text=text.replace('def _build_robot_and_scene_cfg(self, yml_data, table_height):',
                      'def _build_robot_and_scene_cfg(self, yml_data, observed_scene):')
    cls=next(n for n in ast.parse(text).body if isinstance(n,ast.ClassDef) and n.name=='PolicyIK')
    method=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='_build_robot_and_scene_cfg')
    assignment=next(n for n in method.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='scene_model' for t in n.targets))
    lines=text.splitlines(keepends=True)
    start=assignment.lineno-1
    while start and (not lines[start-1].strip() or lines[start-1].lstrip().startswith('#')):start-=1
    lines[start:assignment.end_lineno]=['\n',
        '        if not observed_scene.get("cuboid"):\n',
        '            raise ValueError("Observed support collision patches required for deposit IK")\n',
        '        scene_model = deepcopy(observed_scene)\n',
        '        # Targets and support share the original planner frame bias.\n',
        '        for patch in scene_model["cuboid"].values():\n',
        '            patch["pose"][:3] = (np.asarray(patch["pose"][:3]) + self.frame_bias).tolist()\n']
    return ''.join(lines)
