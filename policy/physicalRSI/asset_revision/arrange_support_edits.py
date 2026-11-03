"""Thread measured support through the original arrange path qualification."""
import ast


def replace_once(text, before, after):
    if text.count(before) != 1:
        raise ValueError('Unexpected arrange source at ' + before[:80])
    return text.replace(before, after)


def edit_arrange_support(relative, text):
    if relative.endswith(('/arrange_rgb/skill_choices.py', '/arrange_rgb/routes.py')):
        text=replace_once(text,'from .perception import rectify',
            'from physicalrsi_observed_geometry.planar_rectification import rectify_support')
        text=replace_once(text,'def free_relays(rgb,table_masks,digits,pads,digit,reference_xy=None):',
            'def free_relays(rgb,table_masks,digits,pads,digit,reference_xy=None,*,camera,support_plane):')
        text=replace_once(text,'rectify(table.astype(np.uint8)*255,z=.765)',
            'rectify_support(table.astype(np.uint8)*255,camera,support_plane)')
        text=replace_once(text,'rectify(rgb,z=.765)','rectify_support(rgb,camera,support_plane)')
    elif relative.endswith('/arrange_rgb/collision_gate.py'):
        text = 'from physicalrsi_observed_geometry.support_collision import plane_clearance\n' + text
        text = replace_once(text, 'def __init__(self,initial_state,arm):',
                            'def __init__(self,initial_state,arm,support_plane):\n        self.support_plane=np.asarray(support_plane,dtype=float)\n        plane_clearance(np.zeros((1,3)),self.support_plane)')
        text = replace_once(text,
            "table_gap=min(float(o[4][:,2].min())-.765 for o in objects if o[1]>=2)",
            "table_gap=min(plane_clearance(o[4],self.support_plane) for o in objects if o[1]>=2)")
    elif relative.endswith(('/arrange_rgb/skill_choice_pool.py', '/arrange_rgb/route_pool.py')):
        function = 'qualify_skill_choice' if 'def qualify_skill_choice(' in text else 'qualify_route'
        text = replace_once(text, f'def {function}(state, legs):', f'def {function}(state, legs, support_plane):')
        text = replace_once(text, 'ContinuousGate(state, arm)', 'ContinuousGate(state, arm, support_plane)')
        text = replace_once(text, 'def ordered_windows(pool, state, candidates):',
                            'def ordered_windows(pool, state, candidates, support_plane):')
        text = replace_once(text, f"pool.submit({function}, state, c['legs'])",
                            f"pool.submit({function}, state, c['legs'], support_plane)")
    elif relative.endswith('/arrange_rgb/single_env.py'):
        text='from physicalrsi_observed_geometry import Camera\n'+text
        tree=ast.parse(text)
        calls=[n for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='free_relays']
        if len(calls)!=1:raise ValueError('Expected one arrange relay call')
        call=calls[0];lines=text.splitlines(keepends=True)
        if call.lineno!=call.end_lineno:raise ValueError('Expected single-line relay call')
        line=lines[call.end_lineno-1];position=call.end_col_offset-1
        lines[call.end_lineno-1]=line[:position]+', camera=Camera.from_observation(self.obs), support_plane=self._observed_support_plane()'+line[position:]
        text=''.join(lines)
        text = 'from physicalrsi_observed_geometry.support_observation import observe_support_geometry\n' + text
        text = 'from physicalrsi_observed_geometry.pad_observation import observe_pad_surface\n' + text
        # Retain only RGB and permitted current calibration, never simulator depth.
        compact = "{'color':v['color']}" if "{'color':v['color']}" in text else "{'color': v['color']}"
        text = replace_once(text, compact,
            "{name:v[name] for name in ('color','intrinsic_matrix','extrinsic_matrix') if name in v}")
        fixed_height = ('z=.765+.003+TIP_OFFSET' if 'z=.765+.003+TIP_OFFSET' in text
                        else 'z = 0.765 + 0.003 + TIP_OFFSET')
        text = replace_once(text, fixed_height,
            'z = float(np.dot(self._observed_support_plane(), [source[0], source[1], 1.])) + 0.003 + TIP_OFFSET')
        pad_offset = 'z if relay else z+.009' if 'z if relay else z+.009' in text else 'z if relay else z + 0.009'
        if text.count(pad_offset)!=2:
            raise ValueError('Expected arrange placement and release pad offsets')
        text=text.replace(pad_offset,'self._observed_destination_height(dest, relay) + 0.003 + TIP_OFFSET')
        tree = ast.parse(text)
        cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == 'Model')
        reset = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == 'reset')
        lines = text.splitlines(keepends=True)
        lines[reset.lineno-1:reset.lineno-1] = ['''    def _observed_support_plane(self):
        if getattr(self, '_support_observation', None) is not self.obs:
            geometry = observe_support_geometry(self.obs, segment_many_http,
                occluders=('robot arm', 'number-shaped object', 'round wooden mat'))
            self._support_plane = geometry['plane']
            self._support_observation = self.obs
            self._pad_surfaces = {}
        return self._support_plane

    def _observed_destination_height(self, destination_xy, relay):
        plane = self._observed_support_plane()
        if relay:
            return float(np.dot(plane, [destination_xy[0], destination_xy[1], 1.]))
        key = tuple(float(value) for value in destination_xy)
        if key not in self._pad_surfaces:
            self._pad_surfaces[key] = observe_pad_surface(self.obs, segment_many_http, plane, destination_xy)
        return self._pad_surfaces[key]['surface_z']

''']
        text = ''.join(lines)
        for call in ('ordered_windows(self.skill_choice_pool, state, candidates)',
                     'ordered_windows(self.route_pool,state,candidates)'):
            if call in text:
                text = replace_once(text, call, call[:-1] + ', support_plane=self._observed_support_plane())')
                break
        else:
            raise ValueError('Missing arrange qualification caller')
    return text
