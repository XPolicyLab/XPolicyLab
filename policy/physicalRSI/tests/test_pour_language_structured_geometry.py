from pathlib import Path
import sys
import ast
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'asset_revision'))
from pour_language_profile import edit_profile_model


def test_structured_execution_receives_episode_shape_and_camera():
    source = '''"""Two execution implementations share episode geometry."""
TABLE = .765
def ground(image, instruction, out, pair_index=0, *, camera):
    bottle_xy = ray(1, 2, TABLE + .2181, camera=camera)
    debug = {}
    return bottle_xy, None, debug
class Model:
    def reset(self): pass
    def build_pair(self):
        self.bxy = b
class StructuredModel(Model):
    def _memory_motion_visual_gates(self, mask, predicted):
        return qualified_axis_fit(mask, predicted)
    def _memory_motion_commit(self):
        return observe(TABLE + .10905)
'''
    seen = {}
    def fit(mask, predicted, **kw):
        seen['fit'] = kw
        return predicted
    def observe(z, **kw):
        seen['observe'] = kw
        return z
    namespace = {'np': np, 'qualified_axis_fit': fit, 'observe': observe}
    result = edit_profile_model(source)
    exec(compile(result, '<structured-profile-revision>', 'exec'), namespace)
    model = namespace['StructuredModel'](); model.reset()
    assert model.bottle_geometry is None
    model.bottle_geometry = {'base': [0, 0, .93], 'height': .31, 'profile': [[0, .04], [.2, .02], [.31, .02]]}
    model.head_camera = object()
    model._memory_motion_visual_gates(None, 'pose')
    assert seen['fit']['profile'] is model.bottle_geometry['profile']
    assert seen['fit']['camera'] is model.head_camera
    assert model._memory_motion_commit() == .93+.31/2
    assert seen['observe']['bottle_height'] == .31
    structured = next(n for n in ast.parse(result).body if isinstance(n, ast.ClassDef) and n.name == 'StructuredModel')
    assert not any(isinstance(n, ast.Name) and n.id == 'TABLE' for n in ast.walk(structured))
