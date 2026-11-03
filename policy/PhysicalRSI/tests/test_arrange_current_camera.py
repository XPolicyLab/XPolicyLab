import ast
from pathlib import Path
from types import SimpleNamespace
import sys
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'asset_revision'))
from arrange_camera import edit_arrange_camera
from arrange_support import edit_arrange_support
from physicalrsi_observed_geometry import Camera,GeometryUnavailable
from physicalrsi_observed_geometry.planar_rectification import rectify_support

SOURCE=Path('/cpfs/physicalrsi-asset-audit/v4/implementations/code-skills/implementations/arrange_rgb')

@pytest.mark.skipif(not SOURCE.exists(),reason='Release fixture is not installed')
@pytest.mark.parametrize('version',['7990151605e857df','8f3f2aab6a15eac1'])
def test_actual_reader_camera_is_instance_local_and_required(version):
    p=SOURCE/version/'arrange_rgb/perception.py'
    tree=ast.parse(edit_arrange_camera(str(p),p.read_text()))
    rect=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='rectify')
    cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='Reader')
    cls.body=[n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='rectify']
    ns={'rectify_support':rectify_support,'GeometryUnavailable':GeometryUnavailable}
    exec(compile(ast.Module(body=[rect,cls],type_ignores=[]),'actual reader methods','exec'),ns)
    first,second=ns['Reader'](),ns['Reader']()
    image=np.zeros((80,100,3),np.uint8);image[:,:,0]=np.arange(100)
    transform=np.eye(4);transform[2,3]=1.7
    first.camera=Camera(image,[[60,0,50],[0,60,40],[0,0,1]],transform)
    with pytest.raises(GeometryUnavailable):second.rectify(image)
    transform=transform.copy();transform[0,3]=.1
    second.camera=Camera(image,first.camera.k,transform)
    assert not np.array_equal(first.rectify(image),second.rectify(image))
    first.camera=None
    with pytest.raises(GeometryUnavailable):first.rectify(image)
    # Check both shipped observation/reset implementations, including memory blocks.
    p=SOURCE/version/'arrange_rgb/single_env.py'
    changed=edit_arrange_camera(str(p),edit_arrange_support(str(p),p.read_text()))
    compile(changed,str(p),'exec')
    assert changed.count('self.reader.camera = Camera.from_observation(self.obs)')==1
    assert changed.count('self.reader.camera = None')==1
    p=SOURCE/version/'arrange_rgb/semantic_recall.py'
    changed=edit_arrange_camera(str(p),p.read_text());compile(changed,str(p),'exec')
    assert 'from .perception import rectify' not in changed
    assert 'reader.rectify(' in changed
