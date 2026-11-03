import ast
from pathlib import Path
from types import SimpleNamespace
import sys
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'asset_revision'))
from pickup_geometry_edits import remove_pickup_geometry_overrides
SOURCE=Path('/cpfs/physicalrsi-asset-audit/v4/implementations/code-skills/implementations/rdj_master')

@pytest.mark.skipif(not SOURCE.exists(),reason='Release fixture not installed')
def test_all_release_variants_remove_lookup_and_preserve_generic_candidates():
    files=list(SOURCE.glob('*/rdj_master/tasks/general_pickup/rgb_cap.py'));assert files
    for path in files:
        before=path.read_text();after=remove_pickup_geometry_overrides(before)
        assert '_CALIBRATED_PRIMARY_GEOMETRY' not in after and '.765' not in after
        tree=ast.parse(after);method=next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=='_plans')
        method.returns=None
        for arg in method.args.args:arg.annotation=None
        candidates=[SimpleNamespace(arm='left',contact_pose=(.2,.1,1.1,1,0,0,0)),SimpleNamespace(arm='right',contact_pose=(.3,.2,1.2,1,0,0,0))]
        scope={'_SAME_ARM_SKILL_DESCRIPTIONS':set(),'_PRINCIPAL_AXIS_DESCRIPTIONS':set(),
               '_plan_from_skill':lambda d,x,y,e:e,'_principal_axis_quaternion':lambda d,a:None,'_uses_complementary_retry':lambda d:False}
        exec(compile(ast.Module(body=[method],type_ignores=[]),'actual generic planner','exec'),scope)
        owner=SimpleNamespace(_skills=SimpleNamespace(select=lambda *a,**kw:candidates))
        for description in ['unknown object','bell pepper','doll','dove','noodle','power bank','scissors','elephant','race car','trowel']:
            target=SimpleNamespace(description=description,appearance=(),box_xyxy=(0,0,10,10),keypoints_xy=(),descriptors=())
            result=scope['_plans'](owner,target,.2,.1)
            assert result[0] is candidates[0] and result[1] is candidates[1]
