from copy import deepcopy
from pathlib import Path
import sys
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'asset_revision'))
from physicalrsi_observed_geometry.mount_clearance import clear_fixed_mount


def fixture():
    scene={'cuboid':{'table':{'dims':[2,2,.05],'pose':[0,0,-.025,1,0,0,0]},
                      'obstacle':{'dims':[.1,.1,.1],'pose':[.4,.4,.05,1,0,0,0]}}}
    robot={'robot_cfg':{'kinematics':{'base_link':'base','collision_link_names':['base','arm'],
          'collision_spheres':{'base':[{'center':[.02,-.01,.03],'radius':.05}],
                               'arm':[{'center':[.4,0,0],'radius':.1}]}}}}
    return scene,robot


def test_only_mount_footprint_is_removed_and_links_retained():
    scene,robot=fixture();before=deepcopy((scene,robot))
    result=clear_fixed_mount(scene,robot)
    assert (scene,robot)==before
    assert result['cuboid']['obstacle']==scene['cuboid']['obstacle']
    patches=[b for n,b in result['cuboid'].items() if n.startswith('table_')]
    assert len(patches)==4
    assert sum(b['dims'][0]*b['dims'][1] for b in patches)==pytest.approx(4-.01)
    for x in np.linspace(-.95,.95,40):
        for y in np.linspace(-.95,.95,40):
            covered=any(abs(x-b['pose'][0])<b['dims'][0]/2 and abs(y-b['pose'][1])<b['dims'][1]/2 for b in patches)
            assert covered==not_inside_mount(x,y)
    assert all(b['pose'][2]+b['dims'][2]/2==pytest.approx(0) for b in patches)


def not_inside_mount(x,y):return not (-.03<x<.07 and -.06<y<.04)


def test_no_exclusion_when_base_does_not_intersect_support():
    scene,robot=fixture();robot['robot_cfg']['kinematics']['collision_spheres']['base'][0]['center'][2]=.2
    assert clear_fixed_mount(scene,robot)==scene


def test_moving_link_does_not_define_mount_exclusion():
    scene,robot=fixture();robot['robot_cfg']['kinematics']['collision_spheres']['base']=[]
    assert clear_fixed_mount(scene,robot)==scene


def test_rotated_support_is_not_silently_misinterpreted():
    scene,robot=fixture();scene['cuboid']['table']['pose'][3:]=[.707,0,0,.707]
    with pytest.raises(ValueError,match='Axis-aligned'):clear_fixed_mount(scene,robot)
