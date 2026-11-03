from pathlib import Path
import sys
from types import SimpleNamespace
import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'asset_revision'))
from physicalrsi_observed_geometry import GeometryUnavailable
from physicalrsi_observed_geometry.support_observation import observe_support_masks


def observation():
    camera={'color':np.zeros((32,32,3),np.uint8),
            'intrinsic_matrix':np.array([[20.,0,16],[0,20.,16],[0,0,1.]]),
            'extrinsic_matrix':np.eye(4)}
    return {'vision':{'cam_head':camera,'cam_left_wrist':camera}}


def group(prompt,mask,score):
    return SimpleNamespace(prompt=prompt,masks=[mask],scores=[score])


def test_material_object_cannot_displace_explicit_table():
    table=np.zeros((32,32),bool);table[3:29,3:29]=True
    object_mask=np.zeros_like(table);object_mask[14:18,14:18]=True
    groups=[group('table',table,.6),group('wooden surface',object_mask,.99),
            group('object',object_mask,.9)]
    _,masks=observe_support_masks(observation(),lambda *args:groups,occluders=('object',))
    for mask in masks:
        assert mask[5,5]
        assert not mask[15,15]
        assert mask.sum()>object_mask.sum()


def test_material_fallback_when_no_explicit_table_is_detected():
    surface=np.zeros((32,32),bool);surface[5:24,6:26]=True
    _,masks=observe_support_masks(observation(),lambda *args:[group('wooden surface',surface,.7)])
    for mask in masks:np.testing.assert_array_equal(mask,surface)


def test_fully_excluded_support_is_reported_before_geometry_fitting():
    surface=np.ones((32,32),bool)
    groups=[group('table',surface,.8),group('robot arm',surface,.9)]
    with pytest.raises(GeometryUnavailable,match='fully excluded'):
        observe_support_masks(observation(),lambda *args:groups)
