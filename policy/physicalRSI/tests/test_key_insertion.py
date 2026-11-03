from pathlib import Path
import sys
import numpy as np
import pytest
from scipy.spatial.transform import Rotation
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'asset_revision'))
from physicalrsi_observed_geometry import GeometryUnavailable
from physicalrsi_observed_geometry.key_insertion import observed_bow_boundary,observed_insertion_offset,observed_approach_offset,insertion_feedback


def outline(scale=1.):
    return scale*np.array([[-.012,0,.012],[.012,0,.012],[.012,0,-.008],[.003,0,-.012],[.003,0,-.06],[-.003,0,-.06],[-.003,0,-.012],[-.012,0,-.008]])


def transform(z,x=0.,angle=0.):
    t=np.eye(4);t[:3,3]=[x,0,z];t[:3,:3]=Rotation.from_euler('z',angle).as_matrix();return t


@pytest.mark.parametrize('scale',[.6,1.,1.7])
def test_travel_bound_follows_bow_geometry_not_lock_depth(scale):
    bow,_=observed_bow_boundary(outline(scale))
    r=Rotation.from_euler('y',.15).as_matrix()
    offset=observed_insertion_offset(outline(scale),r)
    assert np.isclose((bow@r.T)[:,2].min()+offset,.003)
    assert offset>.003


def test_uniform_width_has_no_bow_fallback():
    points=np.array([[-.003,0,.01],[.003,0,.01],[.003,0,-.06],[-.003,0,-.06]])
    with pytest.raises(GeometryUnavailable):observed_bow_boundary(points)


def feedback(heights,*,x=0.,angle=0.,target_z=.81):
    memory={};report=None
    for z in heights:report=insertion_feedback(memory,transform(z,x,angle),transform(target_z),transform(.8),outline())
    return report


def test_stop_below_aperture_after_descent_is_only_contact_hypothesis():
    report=feedback(list(np.linspace(.88,.835,12))+[.835]*9)
    assert report['sustained_stop_after_insertion']
    assert not report['seating_proven'] and report['hidden_jam_not_excluded']


@pytest.mark.parametrize('heights,kwargs',[
    ([.88]*20,{}),  # Stationary above aperture.
    ([.835]*20,{}),  # No observed insertion motion.
    (list(np.linspace(.88,.835,30)),{}),  # Still progressing.
    (list(np.linspace(.88,.835,12))+[.835]*9,{'x':.006}),
    (list(np.linspace(.88,.835,12))+[.835]*9,{'angle':.05}),
    (list(np.linspace(.88,.81,12))+[.81]*9,{}),  # Travel bound reached freely.
])
def test_unqualified_stop_never_triggers_turn(heights,kwargs):
    assert not feedback(heights,**kwargs)['sustained_stop_after_insertion']


def test_other_episode_progress_does_not_initialize_contact():
    assert feedback(list(np.linspace(.88,.835,12))+[.835]*9)['sustained_stop_after_insertion']
    assert not feedback([.835]*10)['sustained_stop_after_insertion']


@pytest.mark.parametrize('scale',[.6,1.,1.7])
def test_approach_clearance_follows_observed_key_extent(scale):
    points=outline(scale);rotation=Rotation.from_euler('y',.12).as_matrix()
    offset=observed_approach_offset(points,rotation)
    assert np.isclose((points@rotation.T)[:,2].min()+offset,.02)
