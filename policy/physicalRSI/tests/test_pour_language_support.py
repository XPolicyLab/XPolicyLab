from pathlib import Path
import sys
import ast
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'asset_revision'))
from pour_language_support_edits import edit_finger_geometry, bind_finger_clearance
from physicalrsi_observed_geometry import GeometryUnavailable


def clearance(vertices):
    source = edit_finger_geometry('def finger_clearance(ee):\n    return 0.765\n')
    namespace = {'np': np, 'finger_vertices': lambda: vertices}
    exec(compile(source, '<revised-finger-check>', 'exec'), namespace)
    return namespace['finger_clearance']


def test_clearance_tracks_support_translation_and_slope_at_each_vertex():
    vertices = np.array([[-.1, 0., .9], [.1, 0., .91], [0., .1, .95]])
    check = clearance(vertices)
    np.testing.assert_allclose(check(np.eye(4), support_plane=[0, 0, .765]), .135)
    np.testing.assert_allclose(check(np.eye(4), support_plane=[0, 0, .865]), .035)
    # A rising plane makes the higher right finger the lowest-clearance point.
    np.testing.assert_allclose(check(np.eye(4), support_plane=[.4, 0, .88]), -.01)
    translated = np.eye(4); translated[2, 3] = .2
    np.testing.assert_allclose(check(translated, support_plane=[.4, 0, 1.08]), -.01)


@pytest.mark.parametrize('plane', [None, [0, 0, float('nan')], [0, 0]])
def test_no_fixed_table_fallback(plane):
    with pytest.raises(GeometryUnavailable):
        clearance(np.zeros((2, 3)))(np.eye(4), support_plane=plane)


def test_model_and_observer_receive_their_episode_plane():
    class Owner:
        observed_support_plane = [.1, -.2, .8]
    for owner in ['self', 'model']:
        source = bind_finger_clearance('result = finger_clearance(pose)', owner, 1)
        seen = []
        namespace = {owner: Owner(), 'pose': np.eye(4), 'finger_clearance': lambda pose, **kw: seen.append(kw['support_plane'])}
        exec(compile(source, '<bound-check>', 'exec'), namespace)
        assert seen == [Owner.observed_support_plane]
    with pytest.raises(ValueError):
        bind_finger_clearance('result = 1', 'self', 1)
