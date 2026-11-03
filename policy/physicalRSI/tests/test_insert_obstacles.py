from pathlib import Path
from types import SimpleNamespace
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'asset_revision'))
from source_edits import retain_insert_pose_choices, insert_revision_imports
from physicalrsi_observed_geometry.insert_obstacles import choose_order, activate_completed_placements


def candidate(label, slot, cost):
    return SimpleNamespace(label=label, slot_index=slot, slot_position=(slot*.1, 0., 0.), cost=cost)


def test_retain_more_expensive_pose_when_cheapest_blocks_other_tube_in_both_orders():
    cheap, clear, other = candidate('a', 0, 1.), candidate('a', 0, 3.), candidate('b', 1, 1.)
    namespace = {}
    exec(retain_insert_pose_choices('def _tube_candidates(candidates):\n return tuple(candidates[:12])\n'), namespace)
    group = namespace['_tube_candidates']([cheap, clear])
    assert group == (cheap, clear)
    selected = choose_order([group, [other]], lambda c: c,
        lambda current, previous: current is not cheap and previous is not cheap, .06)
    assert len(selected) == 2 and clear in selected and cheap not in selected


def test_maximum_placements_precedes_cost_and_partial_when_all_pairs_blocked():
    groups = [[candidate('a',0,20.)], [candidate('b',1,1.)], [candidate('c',2,2.)]]
    selected = choose_order(groups, lambda c: c, lambda c,p: not (c.label=='b' and p.label=='a'), .06)
    assert len(selected)==3
    assert next(i for i,c in enumerate(selected) if c.label=='b') < next(i for i,c in enumerate(selected) if c.label=='a')
    partial = choose_order(groups, lambda c:c, lambda c,p:False, .06)
    assert len(partial)==1 and partial[0].label=='b'


def test_only_completed_releases_activate_obstacles_and_environment_isolation():
    scene=SimpleNamespace(active=());other=SimpleNamespace(active=('unchanged',))
    runtime=SimpleNamespace(ik=SimpleNamespace(placed_tube_scenes={0:scene,1:other}),
        action_counts={0:9},contexts={0:{'tube0':{'slot_index':2}}})
    plan={'stages':[{'name':'tube0:guide','steps':6},{'name':'tube0:release','steps':4}]}
    activate_completed_placements(runtime,0,plan);assert scene.active==()
    runtime.action_counts[0]=10;activate_completed_placements(runtime,0,plan)
    assert scene.active==( ('tube0',2), ) and other.active==('unchanged',)


def test_import_insertion_keeps_future_import_valid():
    source='"""module"""\nfrom __future__ import annotations\nx=1\n'
    compile(insert_revision_imports(source,'import math\n'),'source','exec')
