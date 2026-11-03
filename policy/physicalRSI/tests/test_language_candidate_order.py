from pathlib import Path
from types import SimpleNamespace
import ast
import sys
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'asset_revision'))
from source import controller_edit

@pytest.mark.parametrize('target_index', [0,1,2,3])
def test_executor_preserves_geometry_rank_over_grasp_family(target_index):
    source = """class Controller:
    @staticmethod
    def _ordered_moves(moves, target_index):
        if target_index != 2:
            return moves
        return tuple(sorted(moves, key=lambda move: not move.candidate_family.startswith('block_axis_1:')))
"""
    path='/rdj_v1_stack_blocks_by_language/vendor/rdj_master/tasks/stack_blocks/stage_controller.py'
    edited=controller_edit(path,source,None)
    ns={};exec(edited,ns)
    upright=SimpleNamespace(candidate_family='right_arm_perf:right:offset_000',candidate_rank=0)
    tilted=SimpleNamespace(candidate_family='block_axis_1:right:offset_000->place_right_arm_perf',candidate_rank=1)
    moves=(upright,tilted)
    assert ns['Controller']._ordered_moves(moves,target_index) == moves
