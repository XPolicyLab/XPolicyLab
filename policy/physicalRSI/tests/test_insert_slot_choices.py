"""Candidate compression must preserve the global feasible assignment objective."""
from pathlib import Path
from types import SimpleNamespace
from itertools import product
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'asset_revision'))
from source import retain_insert_slot_choices


def compress(group):
    namespace={}
    exec(retain_insert_slot_choices('def _tube_candidates(candidates):\n    candidates.sort(key=lambda c:c.cost)\n    return tuple(candidates[:12])\n'),namespace)
    return namespace['_tube_candidates'](list(group))


def objective(groups):
    best=(0,0.)
    for choice in product(*(tuple(g)+(None,) for g in groups)):
        selected=[c for c in choice if c is not None]
        if len({c.slot_index for c in selected})!=len(selected):continue
        if any(abs(a.position-b.position)<.06 for i,a in enumerate(selected) for b in selected[i+1:]):continue
        best=min(best,(-len(selected),sum(c.cost for c in selected)))
    return best


def test_preserves_remote_slot_when_cheap_grasp_variants_fill_top_twelve():
    c=lambda slot,cost:SimpleNamespace(slot_index=slot,cost=cost,position=slot*.1)
    group=[c(0,i*.01) for i in range(12)]+[c(1,2.)]
    other=[c(0,0.)]
    assert objective([group[:12],other])[0]==-1
    assert objective([compress(group),other])==objective([group,other])==(-2,2.)


def test_compression_preserves_exhaustive_assignment_with_spacing_constraints():
    rng=np.random.default_rng(72)
    for _ in range(12):
        positions=rng.uniform(0,.2,5)
        groups=[[SimpleNamespace(slot_index=int(slot),position=positions[slot],cost=float(rng.uniform(0,8)))
                 for slot in rng.integers(0,5,9)] for _ in range(3)]
        assert objective([compress(group) for group in groups])==objective(groups)
