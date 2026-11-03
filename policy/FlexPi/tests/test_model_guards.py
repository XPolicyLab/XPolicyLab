"""Offline checks for model.py's input guards: instruction decoding, the stats coverage check, the prompt cache.

Run from the benchmark root with the policy env, e.g.

    cd RoboDojo
    PYTHONPATH=$PWD:$PWD/XPolicyLab/policy/FlexPi/FlexPi:$PWD/XPolicyLab/policy/FlexPi/FlexPi/src \
        conda run -n flexpi python XPolicyLab/policy/FlexPi/tests/test_model_guards.py

No simulator, no weights. Also collectable by pytest.
"""

import numpy as np

from XPolicyLab.policy.FlexPi.model import _cache_prompt_encodes, _check_stats_coverage, _get_instruction
from XPolicyLab.policy.FlexPi.robodojo_real_tasks import TRAINING_INSTRUCTIONS

SWEEP = "Pick up the broom, hand it over to the right hand, then use the dustpan to sweep the blocks."


def _raises(fn, *args):
    try:
        fn(*args)
    except ValueError as exc:
        return str(exc)
    raise AssertionError(f"{fn.__name__}{args} did not raise")


def test_instruction_decoding():
    fallback = "complete the task"
    assert _get_instruction({"instruction": SWEEP}, fallback) == SWEEP
    assert _get_instruction({"instruction": SWEEP.encode()}, fallback) == SWEEP
    assert _get_instruction({"instruction": np.bytes_(SWEEP.encode())}, fallback) == SWEEP
    assert _get_instruction({"instruction": np.array(SWEEP)}, fallback) == SWEEP
    assert _get_instruction({"instructions": [SWEEP.encode()]}, fallback) == SWEEP
    assert _get_instruction({"instruction": f"  {SWEEP}\n"}, fallback) == SWEEP
    for missing in ({}, {"instruction": None}, {"instruction": ""}, {"instructions": []}):
        assert _get_instruction(missing, fallback) == fallback
    assert _get_instruction({"instruction": b"cap\xffpen"}, fallback) == "cap�pen"  # no raise on bad UTF-8


def test_training_instructions():
    assert {robot: len(tasks) for robot, tasks in TRAINING_INSTRUCTIONS.items()} == {
        "arx_x5": 6, "piper": 6, "piper_x": 6,
    }
    assert TRAINING_INSTRUCTIONS["piper_x"]["sweep_blocks"] == SWEEP
    texts = [t for tasks in TRAINING_INSTRUCTIONS.values() for t in tasks.values()]
    assert len(set(texts)) == 18 and all(t == t.strip() for t in texts)


def test_stats_coverage():
    per_robot = {"coverage": {"robodojo_realworld_34d_relabeled_piper_x": 600}}
    pooled = {"coverage": {"robodojo_realworld_34d_relabeled": 1794}}
    name = _check_stats_coverage(per_robot, "relabeled", "piper_x")
    assert name == "robodojo_realworld_34d_relabeled_piper_x"
    assert "eef_frame=relabeled" in _raises(_check_stats_coverage, per_robot, "world_link6", "piper_x")
    # piper is a prefix of piper_x: the robot is the whole suffix, not a prefix match.
    assert "piper_x checkpoint" in _raises(_check_stats_coverage, per_robot, "relabeled", "piper")
    assert "piper checkpoint" in _raises(
        _check_stats_coverage, {"coverage": {"robodojo_realworld_34d_relabeled_piper": 595}}, "relabeled", "piper_x"
    )
    for robot in ("arx_x5", "piper", "piper_x"):
        assert _check_stats_coverage(pooled, "relabeled", robot) == "robodojo_realworld_34d_relabeled"
    _raises(_check_stats_coverage, pooled, "world_link6", "arx_x5")
    # The sim and raw-frame corpora store the observed pose: world_link6 only.
    for corpus in ("robodojo_34d", "robodojo_realworld_34d"):
        stats = {"coverage": {corpus: 600}}
        assert _check_stats_coverage(stats, "world_link6", "arx_x5") == corpus
        assert "eef_frame=world_link6" in _raises(_check_stats_coverage, stats, "relabeled", "arx_x5")
    # Every key is checked, not only the first.
    mixed = {"coverage": {"robodojo_realworld_34d_relabeled_arx_x5": 599, "robodojo_realworld_34d_relabeled_piper": 595}}
    assert "piper checkpoint" in _raises(_check_stats_coverage, mixed, "relabeled", "arx_x5")
    # No coverage, or a corpus none of these: nothing to check.
    for stats in ({}, {"coverage": None}, {"coverage": {"robotwin_50": 5000}}):
        assert _check_stats_coverage(stats, "relabeled", "arx_x5") is None


def test_prompt_cache():
    class Module:
        def __init__(self):
            self.calls = []

        def encode_prompt(self, prompt):
            self.calls.append(prompt)
            return (np.full(2, len(self.calls)), np.ones(2, bool))

    m = Module()
    _cache_prompt_encodes(m)
    first = m.encode_prompt("a")
    assert m.encode_prompt("a") is first
    m.encode_prompt("b")
    assert m.encode_prompt("a") is first and m.calls == ["a", "b"]


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok  {name}")
