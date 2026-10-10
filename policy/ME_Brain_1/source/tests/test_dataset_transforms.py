import numpy as np

from focus_vlwa.data.dataset import _prompt


def test_prompt_coordinates_match_training_rounding() -> None:
    sample = {"task": np.asarray('"pick the object at [832, 583]"')}
    assert _prompt(sample) == "pick the object at [800,600]"
