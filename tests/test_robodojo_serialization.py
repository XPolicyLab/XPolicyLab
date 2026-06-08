import json

import numpy as np

from robodojo.serialization import to_jsonable


def test_to_jsonable_converts_nested_ndarrays():
    value = {
        "actions": [
            {
                "arm_joint_state": np.zeros(6, dtype=np.float32),
                "ee_joint_state": np.array([0.5], dtype=np.float64),
            }
        ]
    }

    converted = to_jsonable(value)

    assert converted["actions"][0]["arm_joint_state"] == [0.0] * 6
    assert converted["actions"][0]["ee_joint_state"] == [0.5]
    json.dumps(converted)
