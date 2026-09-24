"""Keep both transport formats RGB across the adapter boundary."""

import cv2
import numpy as np
import pytest

from XPolicyLab.policy.ME_Brain_1.hist_live import _decode_raw_image
from XPolicyLab.policy.ME_Brain_1.model import ensure_chw_uint8
from XPolicyLab.utils.process_data import decode_image_bit, encode_image_bit


@pytest.mark.parametrize("legacy", [False, True])
def test_client_history_uses_shared_rgb_decoding(legacy):
    pixels = np.zeros((32, 32, 3), dtype=np.uint8)
    pixels[:] = [210, 40, 10]
    # Legacy bytes are generated and consumed only inside this regression test.
    payload = cv2.imencode('.jpg', pixels)[1].tobytes() if legacy else encode_image_bit(pixels)
    decoded = decode_image_bit(payload)
    np.testing.assert_array_equal(_decode_raw_image(payload), decoded)
    np.testing.assert_array_equal(ensure_chw_uint8(decoded), decoded.transpose(2, 0, 1))
    assert decoded[..., 0].mean() > decoded[..., 2].mean()


def test_model_rejects_encoded_buffers():
    pixels = np.zeros((32, 32, 3), dtype=np.uint8)
    with pytest.raises(ValueError, match="rank-three image"):
        ensure_chw_uint8(encode_image_bit(pixels))
