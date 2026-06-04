import msgpack
import msgpack_numpy
import numpy as np
import pytest

from robodojo.protocol.codec import decode_envelope, encode_frame
from robodojo.protocol.exceptions import ErrorCode, WsError
from robodojo.protocol.messages import MessageType
from robodojo.protocol.schemas import Frame


def test_roundtrip_numpy_and_bytes():
    obs = {
        "state": np.zeros(7, dtype=np.float32),
        "flag": True,
        "meta": {"buf": b"abc"},
    }
    frame = Frame(
        message_type=MessageType.INFER,
        request_id="req-test",
        evaluation_id="eval-1",
        payload={"observation": obs},
    )
    raw = encode_frame(frame)
    decoded = decode_envelope(raw)
    assert decoded.message_type == MessageType.INFER
    assert decoded.request_id == "req-test"
    restored = decoded.payload["observation"]["state"]
    assert isinstance(restored, np.ndarray)
    assert restored.shape == (7,)
    assert restored.dtype == np.float32
    assert decoded.payload["observation"]["meta"]["buf"] == b"abc"


def test_wire_dict_uses_protocol_envelope_keys():
    frame = Frame(
        message_type=MessageType.INFER,
        request_id="req-1",
        evaluation_id="e1",
    )
    wire = frame.to_wire_dict()
    assert set(wire) == {
        "message_type",
        "evaluation_id",
        "action_case_id",
        "trial_id",
        "repeat_index",
        "step",
        "sent_at",
        "payload",
        "message_id",
    }
    assert wire["message_type"] == "infer"
    assert wire["message_id"] == "req-1"


def test_missing_type_raises():
    with pytest.raises(WsError):
        Frame.from_wire_dict(
            {
                "message_id": "r1",
                "evaluation_id": "e1",
                "payload": {},
            }
        )


def test_decode_envelope_wraps_invalid_schema_errors():
    raw = msgpack.packb(
        {
            "message_type": "infer",
            "message_id": "r1",
            "evaluation_id": "e1",
            "step": "not-an-int",
            "payload": {},
        },
        use_bin_type=True,
    )

    with pytest.raises(WsError) as exc_info:
        decode_envelope(raw)

    assert exc_info.value.code == ErrorCode.INVALID_FRAME


def test_object_dtype_numpy_rejected_on_encode():
    frame = Frame(
        message_type=MessageType.INFER,
        request_id="req-1",
        evaluation_id="e1",
        payload={"x": np.array([{"unsafe": True}], dtype=object)},
    )

    with pytest.raises(WsError, match="object dtype"):
        encode_frame(frame)


def test_object_dtype_numpy_rejected_on_decode():
    wire = Frame(
        message_type=MessageType.INFER,
        request_id="req-1",
        evaluation_id="e1",
        payload={"x": np.array([{"unsafe": True}], dtype=object)},
    ).to_wire_dict()
    raw = msgpack.packb(wire, default=msgpack_numpy.encode, use_bin_type=True)

    with pytest.raises(WsError, match="object dtype"):
        decode_envelope(raw)
