from robodojo.protocol.exceptions import ErrorCode, WsError
from robodojo.protocol.messages import REQUEST_RESPONSE_PAIRS, MessageType
from robodojo.protocol.schemas import Frame


def test_request_response_pairs_cover_core_flow():
    assert REQUEST_RESPONSE_PAIRS[MessageType.HELLO] == MessageType.HELLO_ACK
    assert REQUEST_RESPONSE_PAIRS[MessageType.INFER] == MessageType.INFER_RESULT
    assert REQUEST_RESPONSE_PAIRS[MessageType.HEARTBEAT] == MessageType.HEARTBEAT_ACK
    assert MessageType.INFER in REQUEST_RESPONSE_PAIRS


def test_error_frame_payload():
    frame = Frame(
        message_type=MessageType.ERROR,
        request_id="err-1",
        evaluation_id="eval-1",
        payload={
            "code": ErrorCode.INFER_FAILED.value,
            "message": "bad shape",
            "details": {"key": "arm_joint_state"},
        },
    )
    assert frame.payload["code"] == "infer_failed"


def test_ws_error_attributes():
    err = WsError(ErrorCode.TIMEOUT, "timed out")
    assert err.code == ErrorCode.TIMEOUT
