import pytest

from robodojo.policy_server import PolicyServer
from robodojo.protocol.messages import MessageType
from robodojo.protocol.schemas import Frame


class DummyModel:
    def prepare_case(self, case_meta):
        return {"prepared": case_meta.get("action_case_id")}

    def reset(self):
        return None

    def update_obs(self, obs):
        self.obs = obs

    def get_action(self):
        return {"value": self.obs.get("step", 0)}


def _frame(message_type, payload=None, **kwargs):
    return Frame(
        message_type=message_type,
        request_id=kwargs.pop("request_id", "req-1"),
        evaluation_id=kwargs.pop("evaluation_id", "eval-integration"),
        payload=payload or {},
        **kwargs,
    )


@pytest.mark.asyncio
async def test_policy_server_infer_loop_without_network():
    server = PolicyServer(DummyModel())

    hello = await server.process_frame(_frame(MessageType.HELLO, {"client_name": "eval-client"}))
    assert hello is not None
    assert hello.message_type == MessageType.HELLO_ACK

    prepare = await server.process_frame(
        _frame(MessageType.PREPARE_CASE, {"action_case_id": "case-1"}, action_case_id="case-1")
    )
    assert prepare is not None
    assert prepare.message_type == MessageType.PREPARE_CASE_ACK

    reset = await server.process_frame(
        _frame(MessageType.RESET, {"trial_id": "case-1-r01"}, trial_id="case-1-r01")
    )
    assert reset is not None
    assert reset.message_type == MessageType.RESET_RESULT

    infer = await server.process_frame(
        _frame(
            MessageType.INFER,
            {"observation": {"step": 3}},
            trial_id="case-1-r01",
            action_case_id="case-1",
        )
    )
    assert infer is not None
    assert infer.message_type == MessageType.INFER_RESULT
    assert infer.payload["actions"] == {"value": 3}

    trial_end = await server.process_frame(
        _frame(MessageType.TRIAL_END, {"trial_id": "case-1-r01", "result": "success"})
    )
    assert trial_end is not None
    assert trial_end.message_type == MessageType.TRIAL_END_ACK
