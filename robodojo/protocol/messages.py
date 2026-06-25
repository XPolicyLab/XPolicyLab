"""WebSocket message types."""

from __future__ import annotations

from enum import Enum


class MessageType(str, Enum):
    HELLO = "hello"
    HELLO_ACK = "hello_ack"
    PREPARE_CASE = "prepare_case"
    PREPARE_CASE_ACK = "prepare_case_ack"
    RESET = "reset"
    RESET_RESULT = "reset_result"
    UPDATE_OBS = "update_obs"
    UPDATE_OBS_ACK = "update_obs_ack"
    UPDATE_OBS_BATCH = "update_obs_batch"
    UPDATE_OBS_BATCH_ACK = "update_obs_batch_ack"
    INFER = "infer"
    INFER_RESULT = "infer_result"
    GET_ACTION_BATCH = "get_action_batch"
    GET_ACTION_BATCH_RESULT = "get_action_batch_result"
    TRIAL_END = "trial_end"
    TRIAL_END_ACK = "trial_end_ack"
    HEARTBEAT = "heartbeat"
    HEARTBEAT_ACK = "heartbeat_ack"
    CLOSE = "close"
    ERROR = "error"


REQUEST_RESPONSE_PAIRS: dict[MessageType, MessageType] = {
    MessageType.HELLO: MessageType.HELLO_ACK,
    MessageType.PREPARE_CASE: MessageType.PREPARE_CASE_ACK,
    MessageType.RESET: MessageType.RESET_RESULT,
    MessageType.UPDATE_OBS: MessageType.UPDATE_OBS_ACK,
    MessageType.UPDATE_OBS_BATCH: MessageType.UPDATE_OBS_BATCH_ACK,
    MessageType.INFER: MessageType.INFER_RESULT,
    MessageType.GET_ACTION_BATCH: MessageType.GET_ACTION_BATCH_RESULT,
    MessageType.TRIAL_END: MessageType.TRIAL_END_ACK,
    MessageType.HEARTBEAT: MessageType.HEARTBEAT_ACK,
}
