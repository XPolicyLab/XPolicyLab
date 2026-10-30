"""Official WS server with an explicit one-active-trial admission boundary."""
from __future__ import annotations

import argparse
import asyncio
from pathlib import Path
import sys

if not __package__:
    checkout = Path(__file__).resolve().parents[2]
    sys.path[:0] = [str(checkout.parent), str(checkout)]

from client_server.ws.model_server import PolicyServer, PolicyServerConfig
from client_server.ws.protocol.exceptions import ErrorCode, WsError
from client_server.ws.protocol.messages import MessageType as M


class SingleTrialServer(PolicyServer):
    """A second trial cannot reset or observe the first trial's model state."""
    def __init__(self, model, config):
        super().__init__(model, config)
        self.owner = None
        self.owner_case = None
        self.owner_started = False
        self.retired_trials = set()
        self._trial_lock = asyncio.Lock()
        self._initial_reset_result = None
        self._initial_reset_error = None

    async def _dispatch_frame(self, frame):
        if frame.message_type in {M.HELLO, M.HEARTBEAT, M.CLOSE}:
            return await super()._dispatch_frame(frame)
        # Admission and completion form one state transition. Upstream shields
        # execution from disconnected waiters; cancelling a retry cannot cancel
        # the initialization it is waiting for. Heartbeats remain responsive.
        async with self._trial_lock:
            return await self._dispatch_trial_frame(frame)

    async def _dispatch_trial_frame(self, frame):
        key = (frame.evaluation_id, frame.trial_id)
        if frame.message_type == M.PREPARE_CASE and not frame.trial_id:
            if self.owner is not None:
                raise WsError(ErrorCode.INVALID_FRAME, "Finish the active trial before preparing another case")
            return await super()._dispatch_frame(frame)
        if not frame.trial_id:
            raise WsError(ErrorCode.INVALID_FRAME, "An independent trial_id is required")
        initial_retry = (frame.message_type == M.RESET and self.owner == key
                         and self.owner_case == frame.action_case_id and not self.owner_started)
        if initial_retry:
            # A failed initialization can replay its failure, never success.
            # This branch runs only AFTER the first initialization settles.
            if self._initial_reset_error is not None:
                code, message, details = self._initial_reset_error
                raise WsError(code, message, details=details)
        if key in self.retired_trials and frame.message_type != M.TRIAL_END:
            raise WsError(ErrorCode.INVALID_FRAME, "Ended or uncertain trial; use a new trial_id")
        if initial_retry and self._initial_reset_result is not None:
            return self._reply(frame, M.RESET_RESULT,
                {**self._initial_reset_result.payload, "initial_reset_already_complete": True})
        if self.owner is not None and self.owner != key:
            raise WsError(ErrorCode.INVALID_FRAME, "This process already owns another active trial")
        if frame.message_type in {M.CALL, M.INFER} and self.owner is None:
            raise WsError(ErrorCode.INVALID_FRAME, "Reset the trial before issuing model calls")
        if frame.message_type == M.CALL:
            if frame.payload.get("func_name") not in {"update_obs", "get_action", "release_metadata"}:
                raise WsError(ErrorCode.INVALID_FRAME, "Unsupported evaluation method")
        if frame.message_type == M.RESET:
            if self.owner is not None:
                raise WsError(ErrorCode.INVALID_FRAME, "Finish the active trial and use a new trial_id")
            if len(self.retired_trials) >= 10000:
                raise WsError(ErrorCode.INVALID_FRAME, "Trial history capacity reached; start a fresh service instance")
            self.owner = key
            self.owner_case = frame.action_case_id
            self.owner_started = False
            self._initial_reset_result = None
            self._initial_reset_error = None
        if frame.message_type == M.TRIAL_END and self.owner is None:
            return self._reply(frame, M.TRIAL_END_ACK, {"ok": True, "session_released": True})
        if (frame.message_type == M.INFER or
                (frame.message_type == M.CALL and frame.payload.get("func_name") in {"update_obs", "get_action"})):
            # Enter execution BEFORE invoking the model, including failed calls.
            self.owner_started = True
            self._initial_reset_result = None
        if frame.message_type == M.TRIAL_END:
            # Cleanup may close resources before raising. Retire the trial and
            # invalidate initialization before attempting any cleanup.
            self.retired_trials.add(key)
            self._initial_reset_result = None
            self._initial_reset_error = None
        try:
            if frame.message_type == M.RESET and frame.action_case_id:
                await self._call_model_method(self.model.prepare_case,
                                              {"action_case_id": frame.action_case_id})
            result = await super()._dispatch_frame(frame)
        except BaseException as error:
            if frame.message_type in {M.RESET, M.CALL, M.INFER, M.TRIAL_END}:
                self.retired_trials.add(key)
                self._initial_reset_result = None
            if frame.message_type == M.RESET:
                failure = error if isinstance(error, WsError) else WsError(ErrorCode.RESET_FAILED, str(error))
                self._initial_reset_error = (failure.code, failure.message, failure.details)
                if failure is not error:
                    raise failure from error
            raise
        if frame.message_type == M.RESET:
            self._initial_reset_result = result
        if frame.message_type == M.TRIAL_END:
            self.retired_trials.add(key)
            self.owner = None
            self._initial_reset_result = None
            self._initial_reset_error = None
        return result


def main():
    import yaml
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config_path", required=True)
    parser.add_argument("--overrides", nargs="*", default=[])
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config_path).read_text())
    for item in args.overrides:
        key, separator, value = item.partition("=")
        if not separator:
            raise ValueError("Overrides must use key=value")
        config[key] = yaml.safe_load(value)
    if __package__:
        from .model import Model
    else:
        from XPolicyLab.policy.WorldScape_RSI.model import Model
    model = Model(config)
    server = SingleTrialServer(model, PolicyServerConfig(host=config["host"], port=int(config["port"])))
    try:
        asyncio.run(server.serve_forever())
    finally:
        model.close()


if __name__ == "__main__":
    main()
