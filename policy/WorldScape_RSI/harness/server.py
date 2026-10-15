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

    async def _dispatch_frame(self, frame):
        if frame.message_type in {M.HELLO, M.HEARTBEAT, M.CLOSE}:
            return await super()._dispatch_frame(frame)
        key = (frame.evaluation_id, frame.trial_id)
        if frame.message_type == M.PREPARE_CASE and not frame.trial_id:
            if self.owner is not None:
                raise WsError(ErrorCode.INVALID_FRAME, "Finish the active trial before preparing another case")
            return await super()._dispatch_frame(frame)
        if not frame.trial_id:
            raise WsError(ErrorCode.INVALID_FRAME, "An independent trial_id is required")
        if self.owner is not None and self.owner != key:
            raise WsError(ErrorCode.INVALID_FRAME, "This process already owns another active trial")
        if frame.message_type == M.CALL:
            if self.owner is None:
                raise WsError(ErrorCode.INVALID_FRAME, "Reset the trial before issuing model calls")
            if frame.payload.get("func_name") not in {"update_obs", "get_action", "release_metadata"}:
                raise WsError(ErrorCode.INVALID_FRAME, "Unsupported evaluation method")
        if frame.message_type == M.RESET:
            self.owner = key
            if frame.action_case_id:
                await self._call_model_method(self.model.prepare_case,
                                              {"action_case_id": frame.action_case_id})
        if frame.message_type == M.TRIAL_END and self.owner is None:
            return self._reply(frame, M.TRIAL_END_ACK, {"ok": True, "session_released": True})
        result = await super()._dispatch_frame(frame)
        if frame.message_type == M.TRIAL_END:
            self.owner = None
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
