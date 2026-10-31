"""Native LIBERO submission boundary around the released OpenWAM client.

The official client still constructs the model input and performs EEF10 -> OSC7
conversion. This wrapper only joins a server-issued OEH ticket to the exact
``env.step`` call and its returned native observation/terminal flag. With an
evidence writer (``oeh_eval.libero_evidence.WireEvidence``) it also records the
campaign sidecar live: the payload the official client sends, the native
submission, and the OEH server's own events, which ride back on every reply.
"""
from __future__ import annotations

import json
import time
from typing import Any

import numpy as np

from .contracts import digest
from .libero_boundary import validate_native_action, validate_raw_action

SERVER_EVENTS = "oeh_events"
SERVER_SOURCE = "oeh_policy.libero_server"


def _action_bytes(value: Any) -> bytes:
    validate_native_action(value)
    return np.asarray(value, dtype=np.float32).tobytes()


class ReceiptTransport:
    """Extends the pinned OpenWAM wire client with a no-retry receipt message."""

    def __init__(self, official_transport, evidence=None, steady_timeout=None):
        self.official = official_transport
        self.evidence = evidence
        self.close_reason = "client_closed"
        # The official client's `timeout` bounds each round trip; the first predict may carry the model's
        # warm-up (first_decision_seconds), every later one only decision bounds.
        self.steady_timeout = steady_timeout

    def _forward(self, events):
        if events and self.evidence is not None:
            self.evidence.forward(events, SERVER_SOURCE)

    def _call(self, operation):
        """Run one round trip; hand the server's piggybacked events to the sidecar, also on errors."""
        try:
            reply = operation()
        except Exception as exc:
            try:
                body = json.loads(getattr(exc, "raw_body", "") or "{}")
            except ValueError:
                body = {}
            if isinstance(body, dict):
                self._forward(body.get(SERVER_EVENTS))
            raise
        if isinstance(reply, dict):
            self._forward(reply.pop(SERVER_EVENTS, None))
        return reply

    def ping(self):
        return self._call(self.official.ping)

    def reset(self):
        return self._call(self.official.reset)

    def predict(self, payload):
        # The released transport retries ambiguous obs by default. OEH cannot
        # authorize an action twice, so use its explicit one-shot operation.
        try:
            return self._call(lambda: self.official.predict_once(payload))
        finally:
            if self.steady_timeout is not None:
                self.official.timeout, self.steady_timeout = self.steady_timeout, None

    def receipt(self, ticket, action, terminal, state, state_sha256):
        return self._call(lambda: self.official._roundtrip(
            {"type": "oeh_receipt", "ticket": ticket, "action": validate_native_action(action),
             "terminal": bool(terminal), "state": state, "state_sha256": state_sha256}, reconnect=False))

    def close(self):
        """Close the OEH episode first, so its final events reach the sidecar, then the socket."""
        try:
            self._call(lambda: self.official._roundtrip({"type": "oeh_close", "reason": self.close_reason},
                                                        reconnect=False))
        except Exception as exc:
            if self.evidence is not None:
                self.evidence.append("cleanup_failure", {"stage": "oeh_close", "error": type(exc).__name__})
        finally:
            self.official.close()


class NativePolicyClient:
    """Transport injected into the unchanged official OpenWAMLiberoPolicy."""

    def __init__(self, transport, official_conversion, state_conversion, evidence=None):
        self.transport = transport
        self.convert = official_conversion
        self.state_conversion = state_conversion
        self.evidence = evidence
        self.pending = None
        self.native_action = None
        self.active = False
        self.terminal = False
        self.native_submissions = []

    def ping(self):
        return self.transport.ping()

    def reset(self):
        if self.pending is not None:
            raise RuntimeError("unacknowledged LIBERO native submission")
        result = self.transport.reset()
        if result.get("type") == "reset_ack":
            self.active = True
            if self.evidence is not None:
                self.evidence.policy_started()
        return result

    def predict(self, payload):
        if not self.active or self.pending is not None:
            raise RuntimeError("LIBERO policy lifecycle is not ready")
        if self.evidence is not None:
            self.evidence.observation(payload)  # exactly what the official client sends
        result = self.transport.predict(payload)
        action = validate_raw_action(result.get("action"))
        ticket = result.get("oeh_ticket")
        if not isinstance(ticket, dict) or not ticket.get("action_sha256"):
            raise ValueError("OEH server omitted native action ticket")
        native = self.convert(np.asarray(action, dtype=np.float32))
        validate_native_action(native)
        if digest(validate_native_action(native)) != ticket["action_sha256"]:
            raise ValueError("OEH ticket does not authorize official OSC conversion")
        self.pending = ticket
        self.native_action = np.asarray(native, dtype=np.float32).copy()
        return result

    def acknowledge(self, action, result):
        if self.pending is None:
            raise RuntimeError("native step has no OEH ticket")
        if _action_bytes(action) != _action_bytes(self.native_action):
            raise ValueError("native action differs from admitted official conversion")
        obs, _, done, _ = result
        if not isinstance(done, (bool, np.bool_)):
            raise ValueError("native LIBERO terminal flag must be boolean")
        if not isinstance(obs, dict):
            raise ValueError("native LIBERO step returned no observation")
        # This is evidence of the returned simulator step, never policy input.
        state = self.state_conversion(obs["robot0_eef_pos"], obs["robot0_eef_quat"],
                                      obs["robot0_gripper_qpos"])
        state = np.asarray(state, dtype=np.float32).reshape(-1).tolist()
        if len(state) != 10:
            raise ValueError("official post-step state is not achieved EEF10")
        state_sha256 = digest(state)
        ticket = self.pending
        reply = self.transport.receipt(ticket, action, bool(done), state, state_sha256)
        if reply.get("type") != "oeh_receipt_ack":
            raise RuntimeError("OEH did not acknowledge native submission")
        submission = self.native_submissions[-1]
        if submission["ticket"] != ticket or submission["status"] != "submitted_unacknowledged":
            raise RuntimeError("native submission ledger mismatch")
        submission.update(status="acknowledged", terminal=bool(done), state_sha256=state_sha256)
        self.pending = None
        self.native_action = None
        if done:
            self.active = False
            self.terminal = True

    def close(self):
        if hasattr(self.transport, "close_reason"):
            self.transport.close_reason = "native_terminal" if self.terminal else "client_closed"
        self.transport.close()

    def mark_submitted(self, action):
        if self.pending is None or _action_bytes(action) != _action_bytes(self.native_action):
            raise ValueError("native submission differs from authorized action")
        self.native_submissions.append({"ticket": self.pending,
                                        "action": validate_native_action(action),
                                        "status": "submitted_unacknowledged"})
        if self.evidence is not None:
            self.evidence.append("native_action_submitted", {
                "trace_id": self.pending.get("trace_id"), "sha256": self.pending["action_sha256"],
                "action": validate_native_action(action)})


class NativeEnvGateway:
    """Delegates reset/settle to official env, then receipts policy env.step."""

    def __init__(self, env, client: NativePolicyClient, evidence=None):
        self.env = env
        self.client = client
        self.evidence = evidence

    def __getattr__(self, name):
        return getattr(self.env, name)

    def step(self, action):
        if self.client is None or not self.client.active:
            if self.evidence is not None:
                self.evidence.settle(action)
            return self.env.step(action)
        if self.client.pending is None:
            raise RuntimeError("policy step has no OEH authorization")
        if _action_bytes(action) != _action_bytes(self.client.native_action):
            raise ValueError("native submission differs from authorized action")
        # One native call. An ambiguous exception is recorded by the caller and
        # must never trigger a retry or an invented receipt.
        self.client.mark_submitted(action)
        started = time.monotonic_ns()
        result = self.env.step(action)
        if self.evidence is not None:
            self.evidence.append("native_step_completed", {"trace_id": self.client.pending.get("trace_id"),
                                                           "duration_ns": time.monotonic_ns() - started})
        self.client.acknowledge(action, result)
        return result
