"""Bind isolated policy capabilities to versioned Operations and durable RPC IDs."""

import time
import uuid
from pathlib import Path

from PhysicalRSI_core.contracts import Context, Operation
from PhysicalRSI_core.infra.execution import Execution
from PhysicalRSI_core.infra.rpc.deadline_http import DeadlineHttpRpcClient
from PhysicalRSI_core.infra.storage import atomic_json


class PrimitiveClient:
    def __init__(
        self,
        endpoint,
        *,
        episode,
        harness_revision,
        output,
        primitives,
        expected_identity=None,
    ):
        """primitives maps public names to explicit contract/validation bindings.

        Each binding supplies revision, input/output Contracts, effects,
        validate_input(args, kwargs)->wire kwargs and project_output(raw)->public
        feedback. The trusted server must bind this endpoint to one actor and
        enforce request.execute journaling. No dynamic environment attribute
        access or raw server result is exposed to policy code.
        """
        self.client = DeadlineHttpRpcClient(endpoint)
        self.identity = expected_identity
        if expected_identity is not None:
            if (
                expected_identity.get("episode") != episode
                or expected_identity.get("harness_revision") != harness_revision
                or self.client.call("identity", timeout_s=5) != expected_identity
            ):
                raise ValueError("Primitive endpoint identity mismatch")
        self.execution = Execution(output)
        self.output = Path(output)
        self.episode, self.harness_revision = episode, harness_revision
        self.handlers = {}
        for name, binding in primitives.items():
            self.handlers[name] = self._bind(name, binding)

    def _bind(self, name, binding):
        def invoke(value, context):
            context.check()
            request_id = uuid.uuid4().hex
            path = self.output / "rpc" / (request_id + ".json")
            record = dict(
                request_id=request_id,
                method=name,
                revision=binding["revision"],
                episode=self.episode,
                harness_revision=self.harness_revision,
                input=value,
                state="started",
            )
            atomic_json(path, record)
            remaining = context.deadline - time.monotonic()
            try:
                payload = (
                    dict(actor_binding=self.identity, arguments=value)
                    if self.identity is not None
                    else value
                )
                raw = self.client.invoke(
                    request_id, name, timeout_s=remaining, **payload
                )
                result = binding["project_output"](raw)
                record.update(state="completed", output=result)
                atomic_json(path, record)
                return result
            except Exception as error:
                error.request_id = request_id
                record.update(
                    state="uncertain", error=type(error).__name__ + ": " + str(error)
                )
                atomic_json(path, record)
                raise

        operation = Operation(
            name,
            binding["revision"],
            binding["input"],
            binding["output"],
            invoke,
            effects=frozenset(binding["effects"]),
        )

        def handler(args, kwargs, *, deadline):
            context = Context(
                self.episode,
                deadline=deadline,
                execution=self.execution,
                harness_revision=self.harness_revision,
            )
            context.check()
            value = binding["validate_input"](args, kwargs)
            return operation(value, context)

        return handler
