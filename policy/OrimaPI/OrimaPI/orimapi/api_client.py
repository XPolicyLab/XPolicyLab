"""Current API configuration layered over the frozen supervision protocol."""

import json
import time
import urllib.error
import urllib.request
from urllib.parse import urlparse
from uuid import uuid4

from .supervisor import APISupervisor, SupervisorError
from .llm_auth import CredentialError
from .copilot_auth import CopilotCredential, EDITOR_VERSION, PLUGIN_VERSION, USER_AGENT


class ConfiguredAPISupervisor(APISupervisor):
    """Support Copilot directly or a local bridge, with exact model checks."""

    def __init__(self, *, require_model_match=False, **kwargs):
        if type(require_model_match) is not bool:
            raise ValueError("require_model_match must be a boolean")
        credential = kwargs.get("credential")
        self._copilot = isinstance(credential, CopilotCredential)
        self._local_bridge = getattr(credential, "loopback_only", False)
        if self._local_bridge and urlparse(kwargs["endpoint"]).hostname not in {
            "127.0.0.1",
            "localhost",
            "::1",
        }:
            raise ValueError(
                "Unauthenticated bridge access requires a loopback endpoint"
            )
        super().__init__(**kwargs)
        self.require_model_match = require_model_match
        transport = self._transport

        def checked_transport(payload):
            response = transport(payload)
            if self.require_model_match and response.get("model") != self.model:
                raise SupervisorError(
                    "Supervisor response model does not match the configured model",
                    retryable=False,
                )
            return response

        self._transport = checked_transport

    def _post(self, payload):
        if not self._local_bridge and not self._copilot:
            return super()._post(payload)
        deadline = time.monotonic() + self.timeout_s
        try:
            for attempt in range(2 if self._copilot else 1):
                token = self._credential.get(deadline)
                endpoint = self.endpoint
                headers = {**self.headers, "Content-Type": "application/json"}
                if self._copilot:
                    address = urlparse(endpoint)
                    if address.hostname == "api.githubcopilot.com":
                        endpoint = self._credential.api_base_url + address.path
                    headers.update(
                        {
                            "Authorization": f"Bearer {token}",
                            "Editor-Version": EDITOR_VERSION,
                            "Editor-Plugin-Version": PLUGIN_VERSION,
                            "Copilot-Integration-Id": "vscode-chat",
                            "User-Agent": USER_AGENT,
                            "Openai-Intent": "conversation-panel",
                            "X-GitHub-Api-Version": "2025-04-01",
                            "X-Request-Id": str(uuid4()),
                            "X-Initiator": "agent",
                        }
                    )
                    if any(
                        part.get("type") in {"input_image", "image_url"}
                        for item in payload.get("input", payload.get("messages", []))
                        if isinstance(item, dict)
                        for part in item.get("content", [])
                        if isinstance(part, dict)
                    ):
                        headers["Copilot-Vision-Request"] = "true"
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise SupervisorError("Supervisor request deadline exceeded")
                request = urllib.request.Request(
                    endpoint,
                    data=json.dumps(payload, allow_nan=False).encode(),
                    headers=headers,
                    method="POST",
                )
                try:
                    with urllib.request.urlopen(request, timeout=remaining) as response:
                        if self.stream:
                            result = self._read_stream(response, deadline)
                        else:
                            raw = response.read(self.max_response_bytes + 1)
                            if len(raw) > self.max_response_bytes:
                                raise SupervisorError(
                                    "Supervisor response exceeded the byte limit"
                                )
                            try:
                                result = json.loads(raw)
                            except (ValueError, UnicodeError):
                                raise SupervisorError(
                                    "Supervisor API returned invalid JSON"
                                ) from None
                        if time.monotonic() > deadline:
                            raise SupervisorError(
                                "Supervisor request deadline exceeded"
                            )
                        return result
                except urllib.error.HTTPError as error:
                    code = error.code
                    error.close()
                    if code == 401 and attempt == 0 and self._copilot:
                        self._credential.invalidate(token, deadline=deadline)
                        continue
                    raise SupervisorError(
                        f"Supervisor API returned HTTP {code}",
                        retryable=code in {408, 409, 429} or 500 <= code < 600,
                    ) from None
        except CredentialError as error:
            raise SupervisorError(str(error), retryable=error.retryable) from None
        except (urllib.error.URLError, TimeoutError, OSError):
            raise SupervisorError(
                "Supervisor API transport failed or timed out"
            ) from None
