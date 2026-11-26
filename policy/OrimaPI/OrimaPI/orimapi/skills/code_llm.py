"""Use one configured JSON API transport for code routing and generation."""

import os

from ..llm_auth import CommandCredential, LocalBridgeCredential, validate_auth
from ..copilot_auth import CopilotCredential
from ..api_client import ConfiguredAPISupervisor as APISupervisor
from ..supervisor import DEFAULT_REASONING_EFFORT, SupervisorEpisodeError
from .initial_experience import current_initial_experience


class CodeJSONClient(APISupervisor):
    def query_json(self, context, images, system_prompt):
        initial = current_initial_experience()
        if initial is None or initial.delivered:
            return super().query_json(context, images, system_prompt)
        augmented = initial.prepare(context, images, system_prompt)
        reply = super().query_json(augmented, images, system_prompt)
        # A failed request leaves delivery pending for the skill's own retry.
        # Never add calls, change prompts or alter subsequent skill interaction.
        initial.delivered_request()
        return reply


def create_json_client(settings):
    auth_settings = settings.get("auth", {})
    if "auth_command" in settings:
        auth_settings = {"type": "command", "command": settings["auth_command"]}
    auth = validate_auth(auth_settings)
    credentials = {}
    if auth["type"] == "command":
        credentials["credential"] = CommandCredential(auth)
    elif auth["type"] == "copilot":
        credentials["credential"] = CopilotCredential(auth)
    elif auth["type"] == "none":
        credentials["credential"] = LocalBridgeCredential()
    else:
        key = os.environ.get(settings.get("api_key_env", "OPENAI_API_KEY"))
        if not key:
            raise SupervisorEpisodeError("Missing code-policy API credential")
        credentials["api_key"] = key
    return CodeJSONClient(
        endpoint=settings["endpoint"],
        model=settings["model"],
        **credentials,
        wire_api=settings.get("wire_api", "responses"),
        headers=settings.get("headers", {}),
        stream=settings.get("stream", False),
        reasoning_effort=settings.get("reasoning_effort", DEFAULT_REASONING_EFFORT),
        require_model_match=settings.get("require_model_match", False),
        system_prompt="Return JSON.",
        timeout_s=settings.get("timeout_s", 120),
    )
