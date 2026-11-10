"""Exchange an OrimaPI-only GitHub credential for a short-lived Copilot token."""

import json
import math
from pathlib import Path
import threading
import time
import urllib.error
import urllib.request
from urllib.parse import urlparse

from .llm_auth import CredentialError, _token, validate_auth

TOKEN_ENDPOINT = "https://api.github.com/copilot_internal/v2/token"
EDITOR_VERSION = "vscode/1.110.0"
PLUGIN_VERSION = "copilot-chat/0.26.7"
USER_AGENT = "GitHubCopilotChat/0.26.7"


class CopilotCredential:
    refreshable = True

    def __init__(self, settings):
        self.settings = validate_auth(settings)
        self._lock = threading.Lock()
        self._token = None
        self._expires = 0.0
        self.api_base_url = "https://api.githubcopilot.com"

    def invalidate(self, rejected_token, *, deadline):
        if not self._lock.acquire(timeout=max(0, deadline - time.monotonic())):
            raise CredentialError("Copilot credential wait timed out")
        try:
            if self._token == rejected_token:
                self._token = None
        finally:
            self._lock.release()

    def get(self, deadline):
        if not self._lock.acquire(timeout=max(0, deadline - time.monotonic())):
            raise CredentialError("Copilot credential wait timed out")
        try:
            if self._token is not None and time.monotonic() < self._expires:
                return self._token
            try:
                document = json.loads(Path(self.settings["token_file"]).read_text())
                github_token = _token(document["github_token"])
            except (OSError, ValueError, KeyError, TypeError, CredentialError):
                raise CredentialError(
                    "Configure github_token in supervisor.auth.token_file",
                    retryable=False,
                ) from None
            timeout = min(self.settings["timeout_s"], deadline - time.monotonic())
            if timeout <= 0:
                raise CredentialError("Copilot credential deadline exceeded")
            request = urllib.request.Request(
                TOKEN_ENDPOINT,
                headers={
                    "Authorization": f"token {github_token}",
                    "Accept": "application/json",
                    "Editor-Version": EDITOR_VERSION,
                    "Editor-Plugin-Version": PLUGIN_VERSION,
                    "User-Agent": USER_AGENT,
                    "X-GitHub-Api-Version": "2022-11-28",
                },
            )
            try:
                with urllib.request.urlopen(request, timeout=timeout) as response:
                    lease = json.loads(response.read(65537))
            except urllib.error.HTTPError as error:
                code = error.code
                error.close()
                raise CredentialError(
                    f"Copilot token exchange returned HTTP {code}",
                    retryable=code in {408, 429} or 500 <= code < 600,
                ) from None
            except (OSError, ValueError):
                raise CredentialError("Copilot token exchange failed") from None
            try:
                token = _token(lease["token"])
                expires_in = float(lease["expires_at"]) - time.time()
                refresh_in = float(lease.get("refresh_in", expires_in))
                lifetime = min(expires_in, refresh_in)
                if lifetime <= 0 or not all(
                    map(math.isfinite, (expires_in, refresh_in))
                ):
                    raise ValueError("Expired lease")
                base_url = (
                    lease.get("endpoints", {})
                    .get("api", "https://api.githubcopilot.com")
                    .rstrip("/")
                )
                address = urlparse(base_url)
                if (
                    address.scheme != "https"
                    or not (address.hostname or "").endswith(".githubcopilot.com")
                    or address.username
                    or address.password
                    or address.path
                    or address.query
                    or address.fragment
                ):
                    raise ValueError("Invalid Copilot endpoint")
            except (ValueError, KeyError, TypeError, AttributeError, CredentialError):
                raise CredentialError(
                    "Copilot token exchange returned an invalid lease", retryable=False
                ) from None
            self._token = token
            self.api_base_url = base_url
            self._expires = time.monotonic() + lifetime - min(60, lifetime / 10)
            return token
        finally:
            self._lock.release()
