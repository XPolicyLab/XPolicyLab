# Copyright 2026 The HEART Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Standalone command-line check for the configured LLM backend.

Backs the ``roborsi-check-llm`` console script. The check itself lives in
:mod:`roborsi.planner.check`, which the Dashboard shares; this module only parses
flags and renders the structured result for a terminal.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from roborsi.planner.check import (
    BASE_URL_ENV_BY_PLANNER,
    CHECK_PLANNERS,
    DEFAULT_TIMEOUT_S,
    STATUS_AUTH_FAILED,
    STATUS_INVALID_MODEL,
    STATUS_MISSING_API_KEY,
    STATUS_MISSING_CONFIG,
    STATUS_NETWORK_ERROR,
    STATUS_PROVIDER_ERROR,
    STATUS_SDK_ERROR,
    STATUS_UNSUPPORTED_PROVIDER,
    LlmCheckRequest,
    LlmCheckResult,
    check_llm,
)


def _parser() -> argparse.ArgumentParser:
    default_timeouts = ", ".join(
        f"{planner}: {seconds}s" for planner, seconds in DEFAULT_TIMEOUT_S.items()
    )
    parser = argparse.ArgumentParser(
        prog="roborsi-check-llm",
        description=(
            "Check that the configured LLM backend answers a minimal request."
        ),
    )
    parser.add_argument(
        "--planner",
        default=None,
        choices=list(CHECK_PLANNERS),
        help="LLM backend to check; defaults to api without an Agent preset.",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Model id. For the 'api' planner, prefix the provider "
        "(e.g. anthropic:claude-opus-4-8, openai:gpt-5.5, "
        "openai-chat:glm-5.2). For claude_code/codex this "
        "overrides the backend default model.",
    )
    parser.add_argument(
        "--base-url",
        default=None,
        help=(
            "API base URL, for the 'api' planner only. claude_code and codex take their endpoint from ANTHROPIC_BASE_URL / CODEX_BASE_URL instead; passing this flag with either is an error rather than a silent no-op."
        ),
    )
    parser.add_argument(
        "--timeout-s",
        type=int,
        default=None,
        help=f"Wall-clock cap for the check. Defaults per backend ({default_timeouts}).",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        dest="as_json",
        help="Print the structured result as JSON instead of a report.",
    )
    parser.add_argument(
        "--agent-config", help="Agent preset path or installed bundle:name."
    )
    parser.add_argument(
        "--agent-set",
        action="append",
        default=[],
        metavar="PATH=JSON",
        help="Override an Agent preset value using the evaluation CLI syntax.",
    )
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="Check the pinned Codex CLI executable without a model request.",
    )
    parser.add_argument("--output-dir", help="Preserve Codex probe artifacts here.")
    return parser


def _remediation(result: LlmCheckResult) -> str:
    """Return terminal-facing advice for a failed check.

    The shared check layer stays UI-agnostic, so the wording lives here.

    Args:
        result: The failed check result to advise on.

    Returns:
        A one-line hint, or an empty string when none applies.
    """
    credential = result.credential_env or "the backend's API key"
    base_url_env = result.base_url_env or "the backend's base URL variable"
    hints = {
        STATUS_MISSING_CONFIG: (
            "Pass --model with a provider prefix, e.g. "
            "--model anthropic:claude-opus-4-8."
        ),
        STATUS_UNSUPPORTED_PROVIDER: (
            "Use a provider RoboRSI installs: anthropic:, openai:, or openai-chat:."
        ),
        STATUS_MISSING_API_KEY: f"Set {credential} in this shell, then retry.",
        STATUS_INVALID_MODEL: (
            f"The provider rejected the model id. Check that --model exists "
            f"for this backend and that {credential} has access to it."
        ),
        STATUS_AUTH_FAILED: (
            f"The provider rejected {credential}. Check the key, and that it "
            f"matches the endpoint in {base_url_env} / --base-url."
        ),
        STATUS_NETWORK_ERROR: (
            f"Nothing answered in time. Check connectivity, any proxy, and the "
            f"endpoint in {base_url_env} / --base-url."
        ),
        STATUS_PROVIDER_ERROR: (
            "The provider was reached but refused the request. Check the model "
            "id, your quota, and any rate limit."
        ),
        STATUS_SDK_ERROR: (
            "The backend SDK failed before a reply arrived. Run with the "
            "backend's own CLI to confirm it is installed and logged in."
        ),
    }
    return hints.get(result.status, "")


def _render(result: LlmCheckResult) -> str:
    """Render a check result as an aligned terminal report.

    Args:
        result: The result to render.

    Returns:
        The report text, without a trailing newline.
    """
    rows: list[tuple[str, str]] = [("planner", result.planner)]
    if result.model:
        rows.append(("model", result.model))
    if result.credential_env:
        state = "set" if result.credential_present else "not set"
        rows.append(("credential", f"{result.credential_env} ({state})"))
    if result.base_url:
        rows.append(("base url", result.base_url))
    elif result.base_url_env:
        rows.append(("base url", f"{result.base_url_env} (env)"))

    if result.ok and result.extra.get("check_mode") == "preflight":
        rows.append(("OK", "CLI preflight passed; model access was not checked."))
    elif result.ok:
        latency = f"{result.latency_s:.2f}s" if result.latency_s is not None else "?"
        rows.append(("OK", f'replied in {latency}: "{result.reply}"'))
    else:
        rows.append(("FAILED", result.status))
        if result.detail:
            rows.append(("detail", result.detail))
        if hint := _remediation(result):
            rows.append(("hint", hint))

    width = max(len(label) for label, _ in rows)
    lines = []
    for label, value in rows:
        pad = " " * (width - len(label) + 4)
        indent = " " * (width + 4)
        wrapped = str(value).replace("\n", "\n" + indent)
        lines.append(f"{label}{pad}{wrapped}")
    return "\n".join(lines)


def main() -> int:
    """Run one connectivity check and report it. Returns 0 on success."""
    parser = _parser()
    args = parser.parse_args()
    if args.agent_set and not args.agent_config:
        parser.error("--agent-set requires --agent-config")
    if args.agent_config and any((args.planner, args.model, args.base_url)):
        parser.error(
            "use --agent-set with --agent-config instead of --planner/--model/--base-url"
        )
    codex_cli = None
    llm_config = None
    working_dir = None
    reasoning_effort = "none"
    provider_capture = None
    if args.agent_config:
        from roborsi.agents.config import AgentConfig

        try:
            config = AgentConfig.load(args.agent_config, overrides=args.agent_set)
            options = config.document.get("agent", {})
            args.planner = options.get("planner", "api")
            args.model = options.get("model")
            working_dir = options.get("working_dir")
            if working_dir is not None:
                working_dir = (config.path.parent / Path(working_dir)).resolve()
            reasoning_effort = options.get("reasoning_effort", "none")
            if options.get("provider_capture") is not None:
                from roborsi.planner.provider_capture import ProviderCaptureConfig

                provider_capture = ProviderCaptureConfig.from_dict(
                    options["provider_capture"]
                )
            if args.planner == "codex_cli":
                from roborsi.planner.codex_cli_config import CodexCliConfig

                codex_cli = CodexCliConfig.from_dict(options.get("codex_cli", {}))
            elif args.planner == "api":
                llm_config = config.runtime().llm
        except (OSError, ValueError, TypeError) as exc:
            parser.error(str(exc))
    args.planner = args.planner or "api"
    if args.preflight_only and args.planner != "codex_cli":
        parser.error("--preflight-only requires codex_cli")
    if args.output_dir and args.planner not in {"codex", "codex_cli"}:
        parser.error("--output-dir requires codex or codex_cli")
    # Mirrors the run CLI: the flag reaches the api model only, so accepting
    # it for the SDK backends would test an endpoint the run cannot use.
    if args.base_url and args.planner in BASE_URL_ENV_BY_PLANNER:
        parser.error(
            "--base-url applies to the 'api' planner only; "
            f"{args.planner} reads its endpoint from "
            f"{BASE_URL_ENV_BY_PLANNER[args.planner]} instead"
        )
    result = check_llm(
        LlmCheckRequest(
            planner=args.planner,
            model=args.model,
            base_url=args.base_url,
            timeout_s=args.timeout_s,
            llm_config=llm_config,
            codex_cli=codex_cli,
            working_dir=working_dir,
            reasoning_effort=reasoning_effort,
            preflight_only=args.preflight_only,
            output_dir=args.output_dir,
            provider_capture=provider_capture,
        )
    )
    if args.as_json:
        print(json.dumps(result.as_dict(), indent=2, ensure_ascii=False))
    else:
        print(_render(result))
    return 0 if result.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
