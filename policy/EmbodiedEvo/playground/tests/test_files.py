"""Host-side rollout accounting must not invalidate simulator episodes."""

import json
import tempfile
import unittest
from pathlib import Path

from orchestrator.files import codex_token_usage


class TokenUsageTests(unittest.TestCase):
    def test_rate_limit_only_events_keep_last_cumulative_usage(self):
        with tempfile.TemporaryDirectory() as directory:
            home = Path(directory)
            sessions = home / ".codex/sessions"
            sessions.mkdir(parents=True)
            events = [
                None,
                {"type": "event_msg", "payload": None},
                {"type": "event_msg", "payload": {"type": "token_count", "info": None}},
                {"type": "event_msg", "payload": {"type": "token_count", "info": {
                    "total_token_usage": {"input_tokens": 10, "total_tokens": 12},
                }}},
                {"type": "event_msg", "payload": {"type": "token_count", "info": None}},
                {"type": "event_msg", "payload": {"type": "token_count", "info": {
                    "total_token_usage": {"input_tokens": 20, "total_tokens": 25},
                }}},
                {"type": "event_msg", "payload": {"type": "token_count", "info": None}},
            ]
            (sessions / "rollout.jsonl").write_text(
                "\n".join(json.dumps(event) for event in events) + "\n"
            )
            result = codex_token_usage(home)
            self.assertEqual(result["input_tokens"], 20)
            self.assertEqual(result["total_tokens"], 25)
            self.assertEqual(result["rollout_files"], 1)
            self.assertEqual(result["token_events"], 2)
            self.assertEqual(result["parse_errors"], 0)
