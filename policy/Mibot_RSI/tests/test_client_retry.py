"""Offline regressions for the exported Responses client; no model or simulator calls."""
import email.utils
import importlib.util
import io
import json
from pathlib import Path
import unittest
import urllib.error

CLIENT_PATH = Path(__file__).resolve().parents[1] / "runtime/v2/runtime/client.py"
spec = importlib.util.spec_from_file_location("mibot_exported_retry_client", CLIENT_PATH)
client_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(client_module)

class ScriptedOpener:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.requests = []

    def open(self, request, timeout):
        self.requests.append((request.data, timeout))
        status, headers, body = next(self.responses)
        raw = json.dumps(body).encode()
        if status != 200:
            raise urllib.error.HTTPError(request.full_url, status, "test response", headers, io.BytesIO(raw))
        return io.BytesIO(raw)

def config(**overrides):
    return {**client_module.DEFAULTS, "base_url": "https://relay.example/v1",
            "api_key": "offline-test-only", "api_retries": 2,
            "api_retry_base_delay_s": 1, "api_retry_max_delay_s": 2, **overrides}

class RetryRegressionTests(unittest.TestCase):
    def test_success_has_one_request_and_no_backoff(self):
        sleeps = []
        client = client_module.ResponsesClient(config(), sleeper=sleeps.append)
        client.opener = ScriptedOpener([(200, {}, {"status": "completed"})])
        def unexpected_delay(*args, **kwargs):
            self.fail("successful requests must not enter backoff")
        client._delay = unexpected_delay
        self.assertEqual(client.post({"input": []})["status"], "completed")
        self.assertEqual(len(client.opener.requests), 1)
        self.assertEqual(sleeps, [])

    def test_long_outage_recovers_after_more_than_1024_retries(self):
        count = 1100
        failures = [(429 if n % 2 else 503, {}, {"error": {"type": "server_error"}}) for n in range(count)]
        sleeps, events = [], []
        client = client_module.ResponsesClient(config(api_retries=count), sleeper=sleeps.append,
                                               retry_observer=events.append, random_value=lambda: .5)
        client.opener = ScriptedOpener(failures + [(200, {}, {"status": "completed"})])
        self.assertEqual(client.post({"input": [{"role": "user", "content": "unchanged"}]})["status"], "completed")
        self.assertEqual(len(client.opener.requests), count + 1)
        self.assertEqual(len({raw for raw, _ in client.opener.requests}), 1)
        self.assertEqual(len(events), count)
        self.assertTrue(all(0 <= delay <= 2 for delay in sleeps))

    def test_large_attempt_counts_saturate(self):
        client = client_module.ResponsesClient(config(), random_value=lambda: .5)
        for attempt in (1024, 2000, 1000000):
            with self.subTest(attempt=attempt):
                self.assertEqual(client._delay(attempt), (2.0, "exponential-jitter"))

    def test_retry_after_seconds_is_not_clipped_to_local_backoff_cap(self):
        sleeps = []
        client = client_module.ResponsesClient(config(), sleeper=sleeps.append)
        client.opener = ScriptedOpener([(429, {"Retry-After": "60"}, {"error": {"type": "rate_limit_error"}}),
                                       (200, {}, {"status": "completed"})])
        client.post({"input": []})
        self.assertEqual(sleeps, [60.0])

    def test_retry_after_http_date_is_preserved(self):
        now = 1700000000
        client = client_module.ResponsesClient(config(), wall_time=lambda: now)
        future = email.utils.formatdate(now + 120, usegmt=True)
        self.assertEqual(client._delay(0, {"Retry-After": future}), (120.0, "retry-after"))

    def test_retry_after_remains_bounded_by_total_retry_deadline(self):
        clock = [0.0]
        sleeps = []
        def advance(seconds):
            sleeps.append(seconds)
            clock[0] += seconds
        client = client_module.ResponsesClient(config(api_retry_total_timeout_s=10), sleeper=advance,
                                               monotonic=lambda: clock[0])
        client.opener = ScriptedOpener([(429, {"Retry-After": "60"}, {"error": {"type": "rate_limit_error"}})])
        with self.assertRaisesRegex(client_module.ModelAPIError, "time budget exhausted"):
            client.post({"input": []})
        self.assertEqual(sleeps, [10.0])
        self.assertEqual(len(client.opener.requests), 1)

    def test_ordinary_jittered_backoff_matches_previous_values(self):
        for value in (0, .5, 1):
            client = client_module.ResponsesClient(config(api_retry_max_delay_s=60), random_value=lambda: value)
            for attempt in range(8):
                expected = min((2 ** attempt) * (.75 + .5 * value), 60)
                self.assertEqual(client._delay(attempt), (expected, "exponential-jitter"))

    def test_zero_base_remains_zero_for_large_attempt(self):
        client = client_module.ResponsesClient(config(api_retry_base_delay_s=0))
        self.assertEqual(client._delay(2000), (0.0, "exponential-jitter"))

    def test_permanent_request_error_does_not_retry(self):
        sleeps = []
        client = client_module.ResponsesClient(config(), sleeper=sleeps.append)
        client.opener = ScriptedOpener([(400, {}, {"error": {"type": "invalid_request_error"}})])
        with self.assertRaises(client_module.ModelAPIError) as error:
            client.post({"input": []})
        self.assertEqual(error.exception.status, 400)
        self.assertEqual(len(client.opener.requests), 1)
        self.assertEqual(sleeps, [])

if __name__ == "__main__":
    unittest.main()
