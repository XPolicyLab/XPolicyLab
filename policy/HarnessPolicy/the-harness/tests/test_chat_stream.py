from pathlib import Path
import json
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from harness_policy.runtime.chat import ChatModel, _read_sse


def _events(*chunks):
    lines = [b": OPENROUTER PROCESSING\n", b"\n"]
    for chunk in chunks:
        lines += [b"data: " + json.dumps(chunk).encode() + b"\n", b"\n"]
    return lines + [b"data: [DONE]\n"]


class StreamTests(unittest.TestCase):
    def test_chunks_assemble_into_one_reply(self):
        body = _read_sse(_events(
            {"id": "gen-1", "choices": [{"delta": {"content": "Hel"}, "finish_reason": None}]},
            {"id": "gen-1", "choices": [{"delta": {"content": "lo"}, "finish_reason": "stop"}]},
            {"id": "gen-1", "choices": [], "usage": {"prompt_tokens": 5, "completion_tokens": 2, "cost": 0.01}},
        ), {"id": None, "usage": None}, [])
        self.assertEqual(body["choices"], [{"message": {"content": "Hello"}, "finish_reason": "stop"}])
        self.assertEqual((body["id"], body["usage"]["cost"]), ("gen-1", 0.01))

    def test_reasoning_chunks_are_kept_apart_from_the_reply(self):
        body = _read_sse(_events(
            {"id": "g", "choices": [{"delta": {"reasoning": "17*23 is "}, "finish_reason": None}]},
            {"id": "g", "choices": [{"delta": {"reasoning": "391."}, "finish_reason": None}]},
            {"id": "g", "choices": [{"delta": {"content": "391"}, "finish_reason": "stop"}]},
        ), {"id": None, "usage": None}, [])
        self.assertEqual(body["choices"][0]["message"], {"content": "391", "reasoning": "17*23 is 391."})

    def test_a_stream_that_stops_early_is_a_transport_failure(self):
        # The provider closed the connection: keep-alives, then nothing, no finish_reason.
        with self.assertRaises(OSError):
            _read_sse([b": OPENROUTER PROCESSING\n", b"\n"], {"id": None, "usage": None}, [])
        partial = [b"data: " + json.dumps({"id": "g", "choices": [{"delta": {"content": "<py"}}]}).encode() + b"\n"]
        with self.assertRaises(OSError):
            _read_sse(partial, {"id": None, "usage": None}, [])

    def test_error_chunk_is_returned(self):
        body = _read_sse(_events({"error": {"message": "overloaded"}}), {"id": None, "usage": None}, [])
        self.assertEqual(body, {"error": {"message": "overloaded"}})

    def test_stream_flag_sets_payload_and_transport(self):
        seen = {}

        def transport(url, headers, payload, timeout):
            seen.update(payload)
            return {"choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}], "usage": {}}

        chat = ChatModel({"backend": "openai_chat", "base_url": "http://127.0.0.1:1/v1", "model": "m",
                          "stream": True}, transport=transport)
        self.assertEqual(chat.complete("s", [{"role": "user", "content": "hi"}]).text, "ok")
        self.assertTrue(seen["stream"])



class RetryTests(unittest.TestCase):
    def _chat(self, failures, **config):
        calls = []

        def transport(url, headers, payload, timeout):
            calls.append(1)
            if len(calls) <= failures:
                raise RuntimeError("Provider request failed: [Errno 104] Connection reset by peer")
            return {"choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}], "usage": {}}

        chat = ChatModel({"backend": "openai_chat", "base_url": "http://127.0.0.1:1/v1", "model": "m",
                          **config}, transport=transport)
        return chat, calls

    def test_retries_counts_attempts(self):
        import harness_policy.runtime.chat as module
        sleep, module.time.sleep = module.time.sleep, lambda seconds: None
        try:
            chat, calls = self._chat(failures=5, retries=2)
            with self.assertRaises(RuntimeError):
                chat.complete("s", [{"role": "user", "content": "hi"}])
            self.assertEqual(len(calls), 3)
            chat, calls = self._chat(failures=2, retries=2)
            self.assertEqual(chat.complete("s", [{"role": "user", "content": "hi"}]).text, "ok")
        finally:
            module.time.sleep = sleep

    def test_retry_seconds_keeps_trying_past_the_count(self):
        import harness_policy.runtime.chat as module
        clock = [0.0]
        sleep, perf = module.time.sleep, module.time.perf_counter
        module.time.sleep = lambda seconds: clock.__setitem__(0, clock[0] + seconds)
        module.time.perf_counter = lambda: clock[0]
        try:
            chat, calls = self._chat(failures=8, retries=1, retry_seconds=600)
            self.assertEqual(chat.complete("s", [{"role": "user", "content": "hi"}]).text, "ok")
            self.assertEqual(len(calls), 9)
            chat, calls = self._chat(failures=100, retries=1, retry_seconds=120)
            with self.assertRaises(RuntimeError):
                chat.complete("s", [{"role": "user", "content": "hi"}])
            self.assertLessEqual(clock[0], 1000)
        finally:
            module.time.sleep, module.time.perf_counter = sleep, perf


if __name__ == "__main__":
    unittest.main()
