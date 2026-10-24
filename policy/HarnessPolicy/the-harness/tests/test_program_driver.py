import base64
from pathlib import Path
import json
import os
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from harness_policy.cli import parser
from harness_policy.core.contracts import Observation
from harness_policy.runtime.chat import ScriptedChat
from harness_policy.runtime.driver import (
    TaskConfig, _window, assemble_prompt, layer_settings, load_task_config, run_program_episode)
from harness_policy.runtime.skills import skills_for_layers  # noqa: E402

PROMPTS = Path(__file__).resolve().parents[1] / "prompts" / "robodojo"


PIXEL = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="


class FakeEnvironment:
    """The worker calls the driver makes; no kernel-side motion."""

    def __init__(self):
        self.claims = 0
        self.observed = 0

    def tools(self):
        return []

    def reset(self, task, seed):
        return Observation("episode-1", 0, "Stack the bowls.", {"step": 0, "step_limit": 10},
                           {"cam_head": PIXEL})

    def call(self, name, **arguments):
        if name == "profile":
            return {"task": "stack_bowls", "step_limit": 10, "control_hz": 25.0, "cameras": ["cam_head"]}
        if name == "claim_outcome":
            self.claims += 1
            return {"token": "secret"}
        if name == "observe":
            self.observed += 1
            return {"state": {"step": 0, "ended": False}, "images": {"cam_head": PIXEL}}
        if name == "outcome":
            assert arguments["token"] == "secret"
            return {"reward": 0.5, "success": False, "process_score": 50.0, "steps": 3,
                    "terminated": False, "truncated": False}
        raise AssertionError(name)


def _config():
    return TaskConfig(
        path=Path("."), environment={"endpoint": "http://127.0.0.1:1"}, main={"model": "scripted"},
        runtime={"bootstrap": "harness_policy.runtime.kernel:PlainSession", "cameras": ["cam_head"],
                 "isolation": False},
        system_prompt="system", task="stack_bowls", seeds=[0], max_model_calls=6, max_wall_seconds=60)


class ProgramDriverTests(unittest.TestCase):
    def test_python_verify_finish_and_outcome(self):
        replies = [
            "no block here",
            "count\n<python>\nx = 2\nprint(x * 21)\n</python>",
            "<python>\nprint(x)\n1 / 0\n</python>",
            "<verify>bowls are stacked</verify>",
            "<finish>done</finish>",
        ]
        env = FakeEnvironment()
        with tempfile.TemporaryDirectory() as root:
            directory = Path(root) / "seed-0"
            summary = run_program_episode(_config(), 0, directory, chat=ScriptedChat(replies),
                                          verifier=ScriptedChat(["NO the bowls are apart"]), environment=env)
            events = [json.loads(line) for line in (directory / "events.jsonl").read_text().splitlines()]
        self.assertEqual(summary["status"], "finished")
        self.assertEqual((summary["model_calls"], summary["cells"], summary["verifier_calls"]), (5, 2, 1))
        self.assertEqual(summary["decisions"],
                         {"start": 1, "format": 1, "program_end": 1, "error": 1, "verify": 1})
        self.assertEqual((summary["reward"], summary["process_score"], summary["steps"]), (0.5, 50.0, 3))
        self.assertEqual(env.claims, 1)
        kinds = [event["event"] for event in events]
        self.assertIn("protocol_error", kinds)
        cells = [event["result"] for event in events if event["event"] == "cell"]
        self.assertEqual(cells[0]["stdout"], "42\n")
        self.assertEqual(cells[1]["status"], "error")
        self.assertIn("ZeroDivisionError", cells[1]["error"])
        verify = next(event for event in events if event["event"] == "verify")
        self.assertTrue(verify["answer"].startswith("NO"))
        # The outcome is never part of a model request.
        requests = json.dumps([event for event in events if event["event"] == "model_request"])
        self.assertNotIn("process_score", requests)

    def test_program_finish_needs_no_further_call(self):
        # The whole task in one program that ends the episode itself: one model call.
        program = ("<python>\nfrom harness_policy.runtime.events import ProgramFinished\n"
                   "print('placed')\nraise ProgramFinished('bowls stacked')\n</python>")
        with tempfile.TemporaryDirectory() as root:
            directory = Path(root) / "seed-0"
            summary = run_program_episode(_config(), 0, directory, chat=ScriptedChat([program]),
                                          verifier=ScriptedChat([]), environment=FakeEnvironment())
            events = [json.loads(line) for line in (directory / "events.jsonl").read_text().splitlines()]
            manifest = json.loads((directory / "manifest.json").read_text())
        self.assertEqual((summary["status"], summary["model_calls"], summary["cells"]), ("finished", 1, 1))
        self.assertEqual(summary["decisions"], {"start": 1})
        finish = next(event for event in events if event["event"] == "finish")
        self.assertEqual((finish["text"], finish["source"]), ("bowls stacked", "program"))
        # The benchmark convention runs after finishing (a session without it reports an error, not a crash).
        self.assertEqual([event["event"] for event in events].index("wrap_up"),
                         [event["event"] for event in events].index("finish") + 1)
        # The embodiment profile is part of the recorded configuration; per-task fields are not.
        self.assertEqual(manifest["embodiment_profile"], {"control_hz": 25.0, "cameras": ["cam_head"]})
        self.assertIn("embodiment_sha256", manifest)

    def test_feedback_has_the_turn_report_and_the_steps(self):
        from harness_policy.runtime.driver import _cell_feedback

        text = _cell_feedback({"status": "ok", "seconds": 1.0, "report": {"text": "Scene this turn: o1 'cup'",
                                                                          "images": {}},
                               "state": {"step": 120}}, 25.0)
        self.assertIn("Scene this turn: o1 'cup'", text)
        self.assertIn("Env steps used: 120 (25 per second); the limit is not reported", text)
        self.assertIn("left: 380", _cell_feedback({"status": "ok", "state": {"step": 120, "step_limit": 500}}))

    def test_program_end_without_finish_asks_again(self):
        chat = RecordingChat(["<python>\nprint('half done')\n</python>", "<finish>done</finish>"])
        with tempfile.TemporaryDirectory() as root:
            summary = run_program_episode(_config(), 0, Path(root) / "seed-0", chat=chat,
                                          verifier=ScriptedChat([]), environment=FakeEnvironment())
        self.assertEqual(summary["decisions"], {"start": 1, "program_end": 1})
        self.assertIn("ended without calling finish()", json.dumps(chat.requests[1]))

    def test_the_calls_after_the_policy_run_are_bounded(self):
        import os
        from unittest import mock

        replies = ["<python>\nhanded()\n</python>"] + ["<python>\npass\n</python>"] * 9
        config = _config()
        config.max_model_calls = 10
        config.runtime = {**config.runtime, "bootstrap": "kernel_fixtures:HandedSession", "post_policy_calls": 2}
        path = os.pathsep.join([str(Path(__file__).resolve().parent), os.environ.get("PYTHONPATH", "")])
        chat = RecordingChat(replies)
        with mock.patch.dict(os.environ, {"PYTHONPATH": path}), tempfile.TemporaryDirectory() as root:
            summary = run_program_episode(config, 0, Path(root) / "seed-0", chat=chat,
                                          verifier=ScriptedChat([]), environment=FakeEnvironment())
            events = [json.loads(line) for line in (Path(root) / "seed-0" / "events.jsonl").read_text().splitlines()]
        # the handover's call, then two more; then the driver runs the policy on and finishes
        self.assertEqual((summary["model_calls"], summary["status"]), (3, "finished"))
        self.assertEqual([e["status"] for e in events if e["event"] == "policy_to_end"], ["ok"])
        self.assertTrue(summary["strategy"]["ran_to_end"])
        self.assertIn("runs on by itself to the end", json.dumps(chat.requests[1]))

    def test_a_handed_over_policy_uses_the_episode_wall_time(self):
        from harness_policy.runtime.kernel import Kernel

        seen = []
        real = Kernel.run

        def spy(kernel, code, timeout):
            seen.append((code, timeout))
            return real(kernel, code, timeout)

        config = _config()
        config.max_wall_seconds = 5400
        config.runtime = {**config.runtime, "cell_timeout_seconds": 1800,
                          "bootstrap": "kernel_fixtures:HandedSession", "post_policy_calls": 0}
        path = os.pathsep.join([str(Path(__file__).resolve().parent), os.environ.get("PYTHONPATH", "")])
        replies = ["<python>\nprint('look')\n</python>", "<python>\nhanded()\nstart('A', 'stacking')\n</python>"]
        with mock.patch.dict(os.environ, {"PYTHONPATH": path}), \
                mock.patch("harness_policy.runtime.driver.Kernel.run", spy), \
                tempfile.TemporaryDirectory() as root:
            run_program_episode(config, 0, Path(root) / "seed-0", chat=ScriptedChat(replies),
                                verifier=ScriptedChat([]), environment=FakeEnvironment())
        short = next(timeout for code, timeout in seen if "print(" in code)
        handed = next(timeout for code, timeout in seen if "start(" in code)
        onward = next(timeout for code, timeout in seen if code == "__policy_to_end__()")
        self.assertLessEqual(short, 1800)
        self.assertGreater(handed, 4000)
        self.assertGreater(onward, 4000)

    def test_budget_is_enforced(self):
        replies = ["<python>\npass\n</python>"] * 10
        config = _config()
        config.max_model_calls = 3
        with tempfile.TemporaryDirectory() as root:
            summary = run_program_episode(config, 0, Path(root) / "seed-0", chat=ScriptedChat(replies),
                                          verifier=ScriptedChat([]), environment=FakeEnvironment())
        self.assertEqual((summary["status"], summary["model_calls"]), ("budget_exhausted", 3))

    def test_outcome_video_is_written_to_the_episode(self):
        mp4 = bytes.fromhex("000000186674797069736f6d0000000069736f6d61766331")
        url = "data:video/mp4;base64," + base64.b64encode(mp4).decode("ascii")

        class RecordingEnvironment(FakeEnvironment):
            def call(self, name, **arguments):
                result = super().call(name, **arguments)
                return {**result, "videos": {"cam_head": url}} if name == "outcome" else result

        with tempfile.TemporaryDirectory() as root:
            directory = Path(root) / "seed-0"
            summary = run_program_episode(_config(), 0, directory, chat=ScriptedChat(["<finish>done</finish>"]),
                                          verifier=ScriptedChat([]), environment=RecordingEnvironment())
            written = (directory / "videos" / "episode-cam_head.mp4").read_bytes()
            text = (directory / "events.jsonl").read_text()
        events = [json.loads(line) for line in text.splitlines()]
        self.assertEqual(summary["process_score"], 50.0)
        self.assertEqual(written, mp4)
        video = next(event for event in events if event["event"] == "episode_video")
        self.assertEqual(video["files"], {"cam_head": "videos/episode-cam_head.mp4"})
        outcome = next(event for event in events if event["event"] == "outcome")
        self.assertNotIn("videos", outcome["outcome"])
        self.assertNotIn("video/mp4", text)

    def test_unscorable_outcome_is_an_error(self):
        reason = "RoboDojo marked task=make_kong layout=1 unstable after step 58; the episode is not scorable"

        class UnstableEnvironment(FakeEnvironment):
            def call(self, name, **arguments):
                result = super().call(name, **arguments)
                return {**result, "unscorable": reason} if name == "outcome" else result

        with tempfile.TemporaryDirectory() as root:
            directory = Path(root) / "seed-0"
            summary = run_program_episode(_config(), 0, directory, chat=ScriptedChat(["<finish>done</finish>"]),
                                          verifier=ScriptedChat([]), environment=UnstableEnvironment())
        self.assertEqual(summary["status"], "error")
        self.assertEqual(summary["error"], f"Unscorable: {reason}")

    def test_video_error_keeps_the_outcome(self):
        class FailingEnvironment(FakeEnvironment):
            def call(self, name, **arguments):
                result = super().call(name, **arguments)
                return {**result, "video_error": "RuntimeError: no ffmpeg"} if name == "outcome" else result

        with tempfile.TemporaryDirectory() as root:
            directory = Path(root) / "seed-0"
            summary = run_program_episode(_config(), 0, directory, chat=ScriptedChat(["<finish>done</finish>"]),
                                          verifier=ScriptedChat([]), environment=FailingEnvironment())
            events = [json.loads(line) for line in (directory / "events.jsonl").read_text().splitlines()]
            self.assertFalse((directory / "videos").exists())
        self.assertEqual((summary["reward"], summary["process_score"], summary["steps"]), (0.5, 50.0, 3))
        outcome = next(event for event in events if event["event"] == "outcome")
        self.assertNotIn("video_error", outcome["outcome"])
        errors = [event["message"] for event in events if event["event"] == "cleanup_error"]
        self.assertEqual(errors, ["video: RuntimeError: no ffmpeg"])

    def test_malformed_exchanges_leave_the_history(self):
        chat = RecordingChat(["thinking only", "still no block", "<python>\nprint(1)\n</python>",
                              "<finish>done</finish>"])
        with tempfile.TemporaryDirectory() as root:
            summary = run_program_episode(_config(), 0, Path(root) / "seed-0", chat=chat,
                                          verifier=ScriptedChat([]), environment=FakeEnvironment())
        self.assertEqual((summary["status"], summary["model_calls"]), ("finished", 4))
        # The third request still shows both format errors; the fourth shows neither.
        self.assertIn("still no block", json.dumps(chat.requests[2]))
        last = json.dumps(chat.requests[3])
        self.assertNotIn("thinking only", last)
        self.assertNotIn("Format error", last)
        self.assertIn("print(1)", last)

    def test_cost_cap_stops_the_episode(self):
        class PricedChat(ScriptedChat):
            def complete(self, system, messages):
                reply = super().complete(system, messages)
                reply.usage["cost"] = 0.4
                return reply

        config = _config()
        config.max_cost_usd = 1.0
        with tempfile.TemporaryDirectory() as root:
            summary = run_program_episode(config, 0, Path(root) / "seed-0",
                                          chat=PricedChat(["<python>\npass\n</python>"] * 10),
                                          verifier=ScriptedChat([]), environment=FakeEnvironment())
        self.assertEqual((summary["status"], summary["model_calls"]), ("cost_exhausted", 3))

    def test_task_description_follows_the_instruction(self):
        config = _config()
        config.task_description = "# Stack Bowls\n\nStack all bowls."
        with tempfile.TemporaryDirectory() as root:
            run_program_episode(config, 0, Path(root) / "seed-0", chat=ScriptedChat(["<finish>done</finish>"]),
                                verifier=ScriptedChat([]), environment=FakeEnvironment())
            events = [json.loads(line) for line in (Path(root) / "seed-0" / "events.jsonl").read_text().splitlines()]
        first = next(event for event in events if event["event"] == "model_request")["new_messages"][0]
        self.assertTrue(first["content"][0]["text"].startswith(
            "Instruction: Stack the bowls.\n\nTask description:\n# Stack Bowls\n\nStack all bowls.\n\nRobot state:"))


class ModelOverrideTests(unittest.TestCase):
    EXPERIMENT = Path(__file__).resolve().parents[1] / "configs" / "experiments" / "dev.toml"
    DEPLOY = str(Path(__file__).resolve().parent / "fixtures" / "deploy-test.toml")

    def load(self, **options):
        return load_task_config(self.EXPERIMENT, deploy=self.DEPLOY, **options)

    def test_model_picks_another_model_file(self):
        default = self.load()
        config = self.load(model="grok-low")
        self.assertEqual((config.main["model"], config.main["reasoning_effort"]), ("x-ai/grok-4.7", "low"))
        self.assertEqual(default.main["backend"], "openai_chat")
        self.assertEqual(default.main["base_url"], "https://api.deepseek.com")
        self.assertEqual(default.main["model"], "deepseek-flash")
        self.assertEqual(default.main["api_key_env"], "DEEPSEEK_API_KEY")
        # The VLM role is the main model itself (one model sees the images).
        self.assertEqual(config.runtime["vlm"]["model"], config.main["model"])
        self.assertEqual(default.runtime["vlm"]["model"], default.main["model"])
        self.assertEqual(config.system_prompt, default.system_prompt)
        deepseek = self.load(model="deepseek-flash")
        self.assertEqual(deepseek.main["base_url"], "https://api.deepseek.com")
        self.assertEqual(deepseek.main["api_key_env"], "DEEPSEEK_API_KEY")

    def test_qwen_url_comes_from_the_environment(self):
        with mock.patch.dict(os.environ, {"QWEN_BASE_URL": "http://eval.example/v1"}):
            config = self.load(model="qwen3.8-27b")
        self.assertEqual(config.main["base_url"], "http://eval.example/v1")
        self.assertEqual(config.main["model"], "qwen3.8-27b")
        self.assertEqual(config.main["extra_body"]["chat_template_kwargs"]["enable_thinking"], False)
        self.assertEqual(config.runtime["vlm"]["base_url"], "http://eval.example/v1")
        self.assertEqual(config.runtime["vlm"]["extra_body"]["chat_template_kwargs"]["enable_thinking"], False)
        # An empty verifier table sits on top of main, so a check stays on this server.
        merged = {**config.main, **config.verifier}
        self.assertEqual(merged["base_url"], "http://eval.example/v1")
        self.assertEqual(merged["extra_body"]["chat_template_kwargs"]["enable_thinking"], False)
        with mock.patch.dict(os.environ):
            os.environ.pop("QWEN_BASE_URL", None)
            bare = self.load(model="qwen3.8-27b")
        self.assertNotIn("base_url", bare.main)
        self.assertEqual(bare.agent_id, config.agent_id)

    def test_a_main_model_id_by_set(self):
        config = self.load(model="deepseek-flash", sets=['model.main.model="z-ai/glm-5.3-flash"'])
        self.assertEqual(config.main["model"], "z-ai/glm-5.3-flash")
        self.assertEqual(config.main["reasoning_effort"], "medium")

    def test_without_model_the_agents_model_is_kept(self):
        config = self.load()
        self.assertEqual(config.main["model"], "deepseek-flash")
        self.assertEqual(config.main["base_url"], "https://api.deepseek.com")

    def test_unknown_model_is_rejected(self):
        for model in ("no-such-model", "  "):
            with self.assertRaises(ValueError):
                self.load(model=model)

    def test_cli_passes_the_model(self):
        args = parser().parse_args(["program", str(self.EXPERIMENT), "--model", "grok-medium"])
        self.assertEqual(args.model, "grok-medium")
        self.assertIsNone(parser().parse_args(["program", str(self.EXPERIMENT)]).model)


class TaskDescriptionFileTests(unittest.TestCase):
    def test_every_task_has_a_description(self):
        from harness_policy.runtime.report import DIMENSIONS

        directory = Path(__file__).resolve().parents[1] / "prompts" / "robodojo_tasks"
        missing = [task for tasks in DIMENSIONS.values() for task in tasks
                   if not (directory / f"{task}.md").is_file()]
        self.assertEqual(missing, [])


class RecordingChat(ScriptedChat):
    def __init__(self, replies):
        super().__init__(replies)
        self.requests = []

    def complete(self, system, messages):
        self.requests.append(json.loads(json.dumps(messages)))
        return super().complete(system, messages)


class ShortContextChat(ScriptedChat):
    """Fails like a self-hosted model out of memory when sent more than `limit` messages."""

    def __init__(self, replies, limit):
        super().__init__(replies)
        self.limit = limit
        self.sizes = []

    def complete(self, system, messages):
        self.sizes.append(len(messages))
        if len(messages) > self.limit:
            raise RuntimeError("Provider HTTP 400: out of GPU memory for this context")
        return super().complete(system, messages)


class ContextShorteningTests(unittest.TestCase):
    def test_out_of_memory_retries_with_fewer_turns(self):
        replies = ["<python>\npass\n</python>"] * 4 + ["<finish>done</finish>"]
        chat = ShortContextChat(replies, limit=5)
        with tempfile.TemporaryDirectory() as root:
            summary = run_program_episode(_config(), 0, Path(root) / "seed-0", chat=chat,
                                          verifier=ScriptedChat([]), environment=FakeEnvironment())
            events = [json.loads(line) for line in (Path(root) / "seed-0" / "events.jsonl").read_text().splitlines()]
        self.assertEqual(summary["status"], "finished")
        shortened = [event for event in events if event["event"] == "context_shortened"]
        self.assertTrue(shortened)
        self.assertEqual(chat.calls, 5)  # every scripted reply was delivered
        # Once shortened, later requests stay within the limit (no repeated failures).
        first_ok = next(i for i, size in enumerate(chat.sizes) if size > 5) + 1
        self.assertTrue(all(size <= 5 for size in chat.sizes[first_ok:]))


class WindowTests(unittest.TestCase):
    def test_keeps_first_message_and_recent_turns(self):
        messages = [{"role": "user", "content": [{"type": "input_text", "text": "instruction"}]}]
        for turn in range(6):
            messages += [{"role": "assistant", "content": f"a{turn}"}, {"role": "user", "content": f"u{turn}"}]
        self.assertIs(_window(messages, None), messages)
        self.assertIs(_window(messages, 6), messages)
        window = _window(messages, 2)
        self.assertEqual([m["content"] for m in window[1:]], ["a4", "u4", "a5", "u5"])
        self.assertIn("8 later messages are omitted", window[0]["content"][-1]["text"])
        self.assertEqual(len(messages[0]["content"]), 1)  # the original is not modified


class PromptLayerTests(unittest.TestCase):
    def test_sections_follow_layers(self):
        full = assemble_prompt(PROMPTS, skills_for_layers({}))
        for name in ("move_eef(", "when(predicate", "run_controller(", "on(event"):
            self.assertIn(name, full)
        fixed = assemble_prompt(PROMPTS, skills_for_layers({"reactive": False, "handlers": False}))
        for name in ("when(predicate", "run_controller(", "on(event", "Use when, watch"):
            self.assertNotIn(name, fixed)
        self.assertIn("move_eef(", fixed)

    def test_learned_layer_and_unknown_layers(self):
        self.assertNotIn("run_policy(", assemble_prompt(PROMPTS, skills_for_layers({})))
        learned = assemble_prompt(PROMPTS, skills_for_layers({"learned": True}))
        self.assertIn("run_policy(", learned)
        self.assertIn("Your first motion is start(strategy, kind)", learned)
        self.assertIn("unless the instruction asks for a read", learned)
        with tempfile.TemporaryDirectory() as root:
            (Path(root) / "core.md").write_text("core")
            with self.assertRaises(ValueError):
                assemble_prompt(Path(root), skills_for_layers({}))
        with self.assertRaises(ValueError):
            layer_settings({"layers": {"memory": True}})


class ScriptedModelsTests(unittest.TestCase):
    def test_a_replay_script_brings_its_verifier_answers(self):
        from harness_policy.runtime.chat import scripted_models

        models = scripted_models({"replies": ["<finish>a</finish>"], "verifier": ["NO apart"]})
        self.assertEqual(models["chat"].complete("", []).text, "<finish>a</finish>")
        self.assertEqual(models["verifier"].complete("", []).text, "NO apart")
        models = scripted_models(["<finish>b</finish>"])
        self.assertTrue(models["verifier"].complete("", []).text.startswith("YES"))


if __name__ == "__main__":
    unittest.main()
