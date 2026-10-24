import contextlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]

PIXEL = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="


def _load_viewer():
    spec = importlib.util.spec_from_file_location("view_runs", ROOT / "scripts" / "view_runs.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_episode(directory, *, runtime, layout_source, success, video=False):
    directory.mkdir(parents=True)
    summary = {"run": str(directory), "task": "stack_bowls", "seed": 0, "success": success,
               "status": "finished" if success else "budget_exhausted", "process_score": 100.0 if success else 25.0,
               "reward": 1.0 if success else 0.25, "model_calls": 2, "steps": 40, "cost_usd": 0.01,
               "wall_seconds": 90.0}
    manifest = {"config": {"agent": {"model": "provider/some-model"}, "runtime": runtime,
                           "system_prompt": "You control a robot.\nSecond line.", "max_model_calls": 8}}
    state = {"layout_source": layout_source, "layout_pack_seed": 0 if layout_source == "eval" else None}
    events = [
        {"seq": 0, "event": "reset", "observation": {"instruction": "Stack the bowls.", "state": state,
                                                    "images": {"cam_head": PIXEL}}},
        {"seq": 1, "event": "model_request", "call": 0, "new_messages": [{"role": "user", "content": [
            {"type": "input_text", "text": "Instruction: Stack the bowls."},
            {"type": "input_text", "text": "Camera: cam_head"}, {"type": "input_image", "image_url": PIXEL}]}]},
        {"seq": 2, "event": "model_reply", "call": 1, "text": "<action> {\"type\": \"observe\"} </action>",
         "usage": {"input_tokens": 10, "output_tokens": 5, "cost": 0.001}, "seconds": 1.5},
        {"seq": 3, "event": "cell", "action": "{\"type\": \"observe\"}",
         "result": {"status": "ok", "stdout": "fingertip arrival_error_m 0.0001", "seconds": 0.2, "state": {"step": 40},
                    "events": [{"skill": "move_eef", "event": {"kind": "COMPLETED"}, "ticks": 3}]}},
        {"seq": 4, "event": "outcome", "outcome": {"success": success, "process_score": 25.0, "steps": 40}},
    ]
    if video:
        (directory / "videos").mkdir()
        (directory / "videos" / "episode-cam_head.mp4").write_bytes(b"not a real mp4")
        events.append({"seq": 5, "event": "episode_video", "files": {"cam_head": "videos/episode-cam_head.mp4"}})
    (directory / "summary.json").write_text(json.dumps(summary))
    (directory / "manifest.json").write_text(json.dumps(manifest))
    (directory / "events.jsonl").write_text("".join(json.dumps(event) + "\n" for event in events))


class ViewRunsTests(unittest.TestCase):
    def test_rung_follows_the_harness_ladder(self):
        rung = _load_viewer().rung
        self.assertEqual(rung({"mode": "direct", "layers": {"learned": False}}), "R0")
        self.assertEqual(rung({"mode": "direct", "layers": {"learned": True}}), "R1")
        self.assertEqual(rung({"layers": {"learned": False, "reactive": False, "handlers": True}}), "R2")
        self.assertEqual(rung({"layers": {"learned": True, "reactive": False, "handlers": True}}), "R3")
        self.assertEqual(rung({"layers": {"learned": True, "reactive": True, "handlers": True}}), "R4")
        self.assertEqual(rung({"layers": {"learned": True, "reactive": True, "handlers": False}}), "R4 − handlers")
        self.assertEqual(rung({"layers": {"learned": True}, "verifier": False}), "R4 − verifier")

    def test_pages_are_written_for_every_run_and_episode(self):
        viewer = _load_viewer()
        with tempfile.TemporaryDirectory() as root:
            runs = Path(root) / "runs"
            _write_episode(runs / "20260930-001309-aaaaaaaa" / "seed-0", layout_source="procedural", success=False,
                           runtime={"mode": "direct", "layers": {"learned": False}}, video=True)
            _write_episode(runs / "20260930-004438-bbbbbbbb" / "seed-0", layout_source="eval", success=True,
                           runtime={"layers": {"learned": True, "reactive": True, "handlers": True}})
            argv, sys.argv = sys.argv, ["view_runs.py", str(runs)]
            try:
                with contextlib.redirect_stdout(io.StringIO()):
                    viewer.main()
            finally:
                sys.argv = argv
            index = (runs / "index.html").read_text()
            run_page = (runs / "20260930-001309-aaaaaaaa" / "index.html").read_text()
            episode = (runs / "20260930-001309-aaaaaaaa" / "seed-0" / "view.html").read_text()
            eval_episode = (runs / "20260930-004438-bbbbbbbb" / "seed-0" / "view.html").read_text()

        # The index aggregates only; episodes live on their run's page.
        self.assertIn("Development layouts · 1", index)
        self.assertIn("Evaluation layouts (official packs) · 1", index)
        self.assertLess(index.index("Evaluation layouts"), index.index("Development layouts"))
        self.assertIn("href='20260930-001309-aaaaaaaa/index.html'", index)
        self.assertNotIn("view.html", index)
        self.assertIn("<b>R0</b>", index)
        self.assertIn("<b>R4</b>", index)
        self.assertIn("0 / 1", index)
        self.assertIn("1 / 1", index)
        self.assertIn("href='seed-0/view.html'", run_page)
        # The episode page escapes model text and shows the prompt, the turn and the video.
        self.assertIn("&lt;action&gt;", episode)
        self.assertNotIn("<action>", episode)
        self.assertIn("System prompt", episode)
        self.assertIn("videos/episode-cam_head.mp4", episode)
        self.assertIn(f"data-rate={viewer.VIDEO_SPEED:g}", episode)
        # Each executed action is a clickable span of the video: control step 40 at 25 Hz = 1.6 s.
        self.assertIn("Jump to turn", episode)
        self.assertIn("data-start=0.00 data-end=1.60", episode)
        self.assertIn("onclick='seekVideos(0.00)'>▶ 0:00.0–0:01.6", episode)
        self.assertIn("thought 1.5 s", episode)  # the model's reply time before the action
        self.assertNotIn("scrollIntoView", episode)  # jumping does not move the page
        # Each turn links to its first message (the model request, seq 1) among the events.
        self.assertIn("href='#ev-1'", episode)
        self.assertIn("id='ev-1'", episode)
        self.assertIn('video.addEventListener("seeking"', episode)  # any camera's drag moves all
        # The turn's cards carry the marker's number, and the link highlights them.
        self.assertIn("<span class=turnbadge>Turn 1</span>", episode)
        self.assertIn("data-turn=1>", episode)
        self.assertIn("onclick='flashTurn(1)'", episode)
        self.assertIn("<button class=r-ok title='ok'", episode)  # marker colored by the action's result
        self.assertNotIn("Jump to turn", eval_episode)  # no video, no markers
        self.assertNotIn("<video", eval_episode)
        # No machine-specific paths in the pages; the refresh command is relative.
        for page in (index, run_page, episode):
            self.assertNotIn(root, page)
            self.assertIn("python3 scripts/view_runs.py runs", page)

    def test_runs_in_subfolders_are_listed_and_link_back(self):
        viewer = _load_viewer()
        with tempfile.TemporaryDirectory() as root:
            runs = Path(root) / "runs"
            nested = runs / "r0-models" / "x-ai-grok-4.7" / "20260930-201500-dddddddd"
            _write_episode(nested / "seed-0", layout_source="procedural", success=False,
                           runtime={"mode": "direct", "layers": {"learned": False}})
            _write_episode(runs / "20260930-001309-aaaaaaaa" / "seed-0", layout_source="procedural", success=True,
                           runtime={"mode": "direct", "layers": {"learned": False}})
            argv, sys.argv = sys.argv, ["view_runs.py", str(runs)]
            try:
                with contextlib.redirect_stdout(io.StringIO()):
                    viewer.main()
            finally:
                sys.argv = argv
            index = (runs / "index.html").read_text()
            run_page = (nested / "index.html").read_text()
            episode = (nested / "seed-0" / "view.html").read_text()

        self.assertIn("Development layouts · 2", index)
        self.assertIn("href='r0-models/x-ai-grok-4.7/20260930-201500-dddddddd/index.html'", index)
        self.assertIn("<small>r0-models/x-ai-grok-4.7</small>", index)
        # Newest first by experiment time, not by folder name.
        self.assertLess(index.index("20260930-201500"), index.index("20260930-001309"))
        self.assertIn("href='../../../index.html'>← all runs", run_page)
        self.assertIn("href='../../../../index.html'>← all runs", episode)
        self.assertIn("(2026-09-30 20:15)", episode)

    def test_overview_and_run_tables_are_sortable(self):
        viewer = _load_viewer()
        with tempfile.TemporaryDirectory() as root:
            runs = Path(root) / "runs"
            _write_episode(runs / "20260930-001309-aaaaaaaa" / "seed-0", layout_source="procedural", success=False,
                           runtime={"mode": "direct", "layers": {"learned": False}})
            argv, sys.argv = sys.argv, ["view_runs.py", str(runs)]
            try:
                with contextlib.redirect_stdout(io.StringIO()):
                    viewer.main()
            finally:
                sys.argv = argv
            index = (runs / "index.html").read_text()
            run_page = (runs / "20260930-001309-aaaaaaaa" / "index.html").read_text()
        for page in (index, run_page):
            self.assertIn("<table class='sortable filterable'>", page)
            self.assertIn('querySelectorAll("table.sortable")', page)
            self.assertNotIn("<script src", page)
        # Formatted cells carry their raw value for sorting.
        self.assertIn("data-sort='90.0'>1.5 min", index)
        self.assertIn("data-sort='0.01'>$0.0100", index)

    def test_overview_and_run_tables_have_dropdown_filters(self):
        viewer = _load_viewer()
        with tempfile.TemporaryDirectory() as root:
            runs = Path(root) / "runs"
            _write_episode(runs / "sweep-a" / "provider-some-model" / "20260930-001309-aaaaaaaa" / "seed-0",
                           layout_source="procedural", success=False, runtime={"mode": "direct", "layers": {}})
            argv, sys.argv = sys.argv, ["view_runs.py", str(runs)]
            try:
                with contextlib.redirect_stdout(io.StringIO()):
                    viewer.main()
            finally:
                sys.argv = argv
            index = (runs / "index.html").read_text()
            run_page = (runs / "sweep-a" / "provider-some-model" / "20260930-001309-aaaaaaaa" / "index.html").read_text()
        for page in (index, run_page):
            self.assertIn("<div class=filters>", page)
            self.assertIn('querySelectorAll("table.filterable")', page)
            self.assertIn('search.type = "search"', page)  # free-text search box
            self.assertIn('box.type = "checkbox"', page)   # multi-select dropdowns
        for name in ("Sweep", "Rung", "Model", "Task", "Ended"):
            self.assertIn(f"data-filter={name}>", index)
        # The sweep is the top-level folder, the model its full id, endings their labels.
        self.assertIn("data-f='sweep-a'", index)
        self.assertIn("data-f='provider/some-model'", index)
        self.assertIn("data-f='R0'", index)
        self.assertIn("data-f='out of budget'", index)
        for name in ("Task", "Result", "Ended"):
            self.assertIn(f"data-filter={name}>", run_page)
        self.assertIn("data-f='fail'", run_page)

    def test_missing_or_partial_files_do_not_break_the_index(self):
        viewer = _load_viewer()
        with tempfile.TemporaryDirectory() as root:
            runs = Path(root) / "runs"
            episode = runs / "20260930-000000-cccccccc" / "seed-0"
            episode.mkdir(parents=True)
            (episode / "summary.json").write_text(json.dumps({"task": "stack_bowls", "status": "error",
                                                             "error": "URLError: connection refused"}))
            (episode / "events.jsonl").write_text('{"event": "reset"\n')  # truncated line
            argv, sys.argv = sys.argv, ["view_runs.py", str(runs)]
            try:
                with contextlib.redirect_stdout(io.StringIO()):
                    viewer.main()
            finally:
                sys.argv = argv
            index = (runs / "index.html").read_text()
            page = (episode / "view.html").read_text()
        self.assertIn("Other (mock runs, or no layout recorded) · 1", index)
        self.assertIn("1 error", index)
        self.assertIn("URLError: connection refused", page)


if __name__ == "__main__":
    unittest.main()


class ServeRunsTests(unittest.TestCase):
    def test_scores_follow_harness_report(self):
        viewer = _load_viewer()
        sys.path.insert(0, str(ROOT / "src"))
        from harness_policy.runtime.report import aggregate

        def episode(directory, task, seed, **summary):
            directory.mkdir(parents=True)
            base = {"run": str(directory), "task": task, "seed": seed, "model_calls": 1, "cost_usd": 0.01}
            (directory / "summary.json").write_text(json.dumps({**base, **summary}))
            (directory / "manifest.json").write_text(json.dumps({"config": {"agent": {"model": "p/m"},
                                                                             "runtime": {"label": "Z1"}}}))
            state = {"layout_source": "eval", "layout_pack_seed": 0}
            (directory / "events.jsonl").write_text(json.dumps({"event": "reset", "observation": {"state": state}}) + "\n")
            return {**base, **summary}

        with tempfile.TemporaryDirectory() as root:
            runs = Path(root) / "runs"
            suite = runs / "suite"
            written = [
                episode(suite / "push_T" / "seed-0", "push_T", 0, status="finished", success=True, reward=1.0,
                        process_score=100.0),
                # A crash with partial progress counts 0, as in harness report.
                episode(suite / "push_T" / "seed-1", "push_T", 1, status="error", success=False, reward=0.5,
                        process_score=50.0, error="RuntimeError: worker died"),
                # Dropped by the simulator: not scored.
                episode(suite / "push_T" / "seed-2", "push_T", 2, status="error", success=False, reward=None,
                        process_score=0.0, error="Unscorable: RoboDojo marked task=push_T layout=2 unstable after step 9"),
                episode(suite / "cover_blocks" / "seed-0", "cover_blocks", 0, status="episode_ended", success=False,
                        reward=0.2, process_score=20.0),
            ]
            broken = suite / "unknown" / "seed-0"
            broken.mkdir(parents=True)
            (broken / "summary.json").write_text("{}")  # unreadable or partial: skipped, as by harness report
            odd = suite / "push_T_odd" / "seed-0"
            odd.mkdir(parents=True)
            (odd / "summary.json").write_text(json.dumps({"task": "push_T", "seed": 3, "reward": "n/a"}))
            argv, sys.argv = sys.argv, ["view_runs.py", str(runs)]
            try:
                with contextlib.redirect_stdout(io.StringIO()):
                    viewer.main()
            finally:
                sys.argv = argv
            index = (runs / "index.html").read_text()
            push_page = (suite / "push_T" / "index.html").read_text()
        # The push_T row: 1 success of 2 scored episodes, score (100 + 0) / 2, and the dropped one noted.
        self.assertIn("1 / 2</span><br><small>+1 dropped</small>", index)
        self.assertIn("<td class=n>50.0</td>", index)
        self.assertIn(">dropped</span>", push_page)
        # The suite row shows harness report's dimension numbers for the same episodes.
        report = aggregate(written)
        memory = report["dimensions"]["Memory"]
        self.assertIn(f"{memory['score']:.1f} / {memory['sr']:.1f}", index)
        self.assertIn("<b>suite</b>", index)
        # Errors counts the crash only; the dropped episode ends "dropped", not "error".
        self.assertRegex(index, r"data-f='dropped\|error\|finished'>(1 (dropped|error|finished)(, )?){3}</td>")
        self.assertRegex(index, r"<td class=n>1</td></tr>")
        # Summaries harness report could not read are skipped in the suite row, not fatal to the page.
        self.assertIn("50.0 / 50.0<br><small>1/12 tasks</small>", index)  # Generalization: push_T
        self.assertIn("<td>Z1</td><td>p/m</td>", index)  # rung and model of the readable runs only
        self.assertIn("<b>suite</b><br><small>2 unreadable skipped</small>", index)

    def test_partial_overall_is_marked_and_counts_its_tasks(self):
        viewer = _load_viewer()
        sys.path.insert(0, str(ROOT / "src"))
        from harness_policy.runtime.report import DIMENSIONS, aggregate

        # One success in every task except the last one of each dimension: overall needs all tasks.
        episodes = [{"task": task, "seed": 0, "status": "finished", "success": True, "reward": 1.0}
                    for tasks in DIMENSIONS.values() for task in tasks[:-1]]
        result = aggregate(episodes)
        self.assertIsNone(result["overall"])
        partial = viewer.partial_overall(result["dimensions"])
        self.assertEqual((partial["score"], partial["tasks"], partial["of"]), (100.0, 37, 42))
        self.assertEqual(viewer.partial_cell(partial), "100.0* / 100.0*<br><small>37/42 tasks</small>")
        # A dimension with no task run leaves it blank too.
        no_open = aggregate([e for e in episodes if e["task"] not in DIMENSIONS["Open"]])
        self.assertIsNone(viewer.partial_overall(no_open["dimensions"]))
        # With every task run it equals harness report's overall.
        full = aggregate(episodes + [{"task": tasks[-1], "seed": 0, "status": "finished", "success": False,
                                      "reward": 0.5} for tasks in DIMENSIONS.values()])
        partial = viewer.partial_overall(full["dimensions"])
        self.assertAlmostEqual(partial["score"], full["overall"]["score"])
        self.assertAlmostEqual(partial["sr"], full["overall"]["sr"])
        self.assertIn("= Overall", viewer.partial_cell(partial))
        self.assertNotIn("*", viewer.partial_cell(partial))

    def test_programs_are_highlighted_without_changing_the_code(self):
        import html
        import re

        viewer = _load_viewer()
        code = ('# Find the bowl, then pick it.\n'
                'obs = get_observation()\n'
                'bowl = localize_3d(obs, "white bowl")  # one call\n'
                'picked = pick("white bowl", attempts=2, lift=0.12)\n'
                'note = f"result {picked}"\n'
                'prompt = "' + "describe the object " * 10 + '"\n'
                'doc = """two\nlines"""\n'
                '# Finish.\n'
                'if picked.get("ok"):\n'
                '    finish("done")\n')

        def text_of(rendered):
            # Long strings show their start; the full text is in data-full.
            rendered = re.sub(r"<span class='s x'[^>]*data-full='([^']*)'[^>]*>.*?</span>",
                              lambda m: m.group(1), rendered)
            return html.unescape(re.sub(r"<[^>]+>", "", rendered.replace("</span><span class=l>", "\n")))

        for source in (code, "def broken(:\n    x = 'unterminated\n", "", '""\r\n', 'x = "a\x0cb"\ny = 1\n'):
            rendered = viewer.code_html(source)
            self.assertEqual(text_of(rendered).rstrip(" "), source.rstrip("\n"), source)
        rendered = viewer.code_html(code)
        self.assertIn("<span class=a>pick</span>", rendered)
        self.assertNotIn("<span class=a>picked</span>", rendered)  # a variable, not a skill
        self.assertIn("<span class=k>if</span>", rendered)
        self.assertIn("<span class=c># one call</span>", rendered)
        self.assertIn("(202 chars)", rendered)  # the long prompt is folded
        outline = viewer.code_outline(code)
        self.assertIn("<b>pick</b> ×1 <small>(L4)</small>", outline)
        self.assertIn("<b>finish</b> ×1 <small>(L11)</small>", outline)
        self.assertIn("<li>Find the bowl, then pick it. <small>(L1)</small></li>", outline)
        self.assertEqual(viewer.code_outline("def broken(:"), "")

        reply = viewer.reply_html("I will pick the bowl.\n<python>\n" + code + "</python>")
        self.assertIn("<pre class=prose>I will pick the bowl.</pre>", reply)
        prose, raw = reply.split("<details class=fold>", 1)
        self.assertNotIn("&lt;python&gt;", prose)
        self.assertIn("Raw reply, exactly as the model returned it", raw)
        self.assertIn("&lt;python&gt;\n# Find the bowl", raw)  # the original text, tags included
        self.assertEqual(viewer.reply_html('locate({"object": "bowl"})'), '<pre>locate({&quot;object&quot;: &quot;bowl&quot;})</pre>')
        self.assertNotIn("<span class=a>", viewer.code_html("obs.show(obs)"))  # a method, not the skill
        fenced = viewer.reply_html("<python>\n\n```python\nhold(1)\n```\n</python>")
        self.assertNotIn("```", fenced.split("<details")[0])  # shown as the kernel runs it
        self.assertIn("<b>hold</b> ×1 <small>(L1)</small>", fenced)
        rejected = viewer.reply_html("<python>\nhold(1)\n</python>\n<python>\nhold(2)\n</python>")
        self.assertIn("The harness rejected this reply", rejected)
        self.assertNotIn("<pre class=code>", rejected)
        tricky = viewer.reply_html("Think </python> carefully.\n<python>\nhold(1)\n</python>\nafter")
        self.assertIn("<pre class=prose>after</pre>", tricky)
        self.assertIn("<pre>", viewer.reply_html('<action>{"type": "observe"}</action> no <python> here'))
        self.assertNotIn("rejected", viewer.reply_html('<action>{"type": "observe"}</action> no <python> here'))
        direct = viewer.event_html({"event": "cell", "seq": 4, "action": '{"type": "observe"}',
                                    "code": "observe()", "result": {"status": "ok"}})
        self.assertIn("{&quot;type&quot;: &quot;observe&quot;}", direct)  # the model's action, shown
        self.assertNotIn("Program:", direct)
        cell = viewer.event_html({"event": "cell", "seq": 3, "code": code, "result": {"status": "finished"}})
        self.assertIn("<summary>Program: 11 lines (the code in the reply above)</summary>", cell)

    def test_cells_that_end_normally_are_green(self):
        viewer = _load_viewer()
        for status, marker, event_class in (("ok", "r-ok", "e-ok"), ("finished", "r-ok", "e-ok"),
                                            ("episode_ended", "r-ok", "e-ok"), ("escalated", "r-esc", "e-esc"),
                                            ("error", "r-bad", "e-bad"), ("protocol_error", "r-bad", "e-bad")):
            cell = {"event": "cell", "result": {"status": status}}
            self.assertEqual(viewer.result_class(cell), (marker, status))
            self.assertEqual(viewer.event_class(cell), event_class)

    def test_range_requests_are_served_for_video_seeking(self):
        import threading
        import urllib.request
        from functools import partial
        from http.server import ThreadingHTTPServer

        spec = importlib.util.spec_from_file_location("serve_runs", ROOT / "scripts" / "serve_runs.py")
        serve_runs = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(serve_runs)
        with tempfile.TemporaryDirectory() as root:
            (Path(root) / "clip.mp4").write_bytes(bytes(range(256)) * 4)
            server = ThreadingHTTPServer(("127.0.0.1", 0), partial(serve_runs.RangeHandler, directory=root))
            threading.Thread(target=server.serve_forever, daemon=True).start()
            url = f"http://127.0.0.1:{server.server_address[1]}/clip.mp4"
            try:
                request = urllib.request.Request(url, headers={"Range": "bytes=10-19"})
                with urllib.request.urlopen(request) as response:
                    self.assertEqual(response.status, 206)
                    self.assertEqual(response.headers["Content-Range"], "bytes 10-19/1024")
                    self.assertEqual(response.read(), bytes(range(10, 20)))
                with urllib.request.urlopen(url) as response:  # a plain request still gets the whole file
                    self.assertEqual(response.status, 200)
                    self.assertEqual(len(response.read()), 1024)
            finally:
                server.shutdown()
                server.server_close()
