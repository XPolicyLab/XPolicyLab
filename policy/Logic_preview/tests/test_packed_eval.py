import json
import os
import signal
import subprocess
import time
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
from unittest.mock import patch
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from packed_eval import Queue, Task, attempt_result, publish, read_result, write_json
from packed_watchdog import update_job_monitor


class PackedResultTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        parent = self.root / "official"
        parent.mkdir()
        self.task = Task("align_blocks", 3, 5, parent)

    def shard(self, name, ids):
        directory = self.root / name
        directory.mkdir()
        details = {}
        for index, layout in enumerate(ids):
            success = layout == 1
            details[str(index)] = dict(layout_id=layout, score=1.0 if success else 0.2, success=success)
            for camera in ("head", "left_wrist", "right_wrist"):
                path = directory / f"episode_{index:07d}_cam_{camera}_{'success' if success else 'fail'}.mp4"
                path.write_text(str(layout))
        (directory / "_result.json").write_text(json.dumps(dict(eval_time=len(ids), details=details)))
        return directory

    def test_out_of_order_workers_preserve_scores_and_video_indices(self):
        last = self.shard("last", [2])
        first = self.shard("first", [0, 1])
        left, meta = read_result(first, [0, 1])
        right, _ = read_result(last, [2])
        publish(self.task, right | left, meta, self.root, "run")
        out = self.task.parent / "run"
        payload = json.loads((out / "_result.json").read_text())
        self.assertEqual([r["layout_id"] for r in payload["details"].values()], [0, 1, 2])
        self.assertEqual(payload["success_rate"], 1 / 3)
        self.assertEqual(payload["score"], (0.2 + 1.0 + 0.2) / 3 * 100)
        self.assertEqual((out / "episode_0000002_cam_head_fail.mp4").read_text(), "2")

    def queue(self, tasks=None, chunk=3, attempts=5):
        args = SimpleNamespace(chunk=chunk, max_attempts=attempts, policy_host="host", port=1000)
        return Queue(args, tasks or [self.task], self.root, {"pipelines": {}})

    def test_retry_keeps_complete_failures_and_recovers_only_missing_video(self):
        queue = self.queue()
        task, ids, folder, lease = queue.claim("0")
        with lease:
            self.assertIsNone(queue.claim("1"))
            self.shard(folder.name, ids)
            next(folder.glob("episode_0000001*right_wrist*")).unlink()
            write_json(folder / "exit.json", 99)
        queue = self.queue()
        task, ids, folder, lease = queue.claim("1")
        with lease:
            self.assertEqual(ids, [1])
            self.shard(folder.name, ids)
        self.assertFalse(queue.claim("2"))
        self.assertTrue(queue.claim("1"))
        self.assertTrue(self.queue().claim("2"))

    def test_unstable_topup_selects_lowest_stable_ids(self):
        queue = self.queue()
        task, ids, folder, lease = queue.claim("0")
        with lease:
            self.shard(folder.name, [2, 0])
            write_json(folder / "native-resume.json", {"abandoned_layout_ids": [1, 3, 4]})
            write_json(folder / "exit.json", 0)
        self.assertFalse(queue.claim("0"))
        task, ids, folder, lease = queue.claim("0")
        with lease:
            self.assertEqual(ids, [3, 4])
            self.shard(folder.name, [4, 3])
        self.assertFalse(queue.claim("0"))
        result, = self.task.parent.glob("*/_result.json")
        rows = json.loads(result.read_text())["details"]
        self.assertEqual([r["layout_id"] for r in rows.values()], [0, 2, 3])

    def test_only_clean_exhaustion_can_classify_omitted_layouts(self):
        folder = self.shard("frag", [0])
        for code, fault, missing in ((99, "", set()), (0, "[Eval] unhandled exception", set()), (0, "", {1})):
            (folder / "client.log").write_text("Unstable nums: 1\nNo more seeds to run, exiting.\n" + fault)
            write_json(folder / "exit.json", code)
            _, _, rejected = attempt_result(folder, [0, 1], self.task, None)
            self.assertEqual(rejected, missing)
        (folder / "_result.json").write_text("{")
        write_json(folder / "native-resume.json", {"details": {"0": dict(layout_id=0, success=False, score=0.2)}})
        rows, _, rejected = attempt_result(folder, [0, 1], self.task, None)
        self.assertEqual(set(rows), {0})
        self.assertFalse(rejected)

    def test_failed_shard_allows_other_tasks_and_explicit_retry(self):
        tasks = [self.task, Task("other", 1, 1, self.task.parent)]
        queue = self.queue(tasks)
        task, ids, folder, lease = queue.claim("0")
        with lease:
            self.assertEqual(task, self.task)
            self.shard(folder.name, [0, 2])
            write_json(folder / "exit.json", 78)
        self.assertFalse(queue.claim("0"))
        other, ids, folder, lease = queue.claim("0")
        with lease:
            self.assertEqual(other.name, "other")
            self.shard(folder.name, ids)
        self.assertFalse(queue.claim("0"))
        with self.assertRaisesRegex(RuntimeError, "78"):
            queue.claim("0")
        with patch.dict("os.environ", LOGIC_PREVIEW_RETRY_ID="repair1"):
            retry = self.queue(tasks)
            task, ids, folder, lease = retry.claim("1")
            with lease:
                self.assertEqual(ids, [1])
                self.assertIsNone(self.queue(tasks).claim("2"))

    def test_invalid_scores_layouts_and_missing_camera(self):
        folder = self.shard("fragment", [0, 1])
        with self.assertRaisesRegex(ValueError, "Unexpected or duplicate"):
            read_result(folder, [0])
        next(folder.glob("*right_wrist*")).unlink()
        with self.assertRaisesRegex(ValueError, "Missing cam_right_wrist"):
            read_result(folder, [0, 1])
        with self.assertRaisesRegex(ValueError, "stable layouts"):
            publish(self.task, {}, {}, self.root, "run")
        self.assertFalse(list(self.task.parent.iterdir()))

    def test_controller_crash_preserves_live_shard_lease(self):
        queue = self.queue()
        code = """
from pathlib import Path
import subprocess, sys, time
from types import SimpleNamespace
from packed_eval import Queue, Task
root = Path(sys.argv[1])
queue = Queue(SimpleNamespace(chunk=3, max_attempts=5, policy_host="host", port=1000),
              [Task("align_blocks", 3, 5, root / "official")], root, {"pipelines": {}})
task, assigned, directory, lease = queue.claim("0")
child = subprocess.Popen(["sleep", "30"], pass_fds=(lease.fileno(),), stdout=subprocess.DEVNULL)
print(child.pid, directory.name, flush=True)
time.sleep(30)
"""
        env = dict(os.environ, PYTHONPATH=str(Path(__file__).resolve().parents[1]))
        parent = subprocess.Popen([sys.executable, "-c", code, str(self.root)],
                                  stdout=subprocess.PIPE, text=True, env=env)
        try:
            pid, name = parent.stdout.readline().split()
            self.shard(name, [0, 2])
            parent.kill()
            parent.wait(timeout=5)
            try:
                self.assertIsNone(queue.claim("1"))
            finally:
                os.kill(int(pid), signal.SIGTERM)
            for _ in range(100):
                recovered = queue.claim("1")
                if recovered is not None:
                    break
                time.sleep(0.02)
            self.assertIsNotNone(recovered)
            task, ids, _, lease = recovered
            with lease:
                self.assertEqual(ids, [1])
        finally:
            parent.kill()
            parent.wait(timeout=5)
            parent.stdout.close()

    def test_pipeline_provenance_must_be_reported_by_client(self):
        folder = self.shard("fragment", [0])
        with self.assertRaisesRegex(ValueError, "pipeline"):
            attempt_result(folder, [0], self.task, {"sim_device": "gpu"})

    def test_changed_identity_rejected(self):
        queue = self.queue()
        with self.assertRaisesRegex(ValueError, "configuration changed"):
            Queue(queue.args, [self.task], self.root, {"pipelines": {}, "seed": 2})

    def test_watchdog_tracks_steps_and_recognizes_driver_fault(self):
        log = self.root / "client.log"
        state = {"log": str(log), "last_progress_at": 0}
        log.write_text("env0 step: 15 / 500\n")
        self.assertIsNone(update_job_monitor(state, 10, None))
        self.assertEqual(state["last_progress_at"], 10)
        self.assertIsNone(update_job_monitor(state, 20, None))
        self.assertEqual(state["last_progress_at"], 10)
        with log.open("a") as handle:
            handle.write("CUDA error 700\n")
        self.assertEqual(update_job_monitor(state, 30, None), "CUDA error 700")
