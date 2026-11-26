import argparse
import json
from pathlib import Path
import queue
import tempfile
import types
import unittest

import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from harness_policy.runtime import suite as run_suite  # noqa: E402

UNSTABLE = 'RuntimeError: Worker HTTP 400: {"error": "RoboDojo layout 0 of task t is unstable after reset"}'
MID_EPISODE = ("Unscorable: RoboDojo marked task=t layout=1 unstable after step 58; "
               "the episode is not scorable")


class FakeSuite:
    """run_slot with the worker and the episodes replaced: `unstable` seeds end in RoboDojo's
    unstable-layout error, every other seed finishes."""

    def __init__(self, unstable, pack=10, mid_episode=(), hung=()):
        self.unstable, self.pack, self.ran, self.workers = set(unstable), pack, [], 0
        self.mid_episode = set(mid_episode)
        self.hung = dict.fromkeys(hung, 1)  # times each of these seeds times out before its first step

    def __enter__(self):
        self.saved = {name: getattr(run_suite, name)
                      for name in ("start_worker", "stop_worker", "experiment_for", "run_program_episode",
                                   "pack_size")}
        run_suite.start_worker = self.start_worker
        run_suite.stop_worker = lambda process: None
        run_suite.experiment_for = lambda *a, **k: types.SimpleNamespace(environment={}, runtime={})
        run_suite.run_program_episode = self.episode
        run_suite.pack_size = lambda args, task: self.pack
        return self

    def __exit__(self, *exc):
        for name, value in self.saved.items():
            setattr(run_suite, name, value)

    def start_worker(self, *args):
        self.workers += 1
        return object()

    def episode(self, config, seed, directory, **kwargs):
        self.ran.append(seed)
        error = UNSTABLE if seed in self.unstable else MID_EPISODE if seed in self.mid_episode else None
        steps = 50
        if self.hung.get(seed):
            self.hung[seed] -= 1
            error, steps = "TimeoutError: timed out", 0
        summary = {"status": "error" if error else "finished", "success": False, "reward": 0.0,
                   "model_calls": 0, "cost_usd": 0.0, "wall_seconds": 1.0, "error": error, "steps": steps}
        directory.mkdir(parents=True)
        (directory / "summary.json").write_text(json.dumps(summary))
        return summary


def run(output, seeds, layout_source="eval"):
    args = argparse.Namespace(harness="harness.toml", model=None, seed_list=seeds, rerun_errors=False,
                              layout_source=layout_source, benchmark="robodojo",
                              unstable_spares=5, max_model_calls=None, max_cost_usd=None, max_total_cost_usd=None,
                              script_replies=None)
    tasks = queue.Queue()
    tasks.put("t")
    run_suite.run_slot(args, 0, "0", 8780, tasks, output, "template.toml")


class RunSuiteTests(unittest.TestCase):
    def test_unstable_layout_is_replaced_by_the_next_unused_one(self):
        with tempfile.TemporaryDirectory() as tmp, FakeSuite(unstable={0}) as suite:
            output = Path(tmp)
            run(output, [0, 1])
            self.assertEqual(suite.ran, [0, 1, 2])
            self.assertEqual(run_suite.replacements(output / "t"), {0: 2})
            # The worker survives an unstable layout: no restart.
            self.assertEqual(suite.workers, 1)
            names = sorted(path.name for path in (output / "t").iterdir() if path.is_dir())
            self.assertEqual(len(names), 3)
            self.assertTrue(names[0].startswith("seed-0.unstable-"))
            self.assertEqual(names[1:], ["seed-1", "seed-2"])

    def test_a_hung_worker_is_restarted_and_the_layout_runs_again(self):
        # A reset that times out (the simulator hung) fails before the first step; every later
        # episode on that worker would time out too.
        with tempfile.TemporaryDirectory() as tmp, FakeSuite(unstable=(), hung={1}) as suite:
            output = Path(tmp)
            run(output, [0, 1, 2])
            self.assertEqual(suite.ran, [0, 1, 1, 2])
            self.assertEqual(suite.workers, 2)
            summary = json.loads((output / "t" / "seed-1" / "summary.json").read_text())
            self.assertEqual(summary["status"], "finished")
            self.assertTrue(any(path.name.startswith("seed-1.error-") for path in (output / "t").iterdir()))

    def test_resume_reuses_the_replacement(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            with FakeSuite(unstable={0}):
                run(output, [0, 1])
            with FakeSuite(unstable={0}) as resumed:
                run(output, [0, 1])
            # Nothing left to run: seed 0 is answered by seed 2, and no further spare is used.
            self.assertEqual(resumed.ran, [])
            self.assertEqual(resumed.workers, 0)

    def test_chained_replacements_and_interrupted_replacement(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            with FakeSuite(unstable={0, 2}) as suite:
                run(output, [0, 1])
            self.assertEqual(suite.ran, [0, 1, 2, 3])
            self.assertEqual(run_suite.replacements(output / "t"), {0: 2, 2: 3})
            # An interrupted replacement (no summary) is run again on resume, nothing else.
            (output / "t" / "seed-3" / "summary.json").unlink()
            with FakeSuite(unstable=set()) as resumed:
                run(output, [0, 1])
            self.assertEqual(resumed.ran, [3])

    def test_mid_episode_instability_is_replaced_too(self):
        with tempfile.TemporaryDirectory() as tmp, FakeSuite(unstable=set(), mid_episode={1}) as suite:
            output = Path(tmp)
            run(output, [0, 1])
            self.assertEqual(suite.ran, [0, 1, 2])
            self.assertEqual(run_suite.replacements(output / "t"), {1: 2})
            self.assertEqual(suite.workers, 1)

    def test_no_spare_past_the_pack(self):
        with tempfile.TemporaryDirectory() as tmp, FakeSuite(unstable={0}, pack=2) as suite:
            output = Path(tmp)
            run(output, [0, 1])
            # Layouts 0 and 1 are the whole pack: the unstable one stays an error episode.
            self.assertEqual(suite.ran, [0, 1])
            self.assertEqual(run_suite.replacements(output / "t"), {})
            self.assertTrue((output / "t" / "seed-0" / "summary.json").exists())

    def test_a_layout_used_as_a_replacement_is_not_counted_twice(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            with FakeSuite(unstable={0}):
                run(output, [0, 1])  # layout 2 runs in place of 0
            with FakeSuite(unstable=set()) as later:
                run(output, [2, 3])
            # Layout 2 already counts for seed 0, so seed 2 takes the next free layout.
            self.assertEqual(sorted(later.ran), [3, 4])
            self.assertEqual(run_suite.replacements(output / "t"), {0: 2})
            self.assertEqual(run_suite.replacements(output / "t", "displaced"), {2: 4})

    def test_an_interrupted_replacement_still_runs_after_its_layout_is_requested(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            with FakeSuite(unstable={0}):
                run(output, [0, 1])
            (output / "t" / "seed-2" / "summary.json").unlink()  # seed 0's replacement was interrupted
            with FakeSuite(unstable=set()) as later:
                run(output, [0, 1, 2, 3])
            # Layout 2 is rerun for seed 0; seed 2 runs on layout 4; nothing is lost or counted twice.
            self.assertEqual(sorted(later.ran), [2, 3, 4])
            with FakeSuite(unstable=set()) as resumed:
                run(output, [0, 1, 2, 3])
            self.assertEqual(resumed.ran, [])

    def test_batches_into_one_output_count_every_seed_once(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            with FakeSuite(unstable={0}):
                run(output, [0, 1])  # 2 runs for 0
            with FakeSuite(unstable=set()):
                run(output, [2, 3])  # 2 is taken: 4 runs for 2
            with FakeSuite(unstable=set()) as third:
                run(output, [4])  # 4 is taken too: 5 runs for 4
            self.assertEqual(third.ran, [5])
            counted = sorted(p.name for p in (output / "t").iterdir() if p.name.removeprefix("seed-").isdigit())
            self.assertEqual(counted, ["seed-1", "seed-2", "seed-3", "seed-4", "seed-5"])

    def test_a_displaced_seeds_unstable_replacement_is_not_reused(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            with FakeSuite(unstable={0}):
                run(output, [0, 1])  # 2 runs for 0
            with FakeSuite(unstable={4}):
                run(output, [2, 3])  # 4 runs for 2, is unstable, 5 runs for it
            with FakeSuite(unstable=set()) as third:
                run(output, [4])
            self.assertEqual(third.ran, [6])

    def test_procedural_layouts_have_no_spares(self):
        with tempfile.TemporaryDirectory() as tmp, FakeSuite(unstable={0}) as suite:
            output = Path(tmp)
            run(output, [0, 1], layout_source="procedural")
            self.assertEqual(suite.ran, [0, 1])
            self.assertEqual(run_suite.replacements(output / "t"), {})


if __name__ == "__main__":
    unittest.main()
