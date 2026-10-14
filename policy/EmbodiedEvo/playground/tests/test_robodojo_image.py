"""Check the pinned image's mounts, GPU selection and supervisor FD contract."""

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from orchestrator import robodojo_image
from orchestrator.robodojo_apptainer import apptainer_command


class ImageRuntimeTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        (self.root / "vendor/robodojo").mkdir(parents=True)
        self.profile = {
            "image": "sha256:" + "a" * 64,
            "python": "/opt/conda/bin/python",
            "source_root": "/workspace/RoboDojo",
            "cache_dir": str(self.root / "cache"),
        }
        self.args = [
            "-m", "orchestrator.session", "--run-dir", str(self.root / "runs/trial"),
            "--socket", str(self.root / "socket/session.sock"), "--ready-fd", "12",
            "--phase", "playground", "--seed", "0", "--budget", "100",
        ]

    def command(self, gpu="4"):
        with (
            patch.object(robodojo_image, "REPO", self.root),
            patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": gpu}),
        ):
            return robodojo_image.docker_command(
                self.args, self.profile, self.root / "bridge", "embodiedevo-sim-test"
            )

    def test_image_keeps_native_protocol_and_repository_source(self):
        command = self.command()
        self.assertEqual(command[command.index("--gpus") + 1], "device=4")
        self.assertEqual(command[command.index("--network") + 1], "none")
        self.assertIn("CUDA_VISIBLE_DEVICES=0", command)
        self.assertIn(f"{self.root}:{self.root}:ro", command)
        self.assertIn(f"{self.root}/vendor/robodojo:/workspace/RoboDojo:ro", command)
        self.assertIn(self.profile["image"], command)
        shell = command[-1]
        self.assertIn("-m orchestrator.session", shell)
        self.assertIn("--phase playground --seed 0 --budget 100", shell)
        self.assertIn("exec 3>/run/embodiedevo/events", shell)
        self.assertIn("--ready-fd 3", shell)
        self.assertNotIn("--ready-fd 12", shell)
        self.assertNotIn("/var/run/docker.sock", " ".join(command))

    def test_rejects_unpinned_image_and_multiple_gpus(self):
        with self.assertRaisesRegex(ValueError, "one physical GPU"):
            self.command("0,1")
        self.profile["image"] = "robodojo:latest"
        with self.assertRaisesRegex(ValueError, "Pin the simulator image"):
            self.command()

    def test_preserves_slurm_cpu_allocation_in_docker(self):
        with patch.dict(os.environ, {'EMBODIEDEVO_SIMULATOR_CPUSET': '0-3,128-131'}):
            command = self.command()
        self.assertEqual(command[command.index('--cpuset-cpus') + 1], '0-3,128-131')
        self.assertEqual(command[command.index('--gpus') + 1], 'device=4')

    def test_campaign_runtime_applies_measured_thread_limits_before_import(self):
        self.profile.update(kit_usd_threads=8, video_threads=2, renderer_barrier=True)
        command = self.command()
        self.assertIn("PXR_WORK_THREAD_LIMIT=8", command)
        self.assertIn("EMBODIEDEVO_ROBODOJO_VIDEO_THREADS=2", command)
        self.assertIn("EMBODIEDEVO_ROBODOJO_RENDERER_BARRIER=1", command)
        self.assertTrue(any("carb.tasking.plugin/threadCount=8" in p for p in command))

    def test_container_gate_precedes_all_native_imports(self):
        self.args += ['--container-start-gate', str(self.root / 'runs/START_NATIVE_128')]
        shell = self.command()[-1]
        self.assertLess(shell.index('container-ready'), shell.index('while [ ! -e'))
        self.assertLess(shell.index('done;'), shell.index('-m simulators.image_entrypoint'))
        self.assertIn('STOP', shell)
        self.assertIn('exit 143', shell)

    def test_apptainer_keeps_mounts_and_initializes_usd_limit_before_imports(self):
        argv = ['-m', 'scripts.stress_robodojo', 'worker', '--run-dir',
                str(self.root / 'runs/trial'), '--gpu', '4', '--cpu-threads', '8']
        with patch.object(robodojo_image, 'REPO', self.root), patch.dict(
                os.environ, {'CUDA_VISIBLE_DEVICES': '4'}):
            command = apptainer_command(argv, self.profile, self.root/'bridge', self.root/'image.sif')
        self.assertIn('--nvccli', command)
        self.assertIn('--containall', command)
        self.assertIn(f'{self.root}/vendor/robodojo:/workspace/RoboDojo:ro', command)
        self.assertIn('CUDA_VISIBLE_DEVICES=0', command)
        self.assertIn('PXR_WORK_THREAD_LIMIT=8', command)
        self.assertIn('--gpu 0', command[-1])
        self.assertNotIn('docker', command)

    def test_smoke_does_not_create_run_directory_or_require_event_fd(self):
        self.args = [
            "-m", "scripts.smoke_robodojo", "--run-dir", str(self.root / "runs/smoke"),
            "--gpu", "2",
        ]
        command = self.command("")
        self.assertEqual(command[command.index("--gpus") + 1], "device=2")
        self.assertIn("--gpu 0", command[-1])
        self.assertFalse((self.root / "runs/smoke").exists())
        self.assertNotIn("exec 3>", command[-1])


if __name__ == "__main__":
    unittest.main()
