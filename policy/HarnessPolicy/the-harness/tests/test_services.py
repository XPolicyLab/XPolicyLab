"""Services: manifests, deploy files (placement, URLs, client configs, launch
commands) and the supervisor starting and stopping a real process."""

from pathlib import Path
import socket
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from harness_policy.services import supervisor  # noqa: E402
from harness_policy.services.deploy import client_config, launch, load_deploy, url  # noqa: E402
from harness_policy.services.registry import load_services  # noqa: E402


def free_port():
    with socket.socket() as handle:
        handle.bind(("127.0.0.1", 0))
        return handle.getsockname()[1]


def write_deploy(directory, text):
    path = Path(directory) / "deploy.toml"
    path.write_text(text)
    return path


class ManifestTests(unittest.TestCase):
    def test_every_manifest_loads(self):
        services = load_services()
        self.assertTrue({"sam3", "molmo2", "g05", "openwam"} <= set(services))
        for spec in services.values():
            self.assertGreater(spec.port, 0)
            if spec.kind != "mcp":  # a tool server has no weights
                self.assertTrue(spec.weights, f"{spec.name} says where its weights come from")

    def test_client_configs_match_what_harnesses_used(self):
        with tempfile.TemporaryDirectory() as directory:
            deploy = load_deploy(write_deploy(directory, """
weights = "/w"
gpus = [0, 1]
[services.molmo2]
gpu = 0
[services.sam3]
gpu = 0
[services.g05]
gpu = 1
[services.openwam]
mode = "external"
url = "http://10.0.0.7:8820/"
"""))
        self.assertEqual(client_config(deploy, "molmo2"), {
            "backend": "openai_chat", "base_url": "http://127.0.0.1:8830/v1", "model": "Molmo2-8B",
            "point_format": "molmo", "timeout_seconds": 300})
        self.assertEqual(client_config(deploy, "sam3"), {"url": "http://127.0.0.1:8810"})
        self.assertEqual(url(deploy, "g05"), "ws://127.0.0.1:19005")
        description = client_config(deploy, "g05")["description"]
        self.assertTrue(description.startswith("A visuomotor policy trained on whole RoboDojo tasks"))
        self.assertIn("Unreliable (it rarely helps)", description)
        self.assertEqual(url(deploy, "openwam"), "http://10.0.0.7:8820")
        self.assertIsNone(client_config(deploy, "simate"))  # not listed: off

    def test_launch_of_an_xpolicylab_policy(self):
        with tempfile.TemporaryDirectory() as directory:
            deploy = load_deploy(write_deploy(directory, """
weights = "/w"
robodojo = "/rd"
gpus = [2]
[services.g05]
gpu = 2
"""))
        argv, environment, cwd = launch(deploy, "g05")
        self.assertEqual(argv[2:5], ["/rd/XPolicyLab/setup_policy_server.py", "--config_path",
                                     "/rd/XPolicyLab/policy/G05/deploy.yml"])
        self.assertIn("ckpt_path=/w/g05/checkpoints/checkpoint", argv)
        self.assertIn("port=19005", argv)
        self.assertEqual(environment["CUDA_VISIBLE_DEVICES"], "2")
        self.assertEqual(cwd, "/rd/XPolicyLab/policy/G05/G05")

    def test_an_xpolicylab_policy_with_its_own_server(self):
        # OpenWAM-alpha: XPolicyLab's protocol, served by our loader instead of setup_policy_server.py.
        with tempfile.TemporaryDirectory() as directory:
            deploy = load_deploy(write_deploy(directory, """
weights = "/w"
robodojo = "/rd"
gpus = [3]
[services.openwam-policy]
gpu = 3
"""))
        argv, environment, cwd = launch(deploy, "openwam-policy")
        self.assertTrue(argv[2].endswith("services/servers/serve_openwam_policy.py"))
        self.assertEqual(argv[3:], ["--ckpt-dir", "/w/openwam", "--port", "19010"])
        self.assertEqual(environment["HARNESS_ROBODOJO_ROOT"], "/rd")
        self.assertEqual(environment["CUDA_VISIBLE_DEVICES"], "3")
        self.assertTrue(url(deploy, "openwam-policy").startswith("ws://"))

    def test_automatic_placement_and_conflicts(self):
        with tempfile.TemporaryDirectory() as directory:
            deploy = load_deploy(write_deploy(directory, """
gpus = [0, 1]
gpu_memory_gb = 24
[services.g05]
gpu = "auto"
[services.openwam]
gpu = "auto"
[services.sam3]
gpu = "auto"
"""))
            gpus = {name: deploy.placement(name).gpu for name in ("g05", "openwam", "sam3")}
            self.assertNotEqual(gpus["g05"], gpus["openwam"])  # 16.4 + 15.3 GB do not fit one 24 GB GPU
            for text, message in (('[services.g05]\ngpu = 0\nport = 8810\n[services.sam3]\ngpu = 0\n', "port"),
                                  ('[services.nope]\ngpu = 0\n', "unknown service"),
                                  ('[services.sam3]\nmode = "external"\n', "url"),
                                  ('[services.sam3]\nmode = "resident"\n', "gpu"),
                                  ('gpus = [0]\n[services.g05]\ngpu = "auto"\n[services.openwam]\ngpu = "auto"\n',
                                   "No GPU")):
                with self.subTest(message=message), self.assertRaisesRegex(ValueError, message):
                    load_deploy(write_deploy(directory, text))

    def test_no_deploy_means_no_services(self):
        from unittest import mock

        with mock.patch.dict("os.environ", {}, clear=False) as environ:
            environ.pop("HARNESS_DEPLOY", None)
            deploy = load_deploy(None)
        self.assertEqual(deploy.name, "none")
        self.assertIsNone(url(deploy, "sam3"))


class SupervisorTests(unittest.TestCase):
    def test_start_status_and_stop_a_service(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "services"
            root.mkdir()
            (root / "echo.toml").write_text(f"""
kind = "segmenter"
protocol = "http"
command = ["{{python}}", "-m", "http.server", "{{port}}", "--bind", "127.0.0.1"]
port = {free_port()}
vram_gb = 0
load_seconds = 10
[[weights]]
dir = "none"
local = "nothing to fetch"
""")
            path = write_deploy(directory, f'state = "{Path(directory) / "state"}"\n[services.echo]\nmode = "resident"\n')
            deploy = load_deploy(path, services_root=root)
            rows = supervisor.up(deploy, log=lambda message: None)
            self.assertTrue(rows[0]["up"])
            self.assertTrue(rows[0]["started_by_harness"])
            # A second up leaves the running service alone.
            self.assertEqual(supervisor.up(deploy, log=lambda message: None)[0]["pid"], rows[0]["pid"])
            supervisor.down(deploy, log=lambda message: None)
            self.assertFalse(supervisor.status(deploy)[0]["up"])


if __name__ == "__main__":
    unittest.main()
