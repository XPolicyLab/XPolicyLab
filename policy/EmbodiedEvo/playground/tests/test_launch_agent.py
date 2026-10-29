import os
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from orchestrator import launch_agent, session_runner
from orchestrator.config import load_task


class LaunchModelTests(unittest.TestCase):
    def test_explicit_dedicated_provider_never_uses_or_changes_host_credentials(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            host = root / ".codex"
            host.mkdir()
            (host / "config.toml").write_text('model_provider = "host"\n')
            (host / "auth.json").write_text('{"OPENAI_API_KEY": "host-secret"}')
            dedicated = root / ".config/embodiedevo/codex"
            dedicated.mkdir(parents=True)
            (dedicated / "config.toml").write_text('model_provider = "container"\n')
            before = {p.name: p.read_bytes() for p in host.iterdir()}
            with (
                patch.object(Path, "home", return_value=root),
                patch.dict(os.environ, {
                    "CODEX_HOME": str(host), "EMBODIEDEVO_CODEX_HOME": str(dedicated)
                }, clear=True),
            ):
                # Missing dedicated auth must not leak the host's API key.
                staged = launch_agent.stage_home(root / "staged") / ".codex"
                self.assertFalse((staged / "auth.json").exists())
                self.assertIn('model_provider = "container"', (staged / "config.toml").read_text())
                (dedicated / "auth.json").write_text('{"OPENAI_API_KEY": "container-secret"}')
                staged = launch_agent.stage_home(root / "staged-auth") / ".codex"
                self.assertEqual(
                    (staged / "auth.json").read_bytes(), (dedicated / "auth.json").read_bytes()
                )
                self.assertEqual((staged / "auth.json").stat().st_mode & 0o777, 0o600)
            self.assertEqual(before, {p.name: p.read_bytes() for p in host.iterdir()})

    def test_provider_override_and_host_fallback(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            host = root / ".codex"
            host.mkdir()
            (host / "config.toml").write_text('model_provider = "host"\n')
            override = root / "override"
            override.mkdir()
            (override / "config.toml").write_text('model_provider = "override"\n')
            with (
                patch.object(Path, "home", return_value=root),
                patch.dict(os.environ, {"CODEX_HOME": str(host)}, clear=True),
            ):
                staged = launch_agent.stage_home(root / "fallback")
                self.assertIn(
                    'model_provider = "host"', (staged / ".codex/config.toml").read_text()
                )
                with patch.dict(os.environ, {"EMBODIEDEVO_CODEX_HOME": str(override)}):
                    staged = launch_agent.stage_home(root / "selected")
                    self.assertIn(
                        'model_provider = "override"', (staged / ".codex/config.toml").read_text()
                    )
                with (
                    patch.dict(os.environ, {"EMBODIEDEVO_CODEX_HOME": str(root / "missing")}),
                    self.assertRaisesRegex(ValueError, "Missing container Codex config"),
                ):
                    launch_agent.stage_home(root / "rejected")

    def test_session_command_uses_the_task_model_and_phase_mounts(self):
        cfg = load_task("RoboDojo")
        cfg = replace(
            cfg,
            agent=replace(
                cfg.agent, model="gpt-6-astra", reasoning_effort="xhigh", isolation="docker"
            ),
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            # The staged CLI default must not supersede the task setting.
            home = root / "home"
            (home / ".codex").mkdir(parents=True)
            (home / ".codex/config.toml").write_text('model = "gpt-5.6-sol"\n')
            for phase in ("playground", "test"):
                with (
                    self.subTest(phase=phase),
                    patch.object(launch_agent, "require_image"),
                    patch.object(launch_agent, "ensure_network"),
                    patch.object(launch_agent, "stage_home", return_value=home),
                    patch.dict(os.environ, {"EMBODIEDEVO_AGENT_CPUSET": "0-3,128-131"}),
                ):
                    _, argv = session_runner.agent_command(cfg, root, root / "socket", phase)

                inner = argv[argv.index("codex") :]
                self.assertIn('model="gpt-6-astra"', inner)
                self.assertIn('model_reasoning_effort="xhigh"', inner)
                self.assertNotIn('model="gpt-5.6-sol"', inner)
                self.assertIn("EMBODIEDEVO_SOCKET=/run/embodiedevo/session.sock", argv)
                self.assertIn(f"type=bind,src={root / 'socket'},dst=/run/embodiedevo,readonly", argv)
                for component in ("instruction.md", "primitives", "scripts", "observations"):
                    self.assertIn(
                        f"type=bind,src={root / 'workspace' / component},"
                        f"dst=/workspace/{component},readonly",
                        argv,
                    )
                for component in ("skills", "lessons"):
                    self.assertEqual(
                        f"type=bind,src={root / 'workspace' / component},"
                        f"dst=/workspace/{component},readonly" in argv,
                        phase == "test",
                    )
                self.assertNotIn("--gpus", argv)
                self.assertEqual(argv[argv.index("--cpuset-cpus") + 1], "0-3,128-131")

    def test_host_isolation_is_rejected_by_the_session_runner(self):
        cfg = load_task("RoboDojo")
        cfg = replace(cfg, agent=replace(cfg.agent, isolation="host"))
        with self.assertRaisesRegex(ValueError, "require Docker isolation"):
            session_runner.validate(cfg)


class StagedHomeTests(unittest.TestCase):
    def test_stages_credentials_and_provider_without_host_configuration(self):
        import os
        import tomllib

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            source.mkdir()
            (source / "auth.json").write_text('{"test": "fixture"}')
            (source / "config.toml").write_text(
                'model = "model-for-test"\nmodel_provider = "custom"\n'
                'sqlite_home = "/private/host/database"\n'
                'developer_instructions = "host-only instructions"\n'
                '[model_providers.custom]\nname="Custom"\n'
                'base_url="https://example.invalid/v1"\n'
                "supports_websockets=true\n"
                '[plugins."local-plugin"]\nenabled=true\n'
                '[mcp_servers.local]\ncommand="host-only-command"\n'
                '[projects."/host/path"]\ntrust_level="trusted"\n'
            )
            with (
                patch.object(Path, "home", return_value=root),
                patch.dict(os.environ, {"CODEX_HOME": str(source)}, clear=True),
            ):
                staged = launch_agent.stage_home(root / "staged")
            cfg = tomllib.loads((staged / ".codex/config.toml").read_text())
            self.assertEqual(
                cfg["model_providers"]["custom"]["base_url"], "https://example.invalid/v1"
            )
            self.assertTrue(cfg["model_providers"]["custom"]["supports_websockets"])
            self.assertEqual(set(cfg["projects"]), {"/workspace"})
            for key in ("sqlite_home", "developer_instructions", "plugins", "mcp_servers"):
                self.assertNotIn(key, cfg)
            self.assertEqual((staged / ".codex/auth.json").stat().st_mode & 0o777, 0o600)


class DockerPreflightTests(unittest.TestCase):
    def test_agent_proxy_overrides_host_proxy_and_none_disables_both(self):
        for network, explicit in (("filtered", True), ("filtered", False), ("none", True)):
            with (
                self.subTest(network=network, explicit=explicit),
                patch.dict(
                    os.environ,
                    {"HTTPS_PROXY": "http://host-private:1234", **(
                        {"EMBODIEDEVO_AGENT_HTTPS_PROXY": "http://172.18.0.1:17890"}
                        if explicit else {}
                    )},
                    clear=True,
                ),
            ):
                argv = launch_agent.docker_argv(
                    Path("/workspace"), Path("/agent-home"), ["codex"],
                    image="agent", network=network, cpus=2, memory_mb=1024,
                )
                self.assertEqual(
                    "HTTPS_PROXY=http://172.18.0.1:17890" in argv,
                    explicit and network != "none",
                )
                self.assertEqual(
                    "HTTPS_PROXY=http://host-private:1234" in argv,
                    not explicit and network != "none",
                )
                self.assertNotIn("--gpus", argv)

    def test_http_proxy_is_used_when_https_proxy_is_unset(self):
        with patch.dict(os.environ, {"HTTP_PROXY": "http://host-proxy:8888"}, clear=True):
            argv = launch_agent.docker_argv(
                Path("/workspace"), Path("/agent-home"), ["codex"],
                image="agent", network="filtered", cpus=2, memory_mb=1024,
            )
        self.assertIn("HTTPS_PROXY=http://host-proxy:8888", argv)
        self.assertIn("HTTP_PROXY=http://host-proxy:8888", argv)

    def test_empty_server_version_is_a_daemon_failure_even_with_zero_exit(self):
        import subprocess

        response = subprocess.CompletedProcess([], 0, "\n", "socket: permission denied")
        with (
            patch.object(launch_agent.shutil, "which", return_value="/usr/bin/docker"),
            patch.object(launch_agent, "_docker", return_value=response) as docker,
            self.assertRaisesRegex(RuntimeError, "cannot reach.*permission denied"),
        ):
            launch_agent.require_image("embodiedevo-agent:xpolicy")
        self.assertEqual(docker.call_count, 1)
