"""CPU integration checks for public packaging; no live API or simulator."""
from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import configure
import install_policy
import verify_release


class Packaging(unittest.TestCase):
    def test_variant_source_relationship(self):
        report = verify_release.verify_source(ROOT)
        self.assertEqual(report["shared_service"], "verified")
        manifest = json.loads((ROOT / "metadata/variant_manifest.json").read_text())
        self.assertNotIn("agent/api/run_policy.py", manifest["modified_files"])
        # Only the SAM3 mask decoder changes in the retained robot implementation.
        self.assertEqual({rel for rel in manifest["modified_files"] if rel.startswith("vendor/")},
                         {"vendor/robot15/hexashell/vision/sam3_client.py"})
        self.assertFalse((ROOT / "tasks").exists())
        self.assertFalse((ROOT / "agent/api/skills").exists())
        self.assertFalse((ROOT / "agent/recipes").exists())

    def test_configuration_quotes_model_and_never_reads_key(self):
        with tempfile.TemporaryDirectory() as temporary:
            tmp = Path(temporary)
            settings = configure.load_settings(ROOT / "configs/model_api.example.json")
            marker = tmp / "must-not-exist"
            settings["model"] = f"model $(touch {marker}) `touch {marker}` 'quoted'"
            values = configure.environment(settings, repo=tmp / "Robo Dojo", isaac_python=Path(sys.executable),
                                           api_base="https://api.example.com/custom/v1/",
                                           key_file=tmp / "nonexistent key file", sam3_url="http://127.0.0.1:18300")
            path = tmp / "config.env"
            configure.write_config(path, values)
            result = subprocess.run(["bash", "-c", 'set -a; source "$1"; exec "$2" -c "$3"', "check",
                                     str(path), sys.executable,
                                     'import os,json; print(json.dumps(dict(os.environ)))'],
                                    text=True, capture_output=True, check=True)
            exported = json.loads(result.stdout)
            for key, value in values.items():
                self.assertEqual(exported[key], value)
            self.assertFalse(marker.exists())
            self.assertEqual(path.stat().st_mode & 0o777, 0o600)
            with self.assertRaises(FileExistsError):
                configure.write_config(path, values)

    def test_api_base_validation(self):
        for value in ("https://user:secret@example.com/v1", "https://example.com/v1?key=x",
                      "https://example.com/v1#secret", "https://example.com/v1/chat/completions",
                      "file:///tmp/api", "https://api.example.com/space here"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                configure.split_api_base(value)
        self.assertEqual(configure.split_api_base("https://api.example.com"), ("https://api.example.com", "/"))

    def test_install_and_native_seed_layout_forwarding(self):
        with tempfile.TemporaryDirectory() as temporary:
            tmp = Path(temporary)
            repo = tmp / "Robo Dojo"
            xpl = repo / "XPolicyLab"
            utils = xpl / "utils"
            utils.mkdir(parents=True)
            # Native policy startup executes this mock instead of the GPU server.
            (xpl / "setup_policy_server.py").write_text(
                "import os,sys,json,pathlib\n"
                "pathlib.Path(os.environ['POLICY_CAPTURE']).write_text(json.dumps({'argv':sys.argv[1:],"
                "'layout':os.environ.get('ROBODOJO_LAYOUT_IDS'),'gpu':os.environ.get('CUDA_VISIBLE_DEVICES')}))\n")
            (utils / "get_free_port.sh").write_text("echo 18399\n")
            (utils / "wait_for_policy_server.sh").write_text(
                'for i in {1..200}; do test -s "$POLICY_CAPTURE" && exit 0; sleep 0.01; done; exit 1\n')
            (utils / "setup_env_client.sh").write_text(
                'exec "$TEST_PYTHON" -c \'import os,sys,json,pathlib; '
                'pathlib.Path(os.environ["CLIENT_CAPTURE"]).write_text(json.dumps({"argv":sys.argv[1:],'
                '"layout":os.environ.get("ROBODOJO_LAYOUT_IDS")}))\' "$@"\n')
            destination = install_policy.install(repo)
            self.assertTrue((destination / "LICENSE").is_file())
            self.assertTrue((destination / "licenses/RoboDojo-LICENSE").is_file())
            with self.assertRaises(FileExistsError):
                install_policy.install(repo)
            key = tmp / "api key"
            secret = "synthetic-key-only-for-test"
            key.write_text(secret)
            config_command = [sys.executable, str(ROOT / "tools/configure.py"),
                              "--robodojo-repo", str(repo), "--isaac-python", sys.executable,
                              "--api-base", "https://api.example.com/v1", "--key-file", str(key)]
            configured = subprocess.run(config_command, text=True, capture_output=True, check=True)
            self.assertNotIn(secret, configured.stdout + configured.stderr + (destination / "config.env").read_text())
            again = subprocess.run(config_command, capture_output=True)
            self.assertNotEqual(again.returncode, 0)
            for seed in (0, 2, 19321):
                with self.subTest(seed=seed):
                    pc, cc = tmp / f"policy-{seed}.json", tmp / f"client-{seed}.json"
                    env = dict(os.environ, POLICY_CAPTURE=str(pc), CLIENT_CAPTURE=str(cc),
                               TEST_PYTHON=sys.executable, ROBODOJO_LAYOUT_IDS="evaluator-owned-layout",
                               PYTHONDONTWRITEBYTECODE="1")
                    command = [sys.executable, str(ROOT / "tools/evaluate.py"), "--robodojo-repo", str(repo),
                               "--task", "stack_bowls_random", "--seed", str(seed), "--policy-gpu", "1",
                               "--env-gpu", "2", "--eval-env", str(tmp / "Isaac Env")]
                    dry = subprocess.run(command + ["--dry-run"], env=env, text=True,
                                         capture_output=True, check=True, timeout=30)
                    self.assertIn(str(seed), shlex.split(dry.stdout))
                    self.assertFalse(pc.exists())
                    subprocess.run(command, env=env, text=True, capture_output=True, check=True, timeout=30)
                    policy, client = json.loads(pc.read_text()), json.loads(cc.read_text())
                    self.assertIn(f"seed={seed}", policy["argv"])
                    self.assertIn("policy_name=JoyAI_VL", policy["argv"])
                    self.assertIn("task_name=stack_bowls_random", policy["argv"])
                    self.assertEqual(policy["gpu"], "1")
                    self.assertEqual(client["argv"][10], str(seed))
                    self.assertEqual(client["argv"][11], "2")
                    self.assertEqual(client["argv"][5], "stack_bowls_random")
                    for captured in (policy, client):
                        self.assertEqual(captured["layout"], "evaluator-owned-layout")


class ActualRunner(unittest.TestCase):
    def test_agent_launcher_exports_evaluated_settings(self):
        with tempfile.TemporaryDirectory() as temporary:
            tmp = Path(temporary)
            policy = tmp / "policy"
            api = policy / "agent/api"
            api.mkdir(parents=True)
            source = ROOT / "agent/api/run_agent.sh"
            shutil.copyfile(source, api / source.name)
            key = tmp / "key"
            key.write_text("synthetic-key-only-for-test")
            settings = configure.load_settings(ROOT / "configs/model_api.example.json")
            values = configure.environment(settings, repo=tmp, isaac_python=Path(sys.executable),
                                           api_base="https://api.example.com/v1", key_file=key,
                                           sam3_url="http://127.0.0.1:18300")
            configure.write_config(policy / "config.env", values)
            binary = tmp / "bin"
            binary.mkdir()
            # Record Docker arguments and exported settings without contacting a daemon.
            docker = binary / "docker"
            docker.write_text(
                f"#!{sys.executable}\n"
                "import sys,os,json\n"
                "with open(os.environ['DOCKER_CAPTURE'],'a') as f: "
                "f.write(json.dumps({'argv':sys.argv[1:],'env':{k:v for k,v in os.environ.items() "
                "if k.startswith('HEXASHELL_') or k == 'MODEL'}})+'\\n')\n"
                "if sys.argv[1:3] == ['network','inspect'] and '--format' in sys.argv: print('172.20.0.1')\n")
            docker.chmod(0o755)
            capture = tmp / "docker.jsonl"
            env = dict(os.environ, PATH=f"{binary}:{os.environ['PATH']}", DOCKER_CAPTURE=str(capture))
            subprocess.run(["bash", str(api / source.name), str(tmp / "run"), "synthetic prompt"],
                           env=env, capture_output=True, text=True, timeout=30, check=True)
            invocations = [json.loads(line) for line in capture.read_text().splitlines()]
            agent = next(row for row in invocations if '/opt/robo/runner/run_policy.py' in row['argv'])
            self.assertIn("PYTHONPATH=/opt/robo/client:/opt/robo", agent["argv"])
            self.assertIn("joyai-vl/agent:preview", agent["argv"])
            self.assertEqual(agent["env"]["HEXASHELL_STREAM"], "0")
            self.assertEqual(agent["env"]["HEXASHELL_MAX_TOKENS"], "65536")
            self.assertEqual(agent["env"]["HEXASHELL_CHAT_TEMPLATE_KWARGS"], values["HEXASHELL_CHAT_TEMPLATE_KWARGS"])
            self.assertNotIn(f"{key}:/run/model-key:ro", agent["argv"])
            proxy = next(row for row in invocations if "/egress_proxy.py" in row["argv"])
            self.assertIn(f"{key}:/run/model-key:ro", proxy["argv"])

    def test_request_and_reasoning_history(self):
        with tempfile.TemporaryDirectory() as temporary:
            settings = configure.load_settings(ROOT / "configs/model_api.example.json")
            env = configure.environment(settings, repo=Path(temporary), isaac_python=Path(sys.executable),
                                        api_base="https://api.example.com/v1", key_file=Path(temporary) / "key",
                                        sam3_url="http://127.0.0.1:18300")
            env["MODEL_API_URL"] = "http://mock.invalid/v1/chat/completions"
            fake_robo = types.ModuleType("robo")
            fake_robo.OBS_DIR = temporary
            fake_robo.EXIT_OK = 0
            fake_robo.EXIT_EPISODE_OVER = 1
            fake_robo.download_obs = lambda: None
            commands = []

            def robot_post(command):
                commands.append(command)
                return {"exit_code": 0, "over": False, "step": 0, "step_limit": 100}

            fake_robo.post = robot_post
            fake_robo.python_cell = lambda code, structured: {"output": "synthetic result", "images": []}
            spec = importlib.util.spec_from_file_location("joyai_vl_test_runner", ROOT / "agent/api/run_policy.py")
            module = importlib.util.module_from_spec(spec)
            captures = []
            replies = [
                {"content": "", "reasoning_content": "synthetic reasoning to retain", "tool_calls": [
                    {"id": "python-1", "type": "function", "function": {"name": "python", "arguments": '{"code":"print(1)"}'}}]},
                {"content": "", "tool_calls": [
                    {"id": "finish-1", "type": "function", "function": {"name": "finish", "arguments":
                     '{"status":"keep_partial","summary":"synthetic check"}'}}]},
            ]

            def capture_request(request, timeout):
                captures.append(json.loads(request.data))
                self.assertEqual(timeout, 900)
                self.assertEqual(request.full_url, env["MODEL_API_URL"])
                response = io.BytesIO(json.dumps({"choices": [{"message": replies[len(captures) - 1]}]}).encode())
                response.headers = {"Content-Type": "application/json"}
                return response

            with patch.dict(sys.modules, {"robo": fake_robo}), patch.dict(os.environ, env, clear=True):
                spec.loader.exec_module(module)
                with patch.object(module.urllib.request, "urlopen", side_effect=capture_request), contextlib.redirect_stdout(io.StringIO()):
                    module.main("Synthetic task; no environment or model is running.")
            self.assertEqual(len(captures), 2)
            first, second = captures
            self.assertEqual(first["model"], "JoyAI-VL")
            self.assertEqual(first["max_tokens"], 65536)
            self.assertEqual(first["temperature"], 0)
            self.assertFalse(first["stream"])
            self.assertNotIn("stream_options", first)
            self.assertEqual(first["reasoning_effort"], "max")
            self.assertEqual(first["chat_template_kwargs"], settings["chat_template_kwargs"])
            self.assertEqual([t["function"]["name"] for t in first["tools"]], ["python", "finish"])
            system = (ROOT / "agent/api/prompts/robodojo/system.md").read_text().strip() + "\n"
            self.assertEqual(first["messages"][0]["content"], system)
            self.assertIn("100 replies and 90 minutes", first["messages"][1]["content"][0]["text"])
            assistant = next(m for m in second["messages"] if m["role"] == "assistant")
            self.assertEqual(assistant["reasoning_content"], "synthetic reasoning to retain")
            self.assertIn({"cmd": "done"}, commands)


if __name__ == "__main__":
    unittest.main()
