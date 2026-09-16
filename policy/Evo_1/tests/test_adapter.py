"""CPU contracts with upstream preprocessing; the network is a deterministic double."""
import asyncio
import copy
import importlib.util
import json
import os
from pathlib import Path
import sys
import shutil
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np
import torch

from XPolicyLab.policy.Evo_1 import model as adapter
from XPolicyLab.policy.Evo_1.prepare_data import prepare, CAMERA_KEYS
from XPolicyLab.utils.process_data import decode_obs_images, encode_image_bit

SOURCE = Path(os.environ.get("EVO1_SOURCE_DIR", adapter.POLICY_DIR / "upstream")).resolve()
sys.path.insert(0, str(SOURCE / "Evo_1"))
from scripts.Evo1_server import Normalizer

spec = importlib.util.spec_from_file_location("evo1_robotwin_reference", SOURCE / "RoboTwin_evaluation/policy/Evo1/deploy_policy.py")
reference = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reference)
torch.set_num_threads(1)
DIMS = {"arm_dim": [6, 6], "ee_dim": [1, 1]}


class TestNetwork(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(1))
        self.actions = torch.linspace(-1, 1, 50 * 24).reshape(50, 24)
        self.seen = None

    def run_inference(self, **kwargs):
        self.seen = kwargs
        return self.actions.clone()


def observation():
    rng = np.random.default_rng(7)
    images = [rng.integers(0, 256, (24, 32, 3), dtype=np.uint8) for _ in range(3)]
    state = np.linspace(-1, 1, 14, dtype=np.float32)
    return {
        "vision": {k: {"color": v} for k, v in zip(adapter.CAMERAS, images)},
        "state": dict(zip(adapter.STATE_KEYS, (state[:6], state[6:7], state[7:13], state[13:14]))),
        "instruction": "Adjust the bottle.", "additional_info": {"frequency": 30},
    }


class Fixture:
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        feature = {"min": [-2.] * 14, "max": [2.] * 14, "mean": [0.] * 14, "std": [1.] * 14}
        other = {"min": [0.] * 14, "max": [8.] * 14, "mean": [4.] * 14, "std": [2.] * 14}
        self.stats = {"aloha_joint": {
            "robotwin_adjust_bottle": {"observation.state": feature, "action": feature},
            "robotwin_click_bell": {"observation.state": other, "action": other},
        }}
        (self.root / "norm_stats.json").write_text(json.dumps(self.stats))
        self.config = {"bench_name": "RoboTwin", "action_type": "joint", "env_cfg_type": "arx_x5",
                       "task_name": "adjust_bottle", "ckpt_name": str(self.root), "seed": 0}
        self.network = TestNetwork()
        self.normalizer = Normalizer(self.stats, normalization_type="bounds")
        with patch.object(adapter, "get_robot_action_dim_info", return_value=DIMS), \
             patch.object(adapter, "load_backend", return_value=(self.network, self.normalizer)):
            self.model = adapter.Model(self.config)


class AdapterTests(Fixture, unittest.TestCase):
    def test_rgb_and_state_match_native_input(self):
        obs = observation()
        for value in obs["vision"].values():
            value["color"].setflags(write=False)
        self.model.update_obs(obs)
        self.model.get_action()
        seen = self.network.seen
        for index, camera in enumerate(adapter.CAMERAS):
            # Execute the upstream client's actual transport encoding, then its inverse.
            bgr = np.asarray(reference.encode_image_array(obs["vision"][camera]["color"]), np.uint8)
            expected = cv2.resize(bgr, (448, 448))[:, :, [2, 1, 0]]
            expected = torch.from_numpy(expected.transpose(2, 0, 1).copy()).float() / 255
            torch.testing.assert_close(seen["images"][index], expected, rtol=0, atol=0)
        state = np.concatenate([obs["state"][key] for key in adapter.STATE_KEYS])
        np.testing.assert_allclose(seen["state_input"].numpy()[0, :14], state / 2, atol=1e-6)
        np.testing.assert_array_equal(seen["state_input"].numpy()[0, 14:], np.zeros(10))
        np.testing.assert_array_equal(seen["action_mask"].numpy(), [[1] * 14 + [0] * 10])
        self.assertEqual(seen["prompt"], obs["instruction"])

    def test_complete_chunk_smoothing_then_37_step_split(self):
        self.model.update_obs(observation())
        actions = self.model.get_action()
        actual = np.array([np.concatenate([row[k] for k in adapter.STATE_KEYS]) for row in actions])
        raw = self.normalizer.denormalize_action(self.network.actions, "aloha_joint", "robotwin_adjust_bottle")
        expected = reference.smooth_actions(np.asarray(raw.numpy().tolist()))[:37, :14]
        np.testing.assert_array_equal(actual, expected)
        self.assertEqual(actual.shape, (37, 14))

    def test_prepare_case_switch_survives_reset(self):
        self.model.prepare_case({"task_name": "click_bell"})
        rng_before = torch.get_rng_state().clone()
        self.model.reset()
        torch.testing.assert_close(torch.get_rng_state(), rng_before)
        self.model.update_obs(observation())
        actual = self.model.get_action()
        self.assertTrue(all(np.all(v >= 0) for row in actual for v in row.values()))
        self.assertEqual(self.model.current_dataset_key, "robotwin_click_bell")

    def test_unknown_task_fails_without_loading_network(self):
        with patch.object(adapter, "get_robot_action_dim_info", return_value=DIMS), \
             patch.object(adapter, "load_backend") as loader:
            with self.assertRaisesRegex(ValueError, "Missing aloha_joint/robotwin_unknown"):
                adapter.Model({**self.config, "task_name": "unknown"})
            loader.assert_not_called()

    def test_reset_rejects_stale_observation(self):
        self.model.update_obs(observation())
        self.model.reset()
        with self.assertRaisesRegex(RuntimeError, "update_obs"):
            self.model.get_action()

    def test_failed_prepare_cannot_reuse_previous_task_statistics(self):
        with self.assertRaises(ValueError):
            self.model.prepare_case({"task_name": "unknown"})
        self.model.reset()
        with self.assertRaises(ValueError):
            self.model.update_obs(observation())

    def test_invalid_state_and_images_fail(self):
        for broken in ("state", "image"):
            obs = observation()
            if broken == "state":
                obs["state"]["right_arm_joint_state"] = [float("nan")] * 6
            else:
                obs["vision"]["cam_head"]["color"] = np.zeros((24, 32, 3), dtype=np.float32)
            with self.assertRaises(ValueError):
                self.model.update_obs(obs)

    def test_invalid_model_output_fails(self):
        self.model.update_obs(observation())
        for value in (torch.zeros(5, 24), torch.full((50, 24), float("nan"))):
            self.network.actions = value
            with self.assertRaisesRegex(ValueError, "undersized or nonfinite"):
                self.model.get_action()

    def test_explicit_norm_suffix_and_instruction_fallback(self):
        self.model.stats["aloha_joint"]["robotwin_click_bell_clean"] = self.stats["aloha_joint"]["robotwin_click_bell"]
        self.normalizer.stats_map = self.model.stats
        self.model.model_cfg["dataset_key_suffix"] = "_clean"
        self.model.prepare_case({"task_name": "click_bell"})
        obs = observation()
        obs["instructions"] = [obs.pop("instruction")]
        self.model.update_obs(obs)
        self.model.get_action()
        self.assertEqual(self.model.current_dataset_key, "robotwin_click_bell_clean")
        self.assertEqual(self.network.seen["prompt"], "Adjust the bottle.")

    def test_unsupported_modes_fail(self):
        for cfg in ({"action_type": "ee"}, {"bench_name": "RoboDojo"}, {"env_cfg_type": "piper"}):
            with self.assertRaises(ValueError):
                adapter.Model({**self.config, **cfg})
        with self.assertRaises(NotImplementedError):
            self.model.update_obs_batch([observation()])

    def test_loader_reads_both_upstream_checkpoint_formats_strictly(self):
        class LoaderNetwork(torch.nn.Module):
            def __init__(self, config):
                super().__init__()
                self.config = config
                self.anchor = torch.nn.Parameter(torch.zeros(1))

            def to(self, device):
                self.requested_device = device
                return self

        config = {"per_action_dim": 24, "state_dim": 24, "image_size": 448, "horizon": 50}
        (self.root / "config.json").write_text(json.dumps(config))
        import scripts.Evo1
        with patch.object(scripts.Evo1, "EVO1", LoaderNetwork), \
             patch.object(torch.cuda, "is_available", return_value=True):
            for name, key in (("checkpoint.pt", "model_state_dict"), ("mp_rank_00_model_states.pt", "module")):
                torch.save({key: {"anchor": torch.tensor([3.])}}, self.root / name)
                model, normalizer = adapter.load_backend(self.root, {"upstream_dir": str(SOURCE)})
                self.assertEqual(model.anchor.item(), 3.)
                self.assertEqual(model.config.num_inference_timesteps, 50)
                self.assertFalse(model.config.finetune_vlm)
                self.assertEqual(model.requested_device, "cuda")
                self.assertIsInstance(normalizer, Normalizer)
            torch.save({"module": {"wrong_name": torch.tensor([3.])}}, self.root / "mp_rank_00_model_states.pt")
            with self.assertRaises(RuntimeError):
                adapter.load_backend(self.root, {"upstream_dir": str(SOURCE)})


class WebSocketTests(Fixture, unittest.IsolatedAsyncioTestCase):
    async def check_roundtrip(self, encoded):
        from client_server.ws.model_server import PolicyServer, PolicyServerConfig
        from client_server.ws.model_client import WsModelClient
        server = PolicyServer(self.model, PolicyServerConfig(host="127.0.0.1", port=0))
        await server.start()
        client = None
        try:
            client = await asyncio.to_thread(WsModelClient, url=server.url,
                                            evaluation_id="evo1-cpu-test", trial_id="0",
                                            action_case_id="evo1-cpu-test-case",
                                            max_connect_attempts=1, request_timeout_s=10)
            obs = observation()
            if encoded:
                for i, key in enumerate(adapter.CAMERAS[:2]):
                    bit = encode_image_bit(obs["vision"][key]["color"])
                    obs["vision"][key]["color"] = bit if i else np.array(list(bit), dtype=np.uint8)
            expected_obs = decode_obs_images(copy.deepcopy(obs))
            await asyncio.to_thread(client.call, "prepare_case", {"task_name": "click_bell"})
            await asyncio.to_thread(client.call, "reset")
            await asyncio.to_thread(client.call, "update_obs", obs)
            actions = await asyncio.to_thread(client.call, "get_action")
            self.assertEqual(len(actions), 37)
            self.assertEqual([len(actions[0][k]) for k in adapter.STATE_KEYS], [6, 1, 6, 1])
            for i, key in enumerate(adapter.CAMERAS):
                expected = cv2.resize(expected_obs["vision"][key]["color"], (448, 448))
                tensor = torch.from_numpy(expected.transpose(2, 0, 1).copy()).float() / 255
                torch.testing.assert_close(self.network.seen["images"][i], tensor, atol=0, rtol=0)
            self.assertEqual(self.model.current_dataset_key, "robotwin_click_bell")
        finally:
            if client is not None:
                await asyncio.to_thread(client.close)
            await server.stop()

    async def test_raw_websocket(self):
        await self.check_roundtrip(False)

    async def test_encoded_websocket(self):
        await self.check_roundtrip(True)


class DataTests(unittest.TestCase):
    def test_installer_imports_from_documented_policy_directory(self):
        # Skip dependency installation, but execute the installer's real import
        # check in a fresh interpreter from the README's working directory.
        with tempfile.TemporaryDirectory() as temporary:
            executable = Path(temporary) / "python"
            executable.write_text(
                "#!/usr/bin/env python3\n"
                "import os, sys\n"
                "if sys.argv[1:3] == ['-m', 'pip']:\n"
                "    sys.exit(0)\n"
                "os.execv(os.environ['EVO1_TEST_PYTHON'], "
                "[os.environ['EVO1_TEST_PYTHON'], *sys.argv[1:]])\n"
            )
            executable.chmod(0o755)
            env = {**os.environ, "PATH": temporary + os.pathsep + os.environ["PATH"],
                   "EVO1_SOURCE_DIR": str(SOURCE), "EVO1_TEST_PYTHON": sys.executable}
            result = subprocess.run(["bash", "install.sh"], cwd=adapter.POLICY_DIR,
                                    env=env, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertIn("[Evo_1] Imports passed", result.stdout)

    def test_statistics_write_only_to_copied_metadata(self):
        import pandas as pd
        from dataset.compute_normstats_streaming import compute_normstats
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            (source / "meta").mkdir(parents=True)
            (source / "data/chunk-000").mkdir(parents=True)
            (source / "videos").mkdir()
            features = {k: {"shape": [14], "dtype": "float32"} for k in ("action", "observation.state")}
            features.update({k: {"shape": [3, 24, 32], "dtype": "video"} for k in CAMERA_KEYS})
            (source / "meta/info.json").write_text(json.dumps({"codebase_version": "v2.1", "features": features}))
            (source / "meta/tasks.jsonl").write_text('{"task_index": 0, "task": "Adjust bottle"}\n')
            (source / "meta/stats.json").write_text('{"original": true}\n')
            before = {p.name: p.read_bytes() for p in (source / "meta").iterdir()}
            pd.DataFrame({"action": [np.ones(14) * i for i in range(60)],
                          "observation.state": [np.ones(14) * (i / 2) for i in range(60)]}).to_parquet(
                source / "data/chunk-000/episode_000000.parquet")
            path = prepare(root / "prepared", [f"adjust_bottle={source}"])
            import yaml
            config = yaml.safe_load(path.read_text())["data_groups"]["aloha_joint"]["robotwin_adjust_bottle"]
            target = Path(config["path"])
            compute_normstats(target, use_delta_actions=False, action_horizon=50, dataset_config=config)
            self.assertEqual(before, {p.name: p.read_bytes() for p in (source / "meta").iterdir()})
            self.assertEqual((target / "data").resolve(), source / "data")
            stats = json.loads((target / "meta/stats.json").read_text())
            self.assertEqual(len(stats["action"]["min"]), 14)
            with self.assertRaises(FileExistsError):
                prepare(root / "prepared", [f"adjust_bottle={source}"])

    def test_training_launcher_preserves_paths_and_existing_results(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            xpl = root / "XPolicyLab"
            policy = xpl / "policy/Evo_1"
            policy.mkdir(parents=True)
            for name in ("train.sh", "train_entry.py", "accelerate.yml"):
                shutil.copyfile(adapter.POLICY_DIR / name, policy / name)
            (xpl / "utils/robot").mkdir(parents=True)
            for name in ("get_action_dim.sh", "robot/_robot_info.json"):
                shutil.copyfile(adapter.POLICY_DIR.parents[1] / "utils" / name, xpl / "utils" / name)
            source = root / "source with spaces"
            (source / "Evo_1").mkdir(parents=True)
            config = root / "config with spaces.yaml"
            config.write_text("{}\n")
            fake_bin = root / "bin"
            fake_bin.mkdir()
            executable = fake_bin / "python"
            executable.write_text("#!/usr/bin/env python3\nimport os,sys,json\nfrom pathlib import Path\nPath(os.environ['EVO1_TEST_CAPTURE']).write_text(json.dumps(sys.argv[1:]))\n")
            executable.chmod(0o755)
            capture = root / "args.json"
            env = {**os.environ, "PATH": str(fake_bin) + os.pathsep + os.environ["PATH"],
                   "EVO1_SOURCE_DIR": str(source), "EVO1_DATA_CONFIG": str(config),
                   "EVO1_TEST_CAPTURE": str(capture)}
            command = ["bash", str(policy / "train.sh"), "RoboTwin", "test_run", "arx_x5", "joint", "7", "0,2", "--lr", "0.00006"]
            subprocess.run(command, env=env, check=True, capture_output=True, text=True)
            args = json.loads(capture.read_text())
            self.assertEqual(args[:2], ["-m", "accelerate.commands.launch"])
            self.assertEqual(args[args.index("--num_processes") + 1], "2")
            self.assertIn("--multi_gpu", args)
            self.assertEqual(args[args.index("--evo1-seed") + 1], "7")
            self.assertEqual(args[args.index("--dataset_config_path") + 1], str(config))
            run_dir = Path(args[args.index("--save_dir") + 1])
            run_dir.mkdir(parents=True)
            sentinel = run_dir / "existing-result.txt"
            sentinel.write_text("preserve me")
            result = subprocess.run(command, env=env, capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("Run directory exists", result.stderr)
            self.assertEqual(sentinel.read_text(), "preserve me")

    def test_training_entry_does_not_shadow_upstream_model_package(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            policy = root / "adapter"
            policy.mkdir()
            shutil.copyfile(adapter.POLICY_DIR / "train_entry.py", policy / "train_entry.py")
            (policy / "model.py").write_text("raise RuntimeError('adapter shadowed upstream model')\n")
            upstream = root / "source" / "Evo_1"
            (upstream / "model").mkdir(parents=True)
            (upstream / "model" / "sentinel.py").write_text("VALUE = 42\n")
            (upstream / "scripts").mkdir()
            (upstream / "scripts" / "train.py").write_text("from model.sentinel import VALUE\nassert VALUE == 42\nprint('UPSTREAM_NAMESPACE_OK')\n")
            result = subprocess.run([sys.executable, str(policy / "train_entry.py"),
                                     "--evo1-source", str(upstream.parent), "--evo1-seed", "0"],
                                    capture_output=True, text=True, check=True)
            self.assertIn("UPSTREAM_NAMESPACE_OK", result.stdout)


if __name__ == "__main__":
    unittest.main()
