from __future__ import annotations

import hashlib
import importlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import yaml


POLICY_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[3]
SOURCE_ROOT = POLICY_DIR / "source_starvla"
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(SOURCE_ROOT))

try:
    from XPolicyLab.policy.StarVLA.runtime_config import (
        resolve_checkpoint_framework,
        resolve_include_state,
        validate_server_runtime_contract,
    )
    from XPolicyLab.policy.StarVLA.scripts.prepare_hf_checkpoint import (
        load_manifest,
        resolve_variant,
        validate_checkpoint_dir,
    )
except ImportError:
    resolve_checkpoint_framework = None
    resolve_include_state = None
    validate_server_runtime_contract = None
    load_manifest = None
    resolve_variant = None
    validate_checkpoint_dir = None


class RuntimeRegistryTest(unittest.TestCase):
    def test_public_and_legacy_mixtures_share_h50_q99_schema(self):
        from starVLA.dataloader.gr00t_lerobot.registry import (
            DATASET_NAMED_MIXTURES,
            ROBOT_TYPE_CONFIG_MAP,
        )

        public_type = DATASET_NAMED_MIXTURES["robodojo_arx_x5_h50_q99"][0][2]
        legacy_type = DATASET_NAMED_MIXTURES["robodojo_v21_all_h50_q99"][0][2]

        self.assertEqual(public_type, "robodojo_arx_x5_h50_q99")
        self.assertEqual(legacy_type, public_type)
        config = ROBOT_TYPE_CONFIG_MAP[public_type]
        self.assertEqual(config.action_indices, list(range(50)))
        self.assertEqual(sum(config.action_key_dims.values()), 14)
        self.assertEqual(sum(config.state_key_dims.values()), 14)


class IncludeStateResolutionTest(unittest.TestCase):
    def test_model_import_resolves_runtime_config_with_repo_pythonpath(self):
        module = importlib.import_module("XPolicyLab.policy.StarVLA.model")
        self.assertTrue(hasattr(module, "Model"))

    def test_observation_decoder_preserves_rgb_channel_order(self):
        module = importlib.import_module("XPolicyLab.policy.StarVLA.model")
        decoded = module._decode_image([[[255, 0, 0]]])
        self.assertEqual(decoded[0, 0].tolist(), [255, 0, 0])

    def setUp(self):
        self.assertTrue(
            callable(resolve_include_state),
            "runtime_config.resolve_include_state is not implemented",
        )
        self.assertTrue(callable(resolve_checkpoint_framework))
        self.temp_dir = tempfile.TemporaryDirectory()
        self.run_dir = Path(self.temp_dir.name)
        self.checkpoint = self.run_dir / "checkpoints" / "model.pt"
        self.checkpoint.parent.mkdir()
        self.checkpoint.touch()

    def tearDown(self):
        self.temp_dir.cleanup()

    def _write_yaml(self, name: str, include_state: bool | None):
        config = {"datasets": {"vla_data": {}}}
        if include_state is not None:
            config["datasets"]["vla_data"]["include_state"] = include_state
        (self.run_dir / name).write_text(yaml.safe_dump(config), encoding="utf-8")

    def test_explicit_value_overrides_checkpoint_config(self):
        self._write_yaml("config.yaml", True)
        self.assertFalse(resolve_include_state("false", self.checkpoint))
        self.assertTrue(resolve_include_state("true", self.checkpoint))

    def test_auto_prefers_config_yaml_then_full_config(self):
        self._write_yaml("config.yaml", False)
        self._write_yaml("config.full.yaml", True)
        self.assertFalse(resolve_include_state("auto", self.checkpoint))

        (self.run_dir / "config.yaml").unlink()
        self.assertTrue(resolve_include_state("auto", self.checkpoint))

    def test_auto_defaults_to_false_without_checkpoint_setting(self):
        self._write_yaml("config.yaml", None)
        self.assertFalse(resolve_include_state("auto", self.checkpoint))

    def test_checkpoint_framework_comes_from_adjacent_config(self):
        config = {"framework": {"name": "QwenPI_v3"}}
        (self.run_dir / "config.yaml").write_text(
            yaml.safe_dump(config), encoding="utf-8"
        )
        self.assertEqual(resolve_checkpoint_framework(self.checkpoint), "QwenPI_v3")


class ReleasedCheckpointManifestTest(unittest.TestCase):
    def setUp(self):
        for function in (load_manifest, resolve_variant, validate_checkpoint_dir):
            self.assertTrue(callable(function))

    def test_manifest_pins_all_released_robodojo_weights(self):
        manifest = load_manifest()
        self.assertEqual(manifest["runtime"]["image_color_order"], "rgb")
        self.assertTrue(manifest["runtime"]["include_state"])
        self.assertEqual(manifest["runtime"]["normalization_key"], "arx_x5")
        self.assertEqual(
            manifest["variants"]["oft"]["checkpoint_sha256"],
            "62b8333c9af58467a911150e908cc726dd3c2660fd461871ae9300296391a3cd",
        )
        self.assertEqual(
            manifest["variants"]["groot"]["checkpoint_sha256"],
            "953df8c0a13b36780327bcd3682bb74e35f2121a6dc63dfb20075d50fa494f74",
        )
        self.assertEqual(
            manifest["variants"]["pi_v3"]["checkpoint_sha256"],
            "7144d490c40edceca31feeedb06434d795ed300b85eef0f2164efa4ed2c12d5f",
        )
        pi_requirements = manifest["variants"]["pi_v3"]["runtime_requirements"]
        self.assertEqual(pi_requirements["pi_v3_forward"], "canonical_interleaved")
        self.assertEqual(resolve_variant("GR00T"), "groot")
        self.assertEqual(resolve_variant("PIv3"), "pi_v3")

    def test_validator_checks_layout_config_statistics_and_weight(self):
        manifest = load_manifest()
        runtime = manifest["runtime"]
        with tempfile.TemporaryDirectory() as temp_dir:
            run_dir = Path(temp_dir)
            checkpoint = run_dir / "checkpoints" / "tiny.pt"
            checkpoint.parent.mkdir()
            checkpoint.write_bytes(b"released-weight")

            compact = {
                "datasets": {"vla_data": {"data_mix": "robodojo_v21_all_h50_q99"}},
                "framework": {
                    "name": "QwenOFT",
                    "action_model": {"action_dim": 14, "action_horizon": 50},
                    "qwenvl": {"base_vlm": runtime["base_vlm"]},
                },
            }
            full = {"datasets": {"vla_data": {"include_state": True}}}
            stats = {
                "arx_x5": {
                    "state": {"q01": [0.0] * 14, "q99": [1.0] * 14},
                    "action": {"q01": [0.0] * 14, "q99": [1.0] * 14},
                }
            }
            config_text = yaml.safe_dump(compact)
            full_text = yaml.safe_dump(full)
            (run_dir / "config.yaml").write_text(config_text, encoding="utf-8")
            (run_dir / "config.full.yaml").write_text(full_text, encoding="utf-8")
            stats_text = json.dumps(stats, sort_keys=True)
            (run_dir / "dataset_statistics.json").write_text(stats_text, encoding="utf-8")
            spec = {
                "framework": "QwenOFT",
                "checkpoint_path": "checkpoints/tiny.pt",
                "checkpoint_size": checkpoint.stat().st_size,
                "checkpoint_sha256": hashlib.sha256(b"released-weight").hexdigest(),
                "config_sha256": hashlib.sha256(config_text.encode()).hexdigest(),
                "config_full_sha256": hashlib.sha256(full_text.encode()).hexdigest(),
                "dataset_statistics_sha256": hashlib.sha256(stats_text.encode()).hexdigest(),
            }

            self.assertEqual(validate_checkpoint_dir(run_dir, spec, runtime), checkpoint)


class ServerRuntimeContractTest(unittest.TestCase):
    def setUp(self):
        self.assertTrue(callable(validate_server_runtime_contract))
        self.metadata = {
            "available_unnorm_keys": ["arx_x5"],
            "runtime_contract": {
                "version": 1,
                "image_color_order": "rgb",
                "state_input": "raw_env",
                "state_normalization": "training_transform",
                "action_output": "unnormalized_env",
                "action_dim": 14,
            },
        }

    def test_accepts_vendored_robodojo_runtime(self):
        contract = validate_server_runtime_contract(
            self.metadata,
            include_state=True,
            action_dim=14,
            unnorm_key="arx_x5",
        )
        self.assertEqual(contract["image_color_order"], "rgb")

    def test_rejects_upstream_server_without_state_normalization_contract(self):
        with self.assertRaisesRegex(ValueError, "vendored server"):
            validate_server_runtime_contract(
                {"action_chunk_size": 50},
                include_state=True,
                action_dim=14,
                unnorm_key="arx_x5",
            )

    def test_pi_v3_forward_requirement_is_checkpoint_specific(self):
        self.metadata["framework"] = "QwenPI_v3"
        self.metadata["runtime_contract"]["pi_v3_forward"] = "legacy_all_cross"

        contract = validate_server_runtime_contract(
            self.metadata,
            include_state=True,
            action_dim=14,
            unnorm_key="arx_x5",
            expected_framework="QwenPI_v3",
        )
        self.assertEqual(contract["pi_v3_forward"], "legacy_all_cross")

        with self.assertRaisesRegex(ValueError, "PI-v3 forward contract"):
            validate_server_runtime_contract(
                self.metadata,
                include_state=True,
                action_dim=14,
                unnorm_key="arx_x5",
                expected_framework="QwenPI_v3",
                expected_pi_v3_forward="canonical_interleaved",
            )

        self.metadata["runtime_contract"]["pi_v3_forward"] = "canonical_interleaved"
        contract = validate_server_runtime_contract(
            self.metadata,
            include_state=True,
            action_dim=14,
            unnorm_key="arx_x5",
            expected_framework="QwenPI_v3",
            expected_pi_v3_forward="canonical_interleaved",
        )
        self.assertEqual(contract["pi_v3_forward"], "canonical_interleaved")


class PiV3CanonicalForwardTest(unittest.TestCase):
    def test_layerwise_head_uses_dit_forward(self):
        import inspect

        from starVLA.model.modules.action_model.LayerwiseFM_ActionHeader import (
            LayerwiseFlowmatchingActionHead,
        )

        for method in (
            LayerwiseFlowmatchingActionHead.forward,
            LayerwiseFlowmatchingActionHead.predict_action,
        ):
            source = inspect.getsource(method)
            self.assertIn("return_pre_output=True", source)
            self.assertNotIn("self.model.transformer_blocks", source)

    def test_dit_interleaves_cross_and_self_attention_for_layerwise_inputs(self):
        import torch
        from torch import nn

        from starVLA.model.modules.action_model.flow_matching_head.cross_attention_dit import (
            DiT,
        )

        class RecordingBlock(nn.Module):
            def __init__(self):
                super().__init__()
                self.encoder_hidden_states = []

            def forward(self, hidden_states, *, encoder_hidden_states, **kwargs):
                self.encoder_hidden_states.append(encoder_hidden_states)
                return hidden_states

        model = DiT(
            num_attention_heads=1,
            attention_head_dim=4,
            output_dim=4,
            num_layers=2,
            dropout=0.0,
            interleave_self_attention=True,
            cross_attention_dim=4,
        )
        cross_block = RecordingBlock()
        self_block = RecordingBlock()
        model.transformer_blocks = nn.ModuleList([cross_block, self_block])

        hidden_states = torch.zeros(1, 3, 4)
        layerwise_encoder = [torch.ones(1, 2, 4), torch.full((1, 2, 4), 2.0)]
        output = model(
            hidden_states=hidden_states,
            encoder_hidden_states=layerwise_encoder,
            timestep=torch.zeros(1, dtype=torch.long),
            return_pre_output=True,
        )

        self.assertTrue(torch.equal(output, hidden_states))
        self.assertTrue(
            torch.equal(cross_block.encoder_hidden_states[0], layerwise_encoder[0])
        )
        self.assertIsNone(self_block.encoder_hidden_states[0])


class RoboDojoArgumentParsingTest(unittest.TestCase):
    def test_released_eval_clears_stale_distributed_training_environment(self):
        entrypoint = (POLICY_DIR / "scripts" / "eval_hf_robodojo.sh").read_text(
            encoding="utf-8"
        )
        launcher = (POLICY_DIR / "setup_eval_policy_server.sh").read_text(
            encoding="utf-8"
        )
        for variable_name in ("RANK", "WORLD_SIZE", "LOCAL_RANK", "PMI_SIZE"):
            self.assertIn(variable_name, entrypoint)
            self.assertIn(variable_name, launcher)
        self.assertIn('unset "${variable_name}"', entrypoint)

    def test_policy_server_limits_default_import_threads(self):
        launcher = (POLICY_DIR / "setup_eval_policy_server.sh").read_text(
            encoding="utf-8"
        )
        self.assertIn('starvla_cpu_threads="${STARVLA_CPU_THREADS:-8}"', launcher)
        self.assertIn(
            'export OMP_NUM_THREADS="${OMP_NUM_THREADS:-${starvla_cpu_threads}}"',
            launcher,
        )
        self.assertIn(
            'export MKL_NUM_THREADS="${MKL_NUM_THREADS:-${starvla_cpu_threads}}"',
            launcher,
        )

    def test_released_eval_preflights_isaac_host_libraries(self):
        entrypoint = (POLICY_DIR / "scripts" / "eval_hf_robodojo.sh").read_text(
            encoding="utf-8"
        )
        helper = (POLICY_DIR / "scripts" / "run_hf_robodojo_env_client.sh").read_text(
            encoding="utf-8"
        )
        self.assertIn("check_isaac_runtime.py", entrypoint)
        self.assertIn("check_isaac_runtime.py", helper)

    def test_released_eval_preflights_official_robodojo_assets(self):
        entrypoint = (POLICY_DIR / "scripts" / "eval_hf_robodojo.sh").read_text(
            encoding="utf-8"
        )
        self.assertIn("Assets/Robots/x5/robot_config.yml", entrypoint)
        self.assertIn("Assets/Eval_Layout/RoboDojo", entrypoint)
        self.assertIn("bash scripts/init_assets.sh", entrypoint)

    def test_released_eval_defaults_to_one_sequential_isaac_environment(self):
        helper = (POLICY_DIR / "scripts" / "run_hf_robodojo_env_client.sh").read_text(
            encoding="utf-8"
        )
        self.assertIn('runtime_num_envs="${STARVLA_ROBODOJO_NUM_ENVS:-1}"', helper)
        self.assertIn('--num_envs "${runtime_num_envs}"', helper)

    def test_starvla_adapter_keeps_release_batch_mode_available(self):
        deploy = yaml.safe_load((POLICY_DIR / "deploy.yml").read_text(encoding="utf-8"))
        self.assertTrue(deploy["eval_batch"])

    def test_device_is_not_abbreviated_to_device_id_during_preparse(self):
        runner = POLICY_DIR / "scripts" / "run_robodojo_main.py"
        with tempfile.TemporaryDirectory() as temp_dir:
            target = Path(temp_dir) / "main.py"
            target.write_text(
                """\
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--device_id", type=int, required=True)
parser.parse_known_args()
parser.add_argument("--device", required=True)
args = parser.parse_args()
print(args.device, args.device_id)
""",
                encoding="utf-8",
            )
            result = subprocess.run(
                [
                    sys.executable,
                    str(runner),
                    str(target),
                    "--device",
                    "cuda:0",
                    "--device_id",
                    "0",
                ],
                check=False,
                capture_output=True,
                text=True,
            )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip(), "cuda:0 0")


if __name__ == "__main__":
    unittest.main()
