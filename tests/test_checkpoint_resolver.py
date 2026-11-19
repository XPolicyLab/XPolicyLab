"""Checkpoint selection must never silently substitute an explicit path."""

from pathlib import Path
import tempfile
import unittest

import yaml

from XPolicyLab.utils.checkpoint_resolver import candidate_checkpoint_roots, resolve_checkpoint_root


class CheckpointResolverTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.policy = Path(temporary.name).resolve()
        self.checkpoints = self.policy / "checkpoints"
        self.checkpoints.mkdir()
        self.config = dict(bench_name="RoboDojo", ckpt_name="cotrain", env_cfg_type="arx_x5", action_type="joint", seed=0)
        self.run = self.checkpoints / "RoboDojo-cotrain-arx_x5-joint-0"
        self.run.mkdir()

    def resolve(self, config, **kwargs):
        return resolve_checkpoint_root(config, self.checkpoints, policy_dir=self.policy, **kwargs)

    def test_missing_explicit_path_cannot_load_existing_named_run(self):
        config = dict(self.config, checkpoint_path="missing")
        with self.assertRaises(FileNotFoundError):
            self.resolve(config)
        self.assertEqual(self.resolve(config, must_exist=False), self.policy / "missing")
        self.assertEqual(candidate_checkpoint_roots(config, self.checkpoints), [self.policy / "missing"])

    def test_first_explicit_key_is_exclusive_even_if_second_exists(self):
        config = dict(self.config, model_path="missing", checkpoint_path=str(self.run))
        with self.assertRaises(FileNotFoundError):
            self.resolve(config)
        self.assertEqual(self.resolve(config, must_exist=False), self.policy / "missing")

    def test_relative_and_absolute_explicit_paths(self):
        weights = self.policy / "weights/model.bin"
        weights.parent.mkdir()
        weights.write_bytes(b"test checkpoint")
        for path in ("weights/model.bin", str(weights)):
            with self.subTest(path=path):
                self.assertEqual(self.resolve(dict(self.config, checkpoint_path=path)), weights)

    def test_ckpt_name_path_is_exclusive(self):
        for path in ("./missing", str(self.policy / "missing")):
            with self.subTest(path=path), self.assertRaises(FileNotFoundError):
                self.resolve(dict(self.config, ckpt_name=path))
        self.assertEqual(self.resolve(dict(self.config, ckpt_name=str(self.run))), self.run)
        self.assertEqual(self.resolve(dict(self.config, ckpt_name="checkpoints/" + self.run.name)), self.run)

    def test_named_runs_keep_training_name_and_verbatim_fallback(self):
        fallback = self.checkpoints / "cotrain"
        fallback.mkdir()
        self.assertEqual(self.resolve(self.config), self.run)
        self.run.rmdir()
        self.assertEqual(self.resolve(self.config), fallback)
        self.assertEqual(self.resolve(dict(self.config, model_path="  ", checkpoint_path=None)), fallback)

    def test_me_brain_default_config_does_not_mask_named_checkpoint(self):
        # The adapter formerly shipped /path/to/focus-vlwa-release. Making
        # explicit paths exclusive must not turn that placeholder into a
        # mandatory path ahead of an otherwise valid named checkpoint.
        deploy = Path(__file__).resolve().parents[1] / "policy/ME_Brain_1/deploy.yml"
        config = yaml.safe_load(deploy.read_text())
        config.update(self.config)
        self.assertIsNone(config["model_path"])
        self.assertEqual(
            self.resolve(config, explicit_keys=("model_path", "checkpoint_path")),
            self.run,
        )


if __name__ == "__main__":
    unittest.main()
