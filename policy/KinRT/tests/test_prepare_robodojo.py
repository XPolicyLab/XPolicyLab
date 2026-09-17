import ast
from copy import deepcopy
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import yaml


POLICY_DIR = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("prepare_robodojo", POLICY_DIR / "prepare_robodojo.py")
helper = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(helper)

GROUND = b'''from pathlib import Path

def resolve_mdl_paths(value):
    path = Path(value).expanduser()
    if path.is_file() and path.suffix.lower() == ".mdl":
        return [str(path.resolve())]
    if path.is_dir():
        return [str(p.resolve()) for p in path.glob("**/*.mdl")]
    return []
'''
TABLE = b'''from pathlib import Path

def resolve_path(value):
    p = Path(value).expanduser()
    if p.exists():
        return str(p.resolve())
    return None
'''
SIMULATION = b'''# Preserve this original file exactly when reverting.
dt: 0.004
render_interval: 10
scene:
  num_envs: 10
  env_spacing: 7
physics:
  solver_type: 1
custom_setting: keep-me
'''


def write_fixture(root, sources):
    for relative, content in sources.items():
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
    (root / "env_cfg/sim").mkdir(parents=True)
    (root / helper.MAPPING_PATH).write_text("config:\n  sim: alternate_sim\n", encoding="utf-8")
    (root / "env_cfg/sim/alternate_sim.yml").write_bytes(SIMULATION)
    (root / "env_cfg/sim/sim_config.yml").write_bytes(b"unrelated: original\n")


def snapshot(root):
    return {str(path.relative_to(root)): path.read_bytes() for path in root.rglob("*") if path.is_file()}


def verify_material_symlink(test, root):
    material = root / "Assets/Material/wood"
    material.mkdir(parents=True)
    blobs = root / "cache/blobs"
    blobs.mkdir(parents=True)
    mdl_blob = blobs / ("a" * 64)
    mdl_blob.write_text("mdl 1.4; export material Mahogany_Planks() = material();")
    texture_blob = blobs / ("b" * 64)
    texture_blob.write_bytes(b"texture fixture")
    mdl = material / "Mahogany_Planks.mdl"
    texture = material / "Mahogany_Planks_BaseColor.png"
    try:
        mdl.symlink_to(mdl_blob)
        texture.symlink_to(texture_blob)
    except OSError as error:
        test.skipTest(f"Symbolic links are unavailable: {error}")
    for relative, name in zip(helper.PATCHES, ("resolve_mdl_paths", "resolve_path")):
        tree = ast.parse((root / relative).read_text(encoding="utf-8"))
        function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name)
        namespace = {"Path": Path, "List": list}
        exec(compile(ast.Module(body=[function], type_ignores=[]), relative, "exec"), namespace)
        outputs = namespace[name](str(mdl))
        if isinstance(outputs, str):
            outputs = [outputs]
        test.assertEqual(outputs, [str(mdl.absolute())])
        test.assertEqual(Path(outputs[0]).stem, "Mahogany_Planks")
        test.assertEqual((Path(outputs[0]).parent / texture.name).read_bytes(), b"texture fixture")
        if name == "resolve_mdl_paths":
            test.assertEqual(namespace[name](str(material)), [str(mdl.absolute())])
    test.assertFalse((mdl.resolve().parent / texture.name).exists())


class PreparationTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name) / "explicit-simulator-checkout"
        self.sources = dict(zip(helper.PATCHES, (GROUND, TABLE)))
        patches = {
            relative: (helper.sha256(self.sources[relative]), replacements)
            for relative, (_digest, replacements) in helper.PATCHES.items()
        }
        self.patcher = mock.patch.object(helper, "PATCHES", patches)
        self.patcher.start()
        self.addCleanup(self.patcher.stop)
        write_fixture(self.root, self.sources)
        self.sim = self.root / "env_cfg/sim/alternate_sim.yml"

    def test_check_is_read_only_and_reports_planned_changes(self):
        before = snapshot(self.root)
        report = helper.prepare_robodojo(self.root, num_envs=1)
        self.assertEqual(snapshot(self.root), before)
        self.assertTrue(all(item["would_change"] and not item["written"] for item in report["material_sources"]))
        self.assertEqual(report["simulation"]["num_envs_before"], 10)
        self.assertEqual(report["simulation"]["num_envs_target"], 1)

    def test_apply_is_idempotent_and_revert_restores_exact_bytes(self):
        before = snapshot(self.root)
        report = helper.prepare_robodojo(self.root, "apply", 1)
        expected = deepcopy(yaml.safe_load(SIMULATION))
        expected["scene"]["num_envs"] = 1
        self.assertEqual(yaml.safe_load(self.sim.read_bytes()), expected)
        self.assertEqual(report["simulation"]["path"], "env_cfg/sim/alternate_sim.yml")
        self.assertEqual((self.root / "env_cfg/sim/sim_config.yml").read_bytes(), b"unrelated: original\n")
        applied = snapshot(self.root)
        repeated = helper.prepare_robodojo(self.root, "apply", 1)
        self.assertEqual(snapshot(self.root), applied)
        self.assertFalse(repeated["simulation"]["written"])
        helper.prepare_robodojo(self.root, "revert")
        self.assertEqual(snapshot(self.root), before)

    def test_material_only_application_does_not_modify_config(self):
        helper.prepare_robodojo(self.root, "apply")
        self.assertEqual(self.sim.read_bytes(), SIMULATION)
        self.assertFalse((self.root / helper.STATE_NAME).exists())
        helper.prepare_robodojo(self.root, "revert")
        self.assertEqual(self.sim.read_bytes(), SIMULATION)

    def test_scene_alias_does_not_change_an_unrelated_mapping(self):
        original = b"scene: &shared\n  num_envs: 10\n  env_spacing: 7\nother_scene: *shared\n"
        self.sim.write_bytes(original)
        helper.prepare_robodojo(self.root, "apply", 1)
        config = yaml.safe_load(self.sim.read_bytes())
        self.assertEqual(config["scene"]["num_envs"], 1)
        self.assertEqual(config["other_scene"]["num_envs"], 10)
        helper.prepare_robodojo(self.root, "revert")
        self.assertEqual(self.sim.read_bytes(), original)

    def test_unknown_second_source_prevents_all_writes(self):
        path = self.root / list(helper.PATCHES)[1]
        path.write_bytes(path.read_bytes() + b"\n# local source edit\n")
        before = snapshot(self.root)
        with self.assertRaisesRegex(ValueError, "Unknown material source"):
            helper.prepare_robodojo(self.root, "apply", 1)
        self.assertEqual(snapshot(self.root), before)

    def test_unknown_config_edit_prevents_revert_and_material_writes(self):
        helper.prepare_robodojo(self.root, "apply", 1)
        self.sim.write_bytes(self.sim.read_bytes() + b"\nnew_setting: user-owned\n")
        before = snapshot(self.root)
        for mode in ("check", "apply", "revert"):
            with self.subTest(mode=mode), self.assertRaisesRegex(ValueError, "simulation YAML changed"):
                helper.prepare_robodojo(self.root, mode)
            self.assertEqual(snapshot(self.root), before)

    def test_unknown_source_edit_is_preserved_when_reverting(self):
        helper.prepare_robodojo(self.root, "apply", 1)
        path = self.root / list(helper.PATCHES)[1]
        path.write_bytes(path.read_bytes() + b"\n# subsequent user edit\n")
        before = snapshot(self.root)
        with self.assertRaisesRegex(ValueError, "Unknown material source"):
            helper.prepare_robodojo(self.root, "revert")
        self.assertEqual(snapshot(self.root), before)

    def test_changed_mapping_is_rejected_after_preparation(self):
        helper.prepare_robodojo(self.root, "apply", 1)
        (self.root / helper.MAPPING_PATH).write_text("config:\n  sim: alternate_sim\nextra: changed\n")
        before = snapshot(self.root)
        with self.assertRaisesRegex(ValueError, "simulator mapping changed"):
            helper.prepare_robodojo(self.root, "revert")
        self.assertEqual(snapshot(self.root), before)

    def test_mapping_cannot_escape_config_directory(self):
        (self.root / helper.MAPPING_PATH).write_text("config:\n  sim: ../../outside\n")
        before = snapshot(self.root)
        with self.assertRaisesRegex(ValueError, "simple config.sim name"):
            helper.prepare_robodojo(self.root, "apply", 1)
        self.assertEqual(snapshot(self.root), before)

    def test_source_symlinks_are_rejected(self):
        source = self.root / next(iter(helper.PATCHES))
        outside = self.root.parent / "outside.py"
        outside.write_bytes(source.read_bytes())
        source.unlink()
        try:
            source.symlink_to(outside)
        except OSError as error:
            self.skipTest(f"Symbolic links are unavailable: {error}")
        with self.assertRaisesRegex(ValueError, "escapes|symbolic links"):
            helper.prepare_robodojo(self.root, "apply", 1)
        self.assertEqual(outside.read_bytes(), GROUND)

    def test_simulation_symlinks_are_rejected(self):
        outside = self.root.parent / "outside.yml"
        outside.write_bytes(SIMULATION)
        self.sim.unlink()
        try:
            self.sim.symlink_to(outside)
        except OSError as error:
            self.skipTest(f"Symbolic links are unavailable: {error}")
        before = (self.root / next(iter(helper.PATCHES))).read_bytes()
        with self.assertRaisesRegex(ValueError, "escapes|symbolic links"):
            helper.prepare_robodojo(self.root, "apply", 1)
        self.assertEqual((self.root / next(iter(helper.PATCHES))).read_bytes(), before)
        self.assertEqual(outside.read_bytes(), SIMULATION)

    def test_invalid_scene_count_prevents_source_writes(self):
        self.sim.write_text("scene:\n  num_envs: true\n")
        before = snapshot(self.root)
        with self.assertRaisesRegex(ValueError, "positive integer"):
            helper.prepare_robodojo(self.root, "apply", 1)
        self.assertEqual(snapshot(self.root), before)

    def test_modified_backup_is_rejected(self):
        helper.prepare_robodojo(self.root, "apply", 1)
        path = self.root / helper.STATE_NAME
        state = json.loads(path.read_bytes())
        state["applied_sha256"] = "0" * 64
        path.write_text(json.dumps(state))
        before = snapshot(self.root)
        with self.assertRaisesRegex(ValueError, "backup hashes"):
            helper.prepare_robodojo(self.root, "revert")
        self.assertEqual(snapshot(self.root), before)

    def test_interrupted_configuration_write_can_be_retried(self):
        helper.prepare_robodojo(self.root, "apply", 1)
        self.sim.write_bytes(SIMULATION)
        helper.prepare_robodojo(self.root, "apply", 1)
        self.assertEqual(yaml.safe_load(self.sim.read_bytes())["scene"]["num_envs"], 1)
        helper.prepare_robodojo(self.root, "revert")
        self.assertEqual(self.sim.read_bytes(), SIMULATION)

    def test_material_paths_preserve_snapshot_names_and_relative_textures(self):
        helper.prepare_robodojo(self.root, "apply")
        verify_material_symlink(self, self.root)


@unittest.skipUnless(os.environ.get("KINRT_ROBODOJO_FIXTURES"), "Set KINRT_ROBODOJO_FIXTURES to check the real tested sources")
class TestedSourceTests(unittest.TestCase):
    def setUp(self):
        fixture_root = Path(os.environ["KINRT_ROBODOJO_FIXTURES"])
        sources = {}
        for relative in helper.PATCHES:
            path = fixture_root / relative
            if not path.exists():
                path = fixture_root / Path(relative).name
            sources[relative] = path.read_bytes()
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name) / "simulator"
        write_fixture(self.root, sources)

    def test_real_source_hashes_and_exact_patch_roundtrip(self):
        before = snapshot(self.root)
        report = helper.prepare_robodojo(self.root)
        self.assertTrue(all(item["prior_state"] == "original" for item in report["material_sources"]))
        helper.prepare_robodojo(self.root, "apply", 1)
        helper.prepare_robodojo(self.root, "revert")
        self.assertEqual(snapshot(self.root), before)

    def test_real_material_functions_preserve_snapshot_symlinks(self):
        helper.prepare_robodojo(self.root, "apply", 1)
        verify_material_symlink(self, self.root)


if __name__ == "__main__":
    unittest.main()
