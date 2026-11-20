"""Exercise the actual offline geometry inputs without a simulator or GPU."""
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
NESTING = ROOT / 'tasks/sort_nesting_dolls_by_size'


class PackagedTaskGeometry(unittest.TestCase):
    def test_toy_preflight_requires_cached_geometry_but_not_unused_raw_usds(self):
        from tasks.play_stacking_toy.bindings import official_policy as binding
        from tasks.play_stacking_toy.resources import declarations
        data = declarations()
        self.assertNotIn('public_toy_models', data)
        with patch.object(binding, 'verify_static_resources', return_value=data) as robot:
            self.assertIs(binding.verify_resources(binding.TASK), data)
        robot.assert_called_once()

    def test_toy_perception_reads_the_exact_checked_cache(self):
        from tasks.play_stacking_toy.resources import asset_root
        from tasks.play_stacking_toy.skills.perception.implementation import U12_ToyPerception
        perception = U12_ToyPerception(0., asset_root())
        self.assertEqual(set(perception.manifest), {'base', 'block0', 'block1', 'block2', 'block3'})
        self.assertTrue(perception.geometry)

    def test_random_profile_installs_exact_original_bytes_and_height_banks(self):
        from tasks.sort_nesting_dolls_by_size.skills.public_shape_profile.implementation import install_random_profile
        declared = json.loads((NESTING / 'assets/random_shape_profile.json').read_text())
        self.assertFalse(Path(declared['path']).is_absolute())
        with tempfile.TemporaryDirectory() as directory:
            actual = install_random_profile(directory)
            self.assertEqual(actual['height_banks'], declared['height_banks'])
            for name, digest in declared['files'].items():
                self.assertEqual(hashlib.sha256((Path(directory) / name).read_bytes()).hexdigest(), digest)

    def test_random_metadata_source_locations_are_portable_logical_references(self):
        path = NESTING / 'assets/random_public_shapes/nesting_dolls_random_public_shapes.json'
        references = []
        def visit(value):
            if isinstance(value, dict):
                for name, item in value.items():
                    if name in ('public_asset', 'path'):
                        references.append(item)
                    visit(item)
            elif isinstance(value, list):
                for item in value: visit(item)
        visit(json.loads(path.read_text()))
        self.assertEqual(len(references), 12)
        self.assertTrue(all(value.startswith('resource://official_assets/') for value in references))

    def fixture(self, directory):
        root = Path(directory)
        shutil.copytree(NESTING / 'assets', root / 'assets')
        return root

    def test_corrupt_random_geometry_is_rejected_before_episode_construction(self):
        from tasks.sort_nesting_dolls_by_size.skills.public_shape_profile.implementation import verify_random_profile
        with tempfile.TemporaryDirectory() as directory:
            root = self.fixture(directory)
            (root / 'assets/random_public_shapes/nesting_dolls_random_public_shapes.npz').write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError, 'SHA256 changed'):
                verify_random_profile(root)

    def test_random_geometry_cannot_escape_the_package_or_use_symlink(self):
        from tasks.sort_nesting_dolls_by_size.skills.public_shape_profile.implementation import verify_random_profile
        for relative in ('../outside', '/outside'):
            with self.subTest(relative=relative), tempfile.TemporaryDirectory() as directory:
                root = self.fixture(directory)
                path = root / 'assets/random_shape_profile.json'
                value = json.loads(path.read_text()); value['path'] = relative
                path.write_text(json.dumps(value))
                with self.assertRaisesRegex(ValueError, 'inside their task'):
                    verify_random_profile(root)
        with tempfile.TemporaryDirectory() as directory:
            root = self.fixture(directory)
            name = root / 'assets/random_public_shapes/nesting_dolls_random_public_shapes.json'
            name.rename(name.with_suffix('.original'))
            name.symlink_to(name.with_suffix('.original'))
            with self.assertRaisesRegex(ValueError, 'regular packaged files'):
                verify_random_profile(root)

if __name__ == '__main__':
    unittest.main()
