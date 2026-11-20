"""The real SAM leaf resolves canonical configuration before any CUDA import."""
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from skills.shared.perception.sam3_nesting_image_model.implementation import image_configuration

ROOT = Path(__file__).resolve().parents[2]


class SingleGpuSamConfiguration(unittest.TestCase):
    def test_real_canonical_configuration_keeps_the_single_gpu_binding(self):
        imported = set(sys.modules)
        config, manifest = image_configuration()
        self.assertEqual(config['cuda_device'],0)
        self.assertEqual(config['maximum_regions_per_label'],12)
        self.assertFalse(config['history_retained'])
        self.assertEqual(manifest['model'],'SAM3')
        self.assertTrue(manifest['files'])
        self.assertEqual((set(sys.modules)-imported) & {'torch','sam3'},set())
        self.assertEqual(json.loads((ROOT/'configs/native-perception.json').read_text())['cuda_device'],1)

    def test_explicit_model_root_relocation_keeps_model_identity(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ,RIEMANN_RESOURCE_ROOTS=json.dumps({'models':directory})):
            config, manifest=image_configuration()
            self.assertEqual(Path(manifest['path']),Path(directory)/'sam3-robodojo-v21')
            self.assertEqual(config['cuda_device'],0)


if __name__ == '__main__':
    unittest.main()
