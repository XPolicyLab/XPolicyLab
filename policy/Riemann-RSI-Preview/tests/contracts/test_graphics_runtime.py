"""Device identity and driver admission; no GPU, SDK or model imports."""
import json
from pathlib import Path
import tempfile
import unittest

from rsi.graphics_runtime import choose_renderer, configure, select_driver


class GraphicsRuntime(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.driver = Path(self.temp.name) / 'nvidia_icd.json'
        self.driver.write_text(json.dumps({'ICD': {'library_path': 'libGLX_nvidia.so.0'}}))

    def test_one_installed_driver_sets_both_loader_variables(self):
        env = {}
        proof = select_driver(env, [self.driver])
        self.assertEqual(env['VK_DRIVER_FILES'], str(self.driver))
        self.assertEqual(env['VK_ICD_FILENAMES'], str(self.driver))
        self.assertEqual(len(proof['driver_file_sha256']), 64)

    def test_conflicting_or_multiple_drivers_fail_before_models(self):
        for env in ({'VK_DRIVER_FILES': '/a', 'VK_ICD_FILENAMES': '/b'},
                    {'VK_DRIVER_FILES': '/a:/b'}):
            with self.assertRaises(ValueError):
                select_driver(env)

    def test_non_nvidia_or_missing_driver_is_not_silently_substituted(self):
        self.driver.write_text('{"ICD":{"library_path":"libother.so"}}')
        with self.assertRaises(ValueError):
            select_driver({'VK_DRIVER_FILES': str(self.driver)})
        with self.assertRaises(ValueError):
            select_driver({'VK_DRIVER_FILES': str(self.driver) + '.missing'})

    def test_uuid_overrides_assumed_physical_index(self):
        devices = [dict(index=0, uuid='GPU-first', vendor_id=0x10de, pci=[0, 40, 0, 0]),
                   dict(index=1, uuid='GPU-leased', vendor_id=0x10de, pci=[0, 2, 0, 0])]
        env = {'VK_ICD_FILENAMES': str(self.driver)}
        proof = configure(env, 'GPU-leased', probe=lambda _: devices)
        self.assertEqual(proof['renderer_index'], 0)
        self.assertEqual(env['RIEMANN_RENDERER_GPU_INDEX'], '0')

    def test_duplicate_missing_or_wrong_vendor_gpu_is_rejected(self):
        row = dict(index=0, uuid='GPU-leased', vendor_id=0x10de)
        for devices in ([], [row, dict(row, index=8)],
                        [dict(row, uuid='GPU-other')], [dict(row, vendor_id=0)]):
            with self.assertRaises(ValueError):
                choose_renderer(devices, 'GPU-leased')

    def test_unknown_or_duplicate_pci_order_fails_before_kit(self):
        first = dict(index=0, uuid='GPU-first', vendor_id=0x10de, pci=[0, 1, 0, 0])
        second = dict(index=1, uuid='GPU-leased', vendor_id=0x10de)
        for row in (second, dict(second, pci=first['pci'])):
            with self.assertRaisesRegex(ValueError, 'PCI identities'):
                choose_renderer([first, row], 'GPU-leased')
