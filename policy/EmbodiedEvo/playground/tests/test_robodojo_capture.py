"""Capture lifetime regressions without a GPU or Isaac Sim imports."""

from types import SimpleNamespace
import unittest
from unittest.mock import Mock

from simulators.robodojo_capture import install_capture_cleanup


class CaptureManager:
    """Native append-on-reset behavior with tracked render resources."""

    def __init__(self):
        self.camera_manager = SimpleNamespace(
            cameras=[["head", "left", "right"]], camera_names=[["h", "l", "r"]])
        self.cameras = self.camera_manager.cameras
        self.camera_names = self.camera_manager.camera_names
        self.num_cams = 3
        self.tiled_cameras = []
        self.tiled_render_products = []
        self.annotator = []
        self.annotator_type = []
        self.annotator_device = []
        self.camera_prim_paths = []
        self._output_buffers = {}
        self.alive = set()
        self.generation = 0

    def reset(self):
        self.generation += 1
        self.camera_prim_paths[:] = self.cameras
        for index in range(self.num_cams):
            path = f"render_{self.generation}_{index}"
            attached = {path}
            annotator = Mock()
            annotator.detach.side_effect = lambda paths, a=attached: a.difference_update(paths)
            product = Mock(path=path)

            def destroy(p=path, a=attached):
                assert not a, "Destroyed product while an annotator was still attached"
                self.alive.remove(p)

            product.destroy.side_effect = destroy
            self.alive.add(path)
            self.tiled_render_products.append(product)
            self.tiled_cameras.append(SimpleNamespace(
                _render_product=product, _render_product_path=path,
                _annotators={"rgba": annotator}, _tiled_render_product=None))
            self.annotator.append([])
            self.annotator_type.append(["rgb"])
            self.annotator_device.append([])
            self._output_buffers[index] = object()

    def destroy(self):
        # Native close also clears the aliased camera lists.
        self.cameras.clear()
        self.camera_names.clear()
        for product in self.tiled_render_products:
            product.destroy()


class CaptureCleanupTests(unittest.TestCase):
    def test_repeated_reset_has_only_current_capture_resources(self):
        manager = CaptureManager()
        install_capture_cleanup(manager)
        for _ in range(12):
            manager.reset()
            self.assertEqual(len(manager.alive), 3)
            self.assertEqual(len(manager.tiled_cameras), 3)
            self.assertEqual(manager.annotator_type, [["rgb"]] * 3)
            self.assertEqual(len(manager._output_buffers), 3)
            self.assertEqual(manager.camera_manager.cameras, [["head", "left", "right"]])
        manager.destroy()
        self.assertFalse(manager.alive)
        self.assertFalse(manager._output_buffers)
        manager.destroy()  # No double detach/destroy after close.

    def test_existing_leaked_resources_and_rebuilt_camera_handles(self):
        manager = CaptureManager()
        manager.reset()
        manager.reset()
        self.assertEqual(len(manager.alive), 6)
        install_capture_cleanup(manager)
        manager.camera_manager.cameras = [["new_head", "new_left", "new_right"]]
        manager.camera_manager.camera_names = [["h2", "l2", "r2"]]
        manager.reset()
        self.assertEqual(len(manager.alive), 3)
        self.assertIs(manager.cameras, manager.camera_manager.cameras)
        self.assertIs(manager.camera_names, manager.camera_manager.camera_names)
        self.assertEqual(manager.camera_prim_paths, manager.cameras)

    def test_cleanup_failure_aborts_reset_and_preserves_retry_state(self):
        manager = CaptureManager()
        install_capture_cleanup(manager)
        manager.reset()
        annotator = manager.tiled_cameras[1]._annotators["rgba"]
        detach = annotator.detach.side_effect
        annotator.detach.side_effect = RuntimeError("detach failed")
        with self.assertRaisesRegex(RuntimeError, "detach failed"):
            manager.reset()
        self.assertEqual(manager.generation, 1)
        self.assertEqual(len(manager.alive), 3)
        annotator.detach.side_effect = detach
        manager.reset()
        self.assertEqual(len(manager.alive), 3)

    def test_install_is_idempotent(self):
        manager = CaptureManager()
        install_capture_cleanup(manager)
        reset = manager.reset
        install_capture_cleanup(manager)
        self.assertIs(manager.reset, reset)


if __name__ == "__main__":
    unittest.main()
