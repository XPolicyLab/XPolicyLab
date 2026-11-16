"""Release native tiled-camera resources before rebuilding them on reset.

The pinned capture manager appends a new CameraView per camera on every reset.
Its CameraView destructor checks _tiled_render_product, while creation actually
sets _render_product. Explicitly detach the real annotators and destroy the real
products here, without changing the vendor snapshot or camera configuration.
"""


def release_capture_resources(manager):
    # Detach before destroying products: annotators hold graph/GPU resources.
    # Remove each completed operation so cleanup can safely be retried if one
    # native operation raises. Never continue reset after a cleanup failure.
    for view in manager.tiled_cameras:
        for name, annotator in list(view._annotators.items()):
            annotator.detach([view._render_product_path])
            del view._annotators[name]
    for product in list(manager.tiled_render_products):
        product.destroy()
        manager.tiled_render_products.remove(product)
    manager.tiled_cameras.clear()
    manager.annotator.clear()
    manager.annotator_type.clear()
    manager.annotator_device.clear()
    manager.camera_prim_paths.clear()
    manager._output_buffers.clear()
    # Do not clear cameras/camera_names: they alias the camera manager's lists.


def install_capture_cleanup(manager):
    """Keep native reset/close, adding deterministic release of capture state."""
    if getattr(manager, "_eah_capture_cleanup", False):
        return
    reset, destroy = manager.reset, manager.destroy

    def reset_capture():
        release_capture_resources(manager)
        # CameraManager may have rebuilt its prims before capture.reset().
        manager.cameras = manager.camera_manager.cameras
        manager.camera_names = manager.camera_manager.camera_names
        manager.num_cams = len(manager.cameras[0])
        return reset()

    def destroy_capture():
        release_capture_resources(manager)
        return destroy()

    manager.reset = reset_capture
    manager.destroy = destroy_capture
    manager._eah_capture_cleanup = True


def capture_resource_counts(manager):
    """Host-only diagnostics, including USD products that Python may orphan."""
    import omni.usd

    stage = omni.usd.get_context().get_stage()
    return {
        "camera_views": len(manager.tiled_cameras),
        "render_products": len(manager.tiled_render_products),
        "annotators": sum(len(view._annotators) for view in manager.tiled_cameras),
        "buffer_groups": len(manager._output_buffers),
        "stage_render_products": sum(prim.GetTypeName() == "RenderProduct" for prim in stage.Traverse()),
    }
