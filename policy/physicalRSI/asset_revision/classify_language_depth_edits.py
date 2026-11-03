"""Remove fixed table-based depth scale from the language classification fork."""
import ast
from pickup_geometry_edits import replace_pickup_depth_scale


def edit_language_depth_perception(text):
    tree=ast.parse(text)
    function=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='calibrated_world_height')
    lines=text.splitlines(keepends=True)
    body=''.join(lines[function.body[0].lineno-1:function.body[-1].lineno-1]).rstrip('\n')
    del lines[function.lineno-1:function.end_lineno]
    text=''.join(lines)
    old='    world_z = calibrated_world_height(predicted_depth)'
    if text.count(old)!=1:raise ValueError('Unexpected classification depth affordance call')
    # Reuse the reviewed pickup depth transformation on the identical fork.
    text=replace_pickup_depth_scale(text.replace(old,body))
    text+='''\n\ndef calibrated_world_height(predicted_depth: np.ndarray, *, observation, segment_fn=None, support_callback=None, prior_plane=None) -> np.ndarray:
    from physicalrsi_observed_geometry.calibrated_depth_scene import observed_world_height
    return observed_world_height(predicted_depth, observation, segment_fn=segment_fn, support_callback=support_callback, prior_plane=prior_plane)
'''
    compile(text,'language classification observed depth','exec')
    return text


def bind_language_depth_observation(text):
    old="{k: {'color': v['color']} for k, v in obs['vision'].items()}"
    if text.count(old)!=1:raise ValueError('Unexpected language classification observation capture')
    text=text.replace(old,"{k: {field: v[field] for field in ('color', 'intrinsic_matrix', 'extrinsic_matrix')} for k, v in obs['vision'].items()}")
    old='calibrated_world_height(depth)'
    if text.count(old)!=1:raise ValueError('Unexpected language classification height cache')
    text=text.replace(old,'calibrated_world_height(depth, observation=self.obs)')
    compile(text,'language classification observation binding','exec')
    return text


def remove_language_surface_height_clamp(text):
    old='        z = float(np.clip(np.median(values_z), 0.78, 0.95))'
    if text.count(old)!=1:raise ValueError('Unexpected language object-height clamp')
    text=text.replace(old,'        from physicalrsi_observed_geometry.calibrated_depth_scene import observed_surface_height\n        z = observed_surface_height(values_z)')
    tree=ast.parse(text);lines=text.splitlines(keepends=True);remove=[]
    for node in ast.walk(tree):
        if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='old_xy' for t in node.targets):remove.append(node)
    if len(remove)!=1:raise ValueError('Unexpected fixed-height diagnostic projection')
    for node in remove:del lines[node.lineno-1:node.end_lineno]
    text=''.join(lines)
    old=", previous_plane_xy=old_xy.tolist(), xy_shift=float(np.linalg.norm(np.array([x, y]) - old_xy))"
    if text.count(old)!=1:raise ValueError('Unexpected fixed-height diagnostic fields')
    text=text.replace(old,'')
    if 'old_xy' in text:raise ValueError('Fixed-height diagnostic reference remains')
    compile(text,'unclamped language surface height','exec');return text


def bind_language_support_segmentation(text):
    old='calibrated_world_height(depth, observation=self.obs)'
    if text.count(old)!=1:raise ValueError('Unexpected language support calibration call')
    text=text.replace(old,'calibrated_world_height(depth, observation=self.obs, segment_fn=self.inventory.sam.segment_many)')
    compile(text,'language semantic support binding','exec');return text


def replace_language_basket_projection_height(text):
    old='        bx, by = pixel_to_world_on_height(bc, world_z=0.8, camera=Camera.from_observation(self.obs))'
    if text.count(old)!=1:raise ValueError('Unexpected fixed basket projection height')
    new=('        from physicalrsi_observed_geometry.calibrated_depth_scene import observed_pixel_height\n'
         '        basket_surface_z = observed_pixel_height(self.inventory_height, bc)\n'
         '        bx, by = pixel_to_world_on_height(bc, world_z=basket_surface_z, camera=Camera.from_observation(self.obs))')
    text=text.replace(old,new)
    compile(text,'observed basket projection height','exec');return text


def replace_language_contact_floor(text):
    old='calibrated_world_height(depth, observation=self.obs, segment_fn=self.inventory.sam.segment_many)'
    if text.count(old)!=1:raise ValueError('Unexpected semantic height calibration call')
    text=text.replace(old,old[:-1]+', support_callback=lambda camera, plane, visible: cache_language_support(self, camera, plane, visible))')
    marker='import cv2\n'
    if text.count(marker)!=1:raise ValueError('Expected model import anchor')
    text=text.replace(marker,marker+'from physicalrsi_observed_geometry.language_support import cache_language_support, observed_language_support, observed_contact_height\n')
    old='        contact = max(0.765 + 0.15757000232458113 + 0.003, min(0.98, z + 0.13257000232458113 - 0.008))\n        contact = max(0.9255700023245811, contact + retry_adjustment)'
    if text.count(old)!=1:raise ValueError('Unexpected language fixed contact floor')
    text=text.replace(old,'        support = observed_language_support(self)\n        support_z = support.height(x, y)\n        contact = observed_contact_height(z, support_z, retry_adjustment)')
    old='tip_clearance=contact - 0.15757000232458113 - 0.765'
    if text.count(old)!=1:raise ValueError('Unexpected tip clearance diagnostic')
    text=text.replace(old,'tip_clearance=contact - 0.15757000232458113 - support_z')
    compile(text,'observed language contact floor','exec');return text


def bind_language_prior_support(text):
    old='support_callback=lambda camera, plane, visible: cache_language_support(self, camera, plane, visible))'
    if text.count(old)!=1:raise ValueError('Unexpected support callback binding')
    text=text.replace(old,old[:-1]+', prior_plane=prior_language_support_plane(self))')
    old='from physicalrsi_observed_geometry.language_support import cache_language_support, observed_language_support, observed_contact_height'
    if text.count(old)!=1:raise ValueError('Unexpected support imports')
    text=text.replace(old,old+', prior_language_support_plane')
    tree=ast.parse(text);init=next(n for n in ast.walk(tree) if isinstance(n,ast.ClassDef) and n.name=='Episode')
    init=next(n for n in init.body if isinstance(n,ast.FunctionDef) and n.name=='__init__')
    lines=text.splitlines(keepends=True)
    lines.insert(init.end_lineno,'        self._observed_relay_support = None\n')
    text=''.join(lines)
    compile(text,'per-episode observed support seed','exec');return text
