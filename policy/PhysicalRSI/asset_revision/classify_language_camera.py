"""Bind classification and relay projections to observation-local calibration."""
import ast
PROJECTIONS={'pixel_to_world_on_height','world_to_pixel_on_height'}


def replace_language_camera_helpers(text):
    tree=ast.parse(text);lines=text.splitlines(keepends=True)
    remove=[n for n in tree.body if (isinstance(n,ast.FunctionDef) and n.name in PROJECTIONS) or (isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='_FOCAL_PIXELS' for t in n.targets))]
    if len(remove)!=3:raise ValueError('Unexpected language camera helper definitions')
    for node in sorted(remove,key=lambda n:n.lineno,reverse=True):del lines[node.lineno-1:node.end_lineno]
    text=''.join(lines)
    needle='from __future__ import annotations\n'
    if text.count(needle)!=1:raise ValueError('Expected language perception future import')
    text=text.replace(needle,needle+'from physicalrsi_observed_geometry.camera_projection import pixel_to_world_on_height, world_to_pixel_on_height\n')
    if '_FOCAL_PIXELS' in text:raise ValueError('Fixed focal reference remains')
    compile(text,'language current-camera helpers','exec');return text


def bind_language_camera_calls(text,kind):
    tree=ast.parse(text);lines=text.splitlines(keepends=True);offsets=[0]
    for line in lines:offsets.append(offsets[-1]+len(line))
    changes=[];counts={'projection':0,'relay':0}
    for fn in ast.walk(tree):
        if not isinstance(fn,ast.FunctionDef):continue
        expr='Camera.from_observation(self.obs)' if kind=='model' else 'camera' if fn.name=='free_relay_sites' else 'Camera.from_observation(ep.obs)'
        for node in ast.walk(fn):
            if not isinstance(node,ast.Call) or not isinstance(node.func,ast.Name):continue
            name=node.func.id
            if name in PROJECTIONS or (kind=='relay' and name=='free_relay_sites'):
                if any(k.arg=='camera' for k in node.keywords):raise ValueError('Projection already camera-bound')
                position=offsets[node.end_lineno-1]+node.end_col_offset-1
                separator='' if text[:position].rstrip().endswith(',') else ','
                changes.append((position,separator+' camera='+expr))
                counts['projection' if name in PROJECTIONS else 'relay']+=1
    expected={'projection':6,'relay':0} if kind=='model' else {'projection':5,'relay':2}
    if counts!=expected:raise ValueError(f'Unexpected classification camera call counts: {counts}')
    for position,value in sorted(changes,reverse=True):text=text[:position]+value+text[position:]
    if kind=='relay':
        old='def free_relay_sites(occupied, baskets, radius):'
        if text.count(old)!=1:raise ValueError('Unexpected relay signature')
        text=text.replace(old,'def free_relay_sites(occupied, baskets, radius, *, camera):')
    # Both modules have plain imports after their module docstring.
    marker='import cv2\n'
    if text.count(marker)!=1:raise ValueError('Expected classification import anchor')
    text=text.replace(marker,marker+'from physicalrsi_observed_geometry import Camera\n')
    compile(text,'language camera binding','exec');return text
