"""Portable single-capture evidence; never join pixels from different times/cameras."""
import copy
import hashlib
import json
import math
from pathlib import Path
from PIL import Image


def asset(root, relative):
    path=(root/relative).resolve() if isinstance(relative,str) else None
    return path if path and path.is_relative_to(root.resolve()) and path.is_file() else None


def capture_key(row):
    return (row['time'],row.get('sample_index'),row.get('camera'),row.get('frame_id'),tuple(row.get('image_size') or []))


def combine_capture(root, rows, *, materialize=True):
    """Union observed bboxes. Transparent gaps are unknown, not reconstructed pixels."""
    if not rows or len({capture_key(r) for r in rows})!=1:raise ValueError('bbox union requires one capture')
    out=copy.deepcopy(rows[0]);ids=sorted({i for r in rows for i in r.get('source_observation_ids',[r.get('observation_id')]) if i})
    out['source_observation_ids']=ids
    if len(rows)==1:return out
    boxes=[r.get('bbox_2d') for r in rows]
    if any(not isinstance(b,list) or len(b)!=4 or not all(isinstance(x,(int,float)) and math.isfinite(x) for x in b) for b in boxes):
        raise ValueError('same-capture union requires valid pixel bboxes')
    bounds=[math.floor(min(b[0] for b in boxes)),math.floor(min(b[1] for b in boxes)),math.ceil(max(b[2] for b in boxes)),math.ceil(max(b[3] for b in boxes))]
    w,h=bounds[2]-bounds[0],bounds[3]-bounds[1]
    if w<=0 or h<=0 or w*h>16777216:raise ValueError('invalid merged crop dimensions')
    name=hashlib.sha256(json.dumps([capture_key(out),ids,boxes,[r.get('assets',{}) for r in rows]],sort_keys=True).encode()).hexdigest()[:24]
    out.update(bbox_2d=bounds,assets={},completeness='complete' if any(r.get('completeness')=='complete' for r in rows) else 'partial',merged_same_capture=True)
    if not materialize:
        out['_components']=rows
        return out
    folder=root/'merged-observations'/name;folder.mkdir(parents=True,exist_ok=True)
    for key,filename in [('bbox_crop_path','bbox.png'),('crop_path','segmented.png')]:
        canvas=Image.new('RGBA',(w,h));available=False
        for r,b in zip(rows,boxes):
            p=asset(root,r.get('assets',{}).get(key))
            if not p:continue
            with Image.open(p) as im:
                image=im.convert('RGBA');target=(math.ceil(b[2])-math.floor(b[0]),math.ceil(b[3])-math.floor(b[1]))
                # Crop coordinates define the original observed footprint, even in old resized fixtures.
                if image.size!=target:image=image.resize(target)
                if key=='crop_path':
                    mask=asset(root,r.get('assets',{}).get('mask_path'))
                    if mask:
                        with Image.open(mask) as source_mask:
                            alpha=source_mask.convert('L')
                            if list(alpha.size)==r.get('image_size'):alpha=alpha.crop((math.floor(b[0]),math.floor(b[1]),math.ceil(b[2]),math.ceil(b[3])))
                            if alpha.size!=target:alpha=alpha.resize(target,Image.Resampling.NEAREST)
                            image.putalpha(alpha)
                canvas.alpha_composite(image,(math.floor(b[0])-bounds[0],math.floor(b[1])-bounds[1]));available=True
        if available:
            path=folder/filename;canvas.save(path);out['assets'][key]=str(path.relative_to(root))
    out['pixel_coverage']='observed_crop_union; transparent_gaps_unknown'
    (folder/'metadata.json').write_text(json.dumps(out,indent=2)+'\n', encoding="utf-8")
    return out


def samples(memory,eid):
    root=memory.directory/'entities'/eid;rows=[];archive=root/'trajectory.jsonl'
    if archive.is_file():
        for line in archive.read_text(encoding="utf-8").splitlines():
            r=json.loads(line)
            if not r.get('observed'):continue
            o=r.get('observation') or {}
            rows.append({**r,'camera':o.get('camera'),'image_size':o.get('image_size'),
                'observation_id':o.get('observation_id'),'completeness':o.get('completeness'),'assets':r.get('assets',{})})
    rows+=copy.deepcopy(memory.entities[eid].get('tracking_observations',[]))
    rows+=copy.deepcopy(memory.identity.tentative.get(eid,[]))
    groups={}
    for r in rows:
        if not r.get('bbox_2d'):continue
        groups.setdefault(capture_key(r),[]).append(r)
    result=[]
    for group in groups.values():
        # Prefer already merged evidence over its original audit records.
        merged_ids={i for r in group if r.get('merged_same_capture') for i in r.get('source_observation_ids',[])}
        selected=[];seen=set()
        for r in group:
            identity=tuple(r.get('source_observation_ids') or [r.get('observation_id') or json.dumps(r.get('assets',{}),sort_keys=True)])
            if not r.get('merged_same_capture') and r.get('observation_id') in merged_ids:continue
            if identity not in seen:selected.append(r);seen.add(identity)
        result.append(combine_capture(root,selected,materialize=False))
    return sorted(result,key=lambda r:(r['time'],r.get('sample_index') or -1))


def semantic_object_input(observation,directory,name):
    """Prefer foreground-only appearance. A complete object never falls back to a table-containing bbox."""
    import numpy as np
    directory.mkdir(parents=True,exist_ok=True)
    bbox=Path(observation.get('bbox_crop_path') or '__missing__')
    segmented=Path(observation.get('crop_path') or '__missing__')
    mask_path=Path(observation.get('mask_path') or '__missing__')
    if segmented.is_file() or bbox.is_file() and mask_path.is_file():
        # Mask RGB before downsampling so table colours cannot bleed into edge pixels.
        with Image.open(bbox if bbox.is_file() and mask_path.is_file() else segmented) as source:
            image=source.convert('RGB')
        if mask_path.is_file():
            with Image.open(mask_path) as source:mask=source.convert('L')
            bounds=observation.get('bbox_2d')
            if list(mask.size)==observation.get('image_size') and bounds:
                mask=mask.crop(tuple(int(v) for v in bounds))
            if mask.size!=image.size:raise ValueError('segmentation and RGB crop lattices differ')
            foreground=np.asarray(mask)>0
            pixels=np.asarray(image).copy();pixels[~foreground]=127
            image=Image.fromarray(pixels)
        target=directory/(name+'-segmented.png');image.save(target)
        return target,'segmented_object'
    if observation.get('completeness')=='complete':
        raise ValueError('complete object requires foreground segmentation; bbox fallback is disabled')
    if not bbox.is_file():raise ValueError('object preview unavailable')
    return bbox,'bbox_context_unknown_or_partial'
