"""Explicit desktop image delivery; a receipt is an attestation, not proof of attention."""
import ast
import hashlib
import json
import re
import shutil
from pathlib import Path

def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def build_delivery(directory, request_id, prompt, bank=None):
    directory=Path(directory)
    images=sorted(directory.glob('m*_p*.jpg'))
    groups=sorted(set(p.name.split('_')[0] for p in images))
    if not groups:
        raise ValueError('No observation images')
    live=[p.name for p in images if p.name.startswith(groups[-1]+'_')]
    raw=[p.name for p in images if p.name not in live]
    match=re.search(r'TOOLBOX PROCEDURE[^\n]*?: (\{[^\n]+\})',prompt)
    flow=ast.literal_eval(match.group(1)) if match else {}
    tool=flow.get('tool');phase=flow.get('phase')
    refs=[];source_hash=None
    if bank:
        bank=Path(bank);spec=json.loads((bank/'manifest.json').read_text())
        source_hash=spec['source_sha256_verified']
        broad = phase not in ('approach','grasp_check','high_check','place_check','release_check','released')
        selected=[{'file':spec['overview'],'label':'All 12 original temporal samples: whole-task overview'}] if broad else []
        panels=spec['groups'].get(tool,{}).get('panels',[])
        if not broad:
            # Keep the dense bank intact; deliver the relevant temporal neighborhood.
            indices = ({'approach':(0,1),'grasp_check':(1,2),'high_check':(2,3),
                        'place_check':(3,4),'release_check':(4,5),'released':(5,6)} if tool=='tape_measure' else
                       {'approach':(0,), 'grasp_check':(1,), 'high_check':(1,),
                        'place_check':(2,), 'release_check':(2,), 'released':(3,)})[phase]
            panels=[panels[min(i,len(panels)-1)] for i in sorted(set(min(i,len(panels)-1) for i in indices))] if panels else []
        for panel in panels:
            selected.append({'file':panel['file'],'label':f"{tool}: {panel['start_s']:.2f}-{panel['end_s']:.2f}s; official-video crops"})
        for item in selected:
            source=bank/item['file']
            if digest(source)!=spec['files_sha256'][item['file']]:
                raise ValueError('Reference bank hash mismatch: '+item['file'])
            name='ref_'+item['file'];shutil.copyfile(source,directory/name)
            refs.append({'file':name,'kind':'official_reference','label':item['label'],'sha256':digest(directory/name)})
    else:
        refs=[{'file':n,'kind':'official_reference','label':'Original ICL frame','sha256':digest(directory/n)} for n in raw]
    # Ken explicitly requested live-first review on 2026-10-03. Keep references
    # available with provenance, without requiring repeated inspection each step.
    required=[{'file':n,'kind':'live_observation','label':'Current camera observation','sha256':digest(directory/n)} for n in live]
    manifest={'schema':2,'request_id':request_id,'tool':tool,'phase':phase,'required_images':required,
              'raw_icl_images':raw,'live_images':live,'source_video_sha256':source_hash,
              'optional_reference_images':refs,
              'reference_mode':'live_first_optional_reference_v4',
              'notice':'Review current live cameras every decision. Official references are optional on ambiguity; do not repeat unchanged examples. A receipt cannot prove model attention.'}
    (directory/'reference_delivery.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    return manifest

def validate_receipt(directory, packet):
    directory=Path(directory);path=directory/'reference_delivery.json'
    if not path.exists():
        return None
    manifest=json.loads(path.read_text());receipt=packet.get('observation_receipt') or {}
    if receipt.get('request_id')!=manifest['request_id'] or packet.get('request_id')!=manifest['request_id']:
        raise ValueError('Missing or stale observation receipt')
    if receipt.get('manifest_sha256')!=digest(path):
        raise ValueError('Observation manifest mismatch')
    expected={i['file']:i['sha256'] for i in manifest['required_images']}
    if receipt.get('images')!=expected:
        raise ValueError('Receipt must include all selected references and current cameras')
    if any(digest(directory/n)!=sha for n,sha in expected.items()):
        raise ValueError('Request image changed after delivery')
    return {'request_id':manifest['request_id'],'manifest_sha256':digest(path),
            'reference_images':sum(i['kind']=='official_reference' for i in manifest['required_images']),
            'live_images':len(manifest['live_images']),'review_attested':True,'actual_model_context_verified':False}
