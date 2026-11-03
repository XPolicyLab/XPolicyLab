"""Remove superseded object data and update source/memory integrity records."""
import hashlib
import json
import re
from pathlib import Path


def finalize(source,destination,original,changes,edit):
    removed=[]
    for module,names in {'PourBallsRGB':('shapes.json','shapes.npz'),
                         'arrange_rgb':('templates.npz','assets/digit_meshes.npz','assets/digit_meshes_provenance.json'),
                         'rdj_pour':('shapes.json','shapes.npz'),
                         'pour_by_language':('core/review_saved_rgb.py','core/review_tilt_rgb.py'),
                         'rdj_store_tools':('box_public.npz',),
                         'rdj_equation':('rgb_reader/templates.npz',),
                         'sort_nesting_dolls':('nesting_rgb/profile_geometry.py','nesting_rgb/doll_profiles.json','nesting_rgb/contours.py','nesting_rgb/attachment.py'),
                         'CoverHarnessRGB':('public_shapes.npz',),
                         'insert_key':('wrist_calibration.json','key_visual_mesh.npz','calibration_provenance.json'),
                         'rdj_v1_stack_blocks_by_language':('public_camera_calibration.json',),
                         'align_blocks_rgb_straightedge':('align_policy/head_camera.json',)}.items():
        if not (source/'implementations'/module).is_dir():
            continue
        for version in (source/'implementations'/module).iterdir():
            for name in names:
                relative=str((version/module/name).relative_to(source))
                target=destination/relative
                if (source/relative).is_file():
                    removed.append({'path':relative,'parent_sha256':hashlib.sha256((source/relative).read_bytes()).hexdigest()})
                    if target.is_file():target.unlink()
                original['text_files'].pop(relative,None)
    removed_paths={item['path'] for item in removed}
    for program in original['programs']:
        program['dependency_files']=[path for path in program['dependency_files'] if path not in removed_paths]
    # Operation revisions describe changed behavior, not just unchanged operation names.
    salt=hashlib.sha256(json.dumps(changes,sort_keys=True).encode()).hexdigest()
    revisions={}
    edited_python=[change['path'] for change in changes if change['path'].endswith('.py')]
    for relative in edited_python:
        text=(destination/relative).read_text()
        for old in re.findall(r"_operation_revision\s*=\s*['\"]([0-9a-f]{64})['\"]",text):
            revisions[old]=hashlib.sha256((salt+old).encode()).hexdigest()
        memory=(source/relative).with_suffix('.memory.json')
        if memory.is_file():
            document=json.loads(memory.read_text())
            for cls in document.get('classes',{}).values():
                old=cls['revision'];revisions[old]=hashlib.sha256((salt+old).encode()).hexdigest()
    for relative in original['text_files']:
        if (relative in edited_python or relative.endswith('.memory.json')
                or relative.endswith('/structured_memory.json') or relative.startswith('memory/')):
            text=((destination if relative in edited_python else source)/relative).read_text();after=text
            for old,new in revisions.items():after=after.replace(old,new)
            if after!=(destination/relative).read_text() or after!=text:edit(relative,after)
    changed={item['path']:item for item in changes}
    for item in changed.values():original['text_files'][item['path']]=item['sha256']
    for program in original['programs']:
        descriptor_path=program['worker_descriptor']
        descriptor=json.loads((destination/descriptor_path).read_text())
        if descriptor.get('name')=='general_pickup':
            reference=json.loads((source/'runtime/descriptors/pour_by_language.json').read_text())
            checkpoint=reference['services'][0]['checkpoint']
            descriptor['environment']['PHYSICALRSI_PICKUP_SAM_CHECKPOINT']=checkpoint
            roots=[p for p in reference['pythonpath'] if '/shared/perception_services/' in p]
            if len(roots)!=1:raise ValueError('Ambiguous perception service package')
            if roots[0] not in descriptor['pythonpath']:descriptor['pythonpath'].append(roots[0])
            prefix=roots[0].removeprefix('{assets}/')+'/'
            for path in original['text_files']:
                if path.startswith(prefix) and path not in program['dependency_files']:program['dependency_files'].append(path)
        identity=descriptor['implementation'];relative=identity['source'].removeprefix('{assets}/')
        identity['sha256']=original['text_files'][relative]
        identity['implementation_revision']=hashlib.sha256((identity['sha256']+salt).encode()).hexdigest()
        descriptor['model_config']['ckpt_name']=identity['implementation_revision']
        initialization=descriptor.get('transport',{}).get('initialization',{})
        if 'controller_source' in initialization:
            relative=initialization['controller_source'].removeprefix('{assets}/')
            initialization['controller_sha256']=original['text_files'][relative]
            if relative not in program['dependency_files']:program['dependency_files'].append(relative)
        edit(descriptor_path,json.dumps(descriptor,indent=2)+'\n')
        original['text_files'][descriptor_path]=hashlib.sha256((destination/descriptor_path).read_bytes()).hexdigest()
        # Bind every modified source into each skill identity, including imported controllers.
        # This is conservative and avoids hiding a changed transitive dependency.
        for relative in changed:
            if relative not in program['dependency_files']:program['dependency_files'].append(relative)
    return removed,original
