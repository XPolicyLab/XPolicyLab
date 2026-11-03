"""Revision packaging must retain skills, protect v4, and bind changed identities."""
import hashlib
import json
from pathlib import Path
import re
import sys

ROOT=Path(__file__).resolve().parents[1]/'asset_revision'
sys.path.insert(0,str(ROOT))
from apply_revision import apply


def test_builder_is_repeatable_preserves_source_and_skill_registry(tmp_path):
    source=tmp_path/'v4';source.mkdir();destination=tmp_path/'candidate'
    for module in ('PourBallsRGB','rdj_pour','CoverHarnessRGB','align_blocks_rgb_straightedge'):
        (source/'implementations'/module).mkdir(parents=True)
    controller='implementations/rdj_insert_policy/version/rdj_insert_policy/policy.py'
    (source/controller).parent.mkdir(parents=True)
    old='a'*64
    text=f"_operation_revision = '{old}'\nvalue = RgbInsertTubesPerception()\n"
    (source/controller).write_text(text)
    memory='memory/program.json';(source/memory).parent.mkdir()
    (source/memory).write_text(json.dumps({'revision':old}))
    descriptor='runtime/descriptors/test.json';(source/descriptor).parent.mkdir(parents=True)
    sha=hashlib.sha256(text.encode()).hexdigest()
    (source/descriptor).write_text(json.dumps({'implementation':{'source':'{assets}/'+controller,'sha256':sha},
        'environment':{},'model_config':{},'pythonpath':['{assets}']}))
    wrist='implementations/insert_key/version/insert_key/wrist_calibration.json'
    (source/wrist).parent.mkdir(parents=True)
    (source/wrist).write_text('{"obsolete": true}')
    mesh='implementations/insert_key/version/insert_key/key_visual_mesh.npz'
    (source/mesh).write_bytes(b'obsolete shape asset')
    template={'schema':'physicalrsi.program-templates/v1','programs':[{'name':'retained-skill',
        'worker_descriptor':descriptor,'dependency_files':[controller,descriptor,memory]}],
        'text_files':{p:hashlib.sha256((source/p).read_bytes()).hexdigest() for p in (controller,descriptor,memory,wrist)}}
    (source/'program-templates.json').write_text(json.dumps(template))
    initial={str(p.relative_to(source)):p.read_bytes() for p in source.rglob('*') if p.is_file()}
    report=apply(source,destination)
    first=(destination/'program-templates.json').read_bytes()
    revised=(destination/controller).read_text()
    revision=re.search(r"_operation_revision = '([0-9a-f]+)'",revised)[1]
    assert revision!=old
    assert json.loads((destination/memory).read_text())['revision']==revision
    assert report['registered_programs']==['retained-skill']
    assert not report['qualified']
    assert not (destination/wrist).exists()
    assert not (destination/mesh).exists()
    assert {wrist,mesh} <= {entry['path'] for entry in report['removed']}
    apply(source,destination)
    assert (destination/'program-templates.json').read_bytes()==first
    for relative,content in initial.items():assert (source/relative).read_bytes()==content
    candidate=json.loads(first)
    for relative,sha in candidate['text_files'].items():
        assert hashlib.sha256((destination/relative).read_bytes()).hexdigest()==sha
