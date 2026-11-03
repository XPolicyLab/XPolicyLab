from pathlib import Path
import hashlib,json,importlib.util
import pytest
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('font_audit',ROOT/'audit_legacy_assets.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)

def test_only_known_unmodified_font_bytes_skip_text_scanning(tmp_path):
    relative='physicalrsi_observed_geometry/fonts/DejaVuSans.ttf'
    target=tmp_path/relative;target.parent.mkdir(parents=True)
    raw=(ROOT/'asset_revision'/relative).read_bytes();target.write_bytes(raw)
    def manifest(content):
        (tmp_path/'program-templates.json').write_text(json.dumps({'schema':'physicalrsi.program-templates/v1','programs':[],'text_files':{relative:hashlib.sha256(content).hexdigest()}}))
    manifest(raw);report=module.audit(tmp_path)
    assert len(report['generic_font_files'])==1 and report['qualified'] is False
    target.write_bytes(raw+b'changed');manifest(raw+b'changed')
    with pytest.raises(ValueError,match='Unrecognized generic font'):module.audit(tmp_path)
