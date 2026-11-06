"""Integrity and path safety of the one public-resource installation path."""
import hashlib
import io
import json
from pathlib import Path
import tarfile
import tempfile
import unittest
from unittest.mock import patch

from rsi.resource_setup import contained, extract_tar, put, finger_triangles
from tools.download_assets import validated_files

ROOT = Path(__file__).resolve().parents[2]


def record(raw, **extra):
    return dict(bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest(),**extra)


class ResourceDeliveryTests(unittest.TestCase):
    def test_every_download_has_a_public_fixed_identity(self):
        data=json.loads((ROOT/'configs/models/downloads.json').read_text())
        rows=validated_files(data,{})
        self.assertTrue(all(x.get('url') and x.get('sha256') for x in rows))
        for x in rows:
            if 'huggingface.co/' in x['url']:
                self.assertRegex(x['url'],r'/resolve/[0-9a-f]{40}/')
        paths={x['path'] for x in rows}
        self.assertIn('models/sam3-robodojo-v21/sam3.pt',paths)
        self.assertIn('task_detectors/imitate_sorting_sequence/model.pt',paths)
        self.assertIn('task_detectors/match_and_pick_from_conveyor/model.pt',paths)
        self.assertIn('downloads/templates.tar.gz',paths)

    def test_every_preparation_input_has_a_download_or_official_source(self):
        plan=json.loads((ROOT/'configs/resource-install.json').read_text())
        downloads={r['path']:r for r in json.loads((ROOT/'configs/models/downloads.json').read_text())['files']}
        for key in ('archive','sam_runtime','protocol_wheel'):
            r=plan[key];self.assertEqual({k:r[k] for k in ('bytes','sha256')},{k:downloads[r['path']][k] for k in ('bytes','sha256')})
        self.assertEqual(len({x['path'] for x in plan['files']}),len(plan['files']))
        for r in plan['files']:
            self.assertIn(r['kind'],('cached','official','finger_triangles','download','wheel'))
            if r['kind']=='download': self.assertIn(r['source'],downloads)
            if r['kind']=='official': self.assertTrue(r['source'].startswith('resource://official_assets/'))

    def test_containment_rejects_parent_absolute_and_symlink(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);(root/'link').symlink_to(root/'elsewhere',target_is_directory=True)
            for path in ('../escape','/escape','link/escape'):
                with self.assertRaises(ValueError):contained(root,path)

    def test_official_resource_mesh_members_preserve_logical_uri(self):
        from rsi.resource_setup import official_records
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / 'tasks/unit').mkdir(parents=True)
            (root / 'configs').mkdir()
            declaration = {
                'public_robot_urdf': {'path': 'resource://official_assets/Robots/x5/X5A.urdf',
                                      'sha256': 'a' * 64},
                'public_robot_meshes': {'meshes/base_link.STL': 'b' * 64},
            }
            (root / 'tasks/unit/resources.json').write_text(json.dumps(declaration))
            (root / 'configs/official-runtime-inputs.json').write_text('{"external_files": {}}')
            with patch('rsi.policy_catalog.catalog', return_value={'unit': {}}):
                self.assertEqual(official_records(root), {
                    'Robots/x5/X5A.urdf': 'a' * 64,
                    'Robots/x5/meshes/base_link.STL': 'b' * 64,
                })

    def test_write_is_idempotent_and_preserves_nonmatching_files(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);r=record(b'first')
            put(root,'a/file',b'first',r);put(root,'a/file',b'first',r)
            with self.assertRaises(ValueError):put(root,'a/file',b'other',record(b'other'))
            self.assertEqual((root/'a/file').read_bytes(),b'first')

    def test_corrupt_content_never_publishes(self):
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaises(ValueError):put(folder,'file',b'bad',record(b'good'))
            self.assertFalse((Path(folder)/'file').exists())

    def archive(self,root,members):
        p=root/'source.tar.gz'
        with tarfile.open(p,'w:gz') as tar:
            for name,kind,raw in members:
                info=tarfile.TarInfo(name);info.type=kind;info.size=len(raw);tar.addfile(info,io.BytesIO(raw))
        return p

    def test_archive_deduplication_materializes_exact_outputs(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);p=self.archive(root,[('objects/a',tarfile.REGTYPE,b'value')])
            extract_tar(p,root/'out',{'objects/a':record(b'value',paths=['one','two'])})
            self.assertEqual((root/'out/one').read_bytes(),(root/'out/two').read_bytes())

    def test_unexpected_missing_repeated_and_link_members_are_rejected(self):
        for members in [[],[('other',tarfile.REGTYPE,b'value')],[('objects/a',tarfile.SYMTYPE,b'')],
                        [('objects/a',tarfile.REGTYPE,b'value'),('objects/a',tarfile.REGTYPE,b'value')]]:
            with self.subTest(members=members),tempfile.TemporaryDirectory() as folder:
                root=Path(folder);p=self.archive(root,members)
                with self.assertRaises(ValueError):extract_tar(p,root/'out',{'objects/a':record(b'value',paths=['one'])})

    def test_official_source_links_are_read_but_outputs_stay_regular_and_hash_checked(self):
        from rsi.resource_setup import prepare
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);code=root/'code';assets=root/'Assets';assets.mkdir()
            (root/'public-mesh').write_bytes(b'public')
            (assets/'mesh').symlink_to(root/'public-mesh')
            (code/'configs/models').mkdir(parents=True)
            (code/'configs/models/sam3.json').write_text('{"files": {}}')
            plan=dict(schema='riemann-resource-install-v1',sam_runtime={'target':'sam'},
                      files=[record(b'public',path='geometry/mesh',kind='official',source='resource://official_assets/mesh')])
            (code/'configs/resource-install.json').write_text(json.dumps(plan))
            (code/'configs/resource-roots.json').write_text('{"roots":{"official_assets":{"default":"official_assets"}}}')
            with patch('rsi.resource_setup.official_records',return_value={}):
                result=prepare(root/'out',official_assets=assets,root=code)
            self.assertTrue(result['complete']);self.assertFalse((root/'out/geometry/mesh').is_symlink())
            (root/'public-mesh').write_bytes(b'private')
            with patch('rsi.resource_setup.official_records',return_value={}),self.assertRaises(ValueError):
                prepare(root/'second-out',official_assets=assets,root=code)

    def test_rule_model_preflight_rejects_a_missing_common_weight(self):
        from rsi.model_assets import verify_rule
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);(root/'configs/models').mkdir(parents=True)
            (root/'tasks/classify_objects').mkdir(parents=True)
            declarations={}
            for name in ('sam3','clip','grounding'):
                base=root/name;base.mkdir();(base/'weight').write_bytes(b'weight')
                declarations[name]=dict(path=str(base),files={'weight':record(b'weight')})
            (root/'configs/models/sam3.json').write_text(json.dumps(declarations['sam3']))
            models={'clip_model_manifest.json':declarations['clip'],'grounding_model_manifest.json':declarations['grounding']}
            (root/'tasks/classify_objects/resources.json').write_text(json.dumps({'models':models}))
            self.assertFalse(verify_rule(root=root)['models_loaded'])
            (root/'sam3/weight').unlink()
            with self.assertRaises(ValueError):verify_rule(root=root)

    def test_robot_yaml_identity_accepts_locations_and_rejects_numeric_changes(self):
        from rsi.resource_setup import robot_yaml_digest
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);a=root/'a.yml';b=root/'b.yml'
            a.write_text('robot: /old/Assets/Robots/x5\nlimit: 1.25\n')
            b.write_text('robot: /new/Assets/Robots/x5\nlimit: 1.25\n')
            def reference(value):
                return value.replace('/old/Assets','resource://official_assets').replace('/new/Assets','resource://official_assets')
            with patch('rsi.resource_locations.reference_for',side_effect=reference):
                self.assertEqual(robot_yaml_digest(a),robot_yaml_digest(b))
                b.write_text(b.read_text().replace('1.25','1.26'))
                self.assertNotEqual(robot_yaml_digest(a),robot_yaml_digest(b))

    def test_official_finger_reconstruction_preserves_numeric_vertices(self):
        import numpy as np
        import struct
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);dtype=np.dtype([('normal','<f4',(3,)),('vertices','<f4',(3,3)),('attribute','<u2')])
            triangles=np.zeros(2,dtype=dtype);triangles['vertices']=np.arange(18,dtype=np.float32).reshape(2,3,3)
            raw=b'x'*80+struct.pack('<I',2)+triangles.tobytes();files=[]
            for i in (7,8):p=root/('link'+str(i)+'.STL');p.write_bytes(raw);files.append(p)
            with np.load(io.BytesIO(finger_triangles(files)),allow_pickle=False) as data:
                for i in (7,8):np.testing.assert_array_equal(data['link'+str(i)],triangles['vertices'])


if __name__=='__main__':unittest.main()
