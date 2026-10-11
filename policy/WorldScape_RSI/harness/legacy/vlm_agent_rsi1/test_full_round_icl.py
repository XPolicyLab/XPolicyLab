import unittest,tempfile,json,hashlib
from pathlib import Path
from full_round_icl import load_messages
class T(unittest.TestCase):
 def setUp(self):
  self.t=tempfile.TemporaryDirectory();self.addCleanup(self.t.cleanup);self.p=Path(self.t.name)
  blob=b'test-only-image-fixture';(self.p/'image.png').write_bytes(blob)
  images=[dict(camera=c,path='image.png',sha256=hashlib.sha256(blob).hexdigest(),mime_type='image/png',original_resolution_preserved=True) for c in ('head','left','right')]
  self.r=dict(decision_id=1,control_step=0,state={'joints':[1]},raw_model_response='full response',original_model_fields={'memory':'x'*50000},submitted_commands_or_flow=['actual'],parsed_executed_actions=['actual'],complete_feedback={'ok':True},available_cameras=['head','left','right'],images=images)
  raw=json.dumps(self.r).encode();(self.p/'r.json').write_bytes(raw)
  term=dict(kind='terminal_observation',is_decision=False,control_step=10,available_cameras=['head','left','right'],images=images);b=json.dumps(term).encode();(self.p/'t.json').write_bytes(b)
  self.m=dict(schema='conveyor.full_round_icl.v2',status='complete',decision_total=1,complete_decisions_exported=1,rounds=[dict(record_path='r.json',record_sha256=hashlib.sha256(raw).hexdigest())],terminal_observation=dict(record_path='t.json',record_sha256=hashlib.sha256(b).hexdigest()));self.write()
 def write(self):(self.p/'manifest.json').write_text(json.dumps(self.m))
 def test_complete_untruncated(self):
  messages,proof=load_messages(self.p);self.assertEqual(proof['images'],6);self.assertIn('x'*50000,messages[0]['content'][0]['text']);self.assertEqual(len(messages),2);self.assertEqual(proof['status'],'constructed_not_proof_of_model_display')
 def test_incomplete_rejected(self):
  self.m['status']='incomplete';self.write()
  with self.assertRaisesRegex(ValueError,'incomplete'):load_messages(self.p)
 def test_limits_fail_not_sample(self):
  with self.assertRaisesRegex(ValueError,'image limit'):load_messages(self.p,max_images=5)
  with self.assertRaisesRegex(ValueError,'content limit'):load_messages(self.p,max_content_bytes=1)
 def test_hash_protects_original(self):
  (self.p/'r.json').write_text('{}')
  with self.assertRaisesRegex(ValueError,'hash mismatch'):load_messages(self.p)
if __name__=='__main__':unittest.main()
