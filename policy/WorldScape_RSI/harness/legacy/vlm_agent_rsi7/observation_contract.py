"""Bind images and decisions to one immutable, measured observation."""
import hashlib,json,re
from pathlib import Path

IDENTITY=('run_id','observation_id','decision','native_step')

def build(directory,run_id,observation_id,current_message_index,current_text):
    directory=Path(directory)
    turn=re.search(r'TURN (\d+)\.',current_text)
    step=re.search(r'steps used: (\d+)',current_text)
    if turn is None or step is None:raise ValueError('missing observation identity')
    images=[]
    for f in sorted(directory.glob(f'm{current_message_index:03d}_p*.jpg')):
        data=f.read_bytes();images.append({'name':f.name,'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()})
    if len(images)!=3:raise ValueError('expected three current cameras')
    tcp={arm.lower():[float(v) for v in values.split(',')] for arm,values in
         re.findall(r'(LEFT|RIGHT) gripper position \(fingertip centre\) = \(([^)]+)\)',current_text)}
    if len(tcp)!=2:raise ValueError('missing current TCP observation')
    last=current_text.split('RESULT OF YOUR LAST COMMANDS:',1)[-1].split('CURRENT STATE:',1)[0]
    return dict(run_id=run_id,observation_id=observation_id,decision=int(turn[1]),native_step=int(step[1]),
                images=images,current_tcp_cm=tcp,recovery_acceptance_required='FAILED' in last)

def verify_files(directory,manifest,expected):
    for k in IDENTITY:
        if manifest.get(k)!=expected.get(k):raise ValueError('stale identity: '+k)
    for row in manifest['images']:
        name=row['name']
        if Path(name).name!=name:raise ValueError('image path escape')
        blob=(Path(directory)/name).read_bytes()
        if len(blob)!=row['bytes'] or hashlib.sha256(blob).hexdigest()!=row['sha256']:
            raise ValueError('image content mismatch: '+name)
    return {k:manifest[k] for k in IDENTITY}|{'images':manifest['images']}

def verify_packet(packet,manifest):
    receipt=packet.get('observation_receipt') or {}
    if packet.get('request_id')!=manifest['observation_id']:raise ValueError('stale request')
    for k in IDENTITY:
        if receipt.get(k)!=manifest[k]:raise ValueError('missing/stale observation receipt: '+k)
    if receipt.get('images')!=manifest['images']:raise ValueError('unverified image hashes')
    response=packet.get('response') or {}
    if manifest['recovery_acceptance_required'] and response.get('abort') is not True:
        recovery=response.get('recovery_acceptance') or {}
        for k in ('observation_id','native_step'):
            if recovery.get(k)!=manifest[k]:raise ValueError('recovery requires current observation')
        if type(recovery.get('object_retained')) is not bool:raise ValueError('retention must be explicitly assessed')
        if recovery.get('current_tcp_cm')!=manifest['current_tcp_cm']:raise ValueError('recovery pose mismatch')
        if not isinstance(recovery.get('evidence'),str) or len(recovery['evidence'].strip())<10:
            raise ValueError('missing fresh visual recovery evidence')

if __name__=='__main__':
    import sys
    p=Path(sys.argv[1]);m=json.loads((p/'observation.json').read_text())
    print(json.dumps(verify_files(p,m,json.loads(sys.argv[2]))))
