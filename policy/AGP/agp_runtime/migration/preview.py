"""CPU robot-only projection draft. No simulator or scene-state dependencies."""
import hashlib
import json
import platform
from importlib.metadata import version
from pathlib import Path
import time
import xml.etree.ElementTree as ET
import numpy as np
from scipy.spatial.transform import Rotation

HERE = Path(__file__).resolve().parent
ASSETS = HERE / 'preview_assets'

def digest(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def validate_request(request):
    if not isinstance(request,dict) or set(request)!={'camera','samples'}: raise ValueError('Expected camera and samples only')
    if request['camera'] not in {'top','left_wrist','right_wrist'}: raise ValueError('Unknown camera')
    samples=request['samples']
    if not isinstance(samples,list) or not 1<=len(samples)<=3 or any(type(i) is not int or i<0 for i in samples) or len(set(samples))!=len(samples): raise ValueError('Expected 1..3 unique nonnegative strict integer samples')
    return {'camera':request['camera'],'samples':sorted(samples)}

def bind_candidate(state,plans,request):
    request=validate_request(request)
    if set(state)!={'left','right'} or set(plans)-set(state) or not plans: raise ValueError('Requires both measured arms')
    if any(p['command']=='gripper' for p in plans.values()): raise ValueError('Stage0 gripper preview unsupported')
    n=max(p['run_samples'] for p in plans.values())
    if max(request['samples'])>n: raise ValueError('Sample exceeds scheduled stage0 prefix')
    prefix={}
    for arm,p in plans.items():
        q=np.asarray(p['points'],float)[:p['run_samples']]
        if q.shape!=(p['run_samples'],6) or not np.isfinite(q).all(): raise ValueError('Invalid prefix')
        prefix[arm]=q.tolist()
    samples=[]
    for k in request['samples']:
        arms={}
        for arm,s in state.items():
            q=s['joints'] if k==0 or arm not in prefix else prefix[arm][min(k,len(prefix[arm]))-1]
            status='captured_start' if k==0 else ('held_start' if arm not in prefix else ('held_endpoint' if k>len(prefix[arm]) else 'predicted'))
            arms[arm]={'joints':list(q),'gripper_measured_m':list(s['gripper_measured_m']),'status':status}
        samples.append({'sample_index':k,'arms':arms})
    return {'stage_index':0,'stage_samples':n,'sample_indices':request['samples'],'samples':samples,'prefix_joints':prefix,'captured_start':{a:{'joints':list(v['joints']),'gripper_measured_m':list(v['gripper_measured_m'])} for a,v in state.items()}}

def rigid(T):
    T=np.asarray(T,float)
    if T.shape!=(4,4) or not np.isfinite(T).all() or not np.allclose(T[3],[0,0,0,1]) or not np.allclose(T[:3,:3].T@T[:3,:3],np.eye(3),atol=1e-6) or not np.isclose(np.linalg.det(T[:3,:3]),1): raise ValueError('Invalid rigid transform')
    return T

def origin(node):
    T=np.eye(4)
    if node is not None:
        xyz=np.fromstring(node.get('xyz','0 0 0'),sep=' '); rpy=np.fromstring(node.get('rpy','0 0 0'),sep=' ')
        if xyz.shape!=(3,) or rpy.shape!=(3,) or not np.isfinite(np.r_[xyz,rpy]).all(): raise ValueError('Unsupported origin')
        T[:3,3]=xyz; T[:3,:3]=Rotation.from_euler('xyz',rpy).as_matrix()
    return T

class Robot:
    def __init__(self,assets=ASSETS):
        import trimesh
        assets=Path(assets).resolve()
        manifest=json.loads((assets/'manifest.json').read_text())
        expected={'X5A.urdf'}|{f'meshes/{n}.STL' for n in ['base_link']+[f'link{i}' for i in range(1,9)]}|{'meshes/camera.glb','meshes/camera_base.glb'}
        if manifest.get('model')!='X5A' or set(manifest['sha256'])!=expected: raise ValueError('Unsupported asset allowlist')
        inventory={str(p.relative_to(assets)) for p in assets.rglob('*') if p.is_file() and p.name!='manifest.json'}
        if inventory!=expected or any(p.is_symlink() for p in assets.rglob('*')): raise ValueError('Robot asset inventory drift')
        for name,sha in manifest['sha256'].items():
            p=(assets/name).resolve()
            if not p.is_relative_to(assets) or digest(p)!=sha: raise ValueError('Robot asset provenance mismatch')
        self.manifest=manifest; self.meshes={}; self.offsets={}; self.joints=[]
        root=ET.parse(assets/'X5A.urdf').getroot()
        names=[l.get('name') for l in root.findall('link')]
        if len(names)!=len(set(names)): raise ValueError('Duplicate links')
        for link in root.findall('link'):
            vs=link.findall('visual'); cs=link.findall('collision')
            if len(vs)!=1 or len(cs)>1: raise ValueError('Unsupported geometry multiplicity')
            for item in vs+cs:
                g=item.find('geometry'); mesh=g.find('mesh')
                if mesh is None or len(g)!=1 or mesh.get('filename') not in expected or not np.array_equal(np.fromstring(mesh.get('scale','1 1 1'),sep=' '),[1,1,1]): raise ValueError('Unsupported mesh or scale')
                origin(item.find('origin'))
            v=vs[0]; name=link.get('name')
            m=trimesh.load(assets/v.find('geometry/mesh').get('filename'),process=False)
            if isinstance(m,trimesh.Scene): m=m.to_geometry()
            if not isinstance(m,trimesh.Trimesh) or not len(m.faces) or not np.isfinite(m.vertices).all(): raise ValueError('Unsupported mesh resolution')
            self.meshes[name]=m; self.offsets[name]=origin(v.find('origin'))
        reached={'base_link'}; actuated=[]
        for j in root.findall('joint'):
            kind=j.get('type'); name=j.get('name'); parent=j.find('parent').get('link'); child=j.find('child').get('link')
            if kind not in {'fixed','revolute','prismatic'} or j.find('mimic') is not None or parent not in reached or child in reached or child not in names: raise ValueError('Unsupported joint tree')
            axis=np.fromstring(j.find('axis').get('xyz'),sep=' ') if j.find('axis') is not None else np.zeros(3)
            if kind!='fixed' and (axis.shape!=(3,) or not np.isfinite(axis).all() or not np.isclose(np.linalg.norm(axis),1)): raise ValueError('Unsupported joint axis')
            if kind!='fixed': actuated.append(name)
            if kind!=('prismatic' if name in {'joint7','joint8'} else ('revolute' if name in {f'joint{i}' for i in range(1,7)} else 'fixed')): raise ValueError('Unsupported joint mapping')
            reached.add(child); self.joints.append((name,kind,parent,child,origin(j.find('origin')),axis))
        if reached!=set(names) or actuated!=[f'joint{i}' for i in range(1,9)]: raise ValueError('Unsupported model joints')
    def fk(self,q,fingers):
        values=np.r_[q,fingers]
        if values.shape!=(8,) or not np.isfinite(values).all(): raise ValueError('Invalid robot coordinates')
        values=dict(zip([f'joint{i}' for i in range(1,9)],values)); out={'base_link':np.eye(4)}
        for name,kind,parent,child,T,axis in self.joints:
            move=np.eye(4)
            if kind=='prismatic': move[:3,3]=axis*values[name]
            elif kind=='revolute': move[:3,:3]=Rotation.from_rotvec(axis*values[name]).as_matrix()
            out[child]=out[parent]@T@move
        return out

def render_overlay(rgb,model,sample,cal):
    import cv2
    from migration.preview_raster import raster
    K=np.asarray(cal['K'],float)
    if K.shape!=(3,3) or not np.isfinite(K).all() or min(K[0,0],K[1,1])<=0 or K[0,1]!=0 or K[1,0]!=0 or not np.array_equal(K[2],[0,0,1]): raise ValueError('Invalid optical pinhole calibration')
    out=rgb.copy(); h,w=rgb.shape[:2]; yy,xx=np.indices((h,w))
    for arm,values in sample['arms'].items():
        T=np.linalg.inv(rigid(cal['T_base_optical_by_arm'][arm])); fk=model.fk(values['joints'],values['gripper_measured_m'])
        for link,m in model.meshes.items():
            transform=T@fk[link]@model.offsets[link]; vertices=m.vertices@transform[:3,:3].T+transform[:3,3]
            depth=np.full((h,w),np.inf); labels=np.full((h,w),-1,np.int32)
            raster(vertices,np.asarray(m.faces),K,depth,labels,0)
            contours,_=cv2.findContours((labels>=0).astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
            edges=np.zeros((h,w),np.uint8); cv2.drawContours(edges,contours,-1,1,1)
            dash=((xx+yy)%8<4) if arm=='left' else ((xx-yy)%10<5)
            out[(edges>0)&dash]=(255,210,80) if arm=='left' else (220,130,255)
    lines=['HYPOTHETICAL robot outlines; includes hidden geometry', 'Fixed captured viewpoint; fingers: captured-width reference', 'Visual meshes, not collision geometry; no clearance/grasp claim',f"Stage 0 sample {sample['sample_index']}; step independently replans"]
    for i,line in enumerate(lines):
        cv2.putText(out,line,(5,16+i*15),cv2.FONT_HERSHEY_SIMPLEX,.34,(0,0,0),3,cv2.LINE_AA)
        cv2.putText(out,line,(5,16+i*15),cv2.FONT_HERSHEY_SIMPLEX,.34,(255,255,255),1,cv2.LINE_AA)
    return out

def make_preview(session,request,source,candidate):
    import cv2
    request=validate_request(request); session=Path(session).resolve()
    if not isinstance(source,dict) or not str(source.get('capture','')).isdigit(): raise ValueError('Invalid capture identity')
    capture=session/'captures'/str(source['capture']); camera=request['camera']
    for arm,start in candidate['captured_start'].items():
        saved=source['state']['arms'][arm]
        if not np.allclose(start['joints'],saved['joints_rad'],rtol=0,atol=1e-8) or not np.allclose(start['gripper_measured_m'],saved['gripper_joints_m'],rtol=0,atol=1e-8): raise ValueError('Capture start state differs')
    paths={}
    for key,name in [('rgb_path',camera+'.png'),('depth_path',camera+'_depth.npy'),('calibration_path','calibration.json')]:
        p=Path(source[key]).resolve()
        if p!=capture/name or not p.is_relative_to(session): raise ValueError('Capture path outside trusted identity')
        paths[key]=p
    rgb=cv2.imread(str(paths['rgb_path']))
    if rgb is None: raise ValueError('Missing captured RGB')
    cal=json.loads(paths['calibration_path'].read_text())[camera]
    model=Robot(); source_manifest={'capture_files':{k:digest(p) for k,p in paths.items()},'robot_assets':model.manifest,'candidate':candidate,'captured_state':source['state'],'renderer_sources':{p.name:digest(p) for p in [Path(__file__),HERE/'preview_raster.py']},'dependencies':dict({n:version(n) for n in ['numpy','trimesh','scipy','numba']},opencv=cv2.__version__),'python':platform.python_version()}
    # Bind the selected viewpoint and original capture identity before naming
    # artifacts: different cameras/captures can contain identical RGB/depth bytes.
    source_manifest['schema_version'] = 2
    source_manifest['source_identity'] = dict(camera=camera, **{
        k:source[k] for k in ['observation_id','revision','capture',
            'captured_monotonic_ns','camera_time_monotonic_ns','sim_time_s']})
    encoded=json.dumps(source_manifest,sort_keys=True,allow_nan=False).encode(); key=hashlib.sha256(encoded).hexdigest()
    directory=session/'candidate_previews'/key; directory.mkdir(parents=True,exist_ok=True)
    images=[{'kind':'captured_rgb','label':f'{camera}: original captured RGB','path':str(paths['rgb_path'])}]
    for sample in candidate['samples']:
        out=directory/f"sample{sample['sample_index']}.png"
        if not cv2.imwrite(str(out),render_overlay(rgb,model,sample,cal)): raise ValueError('PNG write failure')
        images.append({'kind':'derived_hypothetical_robot_projection','label':f"{camera}: derived hypothetical robot projection sample {sample['sample_index']}",'path':str(out)})
    source_manifest['output_sha256']={im['path']:digest(im['path']) for im in images}
    encoded=json.dumps(source_manifest,sort_keys=True,allow_nan=False).encode()
    (directory/'manifest.json').write_bytes(encoded)
    metadata={k:source[k] for k in ['observation_id','revision','capture','captured_monotonic_ns','camera_time_monotonic_ns','sim_time_s']}
    metadata.update(status='available',camera=camera,created_monotonic_ns=time.monotonic_ns(),stage_index=0,stage_samples=candidate['stage_samples'],sample_indices=candidate['sample_indices'],samples=candidate['samples'],source_manifest=str(directory/'manifest.json'),source_manifest_sha256=hashlib.sha256(encoded).hexdigest(),scope='Captured viewpoint; captured finger widths reference only. Includes hidden robot visual geometry; no collision, grasp, visibility, execution or success guarantee.')
    return {'preview':metadata,'images':images}
