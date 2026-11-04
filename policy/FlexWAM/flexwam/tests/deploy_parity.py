"""Compare independent-camera preprocessing and robot action conversion."""
import argparse
import importlib
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from deploy.robot import get_robot


def snapshot(module='deploy.policy', class_name='FlexWAMPolicy'):
    api=importlib.import_module(module)
    cls=getattr(api,class_name)
    rng=np.random.default_rng(13)
    result={}
    for robot,cameras in [('piper',('global','left_wrist','right_wrist')),
                          ('flexiv',('global','global2','left_wrist'))]:
        policy=cls.__new__(cls)
        policy.robot_type=robot
        policy.camera_order=cameras
        policy.video_size_hw=(192,224)
        policy.model=SimpleNamespace(device='cpu',torch_dtype=torch.float32)
        policy.default_prompt='pick the object'
        policy.prompt_template='Robot task: {task}'
        images={name:rng.integers(0,256,(270,360,3),dtype=np.uint8) for name in cameras}
        obs={get_robot(robot).client_keys['cameras'][name]:images[name] for name in cameras}
        image,preview,received,_=policy._build_input_image(obs)
        result[robot+'/image']=image.numpy()
        result[robot+'/preview']=preview
        for name,value in received.items():result[robot+'/received/'+name]=value
        dim=14 if robot=='piper' else 10
        state=rng.normal(size=dim).astype(np.float32)
        delta=rng.normal(size=(8,dim)).astype(np.float32)
        if robot=='piper':
            absolute=api._integrate_joint_delta_actions(state,delta)
        else:
            absolute=api._integrate_flexiv_pose6d_delta_actions(state,delta)
        result[robot+'/absolute']=absolute
        result[robot+'/state']=policy._extract_state({'observation/state':state})
        result[robot+'/prompt']=np.asarray(policy._extract_prompt({}))
        try:
            policy._build_input_image({'observation/image':images[cameras[0]]})
        except ValueError:
            result[robot+'/reject_composite']=np.asarray(True)
        else:
            raise AssertionError('Composite image unexpectedly accepted.')
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--module',default='deploy.policy')
    p.add_argument('--policy-class',default='FlexWAMPolicy')
    p.add_argument('--save',type=Path)
    p.add_argument('--reference',type=Path)
    args=p.parse_args()
    data=snapshot(args.module,args.policy_class)
    if args.save:np.savez(args.save,**data)
    if args.reference:
        reference=np.load(args.reference)
        assert set(reference.files)==set(data)
        for k,v in data.items():np.testing.assert_array_equal(v,reference[k],err_msg=k)
        print(f'PASS: {len(data)} deployment outputs exactly match reference.')
