"""Load a frozen policy copy with shared helpers from the verified official XPolicyLab."""
import os
from pathlib import Path
import runpy
import sys
policy = Path(__file__).resolve().parents[1]

os.environ['OPENCV_FOR_THREADS_NUM']='1'
import cv2
cv2.setNumThreads(1)
candidates=[Path(os.environ.get('Q0_PREVIEW_XPL_ROOT', str(policy.parents[2]))), policy.parents[2]]
shared=next((p for p in candidates if (p/'XPolicyLab/utils/process_data.py').exists()),None)
if shared is None:raise RuntimeError('Set Q0_PREVIEW_XPL_ROOT to the parent of the XPolicyLab checkout')
sys.path.insert(0,str(shared))
import XPolicyLab.policy
XPolicyLab.policy.__path__.insert(0, str(policy.parent))
mode = sys.argv.pop(1)
if mode not in ('runner','smoke_agent'): raise ValueError('Unknown harness mode')
runpy.run_module(f'XPolicyLab.policy.{policy.name}.harness.{mode}',run_name='__main__')
