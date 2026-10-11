"""Local binary network, not the TypeSafe Jev product. No bundled trained weights."""
import hashlib
import re
import numpy as np
import torch
from torch import nn
from PIL import Image

FEATURE_VERSION = "instruction-hash256-rgb8x8x3-state28-mask6-v1"
CLASSES = ["wam", "astra"]
CAMERAS = ("cam_head", "cam_left_wrist", "cam_right_wrist")
STATE_FIELDS = [(a+"_"+k,n) for a in ("left","right") for k,n in
                (("arm_joint_state",6),("ee_joint_state",1),("ee_pose",7))]
INPUT_DIM = 256 + 3*(8*8*3+1) + 2*(6+1+7) + 6

def features(obs):
    # Whitelist only instruction, rendered RGB, measured robot state. No ID, score, seed, layout.
    text = obs.get("instruction")
    if isinstance(text,(list,tuple)):
        if len(text)!=1: raise ValueError("Single environment required")
        text=text[0]
    if not isinstance(text,str) or not text.strip(): raise ValueError("Missing instruction")
    words=np.zeros(256,dtype=np.float32)
    for token in re.findall(r"\w+|[^\s\w]",text.lower()):
        h=hashlib.sha256(token.encode()).digest(); words[int.from_bytes(h[:4],"little")%256]+=1 if h[4]%2 else -1
    words/=max(float(np.linalg.norm(words)),1.)
    parts=[words]; present=0
    for name in CAMERAS:
        image=obs.get("vision",{}).get(name,{}).get("color")
        if image is None: parts += [np.zeros(192,dtype=np.float32),np.array([0.])];continue
        arr=np.asarray(image)
        if arr.ndim==4 and arr.shape[0]==1: arr=arr[0]
        if arr.ndim!=3 or arr.shape[2]!=3 or arr.dtype!=np.uint8:
            raise ValueError("RGB observations must be HxWx3 uint8")
        parts += [np.asarray(Image.fromarray(arr).resize((8,8)),dtype=np.float32).reshape(-1)/255.,np.array([1.])];present+=1
    if not present: raise ValueError("Missing RGB views")
    state=obs.get("state",{})
    for key,n in STATE_FIELDS:
        if key not in state: parts += [np.zeros(n,dtype=np.float32),np.array([0.])];continue
        a=np.asarray(state[key],dtype=np.float32).reshape(-1)
        if a.size!=n or not np.isfinite(a).all(): raise ValueError("Invalid measured state: "+key)
        parts += [np.tanh(a),np.array([1.])]
    x=np.concatenate(parts).astype(np.float32)
    if x.size!=INPUT_DIM or not np.isfinite(x).all():raise ValueError("Invalid features")
    return x

class BinaryJEV(nn.Module):
    def __init__(self):
        super().__init__();self.layers=nn.Sequential(nn.Linear(INPUT_DIM,128),nn.ReLU(),nn.Linear(128,32),nn.ReLU(),nn.Linear(32,2))
    def forward(self,x):return self.layers(x)

class EpisodeRouter:
    def __init__(self,mode="learned",checkpoint=None):
        if mode not in ("learned","astra","wam"):raise ValueError("Unknown route mode")
        self.mode=mode;self.decision=None;self.net=None;self.digest=None
        if mode=="learned":
            if not checkpoint:raise ValueError("JEV weights required; training is a separate step")
            from pathlib import Path
            p=Path(checkpoint)
            if not p.is_file():raise FileNotFoundError("JEV checkpoint missing: "+str(p))
            raw=p.read_bytes();self.digest=hashlib.sha256(raw).hexdigest()
            ckpt=torch.load(p,map_location="cpu",weights_only=True)
            if (ckpt.get("feature_version")!=FEATURE_VERSION or ckpt.get("classes")!=CLASSES
                or ckpt.get("trained") is not True or not ckpt.get("training_manifest_sha256")):
                raise ValueError("Untrained or incompatible JEV checkpoint")
            self.net=BinaryJEV();self.net.load_state_dict(ckpt["state_dict"],strict=True);self.net.eval()
            if not all(torch.isfinite(p).all() for p in self.net.parameters()):raise ValueError("Non-finite weights")
    def reset(self):self.decision=None
    def select(self,obs):
        if self.decision is None:
            probabilities=None
            if self.net is not None:
                with torch.inference_mode():probs=self.net(torch.from_numpy(features(obs))[None]).softmax(-1)[0]
                if not torch.isfinite(probs).all():raise ValueError("Non-finite routing prediction")
                selected=CLASSES[int(probs.argmax())];probabilities=probs.tolist()
            else:selected=self.mode
            self.decision={"executor":selected,"selection_scope":"episode","source":"local_binary_jev" if self.net is not None else "explicit_ablation",
                           "probabilities":probabilities,"classes":CLASSES,"checkpoint_sha256":self.digest,"feature_version":FEATURE_VERSION}
        return dict(self.decision)
