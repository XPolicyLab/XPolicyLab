"""Renew texture only through a verified correspondence to the initial RGB surface."""
import cv2
import numpy as np
from .vision_gate import wrist_unproject,appearance_mask

def initialize(tracker,rgb,pose,source_mask):
 tracker.reference_gray=cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY)
 tracker.reference_pose=np.asarray(pose).copy()
 yy,xx=np.mgrid[:rgb.shape[0],:rgb.shape[1]]
 world=wrist_unproject(np.column_stack([xx.ravel(),yy.ravel()]),pose,tracker.z)
 u=((world[:,0]+.5)*1000).reshape(xx.shape).astype(np.float32)
 v=((.12-world[:,1])*1000).reshape(xx.shape).astype(np.float32)
 tracker.source_surface=cv2.remap(np.asarray(source_mask,np.uint8),u,v,cv2.INTER_NEAREST)
 tracker.source_surface=(tracker.source_surface>0).astype(np.uint8)*255
