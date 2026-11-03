"""Retain recovery margins while removing absolute scene-height assumptions."""
import ast


def replace_language_return_floor(text):
    tree=ast.parse(text)
    node=next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=='joint_return')
    lines=text.splitlines(keepends=True);body=''.join(lines[node.lineno-1:node.end_lineno])
    changes=[
        ('        min_tip = 10.0','        support = observed_language_support(self)\n        min_tip = float(\'inf\')'),
        ('            tip = T.translation + T.rotation[:, 0] * 0.15757000232458113\n            z = float(tip[2] - abs(T.rotation[2, 1]) * 0.04)',
         '            z = observed_finger_clearance(T, support)'),
        ('min_tip >= 0.84','min_tip >= 0.075'),
        ('min_finger_z=min_tip','min_finger_clearance=min_tip'),
    ]
    for old,new in changes:
        if body.count(old)!=1:raise ValueError(f'Unexpected return-height pattern: {old}')
        body=body.replace(old,new)
    lines[node.lineno-1:node.end_lineno]=[body];text=''.join(lines)
    old="plan_open_retreat(self.branch, self.obs['state'], arm)"
    if text.count(old)!=1:raise ValueError('Unexpected recovery call')
    text=text.replace(old,"plan_open_retreat(self.branch, self.obs['state'], arm, support=observed_language_support(self))")
    marker='import cv2\n'
    if text.count(marker)!=1:raise ValueError('Expected model import anchor')
    text=text.replace(marker,marker+'from physicalrsi_observed_geometry.finger_support_clearance import observed_finger_clearance\n')
    compile(text,'observed return floor','exec');return text


def replace_language_open_recovery(text):
    changes=[
        ('def plan_open_retreat(branch, state, arm, count=12):','def plan_open_retreat(branch, state, arm, count=12, *, support):'),
        ('target_z = max(1.025, float(pose[2]) + .08)','target_z = float(pose[2]) + .08'),
        (' or target_z > 1.16',''),
        ('minimum = 10.','minimum = float(\'inf\')'),
        ('            tip = transform.translation + transform.rotation[:, 0] * .15757000232458113\n            minimum = min(minimum, float(tip[2] - abs(transform.rotation[2, 1]) * .04))',
         '            minimum = min(minimum, observed_finger_clearance(transform, support))'),
        ('minimum >= .768','minimum >= .003'),
        ("'min_finger_z': minimum","'min_finger_clearance': minimum"),
    ]
    for old,new in changes:
        if text.count(old)!=1:raise ValueError(f'Unexpected open-recovery pattern: {old}')
        text=text.replace(old,new)
    text='from physicalrsi_observed_geometry.finger_support_clearance import observed_finger_clearance\n'+text
    compile(text,'observed recovery clearance','exec');return text
