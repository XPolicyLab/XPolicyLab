"""Nominal X5 pad geometry, derived exclusively from the robot model."""
import json
from pathlib import Path
import numpy as np

PAD_ZERO=json.loads(Path(__file__).with_name('x5_gripper_geometry.json').read_text())['pad_vertices_tcp_at_q0_m']


def gripper_geometry(tcp_base, measured):
    vertices=[]
    local=[]
    for index,key in enumerate(('7','8')):
        p=np.asarray(PAD_ZERO[key],dtype=float).copy()
        p[:,0]+=(1 if index==0 else -1)*float(measured[index])
        local.append(p.round(6).tolist())
        vertices.append((p @ np.asarray(tcp_base)[:3,:3].T + np.asarray(tcp_base)[:3,3]).round(6).tolist())
    gap=float(measured[0]+measured[1]+PAD_ZERO['7'][0][0]-PAD_ZERO['8'][0][0])
    return {'nominal_pad_gap_m':round(gap,6),'nominal_pad_vertices_base_m':vertices,
            'nominal_pad_vertices_tcp_m':local,
            'meaning':'Robot model geometry at measured pose; nominal pad surfaces, not proof of object contact or possession.'}
