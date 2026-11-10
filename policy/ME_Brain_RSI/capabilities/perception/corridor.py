"""Elevated whole-gripper clearance from current RGB bounds and public heights."""
import math,numpy as np
from ..rgb_reader.geometry import TABLE_TOP,PAD_THICKNESS

def plan(report,start,destination,uncertainty):
 selected=min(report['pieces'],key=lambda p:np.linalg.norm(np.array([p['center'][0]/1000-.5,.12-p['center'][1]/1000])-report['selection']['source']))
 radius=float(selected['radius_pixels']/1000);others=[p for p in report['pieces'] if p['id']!=selected['id']]
 obstacle_height=max([max([c['height'] for c in p['candidates']],default=0.) for p in others],default=0.)
 obstacle_top=float(TABLE_TOP+PAD_THICKNESS+obstacle_height)
 selected_height=float(report['selection']['height'])
 transit_height=max(float(start[2]),obstacle_top+selected_height/2+uncertainty+.010)
 circles=[]
 for p in others:
  center=np.array([p['center'][0]/1000-.5,.12-p['center'][1]/1000]);extent=p['radius_pixels']/1000+radius+uncertainty
  circles.append(dict(center=center.tolist(),inflated_radius=float(extent)))
  if np.linalg.norm(center-destination[:2])<=extent:raise ValueError('rgb_descent_destination_blocked')
 skill_choice=[]
 if transit_height>start[2]+1e-5:
  for z in np.linspace(start[2],transit_height,max(1,math.ceil((transit_height-start[2])/.025))+1)[1:]:skill_choice.append(np.r_[start[:2],z])
 distance=float(np.linalg.norm(destination[:2]-start[:2]))
 for f in np.linspace(0,1,max(1,math.ceil(distance/.04))+1)[1:]:skill_choice.append(np.r_[start[:2]+f*(destination[:2]-start[:2]),transit_height])
 for z in np.linspace(transit_height,destination[2],max(1,math.ceil(abs(transit_height-destination[2])/.025))+1)[1:]:skill_choice.append(np.r_[destination[:2],z])
 return skill_choice,dict(method='height_aware_RGB_whole_gripper_clearance',selected_piece_id=selected['id'],obstacle_top_m=obstacle_top,transit_height_m=transit_height,transit_finger_floor_m=obstacle_top+.005,public_height_source='max_supported_CAD_height_plus_public_pad',selected_radius_m=radius,uncertainty_m=float(uncertainty),obstacles=circles,segments=len(skill_choice),length_xy_m=distance)
